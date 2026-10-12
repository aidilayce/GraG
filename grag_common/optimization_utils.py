"""Geometry, Gaussian and Sum-of-Gaussians utilities shared by Stage 3 and evaluation."""
from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import numpy as np
import pytorch3d.transforms as p3d_t
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_axis_angle
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

from grag_common.project_paths import CONTACT_ZONES_PATH, append_project_paths

append_project_paths()
from third_party.SAM3DO.sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from third_party.SAM3DO.sam3d_objects.utils.visualization import SceneVisualizer

try:
    import wandb
except ImportError:
    wandb = None

opencv_to_pytorch3d = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SH_C0 = 0.28209479177387814

with open(CONTACT_ZONES_PATH, "rb") as f:
    contact_zones = pickle.load(f)["contact_zones"]
contact_idx = np.array([v for zone in contact_zones.values() for v in zone])


def SH2RGB(sh):
    return sh * SH_C0 + 0.5


class MetricLogger:
    """Append scalar losses to ``metrics/stage3.csv`` and optionally to wandb."""

    def __init__(self, out_dir, use_wandb=False, wb_project="hoi-gaussians", wb_run_name=None):
        self.path = Path(out_dir) / "metrics" / "stage3.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = None
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=wb_project, name=wb_run_name, config={})

    def log(self, step: int, stage: str, metrics: dict):
        if self.header is None:
            self.header = ["step", *metrics]
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.header)
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([step] + [float(metrics.get(k, 0.0)) for k in self.header[1:]])
        if self.use_wandb:
            wandb.log({f"{stage}/{k}": v for k, v in metrics.items()}, step=step)

def load_rgb(path, H=None, W=None):
    im = Image.open(path).convert('RGB')
    if H is not None and W is not None:
        im = im.resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.array(im)).float().permute(2,0,1)/255.0


def load_mask(path, H=None, W=None):
    m = Image.open(path).convert('L')
    if H is not None and W is not None:
        m = m.resize((W, H), Image.NEAREST)
    mask = np.array(m)
    # Older HO3D exports stored the object and hand in one label image:
    # background=0, object=50, hand=150.  Treating that image as a generic
    # binary mask either selects the hand (>127) or merges both subjects (>0).
    # Current release masks are binary {0, 255}, so keep their behavior intact.
    values = set(np.unique(mask).tolist())
    if 50 in values and values.issubset({0, 50, 150}):
        mask = mask == 50
    else:
        mask = mask > 127
    return torch.from_numpy(mask).float().unsqueeze(0)


def tight_aabb(xyz, pad_ratio=0.01):
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    size = maxs - mins
    eps = (size.abs().max() * 1e-6).clamp(min=1e-9)
    eps = eps.item()
    size = torch.where(size < eps, torch.full_like(size, eps), size)
    pad = size * pad_ratio
    mins = mins - pad
    size = size + 2*pad
    return torch.cat([mins, size], dim=0)


def normals_to_quaternions(normals, up_vector=None):
    """
    Converts a tensor of normal vectors (N, 3) to wxyz quaternions (N, 4).
    Assumes the "up" direction of the unrotated Gaussian is (0, 0, 1).
    """
    if up_vector is None:
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
    
    # Ensure normals are normalized
    normals = F.normalize(normals, p=2, dim=-1)
    
    # Create a batched 'up' vector
    up = up_vector.expand_as(normals)
    
    # Compute rotation axis (cross product)
    axis = F.normalize(torch.cross(up, normals), p=2, dim=-1)
    
    # Compute rotation angle (dot product)
    dot_prod = torch.sum(up * normals, dim=-1)
    angle = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    
    # Convert axis-angle to quaternion (wxyz)
    half_angle = angle * 0.5
    w = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)
    xyz = axis * sin_half.unsqueeze(-1)
    
    rots_wxyz = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    
    # --- Handle Edge Cases ---
    
    # Case 1: Normals are already [0, 0, 1] (dot_prod ~ 1)
    # `axis` will be NaN, so we must fix it.
    parallel = dot_prod > 0.999999
    rots_wxyz[parallel] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype)
    
    # Case 2: Normals are [0, 0, -1] (dot_prod ~ -1)
    # `axis` will be NaN. We use a 180-deg rotation around X-axis.
    anti_parallel = dot_prod < -0.999999
    rots_wxyz[anti_parallel] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype) # [w, x, y, z]

    # Re-normalize to be safe
    return F.normalize(rots_wxyz, p=2, dim=-1)

def pose_gaussians(gaussian, rotation, translation, scale, in_place=False):
    minimum_kernel_size = float("inf")
    if translation.ndim == 1:
        translation = translation.unsqueeze(0)
    if rotation.ndim == 1:
        rotation = rotation.unsqueeze(0)
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    if scale.shape[1] != 3:
        scale = scale.repeat(1, 3)

    device = rotation.device

    PC = SceneVisualizer.object_pointcloud(
            points_local=gaussian.get_xyz.unsqueeze(0).to(device),
            quat_l2c=rotation,
            trans_l2c=translation,
            scale_l2c=scale,
        )
    xyz_posed = PC.points_list()[0]
    rot_posed = quaternion_multiply(
        quaternion_invert(rotation),
        gaussian.get_rotation.to(device)
    )
    gaus_scale = gaussian.get_scaling.to(device)
    adjusted_scale = gaus_scale * scale
    # assert (scale[0, 0].item() == scale[0, 1].item() == scale[0, 2].item())
    new_min_kernel_size = gaussian.mininum_kernel_size * scale[0, 0].item()
    limit = torch.tensor(new_min_kernel_size * 1.1, device=adjusted_scale.device)
    final_scale = torch.maximum(adjusted_scale, limit)
    
    g_posed = Gaussian(aabb=tight_aabb(xyz_posed), sh_degree=0)
    g_posed.from_xyz(xyz_posed)
    g_posed.from_rotation(rot_posed)
    g_posed.from_scaling(final_scale)
    g_posed._features_dc = gaussian._features_dc.to(device)
    g_posed._features_rest = gaussian._features_rest
    g_posed._opacity = gaussian._opacity.to(device)

    minimum_kernel_size = min(minimum_kernel_size, new_min_kernel_size)
    g_posed.mininum_kernel_size = minimum_kernel_size

    return g_posed

def initialize_gaussians_from_mesh_w_normals(mesh, samples_per_vertex=10, init_scale=0.1, sh_degree=0):
    """
    Correctly initializes a dense Gaussian model from the vertices of a mesh,
    including rotation based on surface normals.
    Handles all trimesh versions by using mesh.nearest.
    """
    
    # 1. Sample dense points (without 'return_index')
    vertices_np, _ = trimesh.sample.sample_surface(mesh, count=mesh.vertices.shape[0] * samples_per_vertex)
    vertices = torch.from_numpy(vertices_np).float().to(device)
    print(f"Initializing hand with {len(vertices)} dense Gaussians.")

    _, _, face_indices = mesh.nearest.on_surface(vertices_np)
    
    normals_np = mesh.face_normals[face_indices]
    normals = torch.from_numpy(normals_np).float().to(device)

    rots = normals_to_quaternions(normals) # (N, 4) wxyz format

    from sklearn.neighbors import NearestNeighbors
    k = 4 # 8–24 works well
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(vertices_np)
    distances, _ = nn.kneighbors(vertices_np)

    # distances[:,0] == 0 (self), so use 1..k-1
    local = distances[:, 1:]

    # robust local spacing (median is much stabler than mean)
    base = np.median(local, axis=1)

    _, idx = nn.kneighbors(vertices_np)
    nbr_idx = idx[:, 1:]   # (N, k-1)

    normals_np_dense = normals_np  # face normals sampled for each vertex sample
    nbr_normals = normals_np_dense[nbr_idx]                 # (N,k-1,3)
    n0 = normals_np_dense[:, None, :]                       # (N,1,3)
    cos = np.clip((nbr_normals * n0).sum(-1), -1.0, 1.0)    # (N,k-1)
    var = 1.0 - np.median(cos, axis=1)                      # 0 = flat, bigger = more variation

    shrink = np.clip(1.0 - 0.8 * var, 0.6, 1.0)

    avg_dist = base * shrink * init_scale
    scales = torch.from_numpy(avg_dist).float().to(device).unsqueeze(1).repeat(1, 3)

    # ==== end old scales one =====

    opacities = torch.full((vertices.shape[0], 1), 0.7, device=device) #0.1 - 0.99
    features_dc = torch.full((vertices.shape[0], 1, 3), 0.5, device=device)
    features_rest = None if sh_degree == 0 else torch.zeros((vertices.shape[0], 3, (sh_degree + 1)**2 - 1), device=device)

    new_aabb = tight_aabb(vertices)
    init_params = {
        'aabb': new_aabb.cpu().tolist(),
        'sh_degree': sh_degree,
        'mininum_kernel_size': 0.0,
        'scaling_bias': 0.01,
        'opacity_bias': 0.1,
        'scaling_activation': 'softplus',
    }
    gauss = Gaussian(**init_params)

    gauss.from_xyz(vertices)
    gauss.from_scaling(scales)
    gauss.from_rotation(rots) 
    gauss.from_opacity(opacities)
    gauss._features_dc = features_dc
    gauss._features_rest = features_rest

    return gauss

def create_high_res_hand_template(mano_model, skinning_grid, subdivision_level=2):
    """
    Build a high-res canonical mesh + LBS weights. We don't reshape posedirs here.
    """
    device = mano_model.v_template.device
    verts_canonical = mano_model.v_template.cpu().numpy()
    faces_canonical = mano_model.faces

    mesh = trimesh.Trimesh(vertices=verts_canonical, faces=faces_canonical)
    for _ in range(subdivision_level):
        mesh = mesh.subdivide() 

    verts_high_res = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    print(f"Subdivided hand mesh: {len(verts_canonical)} -> {len(verts_high_res)} vertices.")

    return {
        'vertices': verts_high_res,                             # (Vd,3)
        'faces': torch.tensor(mesh.faces, dtype=torch.long, device=device)
    }

@torch.no_grad()
def upsample_mano_weights_to_highres(verts_high, mano_verts, mano_weights, mano_faces=None, neighbors=4): #30
    """
    Upsample MANO skinning weights from 778 verts to arbitrary high-res mesh.
    Args:
      verts_high:   (Vh,3) np or torch
      mano_verts:   (778,3)
      mano_weights: (778,J)
      mano_faces:   optional, only used if you want SDF filtering (not needed here)
    Returns:
      W_high: (Vh,J)
    """
    if isinstance(verts_high, np.ndarray):  verts_high = torch.from_numpy(verts_high)
    if isinstance(mano_verts, np.ndarray):  mano_verts = torch.from_numpy(mano_verts)
    if isinstance(mano_weights, np.ndarray):mano_weights = torch.from_numpy(mano_weights)
    verts_high, mano_verts, mano_weights = verts_high.float(), mano_verts.float(), mano_weights.float()

    # kNN interpolation
    dist = torch.cdist(verts_high, mano_verts)         # (Vh, 778)
    knn_d, knn_i = torch.topk(dist, neighbors, largest=False)
    w = 1.0 / (knn_d + 1e-8)
    w = w / w.sum(dim=1, keepdim=True)
    W_hi = torch.sum(mano_weights[knn_i] * w[..., None], dim=1)  # (Vh,J)
    # Normalize rows
    W_hi = W_hi / (W_hi.sum(dim=1, keepdim=True) + 1e-8)
    return W_hi

def mano_joint_transforms(mano_model, betas, hand_pose, global_orient, transl):
    """
    Returns per-joint 4x4 transforms A_j (rest->posed) in MANO space.
    betas: (10,)
    hand_pose: (45,)
    global_orient: (3,)
    transl: (3,)
    """
    device = mano_model.v_template.device

    betas = betas.view(1, -1).to(device)
    hand_pose = hand_pose.view(1, -1).to(device)          # (1,45)
    global_orient = global_orient.view(1, 3).to(device)   # (1,3)
    transl = transl.view(1, 3).to(device)

    # Shape blend
    v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)
    # Joint locations in the canonical rest shape
    J = vertices2joints(mano_model.J_regressor, v_shaped)

    # Build full pose
    full_pose = torch.cat([global_orient, hand_pose], dim=1)          # (1,48)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    # Correct unpacking: returns (joints_posed, transforms)
    J_posed, A = batch_rigid_transform(rot_mats, J, mano_model.parents, dtype=v_shaped.dtype)

    # Apply translation to all transforms
    A[:, :, :3, 3] = A[:, :, :3, 3] + transl[:, None, :]
    return A.squeeze(0)  # (J,4,4)

def skin_points_forward(points_cano, skin_weights, A):
    """
    Linear blend skinning of arbitrary points.
      points_cano:  (N,3) canonical (MANO space)
      skin_weights: (N,J)
      A:            (J,4,4)
    """
    device = points_cano.device
    N = points_cano.shape[0]
    X = torch.cat([points_cano, torch.ones(N,1, device=device, dtype=points_cano.dtype)], dim=1)  # (N,4)
    # apply all joint transforms to each point
    AX = (A @ X.T).permute(2,0,1)[:, :, :3]   # (N,J,3)
    pts = (skin_weights.unsqueeze(-1) * AX).sum(dim=1)  # (N,3)
    return pts

def hand_reflection_mats(is_right: torch.Tensor, device):
    """
    Returns 3x3 and 4x4 reflection matrices S3, S4.
    is_right: scalar tensor {0,1} or shape (,) for the current frame.
    For LEFT hand (0) -> reflect across YZ plane: diag([-1, 1, 1]).
    For RIGHT hand (1) -> identity.
    """
    flip = (is_right.item() == 0)  # left => True
    if not flip:
        S3 = torch.eye(3, device=device)
    else:
        S3 = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device))
    S4 = torch.eye(4, device=device)
    S4[:3, :3] = S3
    return S3, S4

def reflect_quat(q_wxyz: torch.Tensor, S3: torch.Tensor):
    """
    Reflect a quaternion-defined rotation by sandwiching the matrix:
    R' = S3 @ R @ S3. Works on (N,4) quats (w,x,y,z).
    """
    R = p3d_t.quaternion_to_matrix(q_wxyz)                # (N,3,3)
    Rr = S3.T @ R @ S3                                    # S is symmetric, S.T == S
    return p3d_t.matrix_to_quaternion(Rr)

def erode(mask, k=3, it=1):
    # mask: (1,H,W) float in {0,1}
    m = mask
    for _ in range(it):
        m = 1.0 - torch.nn.functional.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
    return m


def unnormalized_K(K_norm, W, H):
    K = K_norm.clone()
    K[0, 0] *= W
    K[1, 1] *= H
    K[0, 2] *= W
    K[1, 2] *= H
    return K


def project_xyz(xyz_world, w2c_gl, K_norm, W, H):
    """
    xyz_world: (N,3) in OpenGL/world of your renderer (same as you use to render).
    w2c_gl: (4,4)
    K_norm: (3,3) normalized intrinsics used by renderer (fx/W, fy/H, cx/W, cy/H)
    Returns: pixel coords (N,2) in [0,W-1]x[0,H-1], valid mask (N,)
    """
    N = xyz_world.shape[0]
    # camera coords
    xyz_h = torch.cat([xyz_world, torch.ones(N,1, device=xyz_world.device)], dim=1)  # (N,4)
    cam = (w2c_gl @ xyz_h.T).T[:, :3]  # (N,3)
    z = cam[:, 2]
    valid = z > 0  # in front of camera

    K = unnormalized_K(K_norm, W, H)   # fx, fy, cx, cy in pixels
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = (cam[:,0] / z) * fx + cx
    v = (cam[:,1] / z) * fy + cy

    uv = torch.stack([u, v], dim=-1)
    # also clip to image bounds
    valid = valid & (uv[:,0] >= 0) & (uv[:,0] <= (W-1)) & (uv[:,1] >= 0) & (uv[:,1] <= (H-1))
    return uv, valid


def project_xyz_w_depth(xyz_w, w2c, K, W, H):
    """
    xyz_w: (N,3) world
    w2c:   (4,4)
    K:     (3,3)
    Returns:
      uv: (N,2) float pixel coords
      zc: (N,)  camera depth (the z in camera frame)
      valid: (N,) bool in front + inside image
    """
    N = xyz_w.shape[0]
    ones = torch.ones((N, 1), device=xyz_w.device, dtype=xyz_w.dtype)
    xw_h = torch.cat([xyz_w, ones], dim=1)                       # (N,4)
    xc_h = (w2c @ xw_h.T).T                                      # (N,4)
    xc = xc_h[:, :3]
    zc = xc[:, 2]

    # perspective
    x = xc[:, 0] / (zc + 1e-8)
    y = xc[:, 1] / (zc + 1e-8)
    uv_h = (K @ torch.stack([x, y, torch.ones_like(x)], dim=0)).T  # (N,3)
    uv = uv_h[:, :2]

    valid = (zc > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, zc, valid

def build_image_iso_sog_quadtree(
    rgb_u8: np.ndarray,
    mask_u8: np.ndarray | None = None,
    max_depth: int = 7,
    var_thresh: float = 0.02,
    min_size: int = 4,
    bbox_pad: int = 2,
    min_valid_ratio: float = 1e-6,
    split_uniform_interior: bool = True,
    boundary_split_min_ratio: float = 0.02,
    # isotropic sigma choices:
    sigma_mode: str = "geom",   # "geom" (sqrt(sx*sy)) or "mean" (0.5*(sx+sy))
):
    """
    Quadtree image SoG with *isotropic* sigma per leaf.
    Returns list with one level:
      {"mu": (N,2), "sigma": (N,1), "color": (N,3), "weight": (N,),
       "rect": (N,4) leaf boxes, "depth": (N,) quadtree depths}
    """
    H, W, _ = rgb_u8.shape
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    col = rgb01

    if mask_u8 is None:
        mask = None
        queue = [(0, 0, W, H, 0)]
    else:
        mask = (mask_u8 > 0) if mask_u8.dtype != np.bool_ else mask_u8
        ys, xs = np.where(mask)
        if xs.size == 0:
            return [{"mu": np.zeros((0,2),np.float32),
                     "sigma": np.zeros((0,1),np.float32),
                     "color": np.zeros((0,3),np.float32),
                     "weight": np.zeros((0,),np.float32),
                     "rect": np.zeros((0,4),np.int32),
                     "depth": np.zeros((0,),np.int32)}]

        x0 = max(int(xs.min()) - bbox_pad, 0)
        x1 = min(int(xs.max()) + 1 + bbox_pad, W)
        y0 = max(int(ys.min()) - bbox_pad, 0)
        y1 = min(int(ys.max()) + 1 + bbox_pad, H)
        queue = [(x0, y0, x1, y1, 0)]

    leaves = []
    while queue:
        x0, y0, x1, y1, d = queue.pop(0)
        w, h = (x1 - x0), (y1 - y0)
        if w <= 0 or h <= 0:
            continue

        if mask is None:
            m = None
            valid_ratio = 1.0
        else:
            m = mask[y0:y1, x0:x1]
            if m.size == 0:
                continue
            valid_ratio = float(m.mean())
            if valid_ratio <= min_valid_ratio:
                continue

        patch = col[y0:y1, x0:x1, :]

        if mask is not None:
            yy_v, xx_v = np.where(m)
            if xx_v.size > 0:
                v = float(patch[yy_v, xx_v, :].reshape(-1, 3).var(axis=0).max())
            else:
                v = 0.0
        else:
            v = float(patch.reshape(-1, 3).var(axis=0).mean())

        boundary_patch = (mask is not None) and (valid_ratio < 0.999) and (valid_ratio > 0.001)
        # Split boundary cells so leaf footprints do not bleed into the background.
        mixed_boundary_patch = (
            boundary_patch
            and (valid_ratio >= boundary_split_min_ratio)
        )
        uniform_interior = (
            split_uniform_interior
            and (mask is not None)
            and (valid_ratio >= 0.999)
            and (min(w, h) > (min_size * 2))
        )

        should_split = (
            (d < max_depth)
            and (min(w, h) > min_size)
            and (mixed_boundary_patch or uniform_interior or (v > var_thresh))
        )

        if should_split:
            xm = (x0 + x1) // 2
            ym = (y0 + y1) // 2
            queue.extend([
                (x0, y0, xm, ym, d+1),
                (xm, y0, x1, ym, d+1),
                (x0, ym, xm, y1, d+1),
                (xm, ym, x1, y1, d+1),
            ])
            continue

        # --- LEAF ---
        if mask is not None:
            yy, xx = np.where(m)
            if xx.size == 0:
                continue

            px = (x0 + xx).astype(np.float32)
            py = (y0 + yy).astype(np.float32)

            mu_x = float(px.mean())
            mu_y = float(py.mean())

            x_min, x_max = float(px.min()), float(px.max())
            y_min, y_max = float(py.min()), float(py.max())
            sx = max(1.0, 0.5 * (x_max - x_min + 1.0))
            sy = max(1.0, 0.5 * (y_max - y_min + 1.0))

            if sigma_mode == "geom":
                s = float(np.sqrt(max(sx * sy, 1e-6)))
            else:
                s = float(0.5 * (sx + sy))

            c = patch[yy, xx, :].mean(axis=0).astype(np.float32)
            wt = float(xx.size)
        else:
            mu_x = 0.5 * (x0 + x1)
            mu_y = 0.5 * (y0 + y1)
            sx = max(1.0, 0.5 * w)
            sy = max(1.0, 0.5 * h)
            s = float(np.sqrt(max(sx * sy, 1e-6))) if sigma_mode == "geom" else float(0.5 * (sx + sy))
            c = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
            wt = float(w * h)

        leaves.append((mu_x, mu_y, s, c, wt, x0, y0, x1, y1, d))

    if len(leaves) == 0:
        return [{"mu": np.zeros((0,2),np.float32),
                 "sigma": np.zeros((0,1),np.float32),
                 "color": np.zeros((0,3),np.float32),
                 "weight": np.zeros((0,),np.float32),
                 "rect": np.zeros((0,4),np.int32),
                 "depth": np.zeros((0,),np.int32)}]

    mu = np.array([[l[0], l[1]] for l in leaves], np.float32)
    sig = np.array([[l[2]] for l in leaves], np.float32)      # (N,1)
    colv = np.array([l[3] for l in leaves], np.float32)
    wts = np.array([l[4] for l in leaves], np.float32)
    rect = np.array([[l[5], l[6], l[7], l[8]] for l in leaves], np.int32)
    depth = np.array([l[9] for l in leaves], np.int32)

    return [{"mu": mu, "sigma": sig, "color": colv, "weight": wts, "rect": rect, "depth": depth}]

def as_iso_scalar(sig):
    """
    Accepts (N,), (N,1), or (N,2) and returns (N,) isotropic scalar in pixels.
    - If (N,2): use geom mean sqrt(sigx*sigy) for stability.
    """
    if sig.ndim == 1:
        return sig
    if sig.ndim == 2 and sig.shape[1] == 1:
        return sig[:, 0]
    if sig.ndim == 2 and sig.shape[1] == 2:
        return torch.sqrt((sig[:, 0].clamp_min(1e-6) * sig[:, 1].clamp_min(1e-6)))
    raise ValueError(f"Unsupported sig shape: {tuple(sig.shape)}")

def project_iso_sog_to_image(mu3d_w, sigma3d_iso, w2c_gl, K_pix, W, H, f_mode: str = "avg"):
    """
    mu3d_w: (N,3) world/gl coords
    sigma3d_iso: (N,) or (N,1) or (N,3) isotropic in world units (if (N,3) assumed equal per-axis)
    Returns:
      mu2d: (N,2) pixels
      sig2d_iso: (N,) pixels
      valid: (N,) bool
    """
  
    uv, zc, valid = project_xyz_w_depth(mu3d_w, w2c_gl, K_pix, W, H)

    if sigma3d_iso.ndim == 2 and sigma3d_iso.shape[1] == 3:
        sigma3d = sigma3d_iso[:, 0]
    else:
        sigma3d = as_iso_scalar(sigma3d_iso)

    fx, fy = K_pix[0, 0], K_pix[1, 1]
    if f_mode == "geom":
        f = torch.sqrt((fx * fy).clamp_min(1e-6))
    else:
        f = 0.5 * (fx + fy)

    z = zc.clamp_min(1e-3)
    sig = f * (sigma3d / z)
    return uv, sig, valid

def compute_img_contrib_topk_iso(
    image_mu, image_sig_iso, image_col, image_w,
    model_mu, model_sig_iso, model_col, model_w,
    use_color=True, topk=32, tau=0.15
):
    """
    Isotropic 2D Gaussian overlap:
      ov = (2π) * ( (σm σi / sqrt(σm^2 + σi^2))^2 ) * exp(-0.5 * ||d||^2/(σm^2+σi^2))
    image_sig_iso: (Ni,) or (Ni,1) or (Ni,2)
    model_sig_iso: (Nm,) or (Nm,1) or (Nm,2)
    Returns h: (Ni,)
    """
    if image_col.shape[1] == 1:
        image_col = image_col.squeeze(1)
    if model_col.shape[1] == 1:
        model_col = model_col.squeeze(1)

    Ni = image_mu.shape[0]
    Nm = model_mu.shape[0]
    if Ni == 0 or Nm == 0:
        return image_mu.new_zeros((Ni,))

    img_sig = as_iso_scalar(image_sig_iso).clamp_min(1e-3)  # (Ni,)
    mdl_sig = as_iso_scalar(model_sig_iso).clamp_min(1e-3)  # (Nm,)

    # top-k in 2D position
    with torch.no_grad():
        d2 = (model_mu[:, None, :] - image_mu[None, :, :]).pow(2).sum(-1)
        k = min(int(topk), Ni)
        nn = torch.topk(d2, k=k, dim=1, largest=False).indices  # (Nm,k)

    img_mu_k  = image_mu[nn]                  # (Nm,k,2)
    img_sig_k = img_sig[nn]                   # (Nm,k)
    img_col_k = image_col[nn]                 # (Nm,k,3)
    img_wt_k  = image_w[nn]                   # (Nm,k)

    d = model_mu[:, None, :] - img_mu_k       # (Nm,k,2)
    S = (mdl_sig[:, None].pow(2) + img_sig_k.pow(2) + 1e-6)  # (Nm,k)
    maha = d.pow(2).sum(-1) / S
    exp_term = torch.exp(-0.5 * maha)

    coef = (mdl_sig[:, None] * img_sig_k) / torch.sqrt(S)    # (Nm,k)
    ov = (2.0 * math.pi) * (coef * coef) * exp_term          # (Nm,k)

    if use_color:
        d2c = (model_col[:, None, :] - img_col_k).pow(2).sum(-1)
        ck = torch.exp(-d2c / (tau * tau))
        ov = ov * ck

    contrib = (model_w[:, None] * img_wt_k) * ov              # (Nm,k)

    nn_flat = nn.reshape(-1).contiguous()
    src_flat = contrib.reshape(-1).contiguous()
    h = torch.zeros((Ni,), device=image_mu.device, dtype=src_flat.dtype)
    h = h.index_add(0, nn_flat, src_flat)
    return h


def quat_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    # enforce consistent sign (w >= 0) to avoid flips
    q = quat_normalize(q)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign

def rotvec_vel_from_quat_seq(q_seq: torch.Tensor) -> torch.Tensor:
    """
    q_seq: (T,4) in (w,x,y,z) OR (x,y,z,w) depending on your convention.
    You must match PyTorch3D's expected convention.
    Returns rotvec velocity: (T-1, 3)
    """
    q_seq = quat_canonicalize(q_seq)
    q_rel = quaternion_multiply(q_seq[1:], quaternion_invert(q_seq[:-1]))   # (T-1,4)
    # axis-angle (rotvec) in radians:
    return quaternion_to_axis_angle(q_rel)   

def make_iso_scaling_from_xyz_scaling(scaling_xyz: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    scaling_xyz: (N,3)
    returns isotropic scaling as (N,3) with equal components.
    """
    if mode == "geom":
        s = torch.sqrt((scaling_xyz[:, 0] * scaling_xyz[:, 1] * scaling_xyz[:, 2]).clamp_min(1e-12)).unsqueeze(-1)
    else:
        s = scaling_xyz.mean(dim=-1, keepdim=True)
    return s.repeat(1, 3)
