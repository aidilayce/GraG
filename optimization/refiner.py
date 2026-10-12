"""Stage 3: hand-object tracking with Sums of Gaussians (paper Sec. 4.3).

Each sliding window minimizes Eq. (2) of the paper,

    -lambda_sog E + lambda_sil L_sil + lambda_depth L_depth + lambda_j2d L_j2d
    + c^{ho} lambda_contact L_contact + lambda_smooth L_smooth,

with the weights of Supp. Sec. 10.2 (``LOSS_WEIGHTS``). The solution is kept
close to the SAM3D / Dyn-HaMR initialization by ``L_init`` (Supp. Sec. 10.2).
Each paper term is a weighted sum of the components listed in ``TERM_WEIGHTS``;
the component weights are relative to the paper weight of their term.
"""
import math
from pathlib import Path

import cv2
import imageio
import numpy as np
import pytorch3d.ops as p3d_ops
import pytorch3d.structures as p3d_structs
import pytorch3d.transforms as p3d_t
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from scipy.spatial import cKDTree

from grag_common.project_paths import append_project_paths

append_project_paths()

from third_party.Dyn_HaMR.dyn_hamr.body_model import run_mano_single
from third_party.SAM3DO.sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from third_party.SAM3DO.sam3d_objects.utils.visualization import SceneVisualizer

from grag_common.optimization_utils import (
    Gaussian,
    MetricLogger,
    SH2RGB,
    as_iso_scalar,
    build_image_iso_sog_quadtree,
    compute_img_contrib_topk_iso,
    contact_idx,
    contact_zones,
    create_high_res_hand_template,
    erode,
    hand_reflection_mats,
    initialize_gaussians_from_mesh_w_normals,
    load_mask,
    load_rgb,
    make_iso_scaling_from_xyz_scaling,
    mano_joint_transforms,
    opencv_to_pytorch3d,
    pose_gaussians,
    project_iso_sog_to_image,
    project_xyz,
    project_xyz_w_depth,
    reflect_quat,
    rotvec_vel_from_quat_seq,
    skin_points_forward,
    tight_aabb,
    upsample_mano_weights_to_highres,
)
from grag_common.runtime_config import device, opencv_to_opengl

PYTORCH3D_TO_OPENGL = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

# Supp. Sec. 10.2, Eq. (2).
LOSS_WEIGHTS = {
    "sog": 0.05,
    "sil": 100.0,
    "depth": 1000.0,
    "j2d": 0.5,
    "contact": 1000.0,
    "smooth": 100.0,
}

# Components of each paper term, relative to its weight in LOSS_WEIGHTS.
TERM_WEIGHTS = {
    "sog": {"energy": 1.0},
    # Eq. (6) boundary distances plus guards against degenerate support:
    # rendered object/hand alpha vs. masks, object SoG centers outside the
    # object mask, and the projected-to-image SoG area ratio.
    "sil": {
        "obj_boundary": 1.0,
        "hand_boundary": 1.0,
        "obj_mask": 500.0,
        "hand_mask": 100.0,
        "obj_sog_outside": 50.0,
        "obj_sog_area": 10.0,
    },
    # Eq. (7) plus centroid anchors: object centroid at the pointmap (or, in
    # contact, the hand) depth, hand centroid and rendered hand depth at the
    # pointmap depth, and the hand-in-front-of-object order inside the masks.
    "depth": {
        "eq7": 1.0,
        "obj_anchor": 200.0,
        "hand_anchor": 40.0,
        "hand_render": 0.5,
        "order": 500.0,
    },
    "j2d": {"joints": 1.0},
    # Contact-zone attraction and penetration of the hand into the object and
    # of the object into the hand.
    "contact": {"near": 1.0, "pen_hand": 25.0, "pen_obj": 200.0},
    # Supp. L_smooth, plus the acceleration of object surface points.
    "smooth": {"motion": 1.0, "obj_points": 4.0},
}

# Initialization prior (Supp. Sec. 10.2), absolute weights.
INIT_WEIGHTS = {
    "obj_trans_xy": 3e3,
    "obj_trans_z": 1e3,
    "obj_rot": 2e2,
    "hand_pose": 1.2e3,
}

OBJ_SCALE_RANGE = (0.75, 1.5)
HAND_SCALE_RANGE = (0.98, 1.04)
OBJ_RESIDUAL_XY = 0.06
HAND_RESIDUAL_XY = 0.055
HAND_RESIDUAL_Z = 0.05


class JointHandObjectRefiner:
    def __init__(self, exp, H, W, frames_meta, hand_res, object_gs, hand_model, renderer,
                 obj_surface_points, obj_surface_normals, data_root, logdir,
                 use_wandb=False, wb_project="hoi-gaussians", wb_run=None,
                 obj_mesh_canonical_verts=None, obj_mesh_faces=None):
        self.out_dir = Path(logdir)
        self.logger = MetricLogger(self.out_dir, use_wandb=use_wandb, wb_project=wb_project, wb_run_name=wb_run)
        self.exp = exp
        self.device = device
        self.H, self.W = H, W
        self.frames = frames_meta
        self.hand_model = hand_model
        self.renderer = renderer
        self.init_obj_gs = object_gs
        self.p3d_to_gl = torch.tensor(PYTORCH3D_TO_OPENGL, device=device, dtype=torch.float32)

        self.rgbs, self.masks_obj, self.masks_hand = [], [], []
        self.Ks, self.Ks_pix, self.w2c_gl, self.pointmaps, self.joints2d_observed = [], [], [], [], []
        init_trans, init_rot, scale_candidates = [], [], []
        for frame_id in sorted(frames_meta):
            meta = frames_meta[frame_id]
            self.rgbs.append(load_rgb(meta["rgb_path"], H, W).to(device))
            # Dilate masks by one pixel to absorb uncertain boundaries.
            for path, out in ((meta["mask_path"], self.masks_obj), (meta["hand_mask_path"], self.masks_hand)):
                mask = load_mask(path, H, W).to(device)
                out.append(F.max_pool2d(mask, kernel_size=3, stride=1, padding=1))

            K_pix = meta["K"].to(device).clone()
            self.Ks_pix.append(K_pix)
            K = K_pix.clone()
            K[0, 0] /= W
            K[1, 1] /= H
            K[0, 2] /= W
            K[1, 2] /= H
            self.Ks.append(K)
            self.w2c_gl.append(meta["w2c_cv"].to(device) @ opencv_to_opengl.to(device))

            pose = meta["pose_obj_pt3d"]
            scale_candidates.append(torch.as_tensor(pose["scale"], dtype=torch.float32, device=device).reshape(-1).abs().mean())
            init_trans.append(pose["translation"].to(device) @ self.p3d_to_gl)
            rot = p3d_t.quaternion_to_matrix(pose["rotation"].to(device)) @ self.p3d_to_gl
            init_rot.append(p3d_t.matrix_to_quaternion(rot))
            self.pointmaps.append(meta["pointmap"].float())
            joints2d = meta.get("joints2d_observed")
            if joints2d is None:
                joints2d = torch.zeros((21, 3))
            self.joints2d_observed.append(torch.as_tensor(joints2d, dtype=torch.float32, device=device))

        self.rgbs = torch.stack(self.rgbs)
        self.masks_obj = torch.stack(self.masks_obj)
        self.masks_hand = torch.stack(self.masks_hand)
        self.Ks = torch.stack(self.Ks)
        self.w2c_gl = torch.stack(self.w2c_gl)
        self.pointmaps = torch.stack(self.pointmaps)
        self.joints2d_observed = torch.stack(self.joints2d_observed)
        self.T = T = len(frames_meta)
        self.contact_flags = self.load_contact_flags(Path(data_root) / exp / "contact_flags_gemini.npy")
        self.delta_obj_z_min = torch.full((T,), -0.10, device=device)
        self.delta_obj_z_max = self.object_depth_residual_cap()

        self.obj_surface_points = obj_surface_points.to(device)
        self.obj_surface_normals = F.normalize(obj_surface_normals.to(device), dim=-1)
        k = min(512, int(self.obj_surface_points.shape[0]))
        self.obj_track_points = p3d_ops.sample_farthest_points(self.obj_surface_points[None], K=k)[0][0]
        self.obj_mesh_canonical_verts = obj_mesh_canonical_verts.to(device=device, dtype=torch.float32)
        self.obj_mesh_faces = obj_mesh_faces.to(device=device, dtype=torch.int64)

        self.is_right = hand_res["is_right"].clone()
        root_orient = hand_res["root_orient"].clone()
        pose_body = hand_res["pose_body"].clone()
        hand_trans = hand_res["trans"].clone()
        pad = T - self.is_right.shape[1]
        if pad:
            self.is_right = F.pad(self.is_right, (0, pad), mode="replicate")
            root_orient = F.pad(root_orient, (0, 0, 0, pad), mode="replicate")
            pose_body = F.pad(pose_body, (0, 0, 0, 0, 0, pad), mode="replicate")
            hand_trans = F.pad(hand_trans, (0, 0, 0, pad), mode="replicate")
        self.hand_betas = torch.nn.Parameter(hand_res["betas"].clone())
        self.hand_global_orient = torch.nn.Parameter(root_orient)
        self.hand_pose = torch.nn.Parameter(pose_body)
        self.hand_trans = torch.nn.Parameter(hand_trans)

        # Gaussians skinned to an upsampled MANO mesh render the hand alpha and depth.
        template = create_high_res_hand_template(self.hand_model, None, subdivision_level=2)
        mesh_high = trimesh.Trimesh(vertices=template["vertices"].cpu().numpy(), faces=template["faces"].cpu().numpy())
        self.hand_gaussians = initialize_gaussians_from_mesh_w_normals(mesh_high, samples_per_vertex=2, init_scale=1, sh_degree=0)
        weights_high = upsample_mano_weights_to_highres(
            mesh_high.vertices,
            self.hand_model.v_template.cpu().numpy(),
            self.hand_model.lbs_weights.cpu().numpy(),
        ).to(device)
        _, nn_idx = cKDTree(mesh_high.vertices).query(self.hand_gaussians.get_xyz.detach().cpu().numpy())
        self.hand_gaussian_weights = weights_high[nn_idx].contiguous()

        self.delta_t_hand = torch.nn.Parameter(torch.zeros((T, 3), device=device))
        self.delta_obj_translations = torch.nn.Parameter(torch.zeros((T, 3), device=device))
        self.hand_scale = torch.nn.Parameter(torch.full((1,), 1.025, device=device))
        self.hand_scale_init = self.hand_scale.detach().clone()
        self.init_contact_vertex_sets()

        self.obj_gl_init_rot_quat = torch.vstack(init_rot)
        self.obj_rotations = torch.nn.Parameter(self.obj_gl_init_rot_quat.clone())
        self.obj_translations = torch.vstack(init_trans)
        self.hand_pose_ref = self.hand_pose.detach().clone()
        self.hand_orient_ref = self.hand_global_orient.detach().clone()
        self.delta_t_hand_ref = self.delta_t_hand.detach().clone()
        # One sequence-wide object scale, initialized at the median SAM3D scale.
        self.obj_scale = torch.nn.Parameter(torch.stack(scale_candidates).median().reshape(1))
        self.obj_scale_init = self.obj_scale.detach().clone()

        self.reanchor = True
        self.reanchor_done = False
        self.freeze_object_scale = False
        self.image_sog = [None] * T
        self.log_step = 0

    # ------------------------------------------------------------------ setup

    def load_contact_flags(self, path: Path) -> torch.Tensor:
        """Gemini contact flags c^{ho}; mask proximity is the offline fallback."""
        if path.exists():
            flags = np.load(str(path))
            if flags.ndim == 1 and flags.size > 0:
                print(f"[INFO] Loaded contact flags: {path} (len={flags.size})")
                return torch.from_numpy((flags > 0).astype(np.uint8)).to(self.device)
        flags = torch.zeros(self.T, dtype=torch.uint8, device=self.device)
        for t in range(self.T):
            hand = self.masks_hand[t, 0] > 0.5
            obj = self.masks_obj[t, 0] > 0.5
            if hand.any() and obj.any():
                hand_d = F.max_pool2d(hand.float()[None, None], 3, 1, 1)[0, 0]
                obj_d = F.max_pool2d(obj.float()[None, None], 3, 1, 1)[0, 0]
                flags[t] = int(bool((hand_d * obj_d).any()))
        print(f"[INFO] No contact flags; {int(flags.sum())}/{self.T} contact frames from mask proximity.")
        return flags

    def init_contact_vertex_sets(self):
        """MANO contact zones (Hasson et al. 2019): all, fingertip side, thumb, other fingers."""
        self.contact_ids = torch.as_tensor(contact_idx, dtype=torch.long, device=self.device)
        finger_ids = [int(v) for k, vals in contact_zones.items() if int(k) != 0 for v in vals]
        tip_ids = torch.unique(torch.as_tensor(finger_ids, dtype=torch.long, device=self.device))
        cano = self.hand_model.v_template.detach().to(self.device)
        palm_center = cano[[95, 22]].mean(dim=0, keepdim=True)
        dist = (cano[tip_ids] - palm_center).norm(dim=-1)
        keep = dist >= torch.quantile(dist, 0.55)
        if int(keep.sum()) >= 5:
            tip_ids = tip_ids[keep]
        self.fingertip_contact_ids = tip_ids
        self.thumb_contact_ids = torch.as_tensor(contact_zones.get(1, []), dtype=torch.long, device=self.device)
        self.finger_contact_ids = torch.as_tensor(
            [int(v) for k, vals in contact_zones.items() if int(k) not in (0, 1) for v in vals],
            dtype=torch.long, device=self.device,
        )

    def object_depth_residual_cap(self) -> torch.Tensor:
        """Sequence-wide cap on the object's camera-depth residual.

        Thin, elongated objects (pens, scissors) are poorly measured by the
        pointmap and need a larger depth range than compact ones.
        """
        caps = []
        for t in range(self.T):
            mask = self.masks_obj[t, 0] > 0.5
            if not mask.any():
                caps.append(0.26)
                continue
            ys, xs = torch.where(mask)
            w = float(xs.max() - xs.min() + 1)
            h = float(ys.max() - ys.min() + 1)
            aspect = max(w, h) / max(min(w, h), 1.0)
            area = float(mask.float().mean())
            if aspect >= 1.9 and area <= 0.035:
                caps.append(0.58)
            elif aspect >= 1.5 and area <= 0.060:
                caps.append(0.36)
            else:
                caps.append(0.26)
        caps = torch.tensor(caps, dtype=torch.float32, device=self.device)
        return torch.full_like(caps, torch.quantile(caps, 0.8).item())

    def clamp_residuals(self):
        d = self.delta_obj_translations.data
        d[:, :2].clamp_(-OBJ_RESIDUAL_XY, OBJ_RESIDUAL_XY)
        d[:, 2] = torch.minimum(torch.maximum(d[:, 2], self.delta_obj_z_min), self.delta_obj_z_max)
        self.delta_t_hand.data[:, :2].clamp_(-HAND_RESIDUAL_XY, HAND_RESIDUAL_XY)
        self.delta_t_hand.data[:, 2].clamp_(-HAND_RESIDUAL_Z, HAND_RESIDUAL_Z)
        lo, hi = OBJ_SCALE_RANGE
        self.obj_scale.data.clamp_(min=float(self.obj_scale_init * lo), max=float(self.obj_scale_init * hi))
        lo, hi = HAND_SCALE_RANGE
        self.hand_scale.data.clamp_(min=float(self.hand_scale_init * lo), max=float(self.hand_scale_init * hi))

    def build_image_sog(self, t: int) -> dict:
        """Image SoG of the masked object (Sec. 4.3.1, Supp. Sec. 10.2)."""
        rgb = (self.rgbs[t].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        roi = (self.masks_obj[t][0] > 0).cpu().numpy().astype("uint8") * 255
        level = build_image_iso_sog_quadtree(
            rgb, roi, max_depth=6, var_thresh=0.04, min_size=8,
            bbox_pad=2, min_valid_ratio=1e-6, sigma_mode="geom", split_uniform_interior=False,
        )[0]
        out = {k: torch.from_numpy(level[k]).to(self.device).float() for k in ("mu", "sigma", "color", "weight")}
        sigma = as_iso_scalar(out["sigma"])
        out["self_overlap"] = math.pi * sigma * sigma
        out["area"] = out["self_overlap"].sum()
        return out

    def object_sog(self, g_obj_canonical, max_points: int = 3000) -> Gaussian:
        """Object SoG: farthest-point sample of the SAM3D Gaussians, isotropic scales."""
        g = g_obj_canonical
        xyz, idx = p3d_ops.sample_farthest_points(g.get_xyz.detach().to(self.device)[None], K=max_points)
        xyz, idx = xyz[0], idx[0].to(g.get_xyz.device)
        with torch.no_grad():
            sampled = Gaussian(aabb=tight_aabb(xyz), sh_degree=0)
            sampled.from_xyz(xyz)
            sampled.from_rotation(g.get_rotation.detach()[idx].to(self.device))
            sampled.from_scaling(make_iso_scaling_from_xyz_scaling(g.get_scaling.detach()[idx].to(self.device)))
            sampled.from_opacity(g.get_opacity.detach()[idx].to(self.device))
            sampled._features_dc = g._features_dc[idx].detach().to(self.device)
            sampled.mininum_kernel_size = g.mininum_kernel_size
        return sampled

    # --------------------------------------------------------- posed geometry

    def get_obj_poses(self, t):
        quat = F.normalize(self.obj_rotations[t:t + 1], dim=-1)
        return quat, self.obj_translations[t] + self.delta_obj_translations[t]

    def transform_obj_points(self, quat, trans, scale, points, normals=None):
        R = p3d_t.quaternion_to_matrix(quat.reshape(-1, 4)[:1])
        s = scale.reshape(-1)[0].reshape(1, 1).expand(1, 3)
        pts = compose_transform(scale=s, rotation=R, translation=trans[None]).transform_points(points[None])[0]
        nrm = None if normals is None else F.normalize(normals @ R[0].T, dim=-1)
        return pts, nrm

    def get_mano_mesh(self, t: int, mean_xyz: torch.Tensor, detach_params: bool = False):
        params = (self.hand_trans[0, t][None], self.hand_global_orient[0, t][None],
                  self.hand_pose[0, t][None].reshape(-1, 45), self.hand_betas)
        if detach_params:
            params = tuple(p.detach() for p in params)
        trans, orient, pose, betas = params
        mano = run_mano_single(self.hand_model, trans, orient, pose, self.is_right[0, t], betas=betas)
        faces = mano["r_faces"] if mano["is_right"] else mano["l_faces"]
        verts = mano["vertices"][0] @ opencv_to_opengl[:3, :3].T.to(self.device)
        verts = (verts - mean_xyz) * self.hand_scale + mean_xyz + self.delta_t_hand[t][None]
        return p3d_structs.Meshes(verts=[verts], faces=[faces])

    def get_posed_hand_gaussians(self, t):
        """Skinned hand Gaussians, their mean (scaling pivot) and the 3D joints."""
        betas = self.hand_betas.squeeze(0)
        orient = self.hand_global_orient[0, t]
        pose = self.hand_pose[0, t].reshape(-1)
        trans = self.hand_trans[0, t]
        is_right = self.is_right[0, t]
        gl = opencv_to_opengl[:3, :3].to(self.device)
        mano = run_mano_single(self.hand_model, trans[None], orient[None], pose[None], is_right, betas=betas[None])
        joints = mano["joints"][0] @ gl.T

        S3, S4 = hand_reflection_mats(is_right, device=self.device)
        A = mano_joint_transforms(self.hand_model, betas, pose, orient, trans)
        xyz = self.hand_gaussians.get_xyz
        rot = self.hand_gaussians.get_rotation
        if is_right.item() == 0:
            A = S4 @ A @ S4
            xyz = xyz @ S3.T
            rot = reflect_quat(rot, S3)
        xyz = skin_points_forward(xyz, self.hand_gaussian_weights, A) @ gl.T
        mean_xyz = xyz.mean(0, keepdim=True)
        shift = self.delta_t_hand[t][None]
        xyz = (xyz - mean_xyz) * self.hand_scale + mean_xyz + shift
        joints = (joints - mean_xyz) * self.hand_scale + mean_xyz + shift

        g = Gaussian(aabb=tight_aabb(xyz), sh_degree=0)
        g.from_xyz(xyz)
        g.from_rotation(reflect_quat(rot, gl))
        g.from_scaling(self.hand_gaussians.get_scaling * self.hand_scale)
        g._features_dc = self.hand_gaussians._features_dc
        g._features_rest = self.hand_gaussians._features_rest
        g._opacity = self.hand_gaussians._opacity
        return g, mean_xyz, joints

    def get_object_mesh(self, t: int):
        quat, trans = self.get_obj_poses(t)
        pc = SceneVisualizer.object_pointcloud(
            points_local=self.obj_mesh_canonical_verts[None], quat_l2c=quat[None],
            trans_l2c=trans[None], scale_l2c=self.obj_scale,
        )
        return p3d_structs.Meshes(verts=[pc.points_list()[0]], faces=[self.obj_mesh_faces])

    def camera_z(self, w2c, point):
        return (w2c @ torch.cat([point, point.new_ones(1)]))[2]

    # --------------------------------------------------------- mask utilities

    @staticmethod
    def as_hw(x: torch.Tensor) -> torch.Tensor:
        while x.ndim > 2 and (x.shape[0] == 1 or x.shape[-1] == 1):
            x = x.squeeze(0) if x.shape[0] == 1 else x.squeeze(-1)
        return x

    @staticmethod
    def dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
        return F.max_pool2d(mask.float()[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0] > 0.5

    @staticmethod
    def signed_distance(mask: torch.Tensor) -> torch.Tensor:
        """Signed distance transform of a mask: negative inside, positive outside."""
        m = (mask > 0.5).cpu().numpy().astype(np.uint8)
        sdf = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) - cv2.distanceTransform(m, cv2.DIST_L2, 3)
        return torch.from_numpy(sdf)[None, None].to(mask.device)

    @staticmethod
    def mask_core(mask, kernels, min_pixels, min_keep_ratio):
        """Eroded mask interior that keeps a minimum support for tiny objects."""
        total = int(mask.sum())
        if total == 0:
            return mask
        min_keep = min(total, max(min_pixels, math.ceil(total * min_keep_ratio)))
        best = mask
        for k in kernels:
            eroded = (erode(mask.float()[None], k=k) > 0.5)[0]
            count = int(eroded.sum())
            if count > 0:
                best = eroded
            if count >= min_keep:
                return eroded
        return best

    @staticmethod
    def visible_hand_mask(hand, obj, min_pixels=16, min_keep_ratio=0.05):
        """Hand mask without the pixels the object occludes."""
        visible = hand & ~obj
        n_hand = int(hand.sum())
        if n_hand > 0 and int(visible.sum()) < max(min_pixels, math.ceil(n_hand * min_keep_ratio)):
            return hand
        return visible

    def pointmap_depth(self, pm_cam, mask, min_points=8, trim=0.10):
        """Robust camera depth of the pointmap inside a mask (trimmed shorth median)."""
        if not mask.any():
            return torch.full((), float("nan"), device=self.device)
        z = pm_cam[mask]
        z = z[torch.isfinite(z).all(dim=-1), 2]
        z = z[z > 1e-8]
        if z.numel() == 0:
            return torch.full((), float("nan"), device=self.device)
        if z.numel() >= max(3, min_points):
            lo, hi = torch.quantile(z, torch.tensor([trim, 1 - trim], device=z.device))
            trimmed = z[(z >= lo) & (z <= hi)]
            if trimmed.numel() > 0:
                z = trimmed
        if z.numel() >= 24:
            zs = torch.sort(z).values
            h = max(2, zs.numel() // 2)
            i = int(torch.argmin(zs[h - 1:] - zs[:zs.numel() - h + 1]))
            return zs[i:i + h].median()
        return z.median()

    # ---------------------------------------------------- contact vertices

    @staticmethod
    def valid_ids(ids, n):
        return ids[(ids >= 0) & (ids < n)]

    def ids_near_object_mask(self, verts, ids, t, obj_mask, hand_mask=None, dilate_px=21,
                             min_keep=1, max_keep=96):
        """Contact vertices that project into the dilated object mask (and the hand mask)."""
        ids = self.valid_ids(ids, verts.shape[0])
        if ids.numel() == 0:
            return ids
        uv, valid = project_xyz(verts[ids], self.w2c_gl[t], self.Ks[t], self.W, self.H)
        x = uv[:, 0].round().long().clamp(0, self.W - 1)
        y = uv[:, 1].round().long().clamp(0, self.H - 1)
        keep = valid & self.dilate(obj_mask, dilate_px)[y, x]
        if hand_mask is not None:
            keep = keep & hand_mask[y, x]
        ids = ids[keep]
        if ids.numel() < min_keep:
            return ids[:0]
        if ids.numel() > max_keep:
            ids = ids[torch.linspace(0, ids.numel() - 1, steps=max_keep, device=ids.device).long()]
        return ids

    def ids_closest_to_object(self, verts, ids, t, obj_mask, max_keep):
        """Fallback: contact vertices whose projection is closest to the object mask center."""
        ids = self.valid_ids(ids, verts.shape[0])
        uv, valid = project_xyz(verts[ids], self.w2c_gl[t], self.Ks[t], self.W, self.H)
        keep = torch.where(valid)[0]
        if keep.numel() == 0:
            return ids[:0]
        ys, xs = torch.where(obj_mask)
        if xs.numel() > 0:
            center = torch.stack([xs.float().mean(), ys.float().mean()])
        else:
            center = torch.tensor([0.5 * (self.W - 1), 0.5 * (self.H - 1)], device=self.device)
        order = torch.argsort((uv[keep] - center).pow(2).sum(-1))[:max_keep]
        return ids[keep[order]]

    def grasp_contact_ids(self, verts, t, obj_mask, max_keep=96):
        """Thumb and opposing-finger vertices next to the object (a pinch/grasp)."""
        out = []
        for ids, fallback_keep in ((self.thumb_contact_ids, 16), (self.finger_contact_ids, 32)):
            sel = self.ids_near_object_mask(verts, ids, t, obj_mask, dilate_px=37, max_keep=max_keep)
            if sel.numel() == 0:
                sel = self.ids_closest_to_object(verts, ids, t, obj_mask, min(fallback_keep, max_keep))
            out.append(sel)
        return out

    def visible_contact_ids(self, verts, t, hand_depth, obj_depth, hand_mask, obj_mask):
        """Contact-zone vertices visible in the hand render and close to the object depth."""
        ids = self.valid_ids(self.contact_ids, verts.shape[0])
        v = verts[ids]
        uv, valid = project_xyz(v, self.w2c_gl[t], self.Ks[t], self.W, self.H)
        x = uv[:, 0].round().long().clamp(0, self.W - 1)
        y = uv[:, 1].round().long().clamp(0, self.H - 1)
        z_v = (self.w2c_gl[t] @ torch.cat([v, v.new_ones(v.shape[0], 1)], -1).T)[2]
        z_hand = hand_depth[y, x].abs()
        z_obj = obj_depth[y, x].abs()
        visible = torch.isfinite(z_hand) & (z_hand > 1e-6) & ((z_v - z_hand).abs() < 0.03)
        near = torch.isfinite(z_obj) & (z_obj > 1e-6) & ((z_hand - z_obj).abs() < 0.12)
        keep = valid & hand_mask[y, x] & self.dilate(obj_mask, 7)[y, x] & visible & near
        return ids[keep]

    def attraction_ids(self, verts, t, cache, hand_depth, obj_depth):
        """Contact vertices for L_near and their weight.

        In order of reliability: a thumb/finger pinch around the object, fingertips
        inside the hand mask next to the object, rendered-visible contact zones,
        and finally all fingertip zones.
        """
        thumb, fingers = self.grasp_contact_ids(verts, t, cache["obj_mask"])
        if thumb.numel() > 0 and fingers.numel() > 0:
            ids = torch.unique(torch.cat([thumb, fingers]))
            if ids.numel() >= 2:
                return ids, 340.0, 130.0
        ids = self.ids_near_object_mask(verts, self.fingertip_contact_ids, t, cache["obj_mask"],
                                        hand_mask=cache["hand_mask_full"], dilate_px=25)
        if ids.numel() >= 1:
            return ids, 220.0, 85.0
        ids = self.visible_contact_ids(verts, t, hand_depth, obj_depth, cache["hand_mask"], cache["obj_mask"])
        if ids.numel() >= 8:
            return ids, 65.0, 45.0
        return self.fingertip_contact_ids, 80.0, 30.0

    # ------------------------------------------------------------ loss terms

    def sog_alignment(self, obj_sog, quat, trans, t, cache):
        """E (Eq. 3) with the hand visibility gate (Sec. 4.3.4), plus support guards."""
        g = pose_gaussians(obj_sog, quat, trans, self.obj_scale)
        sig3d = make_iso_scaling_from_xyz_scaling(g.get_scaling)[:, 0] * 1.75
        mu2d, sig2d, valid = project_iso_sog_to_image(g.get_xyz, sig3d, self.w2c_gl[t], self.Ks_pix[t], self.W, self.H)
        mu2d, sig2d = mu2d[valid], sig2d[valid].clamp(1.0, 18.0)
        color = SH2RGB(g._features_dc)[valid].detach()
        weight = torch.ones(mu2d.shape[0], device=self.device)
        if mu2d.shape[0] > 0:
            x = mu2d[:, 0].round().long().clamp(0, self.W - 1)
            y = mu2d[:, 1].round().long().clamp(0, self.H - 1)
            visible = ~(cache["hand_mask"] & ~cache["obj_mask"])[y, x]
            if int(visible.sum()) >= min(mu2d.shape[0], max(16, math.ceil(0.05 * mu2d.shape[0]))):
                mu2d, sig2d, color, weight = mu2d[visible], sig2d[visible], color[visible], weight[visible]
        img = cache["image_sog"]
        overlap = compute_img_contrib_topk_iso(
            img["mu"], img["sigma"], img["color"], img["weight"],
            mu2d, sig2d, color, weight, use_color=True, topk=64, tau=0.15,
        )
        energy = torch.minimum(overlap, img["self_overlap"]).sum()

        zero = torch.zeros((), device=self.device)
        if mu2d.shape[0] == 0:
            return g, energy, zero, zero
        grid = torch.stack([2 * mu2d[:, 0] / (self.W - 1) - 1, 2 * mu2d[:, 1] / (self.H - 1) - 1], -1)
        inside = F.grid_sample(cache["obj_mask"].float()[None, None], grid.view(1, 1, -1, 2),
                               mode="bilinear", padding_mode="zeros", align_corners=True).view(-1)
        area_w = sig2d.detach().clamp_min(1.0) ** 2
        outside = (area_w / (area_w.mean() + 1e-6) * (1 - inside.clamp(0, 1)).pow(2)).mean()
        log_area = torch.log((math.pi * sig2d * sig2d).sum() / (img["area"] + 1e-6))
        area = 0.3 * F.relu(-log_area).pow(2) + 2.0 * F.relu(log_area - 0.10).pow(2)
        return g, energy, outside, area

    def boundary_loss(self, hand_mesh, g_obj, obj_alpha, t, cache):
        """Eq. (6): signed distance of the projected hand/object boundary to the mask contours.

        Boundary points hidden by the other observed mask are ignored.
        """
        w2c, K = self.w2c_gl[t], self.Ks[t]
        verts = hand_mesh.verts_padded()[0]
        normals = p3d_structs.Meshes(verts=[verts], faces=[hand_mesh.faces_padded()[0]]).verts_normals_packed()
        cam = (w2c @ torch.cat([verts, verts.new_ones(verts.shape[0], 1)], -1).T).T[:, :3]
        facing = (normals @ w2c[:3, :3].T * F.normalize(-cam, dim=-1)).sum(-1)
        uv, valid = project_xyz(verts, w2c, K, self.W, self.H)
        hand_loss = self.sample_boundary(uv[valid & (facing.abs() < 0.20)], cache["sdf_hand"], cache["obj_mask"])

        uv, valid = project_xyz(g_obj.get_xyz, w2c, K, self.W, self.H)
        x = uv[valid][:, 0].long().clamp(0, self.W - 1)
        y = uv[valid][:, 1].long().clamp(0, self.H - 1)
        alpha = obj_alpha > 0.5
        on_edge = torch.zeros_like(x, dtype=torch.bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                on_edge |= alpha[y, x] & ~alpha[(y + dy).clamp(0, self.H - 1), (x + dx).clamp(0, self.W - 1)]
        obj_loss = self.sample_boundary(uv[valid][on_edge], cache["sdf_obj"], cache["hand_mask"])
        return hand_loss, obj_loss

    def sample_boundary(self, uv, sdf, occluder):
        if uv.shape[0] == 0:
            return torch.zeros((), device=self.device)
        keep = ~occluder[uv[:, 1].long().clamp(0, self.H - 1), uv[:, 0].long().clamp(0, self.W - 1)]
        if keep.any():
            uv = uv[keep]
        grid = torch.stack([2 * uv[:, 0] / (self.W - 1) - 1, 2 * uv[:, 1] / (self.H - 1) - 1], -1)
        d = F.grid_sample(sdf, grid.view(1, 1, -1, 2), mode="bilinear", padding_mode="border", align_corners=True)
        return d.abs().mean()

    def mask_loss(self, alpha, mask, ignore=None, outside_weight=1.0):
        """Rendered alpha pushed to 1 inside the mask and to 0 in a band around it."""
        band = self.dilate(mask, 41)
        if ignore is not None:
            band = band & ~ignore
        inside = mask & band
        outside = band & ~inside
        loss = torch.zeros((), device=self.device)
        if inside.any():
            loss = loss + F.smooth_l1_loss(alpha[inside], torch.ones_like(alpha[inside]))
        if outside.any():
            loss = loss + outside_weight * F.smooth_l1_loss(alpha[outside], torch.zeros_like(alpha[outside]))
        return loss

    def hand_mask_loss(self, alpha, cache):
        """Hand alpha vs. its visible mask, plus a centroid pull in normalized coordinates."""
        mask = cache["hand_mask"]
        alpha = alpha.clamp(0, 1)
        if int(mask.sum()) < 16 or not (mask & ~cache["obj_mask"]).any():
            return torch.zeros((), device=self.device)
        loss = self.mask_loss(alpha, mask, ignore=cache["obj_mask"], outside_weight=0.5)
        band = (self.dilate(mask, 41) & ~cache["obj_mask"]).float()
        pred = alpha * band
        if pred.sum() > 1e-6:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.H, device=self.device),
                                    torch.linspace(-1, 1, self.W, device=self.device), indexing="ij")
            pred_c = torch.stack([(pred * xx).sum(), (pred * yy).sum()]) / (pred.sum() + 1e-6)
            gt = mask.float() * band
            gt_c = torch.stack([(gt * xx).sum(), (gt * yy).sum()]) / gt.sum().clamp_min(1e-6)
            loss = loss + 5.0 * (pred_c - gt_c).pow(2).sum()
        return loss

    def joints2d_loss(self, joints, t):
        """L_j2d (Supp. Eq. 13): squared pixel error of the reprojected MANO joints."""
        obs = self.joints2d_observed[t]
        n = min(joints.shape[0], obs.shape[0])
        uv, _, valid = project_xyz_w_depth(joints[:n], self.w2c_gl[t], self.Ks_pix[t], self.W, self.H)
        valid = valid & torch.isfinite(uv).all(-1) & torch.isfinite(obs[:n, :2]).all(-1) & (obs[:n, 2] > 0.05)
        if int(valid.sum()) < 4:
            return torch.zeros((), device=self.device)
        return (uv[valid] - obs[:n, :2][valid]).square().sum()

    def depth_eq7(self, hand_depth, obj_depth, cache):
        """Eq. (7): median rendered depths in the masks vs. pointmap medians, coupled in contact."""
        medians = []
        for depth, mask in ((hand_depth, cache["hand_mask"]), (obj_depth, cache["obj_mask"])):
            z = depth.abs()[mask]
            z = z[torch.isfinite(z) & (z > 1e-8)]
            medians.append(z.median() if z.numel() else None)
        z_hand, z_obj = medians
        loss = torch.zeros((), device=self.device)
        if z_hand is None or z_obj is None:
            return loss
        if torch.isfinite(cache["z_hand"]):
            loss = loss + (z_hand - cache["z_hand"]).square()
        if torch.isfinite(cache["z_obj"]):
            loss = loss + (z_obj - cache["z_obj"]).square()
        if cache["contact"]:
            loss = loss + (z_obj - z_hand).square()
        return loss

    @staticmethod
    def depth_order(front_depth, back_depth, support, front_alpha, back_alpha, margin, max_violation):
        """Penalize the back surface being in front of the front surface inside ``support``."""
        support = support & (front_alpha.detach() > 0.3) & (back_alpha.detach() > 0.3)
        if int(support.sum()) < 16:
            return front_depth.new_zeros(())
        f, b = front_depth[support].abs(), back_depth[support].abs()
        valid = (f > 1e-8) & (b > 1e-8) & torch.isfinite(f) & torch.isfinite(b)
        if int(valid.sum()) < 16:
            return front_depth.new_zeros(())
        violation = F.relu(b[valid] - f[valid] + margin)
        return (max_violation * torch.log1p(violation / max_violation)).pow(2).mean()

    def contact_near(self, hand_verts, hand_normals, ids, quat, trans, weight, side_weight):
        """L_near: pull contact vertices to their nearest object surface point.

        The nearest point is chosen among 16 neighbours, preferring points on the
        outward side of the hand. Distances are soft-capped at 3 cm.
        """
        ids = self.valid_ids(ids, hand_verts.shape[0])
        verts, normals = hand_verts[ids], hand_normals[ids]
        obj_pts, _ = self.transform_obj_points(quat.detach(), trans, self.obj_scale.detach(), self.obj_surface_points)
        if verts.shape[0] == 0:
            return torch.zeros((), device=self.device)
        nn = p3d_ops.knn_points(verts[None], obj_pts[None], K=16)
        cand = nn.idx[0]
        side = ((obj_pts[cand] - verts[:, None]) * normals[:, None]).sum(-1)
        score = nn.dists[0].clamp_min(1e-12).sqrt() + 0.6 * F.relu(0.006 - side)
        nearest = obj_pts[cand[torch.arange(cand.shape[0]), score.argmin(-1)]]
        dist = (verts - nearest).norm(dim=-1)
        near = (0.03 * torch.log1p(F.relu(dist - 0.005) / 0.03)).mean()
        side = F.relu(0.006 - ((nearest - verts) * normals).sum(-1)).mean()
        return torch.nan_to_num(weight * near + side_weight * side)

    def hand_in_object_penetration(self, hand_verts, quat, trans):
        """L_pen (hand side): hand vertices behind the nearest object surface point."""
        obj_pts, obj_nrm = self.transform_obj_points(
            quat.detach(), trans, self.obj_scale.detach(), self.obj_surface_points, self.obj_surface_normals,
        )
        nn = p3d_ops.knn_points(hand_verts[None], obj_pts[None], K=1).idx[0, :, 0]
        signed = ((hand_verts - obj_pts[nn]) * obj_nrm[nn]).sum(-1)
        return F.relu(-signed).mean()

    def object_in_hand_penetration(self, hand_verts, hand_normals, quat, trans, margin=0.012, near=0.08):
        """L_pen (object side): object surface samples behind the nearby hand surface."""
        obj_pts, _ = self.transform_obj_points(quat.detach(), trans, self.obj_scale.detach(), self.obj_track_points)
        nn = p3d_ops.knn_points(obj_pts[None], hand_verts[None], K=4)
        idx = nn.idx[0]
        signed = ((obj_pts[:, None] - hand_verts[idx]) * hand_normals[idx]).sum(-1)
        close = nn.dists[0, :, 0].clamp_min(1e-12).sqrt() < near
        if not close.any():
            return torch.zeros((), device=self.device)
        return torch.nan_to_num(F.relu(margin - signed).amax(-1)[close].pow(2).mean())

    # -------------------------------------------------------- initialization

    @torch.no_grad()
    def reanchor_object_depth(self):
        """Reconcile SAM3D object depth with the Dyn-HaMR hand depth once.

        r = median over contact frames of hand_z / object_z. Scaling the object by
        r and moving it r along each camera ray keeps every projection unchanged,
        so only the metric depth and size change.
        """
        self.reanchor_done = True
        ratios = []
        for t in range(self.T):
            if int(self.contact_flags[t]) != 1:
                continue
            quat, trans = self.get_obj_poses(t)
            obj_pts, _ = self.transform_obj_points(quat, trans, self.obj_scale, self.obj_surface_points)
            obj_z = self.camera_z(self.w2c_gl[t], obj_pts.mean(0))
            _, mean_xyz, _ = self.get_posed_hand_gaussians(t)
            hand_z = self.camera_z(self.w2c_gl[t], self.get_mano_mesh(t, mean_xyz, True).verts_padded()[0].mean(0))
            if torch.isfinite(obj_z) and torch.isfinite(hand_z) and obj_z > 1e-3 and hand_z > 1e-3:
                ratios.append(hand_z / obj_z)
        if not ratios:
            return
        r = float(torch.stack(ratios).median().clamp(0.7, 1.5))
        if abs(r - 1.0) < 0.02:
            print(f"[reanchor] object already at hand depth (r={r:.3f})")
            return
        s_before = float(self.obj_scale)
        self.obj_scale.mul_(r)
        self.obj_scale_init = self.obj_scale_init * r
        for t in range(self.T):
            w2c = self.w2c_gl[t]
            cam = w2c @ torch.cat([self.obj_translations[t] + self.delta_obj_translations[t], torch.ones(1, device=self.device)])
            cam[:3] *= r
            self.obj_translations[t] = (torch.linalg.inv(w2c) @ cam)[:3] - self.delta_obj_translations[t]
        print(f"[reanchor] r={r:.3f} over {len(ratios)} contact frames, "
              f"obj_scale {s_before:.4f}->{float(self.obj_scale):.4f}")

    def apply_residual(self, t, delta):
        proposed = self.delta_obj_translations.data[t] + delta
        proposed[:2].clamp_(-OBJ_RESIDUAL_XY, OBJ_RESIDUAL_XY)
        proposed[2].clamp_(float(self.delta_obj_z_min[t]), float(self.delta_obj_z_max[t]))
        applied = proposed - self.delta_obj_translations.data[t]
        if applied.norm() <= 1e-5:
            return None
        self.delta_obj_translations.data[t] = proposed
        return applied.norm()

    @torch.no_grad()
    def warm_start_window(self, frames, cache):
        """Bounded warm start of the window's object translations.

        Frames without contact slide along the camera ray to the object pointmap
        depth; contact frames move the object surface to the grasp (thumb-finger
        midpoint, or the fingertips touching the object mask).
        """
        shifts = []
        for t in frames:
            c = cache[t]
            if c["contact"] or not torch.isfinite(c["z_obj"]):
                continue
            trans = self.obj_translations[t] + self.delta_obj_translations[t]
            w2c = self.w2c_gl[t]
            cam = w2c @ torch.cat([trans, trans.new_ones(1)])
            if not torch.isfinite(cam[2]) or cam[2] <= 1e-3:
                continue
            cap = 0.32 if float(self.delta_obj_z_max[t]) > 0.30 else 0.12
            dz = (c["z_obj"] - cam[2]).clamp(-cap, cap)
            if dz.abs() < 0.01:
                continue
            cam[:3] *= (cam[2] + dz) / cam[2]
            delta = (torch.linalg.inv(w2c) @ cam)[:3] - trans
            if delta.norm() > cap:
                delta *= cap / (delta.norm() + 1e-8)
            applied = self.apply_residual(t, delta)
            if applied is not None:
                shifts.append(applied)

        for t in frames:
            c = cache[t]
            if not c["contact"]:
                continue
            long_obj = float(self.delta_obj_z_max[t]) > 0.30
            cap, xy_cap = (0.12, 0.020) if long_obj else (0.04, 0.015)
            _, mean_xyz, _ = self.get_posed_hand_gaussians(t)
            mesh = self.get_mano_mesh(t, mean_xyz, detach_params=True)
            verts = mesh.verts_padded()[0]
            normals = F.normalize(mesh.verts_normals_packed(), dim=-1)
            quat, trans = self.get_obj_poses(t)
            obj_pts, _ = self.transform_obj_points(quat, trans, self.obj_scale, self.obj_surface_points)
            thumb, fingers = self.grasp_contact_ids(verts, t, c["obj_mask"])
            if thumb.numel() > 0 and fingers.numel() > 0:
                target = 0.5 * (verts[thumb].mean(0) + verts[fingers].mean(0))
                delta = target - obj_pts[p3d_ops.knn_points(target[None, None], obj_pts[None], K=1).idx[0, 0, 0]]
            else:
                ids = self.ids_near_object_mask(verts, self.fingertip_contact_ids, t, c["obj_mask"], dilate_px=31)
                if ids.numel() == 0:
                    continue
                target = verts[ids].mean(0) + 0.006 * normals[ids].mean(0)
                nearest = obj_pts[p3d_ops.knn_points(verts[ids][None], obj_pts[None], K=1).idx[0, :, 0]]
                delta = torch.median(target[None] - nearest, dim=0).values
            if not torch.isfinite(delta).all():
                continue
            axis = F.normalize(self.w2c_gl[t][2, :3], dim=0)
            along = delta.dot(axis) * axis
            across = delta - along
            if across.norm() > xy_cap:
                across *= xy_cap / (across.norm() + 1e-8)
            delta = along + across
            if delta.norm() > cap:
                delta *= cap / (delta.norm() + 1e-8)
            applied = self.apply_residual(t, delta)
            if applied is not None:
                shifts.append(applied)
        if shifts:
            shifts = torch.stack(shifts)
            print(f"[warm-start] {len(shifts)} frames, mean={shifts.mean():.4f} m, max={shifts.max():.4f} m")

    def build_window_cache(self, frames):
        """Per-frame quantities that do not depend on the optimized variables."""
        cache = {}
        for t in frames:
            if self.image_sog[t] is None:
                self.image_sog[t] = self.build_image_sog(t)
            w2c = self.w2c_gl[t]
            pm = self.pointmaps[t].to(self.device) @ self.p3d_to_gl
            pm_cam = (w2c @ torch.cat([pm.reshape(-1, 3), pm.new_ones(pm.shape[0] * pm.shape[1], 1)], 1).T).T[:, :3]
            pm_cam = pm_cam.reshape(pm.shape)
            hand_full = self.masks_hand[t][0] > 0.5
            obj = self.masks_obj[t][0] > 0.5
            hand = self.visible_hand_mask(hand_full, obj)
            masks_close = bool((self.dilate(hand, 3) & self.dilate(obj, 3)).any())
            ys, xs = torch.where(obj)
            if xs.numel() > 0:
                w, h = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
                aspect, area = max(w, h) / max(min(w, h), 1.0), float(obj.float().mean())
            else:
                aspect, area = 1.0, 0.0
            # Thin objects may sit further in front of the fingers.
            if aspect >= 1.9 and area <= 0.035:
                front_margin = 0.045
            elif aspect >= 1.5 and area <= 0.060:
                front_margin = 0.032
            else:
                front_margin = 0.020
            z_hand = self.pointmap_depth(pm_cam, self.mask_core(hand, (7, 5, 3), 24, 0.04))
            z_obj = self.pointmap_depth(pm_cam, self.mask_core(obj, (9, 7, 5, 3), 12, 0.02))
            cache[t] = {
                "image_sog": self.image_sog[t],
                "camera": self.renderer.build_camera_dict(w2c, self.Ks[t]),
                "hand_mask": hand,
                "hand_mask_full": hand_full,
                "obj_mask": obj,
                "sdf_hand": self.signed_distance(hand.float()),
                "sdf_obj": self.signed_distance(obj.float()),
                "pm_cam": pm_cam,
                "hand_render_mask": self.mask_core(hand, (7, 5, 3), 24, 0.04),
                "z_hand": z_hand.clamp(min=1e-3) if torch.isfinite(z_hand) else z_hand,
                "z_obj": z_obj.clamp(min=1e-3) if torch.isfinite(z_obj) else z_obj,
                "contact": bool(int(self.contact_flags[t]) == 1) or masks_close,
                "order_margin": min(0.018, max(0.004, 0.25 * front_margin)),
            }
        # Pointmap hand depth spikes are pulled to the local median.
        z = [cache[t]["z_hand"] for t in frames]
        if len(z) >= 3:
            for i, t in enumerate(frames):
                if not torch.isfinite(z[i]):
                    continue
                neigh = torch.stack(z[max(0, i - 1):i + 2])
                neigh = neigh[torch.isfinite(neigh)]
                if neigh.numel() >= 2 and (z[i] - neigh.median()).abs() > 0.015:
                    cache[t]["z_hand"] = 0.2 * z[i] + 0.8 * neigh.median()
        return cache

    # ---------------------------------------------------------- optimization

    def optimize_window(self, frames, g_obj_canonical, iters=100, log_every=10):
        """Minimize Eq. (2) over the frames of one sliding window."""
        n = len(frames)
        hand_start = max(5, int(0.20 * iters))
        hand_pose_start = max(hand_start + 8, int(0.60 * iters))
        optimizer = torch.optim.AdamW([
            {"params": [self.obj_scale], "lr": 1e-3},
            {"params": [self.delta_obj_translations], "lr": 2.5e-3},
            {"params": [self.obj_rotations], "lr": 8e-4},
            {"params": [self.hand_scale], "lr": 2e-4},
            {"params": [self.hand_pose], "lr": 4e-4},
            {"params": [self.hand_global_orient], "lr": 5e-4},
            {"params": [self.delta_t_hand], "lr": 9e-4},
        ], betas=(0.9, 0.95), weight_decay=0.0)

        obj_sog = self.object_sog(g_obj_canonical)
        cache = self.build_window_cache(frames)
        if self.reanchor and not self.reanchor_done:
            self.reanchor_object_depth()
        scale_window_start = self.obj_scale.detach().clone()
        self.warm_start_window(frames, cache)

        frame_idx = torch.as_tensor(frames, device=self.device)
        context = torch.arange(max(0, frames[0] - 2), frames[0], device=self.device)
        tw = TERM_WEIGHTS
        for s in range(iters):
            optimizer.zero_grad(set_to_none=True)
            ramp = min(1.0, max(0.0, (s - hand_start) / max(1, iters - hand_start))) if s >= hand_start else 0.0
            parts = {k: torch.zeros((), device=self.device) for k in (
                "energy", "obj_sog_outside", "obj_sog_area", "obj_boundary", "hand_boundary", "obj_mask",
                "hand_mask", "eq7", "obj_anchor", "hand_anchor", "hand_render", "order", "joints",
                "near", "pen_hand", "pen_obj", "hand_keep")}
            obj_track = []
            for t in frames:
                c = cache[t]
                w2c = self.w2c_gl[t]
                g_hand, mean_xyz, joints = self.get_posed_hand_gaussians(t)
                hand_mesh = self.get_mano_mesh(t, mean_xyz)
                hand_verts = hand_mesh.verts_padded()[0]
                quat, trans = self.get_obj_poses(t)

                g_obj, energy, outside, area = self.sog_alignment(obj_sog, quat, trans, t, c)
                obj_out = self.renderer.render_with_camera(g_obj, c["camera"])
                hand_out = self.renderer.render_with_camera(g_hand, c["camera"])
                obj_depth, hand_depth = self.as_hw(obj_out["depth"]), self.as_hw(hand_out["depth"])
                obj_alpha, hand_alpha = self.as_hw(obj_out["sil"]), self.as_hw(hand_out["sil"])

                hand_boundary, obj_boundary = self.boundary_loss(hand_mesh, g_obj, obj_alpha, t, c)
                parts["energy"] += energy
                parts["obj_sog_outside"] += outside
                parts["obj_sog_area"] += area
                parts["obj_boundary"] += obj_boundary
                parts["hand_boundary"] += ramp * hand_boundary
                parts["obj_mask"] += self.mask_loss(obj_alpha, c["obj_mask"])
                parts["hand_mask"] += ramp * self.hand_mask_loss(hand_alpha, c)
                parts["joints"] += ramp * self.joints2d_loss(joints, t)

                # Depth: Eq. (7) on camera depths, anchored to the pointmap and,
                # in contact, to the Dyn-HaMR hand.
                z_obj = self.camera_z(w2c, trans)
                z_hand = self.camera_z(w2c, hand_verts.mean(0))
                z_obj_target = z_hand.detach() if c["contact"] else c["z_obj"]
                with torch.no_grad():
                    surface = obj_depth.abs()[obj_alpha > 0.5]
                    surface = surface[torch.isfinite(surface) & (surface > 1e-6)]
                    # The pointmap sees the visible surface, the anchor moves the centroid.
                    offset = (z_obj - surface.median()).clamp(min=0) if surface.numel() >= 16 else 0.0
                    z_obj_target = z_obj_target + offset
                if torch.isfinite(z_obj_target) and (z_obj - z_obj_target).abs() <= 0.30:
                    parts["obj_anchor"] += (z_obj - z_obj_target).pow(2)
                hand_ok = torch.isfinite(c["z_hand"]) and (z_hand - c["z_hand"]).abs() <= 0.50
                if hand_ok:
                    parts["hand_anchor"] += ramp * (z_hand - c["z_hand"]).pow(2)
                    rendered = hand_depth[c["hand_render_mask"]].abs()
                    rendered = rendered[torch.isfinite(rendered) & (rendered > 1e-8)]
                    if rendered.numel() > 0:
                        parts["hand_render"] += ramp * (rendered.mean() - c["z_hand"]).pow(2)
                parts["eq7"] += self.depth_eq7(hand_depth, obj_depth, c)

                if c["contact"]:
                    parts["order"] += self.depth_order(
                        hand_depth, obj_depth, c["obj_mask"] & ~c["hand_mask"], hand_alpha, obj_alpha,
                        c["order_margin"], 0.12,
                    ) + self.depth_order(
                        obj_depth, hand_depth, c["hand_mask"] & ~c["obj_mask"], obj_alpha, hand_alpha,
                        0.006, 0.08,
                    )
                    # Contact moves the object; the hand follows its own image evidence.
                    verts_d = hand_verts.detach()
                    normals_d = F.normalize(hand_mesh.verts_normals_packed().detach(), dim=-1)
                    ids, w_near, w_side = self.attraction_ids(verts_d, t, c, hand_depth, obj_depth)
                    parts["near"] += self.contact_near(verts_d, normals_d, ids, quat, trans, w_near, w_side)
                    pen_ramp = min(1.0, max(0.0, (s - int(0.30 * iters)) / max(1, int(0.50 * iters))))
                    if pen_ramp > 0:
                        parts["pen_hand"] += pen_ramp * self.hand_in_object_penetration(verts_d, quat, trans)
                        parts["pen_obj"] += pen_ramp * self.object_in_hand_penetration(verts_d, normals_d, quat, trans)

                keep = 6.0 if c["contact"] else 1.5
                parts["hand_keep"] += ramp * keep * (
                    0.5 * (self.hand_global_orient[0, t] - self.hand_orient_ref[0, t]).pow(2).mean()
                    + 0.1 * (self.hand_pose[0, t] - self.hand_pose_ref[0, t]).pow(2).mean()
                    + 0.5 * (self.delta_t_hand[t] - self.delta_t_hand_ref[t]).pow(2).mean()
                )
                obj_track.append(self.transform_obj_points(quat, trans, self.obj_scale, self.obj_track_points)[0])

            smooth = self.smoothness(frame_idx, context, obj_track)
            init = self.init_prior(frame_idx)
            terms = {
                "sog": -tw["sog"]["energy"] * parts["energy"],
                "sil": sum(tw["sil"][k] * parts[k] for k in tw["sil"]),
                "depth": sum(tw["depth"][k] * parts[k] for k in tw["depth"]),
                "j2d": tw["j2d"]["joints"] * parts["joints"],
                "contact": sum(tw["contact"][k] * parts[k] for k in tw["contact"]),
                "smooth": tw["smooth"]["motion"] * smooth["motion"] + tw["smooth"]["obj_points"] * smooth["obj_points"],
            }
            loss = sum(LOSS_WEIGHTS[k] * v for k, v in terms.items()) + init + INIT_WEIGHTS["hand_pose"] * parts["hand_keep"]
            loss = loss / n + self.scale_prior(scale_window_start)

            if log_every and (s % log_every == 0 or s == iters - 1):
                msg = ", ".join(f"{k}={float(LOSS_WEIGHTS[k] * v) / n:.2f}" for k, v in terms.items())
                print(f"[window {frames[0]}-{frames[-1]}] it {s}: loss={float(loss):.3f}, {msg}, "
                      f"obj_scale={float(self.obj_scale):.5f}")
                self.logger.log(step=self.log_step, stage="stage3", metrics={
                    "loss_total": float(loss), **{f"L_{k}": float(LOSS_WEIGHTS[k] * v) / n for k, v in terms.items()},
                })
                self.log_step += 1

            loss.backward()
            if self.freeze_object_scale and self.obj_scale.grad is not None:
                self.obj_scale.grad.zero_()
            if s < hand_start:
                for p in (self.hand_pose, self.delta_t_hand, self.hand_global_orient):
                    if p.grad is not None:
                        p.grad.zero_()
            elif s < hand_pose_start:
                for p in (self.hand_pose, self.hand_global_orient):
                    if p.grad is not None:
                        p.grad.zero_()
            optimizer.step()
            with torch.no_grad():
                self.clamp_residuals()

    def smoothness(self, frame_idx, context, obj_track):
        """L_smooth: hand/object translation and object angular acceleration.

        The two frames before the window enter as fixed context so the
        acceleration is also constrained across window seams.
        """
        zero = torch.zeros((), device=self.device)
        if len(frame_idx) < 2:
            return {"motion": zero, "obj_points": zero}
        obj_t = torch.cat([
            (self.obj_translations[context] + self.delta_obj_translations[context]).detach(),
            self.obj_translations[frame_idx] + self.delta_obj_translations[frame_idx],
        ])
        hand_t = torch.cat([
            (self.hand_trans[0, context] + self.delta_t_hand[context]).detach(),
            self.hand_trans[0, frame_idx] + self.delta_t_hand[frame_idx],
        ])
        obj_w = rotvec_vel_from_quat_seq(torch.cat([self.obj_rotations[context].detach(), self.obj_rotations[frame_idx]]))

        def accel(x):
            return (x[2:] - 2 * x[1:-1] + x[:-2]).square().sum() if x.shape[0] >= 3 else zero

        pts = torch.stack(obj_track)
        pts_vel = pts[1:] - pts[:-1]
        pts_acc = (pts_vel[1:] - pts_vel[:-1]).pow(2).mean() if pts.shape[0] > 2 else zero
        return {
            "motion": accel(hand_t) + accel(obj_t) + 1e-2 * accel(obj_w),
            "obj_points": 0.1 * pts_vel.pow(2).mean() + pts_acc,
        }

    def init_prior(self, frame_idx):
        """Keep the object close to its Stage-2 trajectory (Supp. Sec. 10.2)."""
        obj_t = self.obj_translations[frame_idx] + self.delta_obj_translations[frame_idx]
        delta = obj_t - self.obj_translations[frame_idx]
        init_q = F.normalize(self.obj_gl_init_rot_quat[frame_idx], dim=-1)
        dot = (F.normalize(self.obj_rotations[frame_idx], dim=-1) * init_q).sum(-1).abs().clamp(0, 1)
        return (
            INIT_WEIGHTS["obj_trans_xy"] * delta[:, :2].pow(2).mean()
            + INIT_WEIGHTS["obj_trans_z"] * F.relu(delta[:, 2].abs() - 0.015).pow(2).mean()
            + INIT_WEIGHTS["obj_rot"] * (1 - dot).pow(2).mean()
        )

    def scale_prior(self, scale_window_start):
        """Object scale stays near its (re-anchored) initialization; shrinking costs more."""
        log_s = torch.log(self.obj_scale / self.obj_scale_init)
        log_win = torch.log(self.obj_scale / (scale_window_start + 1e-8))
        log_hand = torch.log(self.hand_scale / (self.hand_scale_init + 1e-8))
        return (
            300.0 * F.relu(-log_s).pow(2) + 160.0 * F.relu(log_s).pow(2)
            + 20.0 * log_win.pow(2) + 600.0 * log_hand.pow(2)
        ).sum()

    # ------------------------------------------------------------- rendering

    def render_video_all_mesh(self, out_path, side_view_y_clockwise_degrees=None, blank_background=False):
        """Render the hand/object meshes over the input video, or from a side view."""
        raster = RasterizationSettings(image_size=(self.H, self.W), blur_radius=0.0, faces_per_pixel=1, bin_size=-1)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster),
            shader=HardPhongShader(device=self.device, lights=self.lights(), blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )
        cv_from_gl = opencv_to_opengl.to(self.device)
        side_ref = self.w2c_gl[self.T // 2] @ cv_from_gl
        obj_to_cv = self.p3d_to_gl @ torch.from_numpy(opencv_to_pytorch3d).to(self.device).float()
        frames = []
        with torch.no_grad():
            for t in range(self.T):
                _, mean_xyz, _ = self.get_posed_hand_gaussians(t)
                hand = self.get_mano_mesh(t, mean_xyz, detach_params=True)
                obj = self.get_object_mesh(t)
                w2c = side_ref if side_view_y_clockwise_degrees is not None else self.w2c_gl[t] @ cv_from_gl

                def to_cam(v):
                    return (w2c @ torch.cat([v, v.new_ones(v.shape[0], 1)], -1).T).T[:, :3]

                hv = to_cam(hand.verts_padded()[0] @ cv_from_gl[:3, :3].T)
                ov = to_cam(obj.verts_padded()[0] @ obj_to_cv)
                if side_view_y_clockwise_degrees is not None:
                    hv, ov = self.side_view(hv, ov, t, side_view_y_clockwise_degrees)
                mesh_hand = p3d_structs.Meshes(verts=[hv], faces=[hand.faces_padded()[0]], textures=self.flat_texture(hv, (0.75, 0.80, 0.90)))
                mesh_obj = p3d_structs.Meshes(verts=[ov], faces=[obj.faces_padded()[0]], textures=self.flat_texture(ov, (0.90, 0.35, 0.30)))
                cams = self.mesh_camera(self.Ks_pix[t])
                renderer.shader.lights = self.lights(cams)
                rgba = renderer(join_meshes_as_scene([mesh_obj, mesh_hand]), cameras=cams)[0]
                rgb, alpha = rgba[..., :3].clamp(0, 1).cpu().numpy(), rgba[..., 3:4].clamp(0, 1).cpu().numpy()
                base = np.zeros_like(rgb) if blank_background else self.rgbs[t].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                out = rgb * alpha + base * (1 - alpha)
                if side_view_y_clockwise_degrees is None and not blank_background:
                    # Draw the object over the hand where the object mask says it is visible.
                    orgba = renderer(mesh_obj, cameras=cams)[0]
                    oa = orgba[..., 3:4].clamp(0, 1).cpu().numpy()
                    comp = orgba[..., :3].clamp(0, 1).cpu().numpy() * oa + base * (1 - oa)
                    draw = (self.masks_obj[t][0].cpu().numpy() > 0.5) & (oa[..., 0] > 1e-3)
                    out[draw] = comp[draw]
                frames.append((np.clip(out, 0, 1) * 255).astype(np.uint8))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out_path, frames, fps=10)

    def side_view(self, hv, ov, t, degrees):
        """Orbit the scene about its center and fit it into the image."""
        theta = -np.deg2rad(degrees)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rot = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=self.device)
        both = torch.cat([hv, ov])
        center = both.mean(0, keepdim=True)
        hv, ov = (hv - center) @ rot.T + center, (ov - center) @ rot.T + center
        xy = torch.cat([hv, ov])[:, :2].mean(0)
        hv[:, :2] -= xy
        ov[:, :2] -= xy
        both = torch.cat([hv, ov])
        fx, fy = float(self.Ks_pix[t][0, 0]), float(self.Ks_pix[t][1, 1])
        need = max(0.20, fx * float(both[:, 0].abs().max()) / max(1.0, 0.44 * self.W),
                   fy * float(both[:, 1].abs().max()) / max(1.0, 0.44 * self.H))
        shift = max(0.0, need - float(both[:, 2].min()))
        hv[:, 2] += shift
        ov[:, 2] += shift
        return hv, ov

    def flat_texture(self, verts, color):
        return TexturesVertex(verts_features=torch.tensor(color, device=self.device).expand(1, verts.shape[0], 3).contiguous())

    def mesh_camera(self, K):
        return cameras_from_opencv_projection(
            R=torch.eye(3, device=self.device)[None], tvec=torch.zeros(1, 3, device=self.device),
            camera_matrix=K.float()[None], image_size=torch.tensor([[self.H, self.W]], device=self.device).float(),
        )

    def lights(self, cameras=None):
        d = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        if cameras is not None:
            d = cameras.get_world_to_view_transform().inverse().transform_normals(d[:, None])[:, 0]
        zeros = torch.zeros((1, 3), device=self.device)
        return DirectionalLights(device=self.device, ambient_color=zeros + 1.0, diffuse_color=zeros,
                                 specular_color=zeros + 1.0, direction=d)
