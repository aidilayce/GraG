'''
Script to post-process the estimated object poses. 
Maybe use FPose refinement network?
A similar pipeline is:
1. Get GT RGB and object mask, crop to object bbox obtained from object mask
2. For each frame, create a few orientation candidates --> original, mirrored left-right, mirrored up-down, mirrored both, rotated 180
3. For each candidate, render the object Gaussian and crop to object bbox from (1)
4. Get DINOv2 embeddings for both GT cropped image and rendered cropped images
5. Choose the candidate with the lowest embedding distance as the final pose for that frame
'''

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from scipy.spatial import cKDTree
from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer
from utils.dino import Dinov2
import imageio
from functools import partial
from kiui.vis import plot_image as plti
plti = partial(plti, save=True)

renderer = GaussianRenderer(
        {
            "resolution": None,  # to be set later
            "near": 0.1,
            "far": 100,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
            "backend": "gsplat",
            # "backend": "inria",
        }
    )

opencv_to_pytorch3d = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
])

pytorch3d_to_opengl = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
])

# helper: project 3D points (N,3) in camera coordinates to image pixels
def project_points_to_pixels(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    # points_cam: (N,3)
    # intrinsics: (3,3)
    xyz = points_cam
    fx = intrinsics[0,0]
    fy = intrinsics[1,1]
    cx = intrinsics[0,2]
    cy = intrinsics[1,2]
    x = (xyz[:,0] * fx) / (xyz[:,2] + 1e-8) + cx
    y = (xyz[:,1] * fy) / (xyz[:,2] + 1e-8) + cy
    pixels = torch.stack([x, y, xyz[:,2]], dim=1)  # (N,3) (u,v,depth)
    return pixels

def mask_from_projected_points(pixels: torch.Tensor, H:int, W:int) -> np.ndarray:
    '''
    pixels: (N,3) tensor
    returns binary mask (H,W) with 1 where at least one point projected
    '''
    u = pixels[:,0].cpu().numpy().round().astype(int)
    v = pixels[:,1].cpu().numpy().round().astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pixels[:,2].cpu().numpy() > 0)
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[v[valid], u[valid]] = 1
    return mask

def iou_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    if g.sum() == 0:
        return 0.0
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / float(union + 1e-9)

def align_gaussian_to_opengl(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    scene_gs.from_xyz(scene_gs.get_xyz @ torch.tensor(pytorch3d_to_opengl, device=scene_gs._xyz.device, dtype=scene_gs._xyz.dtype))
    return scene_gs

def pose_gaussians(gaussian, rotation, translation, scale, in_place=False):
    if not in_place:
        gaussian = deepcopy(gaussian)

    minimum_kernel_size = float("inf")
    if translation.ndim == 1:
        translation = translation.unsqueeze(0)
    if rotation.ndim == 1:
        rotation = rotation.unsqueeze(0)
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)

    PC = SceneVisualizer.object_pointcloud(
            points_local=gaussian.get_xyz.unsqueeze(0),
            quat_l2c=rotation,
            trans_l2c=translation,
            scale_l2c=scale,
        )
    gaussian.from_xyz(PC.points_list()[0])
    # must ... ROTATE
    gaussian.from_rotation(
        quaternion_multiply(
            quaternion_invert(rotation),
            gaussian.get_rotation,
        )
    )
    gaus_scale = gaussian.get_scaling
    
    adjusted_scale = gaus_scale * scale
    assert (scale[0, 0].item() == scale[0, 1].item() == scale[0, 2].item())
    gaussian.mininum_kernel_size *= scale[0, 0].item()
    adjusted_scale = torch.maximum(
        adjusted_scale,
        torch.tensor(
            gaussian.mininum_kernel_size * 1.1,
            device=adjusted_scale.device,
        ),
    )
    gaussian.from_scaling(adjusted_scale)
    minimum_kernel_size = min(
        minimum_kernel_size,
        gaussian.mininum_kernel_size,
    )
    gaussian.mininum_kernel_size = minimum_kernel_size
    return gaussian

# candidate generators (apply small symmetric transforms)
def generate_candidates(rot_quat: torch.Tensor, flip_axes:List[str]=['mirror_x','mirror_y','rot180','flip_ud']) -> List[Tuple[str, torch.Tensor]]:
    """
    rot_quat: torch (4,) quaternion in wxyz
    returns list of (name, quat) candidates
    """
    # We'll apply simple quaternion multipliers for 180-degree rotations about axes.
    # convert to matrix, apply, convert back
    q = rot_quat.unsqueeze(0) if rot_quat.ndim==1 else rot_quat  # (1,4)
    R = quaternion_to_matrix(q)[0]  # (3,3)
    candidates = [('orig', rot_quat.clone())]

    def mat_to_quat(M):
        return matrix_to_quaternion(M.unsqueeze(0))[0]

    if 'mirror_x' in flip_axes:
        M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=R.dtype, device=R.device)) @ R
        candidates.append(('mirror_x', mat_to_quat(M)))
    if 'mirror_y' in flip_axes:
        M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=R.dtype, device=R.device)) @ R
        candidates.append(('mirror_y', mat_to_quat(M)))
    if 'rot180' in flip_axes:
        M = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=R.dtype, device=R.device)) @ R
        candidates.append(('rot180', mat_to_quat(M)))
    if 'flip_ud' in flip_axes:
        # rotate 180 around x or flip vertical
        M = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], dtype=R.dtype, device=R.device) @ R
        candidates.append(('flip_ud', mat_to_quat(M)))

    # dedupe by name
    seen = set()
    out=[]
    for name,q in candidates:
        if name not in seen:
            out.append((name,q))
            seen.add(name)
    return out


def postprocess_sequence(
    poses: List[Dict],               # each dict: "rotation" (torch 4 or 3x3), "translation" (1x3), "scale" (1x3)
    outputs_ref,                     # outputs from decode_slat of reference frame, used to get canonical gaussians
    img_dir: str,                    # path to folder with original images 
    mask_dir: str,                   # path to folder with original masks 
    da3_result_path: str | None = None,
    device='cuda',
):
    if da3_result_path is None:
        raise ValueError("da3_result_path is required")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    dinov2_model = Dinov2().to(device)
    dinov2_model.eval()

    # load da3 pred cams
    da3_result = torch.load(da3_result_path)
    Ks = da3_result['intrinsics']
    Xs = da3_result['extrinsics']
    pm = da3_result['points'] # T, H, W, 3
    H, W = pm.shape[1], pm.shape[2]
    renderer.rendering_options['resolution'] = (H, W)

    # get canonical points in object-local space (from gaussian or mesh)
    # expect outputs_ref['gaussian_4'] or ['gaussian']
    if 'gaussian_4' in outputs_ref:
        canonical_gauss = outputs_ref['gaussian_4'][0]
    elif 'gaussian' in outputs_ref:
        canonical_gauss = outputs_ref['gaussian'][0]
    else:
        raise RuntimeError("can't find gaussian in outputs_ref")
    
    T = len(poses)
    processed_poses = []

    for i in range(T):
        pose = poses[i]
        rot = pose['rotation']         # in quaternion, torch
        trans = pose['translation']    # torch
        scale = pose['scale']          # torch

        gt_img = torchvision.io.read_image(f"{img_dir}/{i:06d}.png").permute(1,2,0).cpu().numpy()  # H,W,3
        gt_mask = torchvision.io.read_image(f"{mask_dir}/{i:06d}.png").permute(1,2,0).cpu().numpy()[:,:,0]  # H,W
        masked_cropped_gt = gt_img * gt_mask[:,:,None]
        bbox = np.where(gt_mask)
        y0, y1 = bbox[0].min(), bbox[0].max()+1
        x0, x1 = bbox[1].min(), bbox[1].max()+1
        cropped_gt = masked_cropped_gt[y0:y1, x0:x1, :] # crop to bbox
        gt_dino_feats = dinov2_model(cropped_gt.astype(np.uint8))

        gs4_posed = pose_gaussians(canonical_gauss, rot, trans, scale)
        gs4_posed_opengl = align_gaussian_to_opengl(gs4_posed, in_place=False) # take from pytorch3d to opengl

        # render img with the estimated camera pose from da3
        X_pt3d_R = Xs[i][:3, :3] @ opencv_to_pytorch3d @ pytorch3d_to_opengl
        X_pt3d_t = Xs[i][:3, 3:] 
        X_pt3d = torch.eye(4)
        X_pt3d[:3, :3] = X_pt3d_R
        X_pt3d[:3, 3:] = X_pt3d_t

        # normalize Ks (for Gaussian renderer, both inria and gsplat backends)
        Ks[i][0, 0] /= W
        Ks[i][1, 1] /= H
        Ks[i][0, 2] /= W
        Ks[i][1, 2] /= H

        color = renderer.render(gs4_posed_opengl, X_pt3d.to(device), Ks[i].to(device))['color']
        color_img = color.permute(1,2,0).cpu().numpy() * 255

        cands = generate_candidates(rot.squeeze() if rot.shape[0]==1 else rot, flip_axes=['mirror_x','mirror_y','rot180','flip_ud'])
        cos_distances = []
        for name, q in cands:
            gs4_cand = pose_gaussians(gs4_posed, q, trans, scale)
            gs4_cand_opengl = align_gaussian_to_opengl(gs4_cand, in_place=False)
            rend = renderer.render(gs4_cand_opengl, X_pt3d.to(device), Ks[i].to(device))['color']
            rend_img = rend.permute(1,2,0).cpu().numpy() * 255  
            rend_img_mask = np.where(rend_img == 255) # background is black
            bbox_rend = np.where((rend_img_mask[0]>=0) & (rend_img_mask[0]<H) & (rend_img_mask[1]>=0) & (rend_img_mask[1]<W))
            cropped_rend = rend_img[bbox_rend[0].min():bbox_rend[0].max()+1, bbox_rend[1].min():bbox_rend[1].max()+1, :]
            # resize cropped_rend to same size as cropped_gt
            cropped_rend_resized = torchvision.transforms.functional.resize(
                torch.from_numpy(cropped_rend.astype(np.uint8)).permute(2,0,1), 
                size=cropped_gt.shape[:2],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ).permute(1,2,0).numpy()
            dino_feats_cand = dinov2_model(cropped_rend_resized.astype(np.uint8))
            cos_dist = 1.0 - torch.nn.functional.cosine_similarity(gt_dino_feats.unsqueeze(0), dino_feats_cand.unsqueeze(0), dim=1)
            cos_distances.append(cos_dist.item())
        cos_distances = np.array(cos_distances)
        best_idx = cos_distances.argmin()
        best_cand = cands[best_idx]
        processed_poses.append(best_cand)
            
    # build final poses in same structure and render video
    final_poses = []
    final_frames = []
    for i, idx in enumerate(T):
        quat = processed_poses[i]
        trans = poses[i]['translation'].to(device)
        scale = poses[i]['scale'].to(device)

        gs4_posed = pose_gaussians(canonical_gauss, quat, trans, scale)
        gs4_posed_opengl = align_gaussian_to_opengl(gs4_posed, in_place=True)

        # render img with the estimated camera pose from da3
        X_pt3d_R = Xs[i][:3, :3] @ opencv_to_pytorch3d @ pytorch3d_to_opengl
        X_pt3d_t = Xs[i][:3, 3:] 
        X_pt3d = torch.eye(4)
        X_pt3d[:3, :3] = X_pt3d_R
        X_pt3d[:3, 3:] = X_pt3d_t

        # normalize Ks (for Gaussian renderer, both inria and gsplat backends)
        Ks[i][0, 0] /= W
        Ks[i][1, 1] /= H
        Ks[i][0, 2] /= W
        Ks[i][1, 2] /= H
    
        color = renderer.render(gs4_posed_opengl, X_pt3d.to(device), Ks[i].to(device))['color']
        final_frames.append((color.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

        # return rotation as quaternion to be consistent with your pipeline:
        final_poses.append({
            'rotation': quat.cpu(),
            'translation': trans.cpu(),
            'scale': scale.cpu(),
            'candidate_name': name,
        })

    imageio.mimsave(
        os.path.join(f"processed_gaussians.gif"),
        final_frames,
        format="GIF",
        duration=1000 / 10,  # default assuming 30fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )

    return final_poses

if __name__ == "__main__":
    PATH = ''
    pose_output_dir = os.path.join(f"{PATH}/sam3d_poses")
    model_output_dir = os.path.join(f"{PATH}/sam3d_model")
    poses = os.path.join(pose_output_dir, f"poses_sam3d_raw.pt")
    geo_refs = os.path.join(model_output_dir, f"object_slat_coords.pt")
    
    postprocess_sequence(
        poses=poses,
        outputs_ref=geo_refs,
        img_dir=os.path.join(PATH, 'rgb'),
        mask_dir=os.path.join(PATH, 'mask'),
        da3_result_path=os.path.join(PATH, 'da3_result.pt'),
        device='cuda',
    )
