import os

import numpy as np
import open3d as o3d

opencv_to_pytorch3d = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
])

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def depths_to_world_points_per_frame(prediction):
    """
    prediction.depth:       (F, H, W)
    prediction.intrinsics:  (F, 3, 3)
    prediction.extrinsics:  (F, 3, 4)  # w2c
    prediction.processed_images: (F, H, W, 3) or (F, H, W, 4)

    Returns:
        points_world: (F, H, W, 3)
        colors:       (F, H, W, 3) in [0,1]
    """
    depth = prediction.depth          # (F, H, W)
    K = prediction.intrinsics         # (F, 3, 3)
    # make extrinsics homogeneous 4x4
    w2c = np.array([_as_homogeneous44(ext) for ext in prediction.extrinsics])  # (F, 4, 4)
    images = prediction.processed_images  # (F, H, W, C)

    F, H, W = depth.shape

    # pixel grid (same for all frames)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # (H, W)
    ones = np.ones_like(xs)
    pix = np.stack([xs, ys, ones], axis=-1).astype(np.float32)       # (H, W, 3)
    pix = np.broadcast_to(pix, (F, H, W, 3))                         # (F, H, W, 3)

    # invert intrinsics per frame
    K_inv = np.linalg.inv(K)                                         # (F, 3, 3)

    # apply K^{-1} to pixel rays
    pix_flat = pix.reshape(F, -1, 3)                                 # (F, N, 3), N = H*W
    rays_cam = (K_inv @ pix_flat.transpose(0, 2, 1)).transpose(0, 2, 1)  # (F, N, 3)
    rays_cam = rays_cam.reshape(F, H, W, 3)                          # (F, H, W, 3)

    # multiply by depth -> camera-space points
    depth_ = depth[..., None]                                        # (F, H, W, 1)
    pts_cam = rays_cam * depth_                                      # (F, H, W, 3)

    # homogeneous and transform to world (w2c -> c2w)
    ones_hw = np.ones_like(depth_)[..., 0:1]                         # (F, H, W, 1)
    pts_cam_h = np.concatenate([pts_cam, ones_hw], axis=-1)          # (F, H, W, 4)
    pts_cam_h_flat = pts_cam_h.reshape(F, -1, 4)[..., None]          # (F, N, 4, 1)

    c2w = np.linalg.inv(w2c)                                         # (F, 4, 4)
    c2w = c2w[:, None, :, :]                                         # (F, 1, 4, 4)

    pts_world_flat = (c2w @ pts_cam_h_flat)[..., 0]                  # (F, N, 4)
    pts_world = pts_world_flat[..., :3].reshape(F, H, W, 3)          # (F, H, W, 3)

    # colors: keep only RGB, normalize to [0,1] if needed
    colors = prediction.processed_images[..., :3].astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0

    # optional: mark invalid depth as NaN, but keep full H×W grid
    invalid = depth <= 0
    pts_world[invalid] = np.nan

    # convert to pytorch3d for sam3d
    pts_world = pts_world @ opencv_to_pytorch3d

    return pts_world, colors

def save_per_frame_pointclouds(prediction, out_dir):
    """
    Saves one .ply colored point cloud per frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    pts_world, colors = depths_to_world_points_per_frame(prediction)
    F, H, W, _ = pts_world.shape

    for f in range(F):
        pts = pts_world[f].reshape(-1, 3)
        cols = colors[f].reshape(-1, 3)

        # drop NaNs (invalid depth) but this *still* comes from a H×W grid
        valid = np.isfinite(pts).all(axis=-1)
        pts_valid = pts[valid]
        cols_valid = cols[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_valid)
        pcd.colors = o3d.utility.Vector3dVector(cols_valid)

        fname = os.path.join(out_dir, f"{f:06d}.ply")
        o3d.io.write_point_cloud(fname, pcd)
        print(f"[frame {f}] wrote {pts_valid.shape[0]} points to {fname}")
