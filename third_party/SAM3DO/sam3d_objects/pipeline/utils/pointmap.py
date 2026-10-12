# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Utility functions for point map processing and intrinsics inference.
Extracted from moge library for use in sam3d_objects pipeline.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
import utils3d


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """MoGe's diagonal-normalized image-plane coordinates."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> tuple[np.float32, float]:
    """Solve MoGe's focal/shift reprojection fit."""
    from scipy.optimize import least_squares

    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift: np.ndarray) -> np.ndarray:
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    solution = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    shift = solution.x.squeeze().astype(np.float32)
    xy_proj = xy / (z + shift)[:, None]
    focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return shift, focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> np.float32:
    """Solve MoGe's fixed-focal z-shift reprojection fit."""
    from scipy.optimize import least_squares

    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift: np.ndarray) -> np.ndarray:
        return (focal * xy / (z + shift)[:, None] - uv).ravel()

    solution = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    return solution.x.squeeze().astype(np.float32)


def recover_focal_shift(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    focal: Optional[torch.Tensor] = None,
    downsample_size: Tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover focal and z shift using the exact MoGe fitting procedure."""
    shape = points.shape
    height, width = points.shape[-3:-1]
    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = F.interpolate(
        uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).squeeze(0).permute(1, 2, 0)
    mask_lr = None
    if mask is not None:
        mask_lr = F.interpolate(
            mask.float().unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1) > 0

    uv_np = uv_lr.cpu().numpy()
    points_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_np = mask_lr.cpu().numpy() if mask_lr is not None else None
    shifts: list[float] = []
    focals: list[float] = []
    for i in range(points.shape[0]):
        points_i = points_np[i] if mask_np is None else points_np[i][mask_np[i]]
        uv_i = uv_np if mask_np is None else uv_np[mask_np[i]]
        if uv_i.shape[0] < 2:
            shifts.append(0.0)
            focals.append(1.0 if focal is None else float(focal_np[i]))
            continue
        if focal is None:
            shift_i, focal_i = solve_optimal_focal_shift(uv_i, points_i)
            focals.append(float(focal_i))
        else:
            shift_i = solve_optimal_shift(uv_i, points_i, focal_np[i])
        shifts.append(float(shift_i))

    shift_out = torch.tensor(shifts, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal is None:
        focal_out = torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        focal_out = focal.reshape(shape[:-3])
    return focal_out, shift_out


def infer_intrinsics_from_pointmap(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    fov_x: Optional[Union[float, torch.Tensor]] = None,
    mask_threshold: float = 0.5,
    force_projection: bool = False,
    apply_mask: bool = False,
    device: Optional[torch.device] = None
) -> dict:
    """
    Infer camera intrinsics from a point map.
    
    Exact implementation matching moge library's inference logic.
    
    Args:
        points: Point map tensor of shape (B, H, W, 3) or (H, W, 3)
        mask: Optional mask tensor of shape (B, H, W) or (H, W)
        fov_x: Optional horizontal field of view in degrees. If None, inferred from points
        mask_threshold: Threshold for binary mask creation
        force_projection: If True, recompute points using depth and intrinsics
        apply_mask: If True, apply mask to output points and depth
        device: Device for computation. If None, uses points.device
    
    Returns:
        Dictionary containing:
        - 'points': Camera-space points
        - 'intrinsics': Camera intrinsics matrix
        - 'depth': Depth map
        - 'mask': Binary mask
    """
    if device is None:
        device = points.device
    
    # Handle batch dimension
    squeeze_batch = False
    if points.dim() == 3:
        points = points.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        squeeze_batch = True
    
    height, width = points.shape[1:3]
    aspect_ratio = width / height
    
    # Always process the output in fp32 precision
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        points, mask, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, mask, fov_x])

        mask_binary = mask > mask_threshold if mask is not None else torch.ones_like(points[..., 0], dtype=torch.bool)
        
        # Add finite check to handle NaN and inf values
        finite_mask = torch.isfinite(points).all(dim=-1)
        mask_binary = mask_binary & finite_mask

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]
        
        # If projection constraint is forced, recompute the point map using the actual depth map
        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            shift_stacked = torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]
            points = points + shift_stacked

        # Apply mask if needed
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

    return_dict = {
        'points': points.squeeze(0) if squeeze_batch else points,
        'intrinsics': intrinsics.squeeze(0) if squeeze_batch else intrinsics,
        'depth': depth.squeeze(0) if squeeze_batch else depth,
        'mask': mask_binary.squeeze(0) if squeeze_batch else mask_binary,
    }
    
    return return_dict
