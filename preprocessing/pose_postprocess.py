import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def rotation_delta_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = F.normalize(q1, dim=-1).mul(F.normalize(q2, dim=-1)).sum(dim=-1).abs()
    return 2 * torch.acos(dot.clamp(-1, 1)) * (180 / np.pi)


def correct_pose_flips(poses: dict[str, dict[str, torch.Tensor]]) -> None:
    """Replace isolated SAM3D half-turns without smoothing translation."""
    previous = None
    half_turns = None
    for key in sorted(poses, key=int):
        rotation = poses[key]["rotation"]
        if previous is None:
            previous = rotation
            half_turns = torch.tensor(
                [[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
                device=rotation.device,
                dtype=rotation.dtype,
            )
            continue
        delta = rotation_delta_deg(rotation, previous)
        if delta > 80:
            candidates = quat_mul(half_turns, rotation.expand_as(half_turns))
            candidate_delta, candidate_idx = rotation_delta_deg(candidates, previous).min(dim=0)
            if candidate_delta < 80:
                previous = candidates[candidate_idx]
        else:
            previous = rotation
        poses[key]["rotation"] = previous


def smooth_pose_rotations(
    poses: dict[str, dict[str, torch.Tensor]],
    radius: int = 2,
    passes: int = 2,
) -> None:
    """Symmetric local quaternion smoothing for a noisy pose cache.

    Quaternion signs are aligned to the center sample before averaging.  This
    is intentionally an explicit post-processing operation; the paper path
    only calls :func:`correct_pose_flips`.
    """
    keys = sorted(poses, key=int)
    if len(keys) < 3 or radius < 1 or passes < 1:
        return
    rotations = torch.stack([
        torch.as_tensor(poses[key]["rotation"]).reshape(-1, 4)[0]
        for key in keys
    ])
    rotations = F.normalize(rotations, dim=-1)
    radius = int(radius)
    for _ in range(int(passes)):
        smoothed = []
        for i in range(len(keys)):
            lo = max(0, i - radius)
            hi = min(len(keys), i + radius + 1)
            q = rotations[lo:hi]
            center = rotations[i]
            q = torch.where((q * center).sum(-1, keepdim=True) < 0, -q, q)
            offsets = torch.arange(lo, hi, device=q.device) - i
            weights = (radius + 1 - offsets.abs()).clamp_min(1).to(q.dtype)
            smoothed.append(F.normalize((q * weights[:, None]).sum(0), dim=0))
        rotations = torch.stack(smoothed)
    for key, rotation in zip(keys, rotations):
        original = poses[key]["rotation"]
        poses[key]["rotation"] = rotation.reshape_as(original)


def rescale_pose_sequence(
    poses: dict[str, dict[str, torch.Tensor]], multiplier: float,
) -> None:
    """Multiply the sequence-wide object scale without changing translation."""
    for pose in poses.values():
        pose["scale"] = pose["scale"] * float(multiplier)


def postprocess_sequence(scene_dir: str | Path, preserve_existing_model: bool = False) -> None:
    """Install SAM3D outputs after correcting pose flips."""
    sam3d_dir = Path(scene_dir) / "sam3d"
    input_path = sam3d_dir / "sam3d_poses_wo_postprocess" / "poses_dict.pt"
    pose_path = sam3d_dir / "sam3d_poses" / "poses_dict.pt"
    if input_path.exists():
        poses = torch.load(input_path, map_location="cpu", weights_only=False)
        correct_pose_flips(poses)
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(poses, pose_path)
    elif not pose_path.exists():
        raise FileNotFoundError(f"Missing SAM3D poses: {input_path}")

    source = sam3d_dir / "sam3d_model_wo_postprocess"
    target = sam3d_dir / "sam3d_model"
    if not preserve_existing_model and source.exists():
        if target.exists():
            shutil.rmtree(target)
        source.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit SAM3D pose-cache diagnostics.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rotation-radius", type=int, default=2)
    parser.add_argument("--rotation-passes", type=int, default=2)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    poses = torch.load(args.input, map_location="cpu", weights_only=False)
    correct_pose_flips(poses)
    smooth_pose_rotations(poses, args.rotation_radius, args.rotation_passes)
    rescale_pose_sequence(poses, args.scale_multiplier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(poses, args.output)


if __name__ == "__main__":
    main()
