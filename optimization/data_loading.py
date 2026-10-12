"""Load and temporally align per-sequence inputs (pointmaps, DynHaMR hand
results, 2D joints) and assemble per-frame metadata for optimization."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from grag_common.project_paths import WORK_DIR


@dataclass(frozen=True)
class ExperimentPaths:
    """Convenience holder for frequently used directories."""

    exp: str
    sam3d_dir: str = "sam3d"
    root: Path = WORK_DIR

    def __post_init__(self) -> None:
        base = self.root / self.exp
        sam3d_root = base / self.sam3d_dir
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "rgb", base / "rgb")
        object.__setattr__(self, "obj_masks", base / "masks")
        object.__setattr__(self, "hand_masks", base / "hand_masks")
        object.__setattr__(self, "obj_pose", sam3d_root / "sam3d_poses" / "poses_dict.pt")
        object.__setattr__(self, "slat_coords_init", sam3d_root / "sam3d_model" / "object_slat_coords.pt")
        object.__setattr__(self, "pointmap_res", base / "da3_pointmaps_intrinsics_extrinsics.pt")
        object.__setattr__(self, "dynhamr_joints2d", base / "dynhamr" / "joints2d.pt")


def load_pointmap_results(exp: str, root: Path = WORK_DIR) -> dict:
    pointmap_path = ExperimentPaths(exp, root=root).pointmap_res
    if not pointmap_path.exists():
        raise FileNotFoundError(f"Could not find pointmap results for experiment '{exp}' at {pointmap_path}")
    return torch.load(pointmap_path, map_location="cpu")


def load_dynhamr_results(
    exp: str,
    fit: str = "smooth_fit",
    dynhamr_dir_name: str = "dynhamr",
    root: Path = WORK_DIR,
) -> np.lib.npyio.NpzFile:
    dynhamr_dir = root / exp / str(dynhamr_dir_name) / "outputs"
    if 'ABLATION' in exp:
        # remove ABLATION from the experiment name to match the folder structure
        exp = exp.replace('_ABLATION', '')
    pattern = f"*/{exp}*/{fit}/*_world_results.npz"
    matches = list(dynhamr_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find any DynHaMR results for experiment '{exp}' in {dynhamr_dir}")
    latest = max(matches, key=lambda path: path.stat().st_mtime_ns)
    return np.load(latest, allow_pickle=True)


def validate_dynhamr_cameras(dynhamr_res, pointmap_res, frame_ids) -> None:
    """Reject hand fits made with cameras other than the current DA3 cameras."""
    cameras = np.asarray(pointmap_res["extrinsics"])[frame_ids]
    rotations = np.asarray(dynhamr_res["cam_R"])
    translations = np.asarray(dynhamr_res["cam_t"])
    rotations_match = np.allclose(rotations, cameras[None, :, :3, :3], atol=1e-5, rtol=1e-5)
    translations_match = np.allclose(translations, cameras[None, :, :3, 3], atol=1e-5, rtol=1e-5)
    if not rotations_match or not translations_match:
        raise ValueError(
            "DynHaMR was fitted with different cameras from the current DA3 cache. "
            "Rerun preprocessing with --dynhamr_overwrite before optimization."
        )


def extract_frame_idx_from_name(name: str) -> int | None:
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        pass
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def infer_dynhamr_frame_indices(
    paths: ExperimentPaths,
    dynhamr_res: np.lib.npyio.NpzFile,
    dynhamr_root: Path | None = None,
) -> np.ndarray:
    """Recover which RGB frame each DynHaMR timestep corresponds to."""
    n_dyn = int(dynhamr_res["trans"].shape[1]) if "trans" in dynhamr_res else 0
    if n_dyn <= 0:
        return np.zeros((0,), dtype=np.int64)

    for key in ["frame_indices", "frame_idx", "frame_ids", "indices"]:
        if key in dynhamr_res.files:
            vals = np.asarray(dynhamr_res[key]).astype(np.int64).reshape(-1)
            if vals.size == n_dyn:
                return vals

    # 1) track_preds/*/*_mano.json usually preserves original frame ids.
    if dynhamr_root is None:
        dynhamr_root = paths.base / "dynhamr"
    track_root = dynhamr_root / "track_preds" / paths.exp
    mano_files = sorted(track_root.glob("*/*_mano.json"))
    if mano_files:
        ids = []
        for p in mano_files:
            tok = p.name.split("_")[0]
            try:
                ids.append(int(tok))
            except ValueError:
                continue
        ids = np.asarray(sorted(set(ids)), dtype=np.int64)
        if ids.size == n_dyn:
            return ids

    # 2) HAMER pkl can encode empty detections per frame; keep only frames with mano predictions.
    pkl_path = dynhamr_root / "hamer_out" / paths.exp / f"{paths.exp}.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            valid_ids: list[int] = []
            if isinstance(pkl_data, dict):
                for k, rec in sorted(pkl_data.items()):
                    frame_idx = extract_frame_idx_from_name(k)
                    if frame_idx is None:
                        continue
                    mano_list = rec.get("mano", []) if isinstance(rec, dict) else []
                    if isinstance(mano_list, list) and len(mano_list) > 0:
                        valid_ids.append(frame_idx)
            ids = np.asarray(valid_ids, dtype=np.int64)
            if ids.size == n_dyn:
                return ids
        except Exception:
            pass

    # Fallback: assume contiguous starting at zero.
    return np.arange(n_dyn, dtype=np.int64)


def fill_nearest_by_index(values: np.ndarray, src_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill from sparse frame ids to dense timeline."""
    if values.shape[0] == 0:
        raise ValueError("Cannot fill from empty source values.")
    src_ids = np.asarray(src_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    right = np.searchsorted(src_ids, target_ids, side="left")
    right = np.clip(right, 0, len(src_ids) - 1)
    left = np.clip(right - 1, 0, len(src_ids) - 1)
    right_dist = np.abs(src_ids[right] - target_ids)
    left_dist = np.abs(src_ids[left] - target_ids)
    choose_left = left_dist <= right_dist
    picked = np.where(choose_left, left, right)
    return values[picked]


def fill_linear_by_index(values: np.ndarray, src_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Linear interpolation (with edge clamping) from sparse frame ids to dense timeline."""
    if values.shape[0] == 0:
        raise ValueError("Cannot fill from empty source values.")
    src_ids = np.asarray(src_ids, dtype=np.float64)
    target_ids = np.asarray(target_ids, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1).astype(np.float32)
    out = np.empty((target_ids.shape[0], flat.shape[1]), dtype=np.float32)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(target_ids, src_ids, flat[:, d])
    return out.reshape((target_ids.shape[0],) + values.shape[1:])


def densify_temporal_tensor(
    tensor: torch.Tensor,
    src_frame_ids: np.ndarray,
    target_len: int,
    *,
    is_binary: bool = False,
    use_linear: bool | None = None,
) -> torch.Tensor:
    if target_len <= 0:
        return tensor
    src_len = int(src_frame_ids.shape[0])
    if src_len == target_len:
        return tensor
    if src_len == 0:
        raise ValueError("DynHaMR source sequence has zero valid frames.")
    sort_idx = np.argsort(src_frame_ids)
    src_ids_sorted = np.asarray(src_frame_ids, dtype=np.int64)[sort_idx]

    # Most DynHaMR tensors are shaped (1, T, ...). Some can be (T, ...).
    if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == src_len:
        head_shape = tensor.shape[:1]
        tail_shape = tensor.shape[2:]
        src_values = tensor.detach().cpu().numpy().reshape(src_len, -1)[sort_idx]
        src_ids_sorted, unique_pos = np.unique(src_ids_sorted, return_index=True)
        src_values = src_values[unique_pos]
        if use_linear is None:
            use_linear = not is_binary
        filler = fill_linear_by_index if use_linear else fill_nearest_by_index
        dense = filler(src_values, src_ids_sorted, np.arange(target_len, dtype=np.int64))
        dense = dense.reshape(head_shape + (target_len,) + tail_shape)
        dense_t = torch.from_numpy(dense).to(device=tensor.device, dtype=tensor.dtype)
        if is_binary:
            dense_t = (dense_t > 0.5).to(dtype=tensor.dtype)
        return dense_t

    if tensor.ndim >= 1 and tensor.shape[0] == src_len:
        tail_shape = tensor.shape[1:]
        src_values = tensor.detach().cpu().numpy().reshape(src_len, -1)[sort_idx]
        src_ids_sorted, unique_pos = np.unique(src_ids_sorted, return_index=True)
        src_values = src_values[unique_pos]
        if use_linear is None:
            use_linear = not is_binary
        filler = fill_linear_by_index if use_linear else fill_nearest_by_index
        dense = filler(src_values, src_ids_sorted, np.arange(target_len, dtype=np.int64))
        dense = dense.reshape((target_len,) + tail_shape)
        dense_t = torch.from_numpy(dense).to(device=tensor.device, dtype=tensor.dtype)
        if is_binary:
            dense_t = (dense_t > 0.5).to(dtype=tensor.dtype)
        return dense_t

    return tensor


def despike_translation_sequence(
    tensor: torch.Tensor,
    max_step_m: float = 0.06,
    z_mad_thresh: float = 6.0,
    max_passes: int = 3,
) -> torch.Tensor:
    """Remove one-frame MANO translation pops while preserving slow DynHaMR motion."""
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] < 3:
        return tensor
    out = tensor.clone()
    trans = out[0]

    for _ in range(int(max_passes)):
        vel = trans[1:] - trans[:-1]
        step = torch.linalg.norm(vel, dim=-1)
        if step.numel() == 0:
            break

        median_step = step.median()
        mad_step = (step - median_step).abs().median().clamp_min(1e-6)
        step_limit = torch.maximum(
            torch.tensor(float(max_step_m), device=tensor.device, dtype=tensor.dtype),
            median_step + float(z_mad_thresh) * mad_step,
        )
        bad_step = step > step_limit

        # Mark an interior frame as a spike if both adjacent jumps are bad and point in opposite directions.
        bad_frames = torch.zeros((trans.shape[0],), device=tensor.device, dtype=torch.bool)
        for i in range(1, trans.shape[0] - 1):
            if not (bool(bad_step[i - 1]) and bool(bad_step[i])):
                continue
            v_prev = trans[i] - trans[i - 1]
            v_next = trans[i + 1] - trans[i]
            if torch.dot(v_prev, v_next) < 0:
                bad_frames[i] = True

        # Also mark isolated depth outliers relative to local linear interpolation.
        interp = 0.5 * (trans[:-2] + trans[2:])
        residual = torch.linalg.norm(trans[1:-1] - interp, dim=-1)
        median_res = residual.median()
        mad_res = (residual - median_res).abs().median().clamp_min(1e-6)
        res_limit = torch.maximum(
            torch.tensor(float(max_step_m), device=tensor.device, dtype=tensor.dtype),
            median_res + float(z_mad_thresh) * mad_res,
        )
        bad_frames[1:-1] |= residual > res_limit

        if not bool(bad_frames.any()):
            break
        trans[bad_frames] = 0.5 * (trans[torch.where(bad_frames)[0] - 1] + trans[torch.where(bad_frames)[0] + 1])

    out[0] = trans
    return out


def load_and_align_joints2d(
    joints2d_path: Path,
    dynhamr_frame_ids: np.ndarray,
    target_len: int,
    hand_index: int = 0,
) -> torch.Tensor:
    raw = torch.load(joints2d_path, map_location="cpu")
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError(f"No hand entries in {joints2d_path}")
        idx = min(max(0, int(hand_index)), len(raw) - 1)
        raw = raw[idx]
    joints = torch.as_tensor(raw).float()
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]
    if joints.ndim != 3:
        raise ValueError(f"Unexpected joints2d shape {tuple(joints.shape)} in {joints2d_path}")

    src_len = joints.shape[0]
    frame_ids = np.asarray(dynhamr_frame_ids, dtype=np.int64)
    if frame_ids.size != src_len:
        frame_ids = np.arange(src_len, dtype=np.int64)
    joints = densify_temporal_tensor(joints, frame_ids, target_len, is_binary=False)
    return joints


def build_frames_metadata(paths: ExperimentPaths, pointmap_res, joints2d_observed: torch.Tensor) -> dict[str, dict]:
    rgb_files = sorted(paths.rgb.glob("*.png"))
    obj_mask_files = sorted(paths.obj_masks.glob("*.png"))
    hand_mask_files = sorted(paths.hand_masks.glob("*.png"))
    obj_poses = torch.load(paths.obj_pose, map_location="cpu")
    # Match masks to frames by stem, not list position: a single missing mask file
    # would otherwise shift every later frame's mask by one. If a frame's hand mask
    # is missing (SAM3 can drop frames), fall back to the temporally nearest one.
    obj_mask_by_stem = {p.stem: p for p in obj_mask_files}
    hand_mask_by_stem = {p.stem: p for p in hand_mask_files}
    hand_mask_stems = sorted(hand_mask_by_stem.keys())

    def nearest_hand_mask(stem: str):
        if stem in hand_mask_by_stem:
            return hand_mask_by_stem[stem]
        if not hand_mask_stems:
            return None
        try:
            tgt = int(stem)
            nearest = min(hand_mask_stems, key=lambda s: abs(int(s) - tgt))
        except ValueError:
            nearest = hand_mask_stems[0]
        print(f"[WARN] hand mask for frame {stem} missing; using nearest ({nearest}).")
        return hand_mask_by_stem[nearest]

    frames_meta: dict[str, dict] = {}
    for idx, rgb_path in enumerate(rgb_files):
        if idx >= len(obj_poses):
            break
        if idx >= len(joints2d_observed):
            break
        mask_path = obj_mask_by_stem.get(rgb_path.stem)
        hand_mask_path = nearest_hand_mask(rgb_path.stem)
        if not (mask_path and mask_path.exists()):
            continue
        
        meta = {
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path),
            "hand_mask_path": str(hand_mask_path),
            "K": pointmap_res["intrinsics"][idx],
            "w2c_cv": pointmap_res["extrinsics"][idx],
            "pointmap": pointmap_res['pointmaps'][idx], # in pytorch3d coordinate system
            "pose_obj_pt3d": obj_poses[f'{idx}'], # obj poses are in pytorch3d format due to sam3d-objects
            "joints2d_observed": joints2d_observed[idx], # 2d joints observed from vitpose obntained in dynhamr
        }
        frames_meta[rgb_path.stem] = meta
    return frames_meta
