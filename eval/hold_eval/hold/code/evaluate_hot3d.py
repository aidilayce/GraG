#!/usr/bin/env python3
"""Evaluate HOT3D predictions against dynamic-camera-correct ground truth.

HOT3D uses a moving head-mounted camera, so the GT hand+object is taken per-frame
in the camera frame from a viewer-cache npz built by
`eval/hot3d/export_hot3d_gt_viewer_cache.py` from the raw HOT3D dataset.
"""
import argparse
import functools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Dir of per-clip GT caches (<clip>.npz) built by export_hot3d_gt_viewer_cache.py.
DEFAULT_GT_VIEWER_CACHE_DIR = Path(
    os.environ.get("HOT3D_VIEWER_CACHE_DIR", "hot3d_results/viewer_cache")
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_path", required=True, type=Path)
    p.add_argument("--gt_viewer_cache_dir", type=Path, default=DEFAULT_GT_VIEWER_CACHE_DIR,
                   help="Dir of <clip>.npz GT caches (from eval/hot3d/export_hot3d_gt_viewer_cache.py).")
    p.add_argument("--include_icp", action="store_true")
    return p.parse_args()


def load_pred_data(pred_dir, eval_m, xdict):
    v3d_c_object = np.load(pred_dir / "pred_v3d_c_object.npy")
    v3d_c_right = np.load(pred_dir / "pred_v3d_c_right.npy")
    j3d_c_right = np.load(pred_dir / "pred_j3d_c_right.npy")

    data_pred = xdict()
    data_pred["v3d_c.object"] = torch.from_numpy(v3d_c_object).float()
    data_pred["v3d_c.right"] = torch.from_numpy(v3d_c_right).float()
    data_pred["j3d_c.right"] = torch.from_numpy(j3d_c_right).float()
    data_pred["full_seq_name"] = pred_dir.name
    data_pred["faces"] = {
        "object": torch.from_numpy(np.load(pred_dir / "pred_faces_object.npy")).long(),
        "right": torch.from_numpy(np.load(pred_dir / "pred_faces_right.npy")).long(),
    }
    root_object = torch.from_numpy(eval_m.compute_bounding_box_centers(v3d_c_object)).float()
    root_right = data_pred["j3d_c.right"][:, 0]
    data_pred["root.object"] = root_object
    data_pred["v3d_ra.object"] = data_pred["v3d_c.object"] - root_object[:, None, :]
    data_pred["root.right"] = root_right
    data_pred["j3d_ra.right"] = data_pred["j3d_c.right"] - data_pred["j3d_c.right"][:, :1, :]
    data_pred["v3d_right.object"] = data_pred["v3d_c.object"] - root_right[:, None, :]
    return data_pred


def resample_temporal_data_nearest(data, target_len):
    if "j3d_c.right" not in data:
        return data
    src_len = int(data["j3d_c.right"].shape[0])
    if target_len <= 0 or src_len <= 0 or target_len == src_len:
        return data
    idx = np.round(np.linspace(0, src_len - 1, num=target_len)).astype(np.int64)
    idx = np.clip(idx, 0, src_len - 1)
    for key, value in list(data.items()):
        if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0 and value.shape[0] == src_len:
            if hasattr(data, "overwrite"):
                data.overwrite(key, value[idx])
            else:
                data[key] = value[idx]
    return data


def load_gt(seq_name, cache_dir, xdict, eval_m):
    """GT xdict from the HOT3D viewer cache: per-frame hand/object verts in the
    camera frame, correctly registered under the dynamic camera."""
    from hold.common.body_models import build_mano_aa

    cache_path = Path(cache_dir) / f"{seq_name}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"No HOT3D GT cache at {cache_path}. Build it with "
            f"eval/hot3d/export_hot3d_gt_viewer_cache.py, or set HOT3D_VIEWER_CACHE_DIR."
        )
    d = np.load(cache_path, allow_pickle=True)

    v3d_h = d["v3d_h"].astype(np.float32)   # (T, 778, 3) camera frame
    v3d_o = d["v3d_o"].astype(np.float32)   # (T, No, 3)  camera frame
    hand_side = str(d["hand_side"]) if "hand_side" in d else (
        "left" if seq_name.endswith("left") else "right"
    )
    T = v3d_h.shape[0]

    mano_layer = build_mano_aa(hand_side == "right", flat_hand=False)
    v_t = torch.from_numpy(v3d_h).float()
    j_t = torch.einsum("bik,ji->bjk", [v_t, mano_layer.J_regressor])
    j3d = mano_layer.vertex_joint_selector(v_t, j_t).detach().cpu().numpy().astype(np.float32)

    # HOLD marks invalid frames with huge coords -> is_valid.
    bad = (
        (np.abs(v3d_h).reshape(T, -1).max(1) > 100.0)
        | (np.abs(v3d_o).reshape(T, -1).max(1) > 100.0)
    )
    is_valid = (~bad).astype(np.float32)

    out = {}
    out["v3d_c.right"] = v3d_h
    out["v3d_c.object"] = v3d_o
    out["j3d_c.right"] = j3d
    out["faces.object"] = d["faces_o"].astype(np.int64)
    out["faces.right"] = d["faces_h"].astype(np.int64)
    out["K"] = d["K"].astype(np.float32)
    out["is_valid"] = is_valid
    out["hot3d.hand_side"] = hand_side
    out["full_seq_name"] = seq_name

    root_j3d = j3d[:, :1]
    root_o = eval_m.compute_bounding_box_centers(v3d_o)[:, None]
    out["root.right"] = root_j3d[:, 0]
    out["root.object"] = root_o[:, 0]
    out["v3d_right.object"] = v3d_o - root_j3d
    out["j3d_ra.right"] = j3d - root_j3d
    out["v3d_ra.object"] = v3d_o - root_o
    return xdict(out).to_torch()


def main():
    args = parse_args()
    # Make the vendored HOLD closure importable (hold.*, common.*, src.*).
    _hold_eval = Path(__file__).resolve().parents[2]  # eval/hold_eval
    for _p in (_hold_eval, _hold_eval / "hold", _hold_eval / "hold" / "code"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from hold.common.xdict import xdict
    import src.utils.eval_modules as eval_m

    data_pred = load_pred_data(args.pred_path, eval_m, xdict)
    seq_name = data_pred["full_seq_name"]
    data_gt = load_gt(seq_name, args.gt_viewer_cache_dir, xdict, eval_m)
    data_gt = resample_temporal_data_nearest(data_gt, int(data_pred["j3d_c.right"].shape[0]))

    eval_fn_dict = {
        "mpjpe_ra_r": eval_m.eval_mpjpe_right,
        "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
        "cd_f_ra": eval_m.eval_cd_f_ra,
        "cd_f_right": functools.partial(eval_m.eval_cd_f_right, normalize_scale=True),
    }
    if args.include_icp:
        eval_fn_dict["icp"] = eval_m.eval_icp_first_frame

    metric_dict = {}
    for eval_fn_name, eval_fn in eval_fn_dict.items():
        print(f"Evaluating {eval_fn_name} ...")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    mean_metrics = {name: float(np.nanmean(values)) for name, values in metric_dict.items()}
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))
    mean_metrics["timestamp"] = datetime.now().strftime("%m-%d %H:%M")
    mean_metrics["seq_name"] = seq_name
    mean_metrics["gt_num_frames"] = int(data_gt["j3d_c.right"].shape[0])
    mean_metrics["pred_num_frames"] = int(data_pred["j3d_c.right"].shape[0])

    json_path = Path(str(args.pred_path) + ".metric.hot3d_compact.json")
    json_path.write_text(json.dumps(mean_metrics, indent=4) + "\n")
    np.save(Path(str(args.pred_path) + ".metric_all.hot3d_compact.npy"), metric_dict)
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()
