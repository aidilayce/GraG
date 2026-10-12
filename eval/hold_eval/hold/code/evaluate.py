import functools
import os

import numpy as np
import torch
import re

import json
import src.utils.eval_modules as eval_m
import src.utils.io.gt as gt

device = "cuda:0"

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    "cd_f_ra": eval_m.eval_cd_f_ra,
    "cd_f_right": functools.partial(eval_m.eval_cd_f_right, normalize_scale=True),
}



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_p", type=str, required=False)
    parser.add_argument(
        "--align_hand",
        action="store_true",
        help="Align predictions to GT using Kabsch on hand joints, apply same transform to object.",
    )
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def _kabsch_align(source, target):
    """
    Compute rigid alignment (R, t) that maps source -> target.
    source/target: (N, 3) torch tensors.
    """
    src_mean = source.mean(dim=0, keepdim=True)
    tgt_mean = target.mean(dim=0, keepdim=True)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    H = src_centered.t() @ tgt_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.t() @ U.t()
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()
    t = tgt_mean.squeeze(0) - (R @ src_mean.squeeze(0))
    return R, t


def _align_pred_to_gt(data_pred, data_gt):
    # Use all frames for a single global alignment based on hand joints
    j_pred = data_pred["j3d_c.right"]
    j_gt = data_gt["j3d_c.right"]
    num_frames = min(j_pred.shape[0], j_gt.shape[0])
    if j_pred.shape[0] != j_gt.shape[0]:
        j_pred = j_pred[:num_frames]
        j_gt = j_gt[:num_frames]
    root_pred = j_pred[:, 0:1]
    root_gt = j_gt[:, 0:1]
    j_pred_rr = j_pred - root_pred
    j_gt_rr = j_gt - root_gt

    src = j_pred_rr.reshape(-1, 3)
    tgt = j_gt_rr.reshape(-1, 3)
    R, _ = _kabsch_align(src, tgt)

    # Align translation by matching root means after rotation
    root_pred_mean = root_pred.mean(dim=0).squeeze(0)
    root_gt_mean = root_gt.mean(dim=0).squeeze(0)
    t = root_gt_mean - (R @ root_pred_mean)

    def _apply(x):
        x_flat = x.reshape(-1, 3).t()
        x_rot = (R @ x_flat).t().reshape(x.shape)
        return x_rot + t

    data_pred.overwrite("j3d_c.right", _apply(j_pred))
    data_pred.overwrite("v3d_c.right", _apply(data_pred["v3d_c.right"][:num_frames]))
    data_pred.overwrite("v3d_c.object", _apply(data_pred["v3d_c.object"][:num_frames]))
    return data_pred


def main():
    from tqdm import tqdm

    args = parse_args()
    import src.utils.io.pred_loader as pred_loader

    data_pred = pred_loader.load_data(args.sd_p)
    seq_name = data_pred["full_seq_name"]
    if 'shorter' in seq_name:
        seq_name = seq_name.replace('_shorter','')
    if 'crop' in seq_name:
        seq_name = seq_name.replace('_crop','')
    data_gt = gt.load_data(seq_name)
    if args.align_hand:
        data_pred = _align_pred_to_gt(data_pred, data_gt)
    out_p = args.sd_p
    if os.environ.get("HOLD_SKIP_ICP", "0").lower() not in {"1", "true", "yes"}:
        eval_fn_dict["icp"] = eval_m.eval_icp_first_frame

    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # # Define the file paths
        # npy_path = out_p + ".metric_all_SHORT_TAKEN.npy"
    json_path = out_p + ".metric.json"
    npy_path = out_p + ".metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm**2), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
