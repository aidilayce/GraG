"""Stage 3 entry point: sliding-window hand-object refinement of one sequence."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["LIDRA_SKIP_INIT"] = "true"

from grag_common.project_paths import LOG_ROOT, append_project_paths

append_project_paths()

from grag_common.runtime_config import WORK_DIR
from optimization.data_loading import (
    ExperimentPaths,
    build_frames_metadata,
    infer_dynhamr_frame_indices,
    load_and_align_joints2d,
    load_dynhamr_results,
    load_pointmap_results,
    validate_dynhamr_cameras,
)
from optimization.hand_setup import prepare_hand_resources
from optimization.object_setup import load_object
from optimization.refiner import JointHandObjectRefiner
from third_party.SAM3DO.sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="sequence folder name under the data root")
    parser.add_argument("--demo_root", type=Path, default=None, help="preprocessed data root (default: GRAG_WORK_DIR)")
    parser.add_argument("--sam3d_dir", default="sam3d", help="SAM3D output folder inside the sequence folder")
    parser.add_argument("--logdir", default=str(LOG_ROOT), help="output root; runs go to <logdir>/<exp>/<timestamp>")
    parser.add_argument("--window_size", type=int, default=8, help="frames per sliding window")
    parser.add_argument("--window_stride", type=int, default=None, help="window stride (default: window_size - 1)")
    parser.add_argument("--iters_per_window", type=int, default=100, help="AdamW steps per window")
    parser.add_argument("--max_windows", type=int, default=None, help="optimize only the first N windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_hand", action="store_true", help="keep the Dyn-HaMR hand fixed")
    parser.add_argument("--no_reanchor", action="store_true", help="skip the one-time object depth reconciliation")
    parser.add_argument("--skip_final_render", "--no_render", action="store_true", help="do not render result videos")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="hoi-gaussians")
    return parser.parse_args()


def window_starts(total_frames: int, size: int, stride: int) -> list[int]:
    last = max(0, total_frames - size)
    starts = [0]
    while starts[-1] < last:
        starts.append(min(starts[-1] + stride, last))
    return starts


def set_seed(seed: int) -> None:
    """Fix RNG-dependent choices. Runs are not bit-identical: the Gaussian
    rasterizer backward uses non-deterministic atomics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(refiner: JointHandObjectRefiner, path: Path) -> None:
    """Optimized state plus the canonical object mesh needed for evaluation."""
    def cpu(v):
        return v.detach().cpu() if torch.is_tensor(v) else v

    frames = {
        fid: {k: ({kk: cpu(vv) for kk, vv in v.items()} if isinstance(v, dict) else cpu(v))
              for k, v in meta.items() if k not in ("pointmap", "joints2d_observed")}
        for fid, meta in refiner.frames.items()
    }
    names = ("obj_rotations", "obj_translations", "delta_obj_translations", "obj_scale", "obj_scale_init",
             "obj_gl_init_rot_quat", "obj_mesh_canonical_verts", "obj_mesh_faces", "hand_betas",
             "hand_global_orient", "hand_pose", "hand_trans", "hand_scale", "delta_t_hand", "is_right")
    state = {"exp": refiner.exp, "H": refiner.H, "W": refiner.W, "T": refiner.T, "frames": frames}
    state.update({name: cpu(getattr(refiner, name)) for name in names})
    torch.save(state, path)
    print(f"[save] {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = args.demo_root or WORK_DIR
    paths = ExperimentPaths(args.exp, sam3d_dir=args.sam3d_dir, root=root)

    dynhamr = load_dynhamr_results(args.exp, root=root)
    pointmaps = load_pointmap_results(args.exp, root=root)
    dynhamr_ids = infer_dynhamr_frame_indices(paths, dynhamr)
    validate_dynhamr_cameras(dynhamr, pointmaps, dynhamr_ids)
    total_frames = int(pointmaps["pointmaps"].shape[0])
    if len(dynhamr_ids) and int(dynhamr_ids.max()) + 1 < total_frames:
        total_frames = int(dynhamr_ids.max()) + 1
        for key in ("pointmaps", "intrinsics", "extrinsics"):
            pointmaps[key] = pointmaps[key][:total_frames]
        print(f"[INFO] Sequence capped at the last Dyn-HaMR frame ({total_frames} frames).")
    # In a two-hand fit the right hand is track 1.
    hand_track = 1 if dynhamr["is_right"].ndim >= 2 and dynhamr["is_right"].shape[0] == 2 else 0
    joints2d = load_and_align_joints2d(paths.dynhamr_joints2d, dynhamr_ids, total_frames, hand_index=hand_track)
    frames_meta = build_frames_metadata(paths, pointmaps, joints2d)
    if not frames_meta:
        raise ValueError(f"No valid frames found for '{args.exp}'.")

    object_gs, verts, faces, surf_pts, surf_nrm = load_object(paths)
    hand_model, hand_res = prepare_hand_resources(dynhamr, dynhamr_frame_ids=dynhamr_ids, total_frames=total_frames)
    H, W = pointmaps["pointmaps"].shape[1:3]
    renderer = GaussianRenderer({
        "resolution": (H, W), "near": 0.1, "far": 100, "ssaa": 1, "bg_color": (0, 0, 0), "backend": "gsplat",
    })

    logdir = Path(args.logdir) / args.exp / f"{args.exp}_{time.strftime('%Y%m%d-%H%M%S')}"
    logdir.mkdir(parents=True, exist_ok=True)
    refiner = JointHandObjectRefiner(
        exp=args.exp, H=H, W=W, frames_meta=frames_meta, hand_res=hand_res, object_gs=object_gs,
        hand_model=hand_model, renderer=renderer, obj_surface_points=surf_pts, obj_surface_normals=surf_nrm,
        data_root=root, logdir=logdir, use_wandb=args.use_wandb, wb_project=args.wandb_project,
        wb_run=logdir.name, obj_mesh_canonical_verts=verts, obj_mesh_faces=faces,
    )
    refiner.reanchor = not args.no_reanchor
    if args.freeze_hand:
        for name in ("hand_pose", "hand_scale", "delta_t_hand"):
            getattr(refiner, name).requires_grad_(False)

    size = max(1, min(args.window_size, refiner.T))
    starts = window_starts(refiner.T, size, args.window_stride or max(1, size - 1))[:args.max_windows]
    start_time = time.time()
    for i, start in enumerate(starts, 1):
        frames = list(range(start, min(start + size, refiner.T)))
        print(f"=== Window {i}/{len(starts)}: frames {frames[0]}-{frames[-1]} ===")
        refiner.optimize_window(frames, object_gs, iters=args.iters_per_window)
    runtime = time.time() - start_time

    save_checkpoint(refiner, logdir / "refiner_final_state.pt")
    summary = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "total_frames": refiner.T, "num_windows": len(starts), "optimization_seconds": round(runtime, 1)}
    (logdir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not args.skip_final_render:
        with torch.no_grad():
            video = logdir / "videos" / f"{args.exp}_final_mesh.mp4"
            refiner.render_video_all_mesh(str(video))
            refiner.render_video_all_mesh(str(video.with_name(f"{args.exp}_final_mesh_side.mp4")),
                                          side_view_y_clockwise_degrees=90.0, blank_background=True)
            print(f"[final] videos in {video.parent}")


if __name__ == "__main__":
    main()
