"""Render checkpoint meshes in the input camera and a stable side view."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "SAM3DO"))

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from optimization.data_loading import (
    ExperimentPaths,
    build_frames_metadata,
    infer_dynhamr_frame_indices,
    load_and_align_joints2d,
    load_dynhamr_results,
    load_pointmap_results,
)
from optimization.hand_setup import prepare_hand_resources
from optimization.object_setup import load_object
from optimization.opt import GaussianRenderer
from optimization.refiner import JointHandObjectRefiner


def render(args: argparse.Namespace) -> None:
    paths = ExperimentPaths(args.seq_name, root=args.demo_root)
    dyn = load_dynhamr_results(args.seq_name, root=args.demo_root)
    pointmaps = load_pointmap_results(args.seq_name, root=args.demo_root)
    frame_ids = infer_dynhamr_frame_indices(paths, dyn)
    total = int(pointmaps["pointmaps"].shape[0])
    joints = load_and_align_joints2d(paths.dynhamr_joints2d, frame_ids, total)
    frames = build_frames_metadata(paths, pointmaps, joints)

    gs, verts, faces, surface, normals = load_object(paths)
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=True)

    hand_model, hand = prepare_hand_resources(dyn, frame_ids, total)
    height, width = pointmaps["pointmaps"].shape[1:3]
    renderer = GaussianRenderer({
        "resolution": (height, width), "near": 0.1, "far": 100,
        "ssaa": 1, "bg_color": (0, 0, 0), "backend": "gsplat",
    })
    refiner = JointHandObjectRefiner(
        exp=args.seq_name, H=height, W=width, frames_meta=frames,
        hand_res=hand, object_gs=gs, hand_model=hand_model,
        renderer=renderer, obj_surface_points=surface,
        obj_surface_normals=normals, data_root=args.demo_root, logdir=args.out_dir,
        obj_mesh_canonical_verts=verts, obj_mesh_faces=faces,
    )
    state_frames = int(state["obj_rotations"].shape[0])
    if state_frames > refiner.T:
        raise ValueError(f"Checkpoint has {state_frames} frames, data has {refiner.T}.")
    with torch.no_grad():
        for name in (
            "hand_betas", "hand_global_orient", "hand_pose", "hand_trans",
            "hand_scale", "delta_t_hand", "delta_obj_translations",
            "obj_rotations", "obj_scale",
        ):
            dst = getattr(refiner, name)
            src = state[name].to(refiner.device)
            if dst.shape == src.shape:
                dst.copy_(src)
            elif dst.ndim > 1 and dst.shape[1] == refiner.T:
                dst[:, :state_frames].copy_(src)
            elif dst.shape[0] == refiner.T:
                dst[:state_frames].copy_(src)
            else:
                raise ValueError(f"Cannot load {name}: checkpoint {src.shape}, model {dst.shape}.")
        refiner.obj_translations = state["obj_translations"].to(refiner.device)
        refiner.obj_mesh_canonical_verts = state["obj_mesh_canonical_verts"].to(refiner.device).float()
        refiner.obj_mesh_faces = state["obj_mesh_faces"].to(refiner.device).long()
        refiner.T = state_frames

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refiner.render_video_all_mesh(str(out / f"{args.seq_name}_input_camera.mp4"))
    refiner.render_video_all_mesh(
        str(out / f"{args.seq_name}_side_view.mp4"),
        side_view_y_clockwise_degrees=90.0,
        blank_background=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    render(parser.parse_args())
