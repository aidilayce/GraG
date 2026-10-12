# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Vendored from the HOT3D toolkit (https://github.com/facebookresearch/hot3d)
# for GraG evaluation; see eval/THIRD_PARTY_NOTICES.md and eval/LICENSE.hot3d.
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from data_loaders.loader_hand_poses import Handedness
from data_loaders.loader_object_library import load_object_library
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from dataset_api import Hot3dDataProvider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from export_hot3d_single_hand_same_object_hold import (
    build_mano_pca_decoders,
    ensure_dir,
    export_object_mesh_to_hold_model,
    find_rgb_stream_id,
    get_camera_pose_and_intrinsics,
    get_sequence_betas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "From a HOT3D clips manifest, export 90deg-CW rotated undistorted videos and "
            "compact HOLD-style per-frame GT (.pt only)."
        )
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--mano_model_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_timestamps(frame_manifest_path: Path) -> List[int]:
    if not frame_manifest_path.is_file():
        return []
    out: List[int] = []
    with open(frame_manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(int(row["timestamp_ns"]))
            except Exception:
                continue
    return out


def rotate_90cw_cam_params(K: np.ndarray, h: int, w: int):
    # 90deg clockwise image rotation:
    # x' = h - 1 - y, y' = x, new_w=h, new_h=w.
    # We keep positive focal lengths by rotating the camera frame:
    # [X'; Y'; Z'] = R_img * [X; Y; Z], with:
    # X'=-Y, Y'=X, Z'=Z.
    R_img = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    K_rot = np.array(
        [[fy, 0.0, float(h - 1) - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K_rot, R_img


def write_rotated_undistorted_clip_and_gt(
    row: Dict[str, str],
    timestamps_ns: List[int],
    data_provider: Hot3dDataProvider,
    mano_provider: MANOHandDataProvider,
    object_library,
    decode_pose_pca,
    betas: np.ndarray,
    stream_id,
    output_root: Path,
    fps: int,
    overwrite: bool,
) -> Optional[Dict[str, str]]:
    hand_side = row["hand_side"]
    object_uid = row["object_uid"]
    full_seq_name = row["full_seq_name"]
    token = row["token"]
    if not full_seq_name or not token:
        return None

    videos_dir = output_root / "videos_undistorted_rot90cw"
    frame_meta_dir = output_root / "frame_manifests_undistorted_rot90cw"
    processed_dir = output_root / "processed_pt_undistorted_rot90cw"
    hold_models_dir = output_root / "models"
    ensure_dir(videos_dir)
    ensure_dir(frame_meta_dir)
    ensure_dir(processed_dir)
    ensure_dir(hold_models_dir)

    video_path = videos_dir / f"{full_seq_name}.mp4"
    frame_meta_path = frame_meta_dir / f"{full_seq_name}.csv"
    processed_path = processed_dir / f"{token}.pt"

    if not overwrite and video_path.is_file() and frame_meta_path.is_file() and processed_path.is_file():
        return {
            "clip_index": row["clip_index"],
            "sequence_name": row["sequence_name"],
            "hand_side": hand_side,
            "full_seq_name": full_seq_name,
            "token": token,
            "object_uid": object_uid,
            "num_frames": row.get("num_frames", ""),
            "start_ts": row.get("start_ts", ""),
            "end_ts": row.get("end_ts", ""),
            "video_path": str(video_path.resolve()),
            "frame_manifest_path": str(frame_meta_path.resolve()),
            "hold_build_path": "",
            "processed_pt_path": str(processed_path.resolve()),
        }

    obj_name = export_object_mesh_to_hold_model(
        object_library=object_library,
        object_uid=object_uid,
        hold_models_dir=hold_models_dir,
        external_models_dir=output_root / "external_models",
    )

    writer = None
    frame_rows = []
    fnames = []
    hand_pose_48 = []
    hand_transl = []
    obj_trans = []
    obj_rot = []
    Ks = []
    F = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    wanted = Handedness.Right if hand_side == "right" else Handedness.Left
    frame_id = 0

    for ts in timestamps_ns:
        image = data_provider.device_data_provider.get_undistorted_image(
            timestamp_ns=int(ts),
            stream_id=stream_id,
        )
        if image is None:
            continue
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        h, w = image.shape[:2]
        img_rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        cam_pack = get_camera_pose_and_intrinsics(data_provider, int(ts), stream_id)
        if cam_pack is None:
            continue
        T_world_camera, K = cam_pack
        T_camera_world = T_world_camera.inverse()
        K_rot, R_img = rotate_90cw_cam_params(K, h, w)

        hand_out = mano_provider.get_pose_at_timestamp(
            timestamp_ns=int(ts),
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_out is None:
            continue
        side_hand = hand_out.pose3d_collection.poses.get(wanted)
        if side_hand is None or side_hand.wrist_pose is None or side_hand.joint_angles is None:
            continue

        object_out = data_provider.object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=int(ts),
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if object_out is None:
            continue
        obj_pose3d = object_out.pose3d_collection.poses.get(object_uid)
        if obj_pose3d is None:
            continue

        T_cam_wrist = T_camera_world @ side_hand.wrist_pose
        R_cam_wrist = np.asarray(T_cam_wrist.to_matrix(), dtype=np.float32)[:3, :3]
        t_cam_wrist = np.asarray(T_cam_wrist.translation(), dtype=np.float32).reshape(3)
        R_cam_wrist_rot = R_img @ R_cam_wrist
        t_cam_wrist_rot = R_img @ t_cam_wrist
        rvec = cv2.Rodrigues(R_cam_wrist_rot)[0].reshape(3).astype(np.float32)

        pca_coeff = np.asarray(side_hand.joint_angles, dtype=np.float32)
        if pca_coeff.shape[0] != 15:
            continue
        hand_pose_aa = decode_pose_pca(pca_coeff).astype(np.float32)
        pose48 = np.concatenate([rvec, hand_pose_aa], axis=0)

        T_cam_obj = T_camera_world @ obj_pose3d.T_world_object
        M_cam_obj = np.asarray(T_cam_obj.to_matrix(), dtype=np.float32)
        R_cv = M_cam_obj[:3, :3]
        t_cv = np.asarray(T_cam_obj.translation(), dtype=np.float32).reshape(3)
        R_cv_rot = R_img @ R_cv
        t_cv_rot = R_img @ t_cv
        R_store = (F @ R_cv_rot).astype(np.float32)
        t_store = (F @ t_cv_rot).astype(np.float32)

        if writer is None:
            h_r, w_r = img_rot.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_r, h_r))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
        writer.write(cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR))

        fnames.append(f"{video_path.resolve()}:{frame_id:06d}")
        hand_pose_48.append(pose48)
        hand_transl.append(t_cam_wrist_rot)
        obj_trans.append(t_store)
        obj_rot.append(R_store)
        Ks.append(K_rot)

        frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": int(ts),
                "video_path": str(video_path.resolve()),
                "object_uid": object_uid,
                "hand_side": hand_side,
                "gt_image_path": "",
                "hold_seq_name": full_seq_name,
            }
        )
        frame_id += 1

    if writer is not None:
        writer.release()
    if frame_id == 0:
        return None

    with open(frame_meta_path, "w", newline="") as f:
        fields = ["frame_id", "timestamp_ns", "video_path", "object_uid", "hand_side", "gt_image_path", "hold_seq_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)

    T = len(fnames)
    payload = {
        "fnames": fnames,
        "hand_pose": torch.from_numpy(np.asarray(hand_pose_48, dtype=np.float32)),
        "hand_beta": torch.from_numpy(np.repeat(betas.reshape(1, 10), T, axis=0).astype(np.float32)),
        "hand_transl": torch.from_numpy(np.asarray(hand_transl, dtype=np.float32)),
        "K": torch.from_numpy(np.asarray(Ks, dtype=np.float32)),
        "obj_trans": torch.from_numpy(np.asarray(obj_trans, dtype=np.float32)),
        "obj_rot": torch.from_numpy(np.asarray(obj_rot, dtype=np.float32)),
        "obj_name": obj_name,
        "is_valid": torch.ones((T,), dtype=torch.float32),
    }
    torch.save(payload, str(processed_path))

    return {
        "clip_index": row["clip_index"],
        "sequence_name": row["sequence_name"],
        "hand_side": hand_side,
        "full_seq_name": full_seq_name,
        "token": token,
        "object_uid": object_uid,
        "num_frames": str(frame_id),
        "start_ts": str(frame_rows[0]["timestamp_ns"]),
        "end_ts": str(frame_rows[-1]["timestamp_ns"]),
        "video_path": str(video_path.resolve()),
        "frame_manifest_path": str(frame_meta_path.resolve()),
        "hold_build_path": "",
        "processed_pt_path": str(processed_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()
    output_root = Path(args.output_root).resolve()
    mano_model_dir = Path(args.mano_model_dir).resolve()
    ensure_dir(output_root)

    object_library = load_object_library(str(dataset_root / "assets"))
    decode_pca = build_mano_pca_decoders(mano_model_dir)

    with open(manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in manifest: {manifest_csv}")

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        seq_name = row["sequence_name"]
        seq_path = dataset_root / seq_name
        frame_manifest_path = Path(row["frame_manifest_path"])
        if not seq_path.is_dir():
            print(f"[skip] sequence missing: {seq_path}")
            continue
        if not frame_manifest_path.is_file():
            print(f"[skip] frame manifest missing: {frame_manifest_path}")
            continue

        print(f"[sequence] {seq_name} | side={row['hand_side']} | clip={row['clip_index']}")
        timestamps_ns = load_timestamps(frame_manifest_path)
        if not timestamps_ns:
            print("  [skip] no timestamps")
            continue

        dp = Hot3dDataProvider(
            sequence_folder=str(seq_path),
            object_library=object_library,
            mano_hand_model=None,
        )
        mano_jsonl = seq_path / "mano_hand_pose_trajectory.jsonl"
        if not mano_jsonl.is_file():
            print("  [skip] missing mano jsonl")
            continue
        mano_provider = MANOHandDataProvider(str(mano_jsonl), None)
        stream_id = find_rgb_stream_id(dp.device_data_provider)
        side = row["hand_side"]
        betas = get_sequence_betas(mano_jsonl, side)

        result = write_rotated_undistorted_clip_and_gt(
            row=row,
            timestamps_ns=timestamps_ns,
            data_provider=dp,
            mano_provider=mano_provider,
            object_library=object_library,
            decode_pose_pca=decode_pca[side],
            betas=betas,
            stream_id=stream_id,
            output_root=output_root,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        if result is None:
            print("  [skip] failed export")
            continue
        out_rows.append(result)
        print(f"  [ok] frames={result['num_frames']} video={result['video_path']}")

    out_manifest = output_root / "clips_manifest_undistorted_rot90cw.csv"
    with open(out_manifest, "w", newline="") as f:
        fields = [
            "clip_index",
            "sequence_name",
            "hand_side",
            "full_seq_name",
            "token",
            "object_uid",
            "num_frames",
            "start_ts",
            "end_ts",
            "video_path",
            "frame_manifest_path",
            "hold_build_path",
            "processed_pt_path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Done. Exported clips: {len(out_rows)}")
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
