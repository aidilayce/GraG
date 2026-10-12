# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from data_loaders.headsets import Headset
from data_loaders.loader_hand_poses import Handedness, load_hand_pose_as_json_lines
from data_loaders.loader_object_library import ObjectLibrary, load_object_library
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from dataset_api import Hot3dDataProvider
from projectaria_tools.core.calibration import LINEAR
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export HOT3D single-hand, single-object clips (rotated MP4) + frame manifests. "
            "Optionally write HOLD gt.load_data-compatible assets."
        )
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--hold_root", type=str, default=os.environ.get("GRAG_HOLD_ROOT"),
                        help="HOLD checkout root for optional GT-compatible asset export "
                             "(or set GRAG_HOLD_ROOT). GT export is skipped if unset.")
    parser.add_argument("--mano_model_dir", type=str, required=True)
    parser.add_argument("--num_clips", type=int, default=20)
    parser.add_argument("--hand_sides", type=str, choices=["right", "left", "both"], default="both")
    parser.add_argument("--min_dom_vis", type=float, default=0.30)
    parser.add_argument("--max_other_vis", type=float, default=0.05)
    parser.add_argument("--min_dom_ratio", type=float, default=0.20)
    parser.add_argument("--min_either_ratio", type=float, default=0.70)
    parser.add_argument("--min_segment_len", type=int, default=45)
    parser.add_argument("--max_gap_ns", type=int, default=120_000_000)
    parser.add_argument("--max_frames_per_clip", type=int, default=300)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--sequence_glob", type=str, default="P*_*")
    parser.add_argument(
        "--exclude_manifest_csv",
        type=str,
        default="",
        help="Optional existing clips manifest; object_uids found here will be excluded.",
    )
    parser.add_argument(
        "--require_unique_objects",
        action="store_true",
        help="If set, each newly exported clip must have a distinct object_uid.",
    )
    parser.add_argument(
        "--clip_index_offset",
        type=int,
        default=0,
        help="Starting index used for clip_index/token generation.",
    )
    parser.add_argument(
        "--hold_gt_for_left",
        action="store_true",
        help=(
            "Export HOLD GT assets for left-hand clips too. "
            "Note: hold/code/evaluate.py + gt.load_data are right-hand-oriented."
        ),
    )
    return parser.parse_args()


def find_rgb_stream_id(device_data_provider):
    image_stream_ids = device_data_provider.get_image_stream_ids()
    for sid in image_stream_ids:
        label = str(device_data_provider.get_image_stream_label(sid)).lower()
        if "rgb" in label:
            return sid
    return image_stream_ids[0]


def rank_candidate_sequences(
    dataset_root: Path,
    sequence_glob: str,
    min_dom_vis: float,
    max_other_vis: float,
    min_dom_ratio: float,
    min_either_ratio: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for seq_dir in sorted(dataset_root.glob(sequence_glob)):
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "recording.vrs").is_file():
            continue
        if not (seq_dir / "box2d_hands.csv").is_file():
            continue
        if not (seq_dir / "mano_hand_pose_trajectory.jsonl").is_file():
            continue

        vis_by_ts: Dict[int, List[float]] = {}
        with open(seq_dir / "box2d_hands.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(row["timestamp[ns]"])
                    hand_idx = int(row["hand_index"])  # 0=left, 1=right
                    vis = float(row["visibility_ratio[%]"])
                except Exception:
                    continue
                cur = vis_by_ts.setdefault(ts, [0.0, 0.0])
                cur[hand_idx] = max(cur[hand_idx], vis)

        if not vis_by_ts:
            continue

        n = len(vis_by_ts)
        left_dom = sum(1 for l, r in vis_by_ts.values() if l >= min_dom_vis and r <= max_other_vis)
        right_dom = sum(1 for l, r in vis_by_ts.values() if r >= min_dom_vis and l <= max_other_vis)
        left_dom_ratio = left_dom / n
        right_dom_ratio = right_dom / n
        either_ratio = sum(1 for l, r in vis_by_ts.values() if l >= 0.2 or r >= 0.2) / n
        both_ratio = sum(1 for l, r in vis_by_ts.values() if l >= 0.2 and r >= 0.2) / n

        if max(left_dom_ratio, right_dom_ratio) < min_dom_ratio:
            continue
        if either_ratio < min_either_ratio:
            continue

        rows.append(
            {
                "sequence_name": seq_dir.name,
                "sequence_path": str(seq_dir),
                "left_dom_ratio": left_dom_ratio,
                "right_dom_ratio": right_dom_ratio,
                "both_ratio": both_ratio,
                "either_ratio": either_ratio,
            }
        )

    rows.sort(key=lambda x: (-max(x["left_dom_ratio"], x["right_dom_ratio"]), x["both_ratio"]))
    return rows


def get_sequence_betas(mano_jsonl_path: Path, hand_side: str) -> np.ndarray:
    ts_payload = load_hand_pose_as_json_lines(str(mano_jsonl_path))
    key = "1" if hand_side == "right" else "0"
    for _, hand_payload in ts_payload.items():
        payload = hand_payload.get(key)
        if payload is None:
            continue
        betas = payload.get("betas")
        if betas is not None and len(betas) == 10:
            return np.asarray(betas, dtype=np.float32)
    return np.zeros((10,), dtype=np.float32)


def infer_nearest_object_uid(object_pose_collection, wrist_world_t: np.ndarray) -> Optional[str]:
    if object_pose_collection is None:
        return None
    poses = object_pose_collection.poses
    if not poses:
        return None
    best_uid = None
    best_dist = None
    for uid, pose3d in poses.items():
        t_obj = np.asarray(pose3d.T_world_object.translation(), dtype=np.float32)
        d = float(np.linalg.norm(t_obj - wrist_world_t))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_uid = uid
    return best_uid


def build_single_object_segments(
    data_provider: Hot3dDataProvider,
    mano_provider,
    stream_id,
    hand_side: str,
    min_dom_vis: float,
    max_other_vis: float,
    min_segment_len: int,
    max_gap_ns: int,
) -> List[List[Dict]]:
    hand_box2d_provider = data_provider.hand_box2d_data_provider
    object_provider = data_provider.object_pose_data_provider
    if hand_box2d_provider is None or mano_provider is None or object_provider is None:
        return []

    ts_list = hand_box2d_provider.get_timestamp_ns_list(stream_id)
    if ts_list is None or len(ts_list) == 0:
        return []

    wanted = Handedness.Right if hand_side == "right" else Handedness.Left
    entries: List[Dict] = []
    for ts in ts_list:
        ts = int(ts)
        box_out = hand_box2d_provider.get_bbox_at_timestamp(
            stream_id=stream_id,
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if box_out is None:
            continue

        box2ds = box_out.box2d_collection.box2ds
        left_vis = box2ds[0].visibility_ratio if 0 in box2ds and box2ds[0].visibility_ratio is not None else 0.0
        right_vis = box2ds[1].visibility_ratio if 1 in box2ds and box2ds[1].visibility_ratio is not None else 0.0

        if hand_side == "right":
            keep = right_vis >= min_dom_vis and left_vis <= max_other_vis
        else:
            keep = left_vis >= min_dom_vis and right_vis <= max_other_vis
        if not keep:
            continue

        hand_out = mano_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_out is None:
            continue
        side_hand = hand_out.pose3d_collection.poses.get(wanted)
        if side_hand is None or side_hand.wrist_pose is None:
            continue

        object_out = object_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if object_out is None:
            continue

        wrist_t = np.asarray(side_hand.wrist_pose.translation(), dtype=np.float32)
        obj_uid = infer_nearest_object_uid(object_out.pose3d_collection, wrist_t)
        if obj_uid is None:
            continue

        entries.append(
            {
                "timestamp_ns": ts,
                "left_vis": float(left_vis),
                "right_vis": float(right_vis),
                "object_uid": str(obj_uid),
            }
        )

    if not entries:
        return []

    segments: List[List[Dict]] = []
    cur = [entries[0]]
    for item in entries[1:]:
        if item["object_uid"] == cur[-1]["object_uid"] and item["timestamp_ns"] - cur[-1]["timestamp_ns"] <= max_gap_ns:
            cur.append(item)
        else:
            if len(cur) >= min_segment_len:
                segments.append(cur)
            cur = [item]
    if len(cur) >= min_segment_len:
        segments.append(cur)

    segments.sort(key=len, reverse=True)
    return segments


def build_mano_pca_decoders(mano_model_dir: Path):
    try:
        import smplx
    except ImportError as exc:
        raise RuntimeError("smplx is required to decode MANO PCA to axis-angle") from exc

    def build(is_right: bool):
        model_p = mano_model_dir / ("MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")
        if not model_p.is_file():
            raise RuntimeError(f"Cannot find MANO model at: {model_p}")
        mano = smplx.create(
            str(model_p),
            "mano",
            use_pca=True,
            is_rhand=is_right,
            num_pca_comps=15,
            flat_hand_mean=False,
        )
        comps = mano.hand_components.detach().cpu().numpy()
        mean = mano.hand_mean.detach().cpu().numpy()

        def decode(pca_coeff: np.ndarray) -> np.ndarray:
            return mean + pca_coeff @ comps

        return decode

    return {"right": build(True), "left": build(False)}


def get_camera_pose_and_intrinsics(data_provider: Hot3dDataProvider, timestamp_ns: int, stream_id):
    device_dp = data_provider.device_data_provider
    headset_pose_dp = data_provider.device_pose_data_provider
    headset_pose = headset_pose_dp.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    if headset_pose is None:
        return None

    T_device_camera, intrinsics = device_dp.get_camera_calibration(stream_id=stream_id, camera_model=LINEAR)
    T_world_camera = headset_pose.pose3d.T_world_device @ T_device_camera

    focal = intrinsics.get_focal_lengths()
    pp = intrinsics.get_principal_point()
    K = np.array(
        [[float(focal[0]), 0.0, float(pp[0])], [0.0, float(focal[1]), float(pp[1])], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return T_world_camera, K


def export_object_mesh_to_hold_model(
    object_library: ObjectLibrary,
    object_uid: str,
    hold_models_dir: Path,
    external_models_dir: Path,
) -> str:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to export object meshes") from exc

    obj_name = f"hot3d_{object_uid}"
    obj_dir_hold = hold_models_dir / obj_name
    obj_path_hold = obj_dir_hold / "textured_simple.obj"

    # Try native HOLD location first; if not writable, fall back to output_root/external_models.
    target_obj_dir = obj_dir_hold
    target_obj_path = obj_path_hold
    obj_name_for_gt = obj_name
    try:
        ensure_dir(obj_dir_hold)
    except PermissionError:
        obj_dir_external = external_models_dir / obj_name
        ensure_dir(obj_dir_external)
        target_obj_dir = obj_dir_external
        target_obj_path = obj_dir_external / "textured_simple.obj"
        # gt.py builds: ../generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj
        # so encode obj_name as relative path from hold_models_dir to external model dir.
        obj_name_for_gt = os.path.relpath(str(target_obj_dir), str(hold_models_dir))

    if target_obj_path.is_file():
        return obj_name_for_gt

    glb_path = ObjectLibrary.get_cad_asset_path(
        object_library_folderpath=object_library.asset_folder_name,
        object_id=object_uid,
    )
    mesh_scene = trimesh.load_mesh(glb_path, process=True, merge_primitives=True, file_type="glb")
    mesh = mesh_scene.to_mesh() if hasattr(mesh_scene, "to_mesh") else mesh_scene
    mesh.export(str(target_obj_path))
    return obj_name_for_gt


def make_token(sequence_name: str, hand_side: str, clip_idx: int) -> str:
    base = f"{sequence_name}_{hand_side}".replace("_", "")
    return f"H{clip_idx:03d}{base[:18]}"


def load_excluded_object_uids(manifest_csv: Path) -> set:
    if not manifest_csv.is_file():
        return set()
    out = set()
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            uid = str(row.get("object_uid", "")).strip()
            if uid:
                out.add(uid)
    return out


def save_clip_assets_and_gt(
    clip_idx: int,
    sequence_name: str,
    hand_side: str,
    data_provider: Hot3dDataProvider,
    mano_provider,
    object_library: ObjectLibrary,
    stream_id,
    segment: Sequence[Dict],
    object_uid: str,
    betas: np.ndarray,
    decode_pose_pca,
    output_root: Path,
    hold_root: Optional[Path],
    max_frames_per_clip: int,
    fps: int,
    write_hold_gt: bool,
) -> Optional[Dict]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python (cv2) is required") from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to save HOLD-compatible .pt files") from exc

    device_dp = data_provider.device_data_provider
    object_provider = data_provider.object_pose_data_provider

    stride = max(1, len(segment) // max_frames_per_clip)
    segment_ds = list(segment)[::stride]

    token = make_token(sequence_name, hand_side, clip_idx)
    full_seq_name = f"hold_{token}_ho3d"

    videos_dir = output_root / "videos"
    frames_meta_dir = output_root / "frame_manifests"
    ensure_dir(videos_dir)
    ensure_dir(frames_meta_dir)

    video_path = videos_dir / f"{full_seq_name}.mp4"
    frame_meta_path = frames_meta_dir / f"{full_seq_name}.csv"

    hold_code_data_build = hold_root / "code" / "data" / full_seq_name / "build" if write_hold_gt else None
    hold_build_rgb = hold_code_data_build / "rgb" if write_hold_gt else None
    hold_build_mask = hold_code_data_build / "mask" if write_hold_gt else None
    hold_processed_dir = hold_root / "generator" / "assets" / "ho3d_v3" / "processed" if write_hold_gt else None
    hold_models_dir = hold_root / "generator" / "assets" / "ho3d_v3" / "models" if write_hold_gt else None
    external_models_dir = output_root / "external_models"

    obj_name = ""
    if write_hold_gt:
        try:
            ensure_dir(hold_build_rgb)
            ensure_dir(hold_build_mask)
            ensure_dir(hold_processed_dir)
            ensure_dir(hold_models_dir.parent)
            ensure_dir(external_models_dir)
        except PermissionError as exc:
            raise RuntimeError(
                "No write permission for HOLD GT paths. "
                "Use a writable --hold_root or disable GT export for this side."
            ) from exc
        obj_name = export_object_mesh_to_hold_model(
            object_library=object_library,
            object_uid=object_uid,
            hold_models_dir=hold_models_dir,
            external_models_dir=external_models_dir,
        )

    writer = None
    frame_rows = []
    boxes = []
    fnames = []
    hand_pose_48 = []
    hand_transl = []
    obj_trans = []
    obj_rot = []
    Ks = []

    F = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    wanted = Handedness.Right if hand_side == "right" else Handedness.Left

    frame_id = 0
    for item in segment_ds:
        ts = int(item["timestamp_ns"])
        image = device_dp.get_image(timestamp_ns=ts, stream_id=stream_id)
        if image is None:
            continue
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        cam_pack = get_camera_pose_and_intrinsics(data_provider, ts, stream_id)
        if cam_pack is None:
            continue
        T_world_camera, K = cam_pack
        T_camera_world = T_world_camera.inverse()

        hand_out = mano_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_out is None:
            continue
        side_hand = hand_out.pose3d_collection.poses.get(wanted)
        if side_hand is None or side_hand.wrist_pose is None or side_hand.joint_angles is None:
            continue

        object_out = object_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
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
        rvec = cv2.Rodrigues(R_cam_wrist)[0].reshape(3).astype(np.float32)

        pca_coeff = np.asarray(side_hand.joint_angles, dtype=np.float32)
        if pca_coeff.shape[0] != 15:
            continue
        hand_pose_aa = decode_pose_pca(pca_coeff).astype(np.float32)
        pose48 = np.concatenate([rvec, hand_pose_aa], axis=0)

        T_cam_obj = T_camera_world @ obj_pose3d.T_world_object
        M_cam_obj = np.asarray(T_cam_obj.to_matrix(), dtype=np.float32)
        R_cv = M_cam_obj[:3, :3]
        t_cv = np.asarray(T_cam_obj.translation(), dtype=np.float32).reshape(3)
        R_store = (F @ R_cv).astype(np.float32)
        t_store = (F @ t_cv).astype(np.float32)

        # Rotated video (90 clockwise)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if writer is None:
            h_r, w_r = rotated.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_r, h_r))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
        overlay = rotated.copy()
        cv2.putText(
            overlay,
            f"{full_seq_name} | side={hand_side} | obj={object_uid} | t={ts}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        gt_image_path = ""
        if write_hold_gt:
            frame_name = f"{frame_id:06d}.png"
            gt_image_path = str((hold_build_rgb / frame_name).resolve())
            cv2.imwrite(gt_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            h, w = image.shape[:2]
            cv2.imwrite(str((hold_build_mask / frame_name).resolve()), np.zeros((h, w), dtype=np.uint8))
            boxes.append([1.0, 1.0, float(w - 1), float(h - 1)])

            fnames.append(gt_image_path)
            hand_pose_48.append(pose48)
            hand_transl.append(t_cam_wrist)
            obj_trans.append(t_store)
            obj_rot.append(R_store)
            Ks.append(K)

        frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": ts,
                "video_path": str(video_path.resolve()),
                "object_uid": object_uid,
                "hand_side": hand_side,
                "gt_image_path": gt_image_path,
                "hold_seq_name": full_seq_name if write_hold_gt else "",
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

    processed_path = ""
    hold_build_path = ""
    if write_hold_gt:
        with open(hold_code_data_build / "corres.txt", "w") as f:
            for fid in range(frame_id):
                f.write(f"{fid:06d}.png\n")
        np.save(str(hold_code_data_build / "boxes.npy"), np.asarray(boxes, dtype=np.float32))

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
        processed_path = str((hold_processed_dir / f"{token}.pt").resolve())
        torch.save(payload, processed_path)
        hold_build_path = str(hold_code_data_build.resolve())

    return {
        "clip_index": clip_idx,
        "sequence_name": sequence_name,
        "hand_side": hand_side,
        "full_seq_name": full_seq_name if write_hold_gt else "",
        "token": token if write_hold_gt else "",
        "object_uid": object_uid,
        "num_frames": frame_id,
        "start_ts": int(frame_rows[0]["timestamp_ns"]),
        "end_ts": int(frame_rows[-1]["timestamp_ns"]),
        "video_path": str(video_path.resolve()),
        "frame_manifest_path": str(frame_meta_path.resolve()),
        "hold_build_path": hold_build_path,
        "processed_pt_path": processed_path,
    }


def main() -> None:
    args = parse_args()
    print(f"args: {args}")

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    hold_root = Path(args.hold_root).resolve() if args.hold_root else None
    mano_model_dir = Path(args.mano_model_dir).resolve()
    ensure_dir(output_root)

    object_library_dir = dataset_root / "assets"
    if not object_library_dir.is_dir():
        raise RuntimeError(f"Missing object library folder: {object_library_dir}")
    object_library = load_object_library(object_library_folderpath=str(object_library_dir))

    decode_pca = build_mano_pca_decoders(mano_model_dir)
    excluded_object_uids = set()
    if args.exclude_manifest_csv:
        excluded_object_uids = load_excluded_object_uids(Path(args.exclude_manifest_csv).resolve())
        print(f"Excluded object_uids from manifest: {len(excluded_object_uids)}")
    used_object_uids = set(excluded_object_uids)

    candidates = rank_candidate_sequences(
        dataset_root=dataset_root,
        sequence_glob=args.sequence_glob,
        min_dom_vis=args.min_dom_vis,
        max_other_vis=args.max_other_vis,
        min_dom_ratio=args.min_dom_ratio,
        min_either_ratio=args.min_either_ratio,
    )
    print(f"Found {len(candidates)} candidate sequences")

    wanted_sides = ["right", "left"] if args.hand_sides == "both" else [args.hand_sides]

    exported_rows: List[Dict] = []
    for cand in candidates:
        if len(exported_rows) >= args.num_clips:
            break

        seq_name = cand["sequence_name"]
        seq_path = cand["sequence_path"]
        print(
            f"\n[sequence] {seq_name} "
            f"(R={cand['right_dom_ratio']:.3f}, L={cand['left_dom_ratio']:.3f}, both={cand['both_ratio']:.3f})"
        )

        try:
            dp = Hot3dDataProvider(
                sequence_folder=seq_path,
                object_library=object_library,
                mano_hand_model=None,
            )
        except Exception as exc:
            print(f"  [skip] init failed: {exc}")
            continue

        if dp.hand_box2d_data_provider is None or dp.object_pose_data_provider is None:
            print("  [skip] missing box2d/object providers")
            continue

        mano_jsonl = Path(seq_path) / "mano_hand_pose_trajectory.jsonl"
        if not mano_jsonl.is_file():
            print("  [skip] missing mano_hand_pose_trajectory.jsonl")
            continue
        # mano_layer=None still provides wrist pose + joint PCA from JSONL (no mesh FK).
        mano_provider = MANOHandDataProvider(str(mano_jsonl), None)

        stream_id = find_rgb_stream_id(dp.device_data_provider)

        for side in wanted_sides:
            if len(exported_rows) >= args.num_clips:
                break

            dom_ratio = cand["right_dom_ratio"] if side == "right" else cand["left_dom_ratio"]
            if dom_ratio < args.min_dom_ratio:
                continue

            segments = build_single_object_segments(
                data_provider=dp,
                mano_provider=mano_provider,
                stream_id=stream_id,
                hand_side=side,
                min_dom_vis=args.min_dom_vis,
                max_other_vis=args.max_other_vis,
                min_segment_len=args.min_segment_len,
                max_gap_ns=args.max_gap_ns,
            )
            if not segments:
                print(f"  [skip] no {side}-hand same-object segment")
                continue

            write_hold_gt = (side == "right" or args.hold_gt_for_left) and hold_root is not None
            if side == "left" and not args.hold_gt_for_left:
                print("  [info] exporting left clip video+manifest only (HOLD GT skipped by default)")

            seg = None
            obj_uid = None
            for cand_seg in segments:
                cand_uid = cand_seg[0]["object_uid"]
                if cand_uid in excluded_object_uids:
                    continue
                if args.require_unique_objects and cand_uid in used_object_uids:
                    continue
                seg = cand_seg
                obj_uid = cand_uid
                break
            if seg is None or obj_uid is None:
                print(f"  [skip] no {side}-hand segment left after object_uid filtering")
                continue

            betas = get_sequence_betas(Path(seq_path) / "mano_hand_pose_trajectory.jsonl", side)
            result = save_clip_assets_and_gt(
                clip_idx=args.clip_index_offset + len(exported_rows),
                sequence_name=seq_name,
                hand_side=side,
                data_provider=dp,
                mano_provider=mano_provider,
                object_library=object_library,
                stream_id=stream_id,
                segment=seg,
                object_uid=obj_uid,
                betas=betas,
                decode_pose_pca=decode_pca[side],
                output_root=output_root,
                hold_root=hold_root,
                max_frames_per_clip=args.max_frames_per_clip,
                fps=args.fps,
                write_hold_gt=write_hold_gt,
            )
            if result is None:
                print(f"  [skip] failed writing clip side={side}")
                continue

            exported_rows.append(result)
            used_object_uids.add(obj_uid)
            print(
                f"  [ok] side={side} frames={result['num_frames']} obj={result['object_uid']} "
                f"video={result['video_path']}"
            )

    manifest_path = output_root / "clips_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
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
        w.writerows(exported_rows)

    print("\nExport complete")
    print(f"Requested clips: {args.num_clips}")
    print(f"Exported clips:  {len(exported_rows)}")
    print(f"Manifest:        {manifest_path}")


if __name__ == "__main__":
    main()
