# Vendored from GraG for release. Builds dynamic-camera-correct HOT3D GT
# (per-frame hand+object verts in camera frame) into a viewer-cache npz, consumed by
# eval/hold_eval/hold/code/evaluate_hot3d.py --gt_source viewer_cache. Runs in the HOT3D
# toolkit env (needs projectaria-tools + HOT3D data_loaders/dataset_api). See eval/README.md.
#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
import os
# Path to the HOT3D toolkit checkout (provides data_loaders / dataset_api).
HOT3D_ROOT = os.environ.get("GRAG_HOT3D_ROOT")
if HOT3D_ROOT is None:
    raise SystemExit("Set GRAG_HOT3D_ROOT to your HOT3D toolkit checkout (see eval/README.md).")
HOT3D_ROOT = Path(HOT3D_ROOT)
if str(HOT3D_ROOT) not in sys.path:
    sys.path.insert(0, str(HOT3D_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_loaders.headsets import Headset
from data_loaders.loader_hand_poses import Handedness
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import loadManoHandModel
from dataset_api import Hot3dDataProvider
from export_hot3d_clips import find_rgb_stream_id
from projectaria_tools.core.calibration import LINEAR
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from src.utils.io import gt_hot3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export HOT3D GT meshes in rotated camera space for AITViewer."
    )
    parser.add_argument("--seq_name", required=True, type=str)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--dataset_root", default=os.environ.get("HOT3D_DATASET_ROOT"), type=str,
                        help="HOT3D dataset root (or set HOT3D_DATASET_ROOT).")
    parser.add_argument(
        "--mano_model_dir",
        default=os.environ.get("GRAG_MANO_MODELS"),
        type=str,
        help="Directory with MANO pkl models (or set GRAG_MANO_MODELS).",
    )
    return parser.parse_args()


def rotate_90cw_cam_params(K: np.ndarray, h: int, w: int):
    r_img = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    K_rot = np.array(
        [[fy, 0.0, float(h - 1) - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K_rot, r_img


def get_camera_calibration(hot3d_data_provider, timestamp_ns, stream_id, camera_model=LINEAR):
    device_data_provider = hot3d_data_provider.device_data_provider
    if hot3d_data_provider.get_device_type() is Headset.Aria:
        return device_data_provider.get_online_camera_calibration(
            stream_id=stream_id,
            timestamp_ns=timestamp_ns,
            camera_model=camera_model,
        )
    if hot3d_data_provider.get_device_type() is Headset.Quest3:
        return device_data_provider.get_camera_calibration(
            stream_id=stream_id,
            camera_model=camera_model,
        )
    return None


def get_camera_pose_and_intrinsics(data_provider, timestamp_ns, stream_id):
    device_pose_provider = data_provider.device_pose_data_provider
    headset_pose = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    if headset_pose is None:
        return None

    calib = get_camera_calibration(
        hot3d_data_provider=data_provider,
        timestamp_ns=timestamp_ns,
        stream_id=stream_id,
        camera_model=LINEAR,
    )
    if calib is None:
        return None
    T_device_camera, intrinsics = calib
    T_world_camera = headset_pose.pose3d.T_world_device @ T_device_camera
    focal = intrinsics.get_focal_lengths()
    pp = intrinsics.get_principal_point()
    K = np.array(
        [[float(focal[0]), 0.0, float(pp[0])], [0.0, float(focal[1]), float(pp[1])], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return T_world_camera, K


def transform_world_to_rotated_camera(vertices_world: np.ndarray, T_camera_world, r_img: np.ndarray):
    verts = np.asarray(vertices_world, dtype=np.float32)
    rot = np.asarray(T_camera_world.to_matrix(), dtype=np.float32)[:3, :3]
    trans = np.asarray(T_camera_world.translation(), dtype=np.float32).reshape(1, 3)
    verts_cam = verts @ rot.T + trans
    return verts_cam @ r_img.T


def main():
    args = parse_args()
    if not args.dataset_root:
        raise SystemExit("--dataset_root (or HOT3D_DATASET_ROOT) is required.")
    if not args.mano_model_dir:
        raise SystemExit("--mano_model_dir (or GRAG_MANO_MODELS) is required.")
    seq_name = args.seq_name
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entry = gt_hot3d._resolve_seq_entry(seq_name)
    frame_manifest_path = entry.get("frame_manifest_path", "")
    if not frame_manifest_path:
        raise FileNotFoundError(f"No frame manifest found for {seq_name}")

    frame_rows = []
    with open(frame_manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ts = (row.get("timestamp_ns") or "").strip()
            fid = (row.get("frame_id") or "").strip()
            if not ts or not fid:
                continue
            frame_rows.append({"timestamp_ns": int(ts), "frame_id": int(fid)})
    if not frame_rows:
        raise RuntimeError(f"No frame rows found for {seq_name}")

    dataset_root = Path(args.dataset_root)
    sequence_folder = dataset_root / entry["sequence_name"]
    object_library = load_object_library(str(dataset_root / "assets"))
    mano_hand_model = loadManoHandModel(args.mano_model_dir)
    data_provider = Hot3dDataProvider(
        sequence_folder=str(sequence_folder),
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=False,
    )
    mano_provider = data_provider.mano_hand_data_provider
    if mano_provider is None:
        raise RuntimeError(f"MANO provider unavailable for {seq_name}")
    stream_id = find_rgb_stream_id(data_provider.device_data_provider)
    wanted = Handedness.Right if entry["hand_side"] == "right" else Handedness.Left

    obj_mesh = trimesh.load(
        gt_hot3d._resolve_object_mesh_path(entry, f"hot3d_{entry['object_uid']}"),
        process=False,
    )
    obj_faces = np.asarray(obj_mesh.faces, dtype=np.int64)
    obj_verts_cano = np.asarray(obj_mesh.vertices, dtype=np.float32)

    image_root = SCRIPT_DIR / "data" / seq_name / "images"
    image_paths_sorted = []
    for ext in ("png", "jpg"):
        image_paths_sorted.extend(sorted(image_root.glob(f"*.{ext}")))
    image_paths_sorted = sorted(image_paths_sorted)
    if len(image_paths_sorted) == 0:
        raise FileNotFoundError(f"No undistorted images found in {image_root}")

    # HOLD/AITViewer image folders can be temporally downsampled relative to the
    # original HOT3D rotated-video frame manifest. Resample timestamps to the
    # actual image count so mesh motion stays synchronized with the background.
    if len(frame_rows) != len(image_paths_sorted):
        idx = np.linspace(0, len(frame_rows) - 1, num=len(image_paths_sorted))
        idx = np.round(idx).astype(np.int64)
        idx = np.clip(idx, 0, len(frame_rows) - 1)
        frame_rows = [frame_rows[i] for i in idx.tolist()]

    image_paths = []
    v3d_h_all = []
    j3d_h_all = []
    v3d_o_all = []
    frame_ids = []
    hand_faces = None
    K_rot_ref = None

    for row, image_path_candidate in zip(frame_rows, image_paths_sorted):
        ts = row["timestamp_ns"]
        frame_id = row["frame_id"]

        image = data_provider.device_data_provider.get_undistorted_image(
            timestamp_ns=ts,
            stream_id=stream_id,
        )
        if image is None:
            continue
        h, w = image.shape[:2]

        cam_pack = get_camera_pose_and_intrinsics(data_provider, ts, stream_id)
        if cam_pack is None:
            continue
        T_world_camera, K = cam_pack
        T_camera_world = T_world_camera.inverse()
        K_rot, r_img = rotate_90cw_cam_params(K, h, w)
        if K_rot_ref is None:
            K_rot_ref = K_rot

        hand_out = mano_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_out is None:
            continue
        hand_pose = hand_out.pose3d_collection.poses.get(wanted)
        if hand_pose is None:
            continue
        hand_vertices_world = mano_provider.get_hand_mesh_vertices(hand_pose)
        hand_landmarks_world = mano_provider.get_hand_landmarks(hand_pose)
        hand_faces_cur, _ = mano_provider.get_hand_mesh_faces_and_normals(hand_pose)
        if hand_landmarks_world is None:
            continue
        hand_vertices_world = np.asarray(hand_vertices_world, dtype=np.float32)
        hand_landmarks_world = np.asarray(hand_landmarks_world, dtype=np.float32)
        if hand_faces is None:
            hand_faces = np.asarray(hand_faces_cur, dtype=np.int64)

        object_out = data_provider.object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if object_out is None:
            continue
        obj_pose = object_out.pose3d_collection.poses.get(entry["object_uid"])
        if obj_pose is None:
            continue
        T_world_object = np.asarray(obj_pose.T_world_object.to_matrix(), dtype=np.float32)
        obj_vertices_world = obj_verts_cano @ T_world_object[:3, :3].T + T_world_object[:3, 3]

        hand_vertices_cam = transform_world_to_rotated_camera(hand_vertices_world, T_camera_world, r_img)
        hand_landmarks_cam = transform_world_to_rotated_camera(hand_landmarks_world, T_camera_world, r_img)
        obj_vertices_cam = transform_world_to_rotated_camera(obj_vertices_world, T_camera_world, r_img)

        image_paths.append(str(image_path_candidate))
        v3d_h_all.append(hand_vertices_cam.astype(np.float32))
        j3d_h_all.append(hand_landmarks_cam.astype(np.float32))
        v3d_o_all.append(obj_vertices_cam.astype(np.float32))
        frame_ids.append(frame_id)

    if len(frame_ids) == 0:
        raise RuntimeError(f"No valid GT frames exported for {seq_name}")

    np.savez(
        out_path,
        seq_name=seq_name,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        fnames=np.asarray(image_paths),
        v3d_h=np.stack(v3d_h_all, axis=0),
        j3d_h=np.stack(j3d_h_all, axis=0),
        v3d_o=np.stack(v3d_o_all, axis=0),
        faces_h=np.asarray(hand_faces, dtype=np.int64),
        faces_o=np.asarray(obj_faces, dtype=np.int64),
        K=np.asarray(K_rot_ref, dtype=np.float32),
        hand_side=np.asarray(entry["hand_side"]),
        fps=np.asarray(15.0, dtype=np.float32),
    )
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
