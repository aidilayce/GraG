"""
Export a GraG ``refiner_final_state.pt`` to the five prediction ``.npy`` files consumed by the HO3D / HOT3D evaluators:

    pred_v3d_c_object.npy   pred_v3d_c_right.npy   pred_j3d_c_right.npy
    pred_faces_object.npy   pred_faces_right.npy

MANO joint regression uses HOLD's ``build_mano_aa`` (vendored under
``eval/hold_eval``), which loads ``MANO_{LEFT,RIGHT}.pkl`` from ``--mano_dir``.
Download the MANO models yourself (non-redistributable); see eval/README.md.
Optional GT-alignment reads HOLD GT via ``--hold_code_dir``.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import trimesh
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
HOLD_EVAL = EVAL_DIR / "hold_eval"
for _p in (
    REPO_ROOT,
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "SAM3DO",
    REPO_ROOT / "third_party" / "SAM3DO" / "notebook",
    HOLD_EVAL,
    HOLD_EVAL / "hold",
    HOLD_EVAL / "hold" / "code",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grag_common.optimization_utils import (
    create_high_res_hand_template,
    hand_reflection_mats,
    mano_joint_transforms,
    skin_points_forward,
    upsample_mano_weights_to_highres,
)
from sam3d_objects.utils.visualization import SceneVisualizer
from third_party.Dyn_HaMR.dyn_hamr.body_model import MANO, run_mano_single
from hold.common import body_models as hold_body_models

build_mano_aa = hold_body_models.build_mano_aa


OPENCV_TO_PYTORCH3D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
PYTORCH3D_TO_OPENGL = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
R_GL2CV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
R_CV2GL = R_GL2CV
PYTORCH3D_TO_OPENCV = PYTORCH3D_TO_OPENGL @ OPENCV_TO_PYTORCH3D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a compact/full GraG refiner_final_state.pt directly to HOLD eval npys."
    )
    parser.add_argument("--seq_name", required=True)
    parser.add_argument("--weights_path", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optionally export only the first N frames (useful for diagnostic renders).",
    )
    parser.add_argument("--demo_root", type=Path, default=REPO_ROOT / "demo_data",
                        help="Root holding <seq_name>/ preprocessing + reconstruction outputs.")
    parser.add_argument("--mano_dir", type=Path, required=True,
                        help="Directory containing MANO_LEFT.pkl / MANO_RIGHT.pkl (download yourself; see eval/README.md).")
    parser.add_argument(
        "--sam3d_dir",
        default=None,
        help="Preferred SAM3D subdirectory under <demo_root>/<seq_name>. "
        "Used for ablation variants with variant-specific object reconstructions.",
    )
    parser.add_argument("--hold_code_dir", type=Path, default=None,
                        help="Path to a HOLD checkout's code/ dir, used only for optional GT alignment. "
                        "If omitted, raw predictions are exported without GT similarity alignment.")
    parser.add_argument(
        "--apply_camera_transform",
        action="store_true",
        help="Apply per-frame w2c_cv before writing vertices. Default preserves the original GraG eval export convention.",
    )
    return parser.parse_args()


def infer_is_right_from_seq_name(name: str) -> bool:
    s = str(name)
    for suffix in ["_ABLATION", "_crop", "_shorter"]:
        s = s.replace(suffix, "")
    if s.endswith("left"):
        return False
    if s.endswith("right"):
        return True
    return True


def load_mano_model(device: torch.device):
    with open(REPO_ROOT / "assets" / "mano_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    mano_cfg_raw = cfg.get("MANO", {})
    mano_cfg = {k.lower(): v for k, v in mano_cfg_raw.items()}
    return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)


def sam3d_roots(seq_name: str, demo_root: Path, sam3d_dir: str | None = None) -> list[Path]:
    seq_root = demo_root / seq_name
    roots = []
    if sam3d_dir:
        roots.append(seq_root / sam3d_dir)
    roots.append(seq_root / "sam3d")
    if seq_root.exists():
        roots.extend(sorted(p for p in seq_root.glob("sam3d*") if p.is_dir() and p.name != "sam3d"))
    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def load_refiner_object_mesh(
    seq_name: str,
    demo_root: Path,
    device: torch.device,
    sam3d_dir: str | None = None,
    state: dict | None = None,
    return_object_to_cv: bool = False,
):
    """Load the canonical object mesh used by opt.py/refiner_sog.py."""
    def ret(verts, faces, object_to_cv: np.ndarray):
        if return_object_to_cv:
            return verts, faces, object_to_cv
        return verts, faces

    if isinstance(state, dict):
        verts_raw = state.get("obj_mesh_canonical_verts")
        faces_raw = state.get("obj_mesh_faces")
        if verts_raw is not None and faces_raw is not None:
            verts = (
                verts_raw.detach().float().to(device)
                if torch.is_tensor(verts_raw)
                else torch.as_tensor(verts_raw, dtype=torch.float32, device=device)
            )
            faces = (
                faces_raw.detach().cpu().numpy().astype(np.int64)
                if torch.is_tensor(faces_raw)
                else np.asarray(faces_raw, dtype=np.int64)
            )
            if int(verts.shape[0]) > 0 and int(faces.shape[0]) > 0:
                print("[INFO] Loaded canonical object mesh from refiner checkpoint")
                # SceneVisualizer.object_pointcloud outputs OpenGL-world coords;
                # we need to convert to OpenCV-world (x same, y→-y, z→-z) before
                # applying the CV w2c matrix.
                return ret(verts, faces, R_GL2CV)
        if "obj_rotations" in state:
            raise ValueError(
                "Compact refiner checkpoint has no canonical object mesh; rerun optimization."
            )

    for sam3d_root in sam3d_roots(seq_name, demo_root, sam3d_dir):
        slat_path = sam3d_root / "sam3d_model" / "object_slat_coords.pt"
        if slat_path.exists():
            data = torch.load(slat_path, map_location="cpu")
            mesh = data.get("mesh") if isinstance(data, dict) else None
            if mesh is not None and hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                verts_raw = mesh.vertices
                faces_raw = mesh.faces
                verts = (
                    verts_raw.detach().float().to(device)
                    if torch.is_tensor(verts_raw)
                    else torch.as_tensor(verts_raw, dtype=torch.float32, device=device)
                )
                faces = (
                    faces_raw.detach().cpu().numpy().astype(np.int64)
                    if torch.is_tensor(faces_raw)
                    else np.asarray(faces_raw, dtype=np.int64)
                )
                if int(verts.shape[0]) > 0 and int(faces.shape[0]) > 0:
                    print(f"[INFO] Loaded refiner SAM3D mesh from {slat_path}")
                    return ret(verts, faces, PYTORCH3D_TO_OPENCV)

        # prefer PLY over GLB because these assets can differ by a 90 degree x-axis convention.
        for mesh_path in (sam3d_root / "result.ply", sam3d_root / "result.glb"):
            if not mesh_path.exists():
                continue
            loaded = trimesh.load(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
                continue
            verts_np = np.asarray(loaded.vertices, dtype=np.float32)
            if mesh_path.suffix.lower() == ".glb":
                rot_x_90 = np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
                verts_np = verts_np @ rot_x_90.T
            verts = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
            faces = np.asarray(loaded.faces, dtype=np.int64)
            print(f"[WARN] Falling back to SAM3D mesh file {mesh_path}")
            return ret(verts, faces, PYTORCH3D_TO_OPENCV)

    for sam3d_root in sam3d_roots(seq_name, demo_root, sam3d_dir):
        for mesh_path in (sam3d_root / "result.ply", sam3d_root / "result.glb"):
            if not mesh_path.exists():
                continue
            loaded = trimesh.load(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
                continue
            verts_np = np.asarray(loaded.vertices, dtype=np.float32)
            if mesh_path.suffix.lower() == ".glb":
                rot_x_90 = np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
                verts_np = verts_np @ rot_x_90.T
            verts = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
            faces = np.asarray(loaded.faces, dtype=np.int64)
            print(f"[WARN] Falling back to SAM3D mesh file {mesh_path}")
            return ret(verts, faces, PYTORCH3D_TO_OPENCV)
    raise FileNotFoundError(f"Missing refiner/SAM3D mesh under {demo_root / seq_name}/sam3d*")


def stack_w2c_cv(frames, t_total: int) -> np.ndarray:
    if not frames:
        print("[WARN] Checkpoint has no frames/w2c_cv; using identity camera transforms.")
        return np.repeat(np.eye(4, dtype=np.float32)[None], t_total, axis=0)
    if isinstance(frames, dict):
        items = [frames[k] for k in sorted(frames.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    else:
        items = list(frames)
    mats = []
    for meta in items[:t_total]:
        w2c = meta.get("w2c_cv") if isinstance(meta, dict) else None
        if w2c is None:
            print("[WARN] A checkpoint frame is missing w2c_cv; using identity camera transforms.")
            return np.repeat(np.eye(4, dtype=np.float32)[None], t_total, axis=0)
        if torch.is_tensor(w2c):
            w2c = w2c.detach().cpu().numpy()
        w2c = np.asarray(w2c, dtype=np.float32)
        if w2c.shape == (3, 4):
            w2c = np.vstack([w2c, np.array([0, 0, 0, 1], dtype=np.float32)])
        if w2c.shape != (4, 4):
            raise ValueError(f"Unexpected w2c_cv shape: {w2c.shape}")
        mats.append(w2c)
    if len(mats) < t_total:
        mats.extend([mats[-1].copy() for _ in range(t_total - len(mats))])
    return np.stack(mats, axis=0)


def resolve_gt_seq_name(seq_name: str) -> str:
    """"_shorter" sequence variants are prefix cuts of the full video: frame i of
    the clip is frame i of the full sequence, so GT alignment needs no offset --
    just the full sequence's GT truncated to the common prefix length."""
    return seq_name.replace("_shorter", "")


def load_hold_gt(full_seq_name: str, hold_code_dir: Path):
    from src.utils.io import gt as gt_io

    cwd = os.getcwd()
    try:
        os.chdir(hold_code_dir)
        return gt_io.load_data(full_seq_name)
    finally:
        os.chdir(cwd)


def try_load_hold_gt(full_seq_name: str, hold_code_dir: Path):
    try:
        return load_hold_gt(full_seq_name, hold_code_dir)
    except Exception as exc:
        print(f"[WARN] Failed to load HOLD GT for {full_seq_name}: {exc}")
        return None


def compute_similarity_transform(pred, gt, valid=None):
    t = min(pred.shape[0], gt.shape[0])
    pred = pred[:t]
    gt = gt[:t]
    if valid is None:
        valid = torch.ones(t, dtype=torch.bool, device=pred.device)
    else:
        valid = valid[:t].reshape(t) > 0.5

    pred_v = pred[valid].reshape(-1, 3)
    gt_v = gt[valid].reshape(-1, 3)
    mu_p = pred_v.mean(dim=0, keepdim=True)
    mu_g = gt_v.mean(dim=0, keepdim=True)
    p0 = pred_v - mu_p
    g0 = gt_v - mu_g
    h = p0.T @ g0
    u, s, vt = torch.linalg.svd(h)
    r = vt.T @ u.T
    if torch.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    scale = s.sum() / ((p0**2).sum() + 1e-8)
    trans = mu_g - scale * (mu_p @ r.T)
    return scale, r, trans


def apply_similarity(points: np.ndarray, scale, rot, trans):
    pts = torch.from_numpy(points).float()
    flat = pts.reshape(-1, 3)
    aligned = scale * (flat @ rot.T) + trans
    return aligned.view_as(pts).cpu().numpy()


def state_frames_len(state: dict) -> int:
    if "T" in state:
        return int(state["T"])
    frames = state.get("frames", {})
    return len(frames)


def tensor_to_device(value, device):
    return value.detach().to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)


def build_refiner_hand_mean_helper(hand_model, device):
    hand_template = create_high_res_hand_template(hand_model, None, subdivision_level=2)
    verts_high = hand_template["vertices"].detach().cpu().numpy()
    faces_high = hand_template["faces"].detach().cpu().numpy()
    high_res_mesh = trimesh.Trimesh(vertices=verts_high, faces=faces_high, process=False)
    xyz_cano_np, _ = trimesh.sample.sample_surface(high_res_mesh, count=verts_high.shape[0] * 2)

    mano_verts = hand_model.v_template.detach().cpu().numpy()
    mano_weights = hand_model.lbs_weights.detach().cpu().numpy()
    w_high = upsample_mano_weights_to_highres(verts_high, mano_verts, mano_weights).to(device)
    kd = cKDTree(verts_high)
    xyz_cano = torch.from_numpy(xyz_cano_np).float().to(device)
    _, nn_idx = kd.query(xyz_cano_np)
    w_gauss = w_high[nn_idx].contiguous().to(device)
    return xyz_cano, w_gauss


def refiner_hand_mean_xyz(
    hand_model,
    xyz_cano,
    w_gauss,
    hand_betas,
    hand_global_orient,
    hand_pose,
    hand_trans,
    is_right_t,
    frame_id,
    cv_to_gl,
):
    betas = hand_betas.squeeze(0)
    global_orient = hand_global_orient[:, frame_id].squeeze(0)
    pose = hand_pose[:, frame_id].reshape(1, -1).squeeze(0)
    transl = hand_trans[:, frame_id].squeeze(0)
    s3, s4 = hand_reflection_mats(is_right_t, device=hand_model.v_template.device)
    transforms = mano_joint_transforms(hand_model, betas, pose, global_orient, transl)
    xyz = xyz_cano
    if is_right_t.item() == 0:
        transforms = s4 @ transforms @ s4
        xyz = xyz @ s3.T
    xyz_posed = skin_points_forward(xyz, w_gauss, transforms)
    xyz_posed_gl = xyz_posed @ cv_to_gl.T
    return xyz_posed_gl.mean(0, keepdim=True)


def main():
    args = parse_args()
    # Resolve to absolute: GT loading chdirs into --hold_code_dir, which would
    # otherwise break a relative --mano_dir.
    hold_body_models.MODEL_DIR = str(args.mano_dir.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.weights_path, map_location="cpu")
    seq_name = args.seq_name
    t_total = state_frames_len(state)
    if args.max_frames is not None:
        t_total = min(t_total, max(1, int(args.max_frames)))
    w2c_cv = stack_w2c_cv(state.get("frames"), t_total)
    w2c_cv_t = torch.from_numpy(w2c_cv).float().to(device)

    is_right_seq = infer_is_right_from_seq_name(seq_name)
    if "is_right" in state:
        is_right_arr = state["is_right"]
        is_right_seq = bool(float(is_right_arr.reshape(-1)[0]) > 0.5)
    is_right_flags = state.get("is_right")
    if is_right_flags is None:
        is_right_flags = torch.full((1, t_total), 1.0 if is_right_seq else 0.0)
    is_right_flags = tensor_to_device(is_right_flags, device).float()

    obj_rotations = tensor_to_device(state["obj_rotations"], device)
    obj_translations = tensor_to_device(state["obj_translations"], device)
    delta_obj_translations = tensor_to_device(state["delta_obj_translations"], device)
    obj_scale = tensor_to_device(state["obj_scale"], device)

    canon_obj_verts, obj_faces, object_to_cv = load_refiner_object_mesh(
        seq_name,
        args.demo_root,
        device,
        args.sam3d_dir,
        state=state,
        return_object_to_cv=True,
    )
    object_to_cv_t = torch.from_numpy(object_to_cv).float().to(device)
    posed_object_verts_cv_all = []
    with torch.no_grad():
        for frame_id in range(t_total):
            quat = F.normalize(obj_rotations[frame_id : frame_id + 1], dim=-1)
            trans = obj_translations[frame_id] + delta_obj_translations[frame_id]
            pointcloud = SceneVisualizer.object_pointcloud(
                points_local=canon_obj_verts.unsqueeze(0),
                quat_l2c=quat.unsqueeze(0),
                trans_l2c=trans.unsqueeze(0),
                scale_l2c=obj_scale,
            )
            posed_gl = pointcloud.points_list()[0]
            posed_world_cv = posed_gl @ object_to_cv_t
            if args.apply_camera_transform:
                ones = torch.ones((posed_world_cv.shape[0], 1), device=device, dtype=posed_world_cv.dtype)
                posed_world_cv = (w2c_cv_t[frame_id] @ torch.cat([posed_world_cv, ones], dim=-1).T).T[:, :3]
            posed_object_verts_cv_all.append(posed_world_cv.detach().cpu().numpy())
    obj_pred = np.stack(posed_object_verts_cv_all, axis=0)

    print("[INFO] Building MANO hand model...")
    hand_model = load_mano_model(device)
    mano_layer_right = build_mano_aa(True, flat_hand=False)
    mano_layer_left = build_mano_aa(False, flat_hand=False)

    hand_betas = tensor_to_device(state["hand_betas"], device)
    hand_global_orient = tensor_to_device(state["hand_global_orient"], device)
    hand_pose = tensor_to_device(state["hand_pose"], device)
    hand_trans = tensor_to_device(state["hand_trans"], device)
    hand_scale = tensor_to_device(state["hand_scale"], device)
    delta_t_hand = tensor_to_device(state["delta_t_hand"], device)
    hand_scale_s = float(hand_scale.reshape(-1)[0])
    if delta_t_hand.ndim == 1:
        delta_t_hand = delta_t_hand.view(1, 3).repeat(t_total, 1)

    hand_verts_cv_all = []
    last_mano_out = None
    cv_to_gl_t = torch.from_numpy(R_CV2GL).float().to(device)
    gl_to_cv_t = torch.from_numpy(R_GL2CV).float().to(device)
    xyz_cano, w_gauss = build_refiner_hand_mean_helper(hand_model, device)
    for frame_id in range(t_total):
        is_right_t = is_right_flags[:, frame_id].squeeze(0)
        mano_out = run_mano_single(
            hand_model,
            hand_trans[:, frame_id, :],
            hand_global_orient[:, frame_id, :],
            hand_pose[:, frame_id, :, :].reshape(1, -1),
            is_right_t,
            betas=hand_betas,
        )
        last_mano_out = mano_out
        verts_cv = mano_out["vertices"][0]
        verts_gl = verts_cv @ cv_to_gl_t.T
        mean_xyz = refiner_hand_mean_xyz(
            hand_model,
            xyz_cano,
            w_gauss,
            hand_betas,
            hand_global_orient,
            hand_pose,
            hand_trans,
            is_right_t,
            frame_id,
            cv_to_gl_t,
        )
        verts_gl_scaled = (verts_gl - mean_xyz) * hand_scale_s + mean_xyz
        verts_gl_final = verts_gl_scaled + delta_t_hand[frame_id].view(1, 3)
        verts_world_cv = verts_gl_final @ gl_to_cv_t.T
        if args.apply_camera_transform:
            ones = torch.ones((verts_world_cv.shape[0], 1), device=device, dtype=verts_world_cv.dtype)
            verts_world_cv = (w2c_cv_t[frame_id] @ torch.cat([verts_world_cv, ones], dim=-1).T).T[:, :3]
        hand_verts_cv_all.append(verts_world_cv.detach().cpu().numpy())

    if last_mano_out is None:
        raise RuntimeError("No MANO frames were exported.")
    hand_faces = last_mano_out["r_faces"] if last_mano_out["is_right"] else last_mano_out["l_faces"]
    hand_pred = np.stack(hand_verts_cv_all, axis=0)

    mano_layer = mano_layer_right if is_right_seq else mano_layer_left
    hand_pred_t = torch.from_numpy(hand_pred).float()
    joints = torch.einsum("bik,ji->bjk", [hand_pred_t, mano_layer.J_regressor])
    joints = mano_layer.vertex_joint_selector(hand_pred_t, joints)
    joints_pred = joints.detach().cpu().numpy()

    eval_seq_name = seq_name
    if "ABLATION" in eval_seq_name:
        eval_seq_name = eval_seq_name.replace("_ABLATION", "")
    if "crop" in eval_seq_name:
        eval_seq_name = eval_seq_name.replace("_crop", "")

    gt_seq_name = resolve_gt_seq_name(eval_seq_name)
    gt_data = try_load_hold_gt(gt_seq_name, args.hold_code_dir) if args.hold_code_dir else None
    if gt_data is not None:
        pred_j = torch.from_numpy(joints_pred).float()
        gt_j = gt_data["j3d_c.right"].float()
        valid = gt_data["is_valid"] if "is_valid" in gt_data else None
        scale, rot, trans = compute_similarity_transform(pred_j, gt_j, valid=valid)
        obj_pred = apply_similarity(obj_pred, scale, rot, trans)
        hand_pred = apply_similarity(hand_pred, scale, rot, trans)
        joints_pred = apply_similarity(joints_pred, scale, rot, trans)
    else:
        print("[WARN] GT alignment skipped. Exporting raw predictions.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "pred_v3d_c_object.npy", obj_pred)
    np.save(args.out_dir / "pred_v3d_c_right.npy", hand_pred)
    np.save(args.out_dir / "pred_j3d_c_right.npy", joints_pred)
    np.save(args.out_dir / "pred_faces_object.npy", obj_faces)
    np.save(args.out_dir / "pred_faces_right.npy", hand_faces.detach().cpu().numpy())
    print(f"[INFO] Exported predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
