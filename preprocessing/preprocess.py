import ast
import sys
import os
import subprocess
import cv2
import numpy as np
import glob
from copy import deepcopy
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import shutil
import torch
import trimesh
import time
import json
import re
import tempfile
import warnings
import imageio
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grag_common.project_paths import (
    SAM3DO_CKPT,
    THIRD_PARTY_ROOT,
    WORK_DIR,
    append_project_paths,
)

append_project_paths()

try:
    # Optional VLM (THUDM/GLM-4.1V-9B-Thinking) for naming the held object;
    # without it the SAM3 prompt falls back to a generic "object".
    from preprocessing.object_naming import main as infer_object_name
except ImportError:
    infer_object_name = None
from third_party.SAM3DO.notebook.inference_sam3d import load_image, load_mask
from third_party.SAM3DO.sam3d_video import Inference
from preprocessing.common import (
    VIDEO_FPS,
    mask_dir_complete_for_frames,
    sorted_frame_paths,
)
from preprocessing.contact_flags import infer_contact_flags_from_video
from preprocessing.pose_postprocess import postprocess_sequence
from optimization import data_loading
gpus_to_use = range(torch.cuda.device_count())

os.environ.setdefault("GRAG_WORK_DIR", str(WORK_DIR))

MAX_SEED = np.iinfo(np.int32).max
DA3_CACHE_FILENAME = "da3_pointmaps_intrinsics_extrinsics.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def object_mask_binary(mask: np.ndarray) -> np.ndarray:
    """Decode binary masks and the legacy HO3D 0/50/150 label format."""
    values = set(np.unique(mask).tolist())
    if 50 in values and values.issubset({0, 50, 150}):
        return mask == 50
    return mask > 0


def resize_image_file_atomic(path: Path, size: tuple[int, int], resample: int) -> None:
    """Resize an image only when needed, replacing it atomically."""
    with Image.open(path) as image:
        image.load()
        if image.size == size:
            return
        image_format = image.format or "PNG"
        resized = image.resize(size, resample)
    temporary = path.with_name(f".{path.name}.tmp")
    resized.save(temporary, format=image_format)
    os.replace(temporary, path)


def load_video_with_opencv(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps == 0:
        print(f"Warning: Could not get original FPS for {video_path}. Assuming 30.")
        original_fps = 30

    target_fps = float(target_fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    # Timestamp-based sampling keeps HOT3D frames aligned when rates differ.
    source_frames_per_output_frame = original_fps / target_fps
    next_source_frame = 0.0

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count + 1e-6 >= next_source_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            next_source_frame += source_frames_per_output_frame

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from video: {video_path}. Check path and video integrity.")

    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()

    return video_tensor

def process_and_save_rgb(video_path, output_dir, fps):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_tensor = load_video_with_opencv(video_path, fps)
    result = video_tensor.to(torch.uint8)
    T_img = result.shape[0]
    for i in range(T_img):
        img_np = result[i].permute(1, 2, 0).cpu().numpy()
        filename = f"{i:06d}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img_np).save(filepath)
    return output_dir

def read_cached_obj_name(scene_dir: Union[str, Path]) -> Optional[str]:
    obj_name_file = Path(scene_dir) / "obj_name.txt"
    if not obj_name_file.exists():
        return None
    obj_name = obj_name_file.read_text().strip()
    return obj_name if obj_name else None

def release_sam3(predictor):
    # SAM3's tracker enters a process-wide bf16 autocast; leave it so later stages
    # run in their own precision and the autocast weight cache is freed.
    predictor.shutdown()
    for module in predictor.model.modules():
        if hasattr(module, "bf16_context"):
            module.bf16_context.__exit__(None, None, None)
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


def get_all_masks(frame_dir, scene_dir, selected_view_idx):
    from third_party.sam3.demo_sam3 import propagate_in_video
    from third_party.sam3.sam3.model_builder import build_sam3_video_predictor
    from third_party.sam3.sam3.visualization_utils import prepare_masks_for_visualization

    output_dir = Path(scene_dir) / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(
        list(frame_dir.glob("*.jpg")) +
        list(frame_dir.glob("*.jpeg")) +
        list(frame_dir.glob("*.png")))
    if not frames:
        raise ValueError(f"No frames found in {frame_dir}")
    images = [Image.open(frame).convert("RGB") for frame in frames]
    selected_view_idx = int(np.clip(int(selected_view_idx), 0, len(frames) - 1))
    obj_name = read_cached_obj_name(scene_dir)
    if obj_name is None and infer_object_name is not None:
        obj_name = infer_object_name(
            str(frames[selected_view_idx]),
            "What is the person holding? Reply with the object name.",
        )
    obj_name = re.sub(r"\([^)]*\)", "", str(obj_name or "object")).strip(" .,;:-") or "object"
    prompt = re.sub(r"^(a|an|the)\s+", "", obj_name, flags=re.IGNORECASE) or "object"

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    # Keep the prompted track when the detector stops re-detecting it (e.g. under the grasp).
    predictor.model.hotstart_unmatch_thresh = len(frames) + 1

    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": images}
    )
    session_id = response["session_id"]

    hand_dir = Path(scene_dir) / "hand_masks"
    hands = [np.asarray(Image.open(hand_dir / f"{f.stem}.png").convert("L")) > 127
             if (hand_dir / f"{f.stem}.png").exists() else None for f in frames]
    kernel = np.ones((15, 15), np.uint8)

    def hand_contact(tracked, obj):
        # Fraction of frames where the track touches the hand: the held object, not a look-alike.
        hits = [hand is not None and obj in tracked.get(idx, {}) and
                (cv2.dilate(tracked[idx][obj].astype(np.uint8), kernel).astype(bool) & hand).any()
                for idx, hand in enumerate(hands)]
        return float(np.mean(hits))

    def track(text):
        # The object can be occluded on the selected view: try other frames, nearest first.
        anchors = sorted(set(np.linspace(0, len(frames) - 1, 8).astype(int).tolist()) | {selected_view_idx},
                         key=lambda i: abs(i - selected_view_idx))
        best = None
        for anchor in anchors:
            predictor.handle_request(request={"type": "reset_session", "session_id": session_id})
            outputs = predictor.handle_request(
                request={"type": "add_prompt", "session_id": session_id, "frame_index": anchor, "text": text}
            ).get("outputs", {})
            if not len(outputs.get("out_obj_ids", [])):
                continue
            tracked = prepare_masks_for_visualization(propagate_in_video(predictor, session_id, anchor))
            for obj, mask in zip(outputs["out_obj_ids"], outputs["out_binary_masks"]):
                score = (hand_contact(tracked, int(obj)), int(mask.sum()))
                if best is None or score > best[0]:
                    best = (score, anchor, int(obj), mask.astype(bool), tracked)
            if best[0][0] >= 0.5 or all(hand is None for hand in hands):
                break
        return best

    try:
        # A wrong or ambiguous name can match only objects the hand never touches;
        # then fall back to a generic prompt and keep the track in contact with the hand.
        text = prompt
        found = track(prompt)
        if prompt != "object" and (found is None or found[0][0] < 0.2):
            generic = track("object")
            if generic is not None and (found is None or generic[0] > found[0]):
                text, found = "object", generic
        if found is None:
            raise RuntimeError(f"SAM3 found no mask for '{prompt}' in {frame_dir}.")
        _, anchor, object_id, previous, tracked_masks = found
        if (anchor, text) != (selected_view_idx, prompt):
            print(f"[masks] tracking '{text}' from frame {anchor} (object name '{prompt}').")
        start_mask = previous
        picked = []
        for idx in range(len(frames)):
            candidates = tracked_masks.get(idx, {})
            if object_id in candidates:
                mask = candidates[object_id].astype(bool)
            elif candidates:
                mask = max(candidates.values(), key=lambda c: np.logical_and(c, previous).sum()).astype(bool)
            else:
                mask = np.zeros_like(previous)
            picked.append(mask)
            previous = mask
        if not picked[anchor].any():
            picked[anchor] = start_mask

        # Frames the prompted track lost are filled from a generic "object" track by
        # overlap with the neighbouring frame, then with the nearest available mask.
        missing = [idx for idx, mask in enumerate(picked) if not mask.any()]
        recovery = track("object") if missing and text != "object" else None
        if recovery is not None:
            for order, step in ((missing, -1), (missing[::-1], 1)):
                for idx in order:
                    neighbour = idx + step
                    if picked[idx].any() or not 0 <= neighbour < len(frames) or not picked[neighbour].any():
                        continue
                    candidates = list(recovery[4].get(idx, {}).values())
                    overlaps = [np.logical_and(c, picked[neighbour]).sum() / max(np.logical_or(c, picked[neighbour]).sum(), 1)
                                for c in candidates]
                    if overlaps and max(overlaps) > 0.1:
                        picked[idx] = candidates[int(np.argmax(overlaps))].astype(bool)
        covered = [idx for idx, mask in enumerate(picked) if mask.any()]
        for idx in missing:
            if not picked[idx].any():
                picked[idx] = picked[min(covered, key=lambda c: abs(c - idx))]
        if missing:
            print(f"[masks] {len(missing)} frames lost by the '{text}' track were filled.")

        for frame, mask in zip(frames, picked):
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
            if eroded.sum() < 0.1 * mask.sum():
                eroded = mask.astype(np.uint8)
            imageio.imwrite(output_dir / f"{frame.stem}.png", eroded * 255)
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session_id})
        release_sam3(predictor)
    return obj_name

def get_hand_masks(frame_dir, scene_dir):
    from third_party.sam3.demo_sam3 import propagate_in_video
    from third_party.sam3.sam3.model_builder import build_sam3_video_predictor
    from third_party.sam3.sam3.visualization_utils import prepare_masks_for_visualization

    output_dir = f'{scene_dir}/hand_masks'
    os.makedirs(output_dir, exist_ok=True)
    frames = sorted(
        list(frame_dir.glob("*.jpg")) +
        list(frame_dir.glob("*.jpeg")) +
        list(frame_dir.glob("*.png"))
    )

    pil_img_list = [Image.open(frame) for frame in frames]

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=pil_img_list,
        )
    )
    session_id = response["session_id"]
    prompt_text_str = "hand"
    frame_idx = 0
    response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str,
                    )
                )
    out = response["outputs"]
    outputs_per_frame = propagate_in_video(predictor, session_id, frame_idx)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
        )
    )
    release_sam3(predictor)
    for idx in range(len(frames)):
        frame_path = frames[idx]
        try:
            all_masks = np.array([*outputs_per_frame[idx].values()])
        except Exception:
            all_masks = out['out_binary_masks']

        if all_masks.shape[0] == 0:
            ref = out.get('out_binary_masks', None)
            if ref is not None and len(ref) > 0:
                h, w = np.asarray(ref[0]).shape[-2:]
            else:
                h, w = imageio.imread(frame_path).shape[:2]
            pred_mask = np.zeros((h, w), dtype=bool)
        elif all_masks.shape[0] == 1:
            pred_mask = all_masks[0]
        else:
            areas = [mask.sum() for mask in all_masks]
            pred_mask = all_masks[np.argmax(areas)]

        pred_mask = cv2.erode(pred_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        mask_path = output_dir + '/' + f"{frame_path.stem}.png"
        imageio.imwrite(mask_path, (pred_mask.astype(np.uint8) * 255))

def select_view(rgb_dir: Path, mask_dir: Path) -> str:
    """Pick the frame with the largest object mask among the sharpest 70% (Laplacian focus)."""
    def path_sort_key(path: Path) -> Tuple[int, Union[int, str]]:
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    rgb_candidates = list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.jpeg"))
    rgb_paths = {p.stem: p for p in sorted(rgb_candidates, key=path_sort_key)}
    candidates: List[Dict[str, Any]] = []
    for mask_path in sorted(mask_dir.glob("*.png"), key=path_sort_key):
        stem = mask_path.stem
        rgb_path = rgb_paths.get(stem)
        if rgb_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_binary = object_mask_binary(mask)
        mask_area = int(mask_binary.sum())
        if mask_area == 0:
            continue
        rgb_gray = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        if rgb_gray is None:
            continue
        ys, xs = np.where(mask_binary)
        if ys.size == 0 or xs.size == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        roi = rgb_gray[y0:y1, x0:x1].astype(np.float64)
        roi_mask = mask_binary[y0:y1, x0:x1].astype(np.uint8)
        laplacian = cv2.Laplacian(roi, cv2.CV_64F, ksize=3)
        masked_laplacian = laplacian[roi_mask]
        focus_score = float(masked_laplacian.var()) if masked_laplacian.size > 0 else 0.0
        candidates.append({
            "area": mask_area,
            "focus": focus_score,
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path)
        })
    if not candidates:
        raise RuntimeError("No valid views found for selection.")
    focus_values = np.array([c["focus"] for c in candidates], dtype=np.float64)
    focus_threshold = float(np.percentile(focus_values, 30.0))
    filtered_candidates = [c for c in candidates if c["focus"] >= focus_threshold]
    if not filtered_candidates:
        filtered_candidates = sorted(candidates, key=lambda c: c["focus"], reverse=True)[:max(1, len(candidates) // 4)]
    selected_candidate = max(filtered_candidates, key=lambda c: (c["area"], c["focus"]))
    selected_view_path = selected_candidate["rgb_path"]
    return selected_view_path

def build_object_crops(
    scene_dir: str,
    crop_size: int = 518,
    pad: float = 0.50,
):
    def bbox_from_mask(mask_u8: np.ndarray):
        ys, xs = np.where(object_mask_binary(mask_u8))
        if ys.size == 0:
            return None
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return x0, y0, x1, y1

    def pad_and_square_bbox(x0, y0, x1, y1, H, W, pad=0.15):
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        side = max(w, h) * (1.0 + pad)

        x0n = int(round(cx - side * 0.5))
        x1n = int(round(cx + side * 0.5))
        y0n = int(round(cy - side * 0.5))
        y1n = int(round(cy + side * 0.5))

        crop_w = x1n - x0n
        crop_h = y1n - y0n
        if crop_w <= W:
            if x0n < 0:
                x1n -= x0n
                x0n = 0
            if x1n > W:
                x0n -= x1n - W
                x1n = W
        if crop_h <= H:
            if y0n < 0:
                y1n -= y0n
                y0n = 0
            if y1n > H:
                y0n -= y1n - H
                y1n = H
        x0n = max(0, x0n); y0n = max(0, y0n)
        x1n = min(W, x1n); y1n = min(H, y1n)
        return x0n, y0n, x1n, y1n

    rgb_dir = os.path.join(f'{scene_dir}', 'rgb')
    mask_dir = os.path.join(f'{scene_dir}', 'masks')
    out_meta_path = os.path.join(f'{scene_dir}', 'obj_crops_meta.json')
    out_rgb_dir = os.path.join(f'{scene_dir}', 'obj_crops')
    os.makedirs(out_rgb_dir, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    meta = {}

    for p in rgb_paths:
        stem = Path(p).stem
        mp = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.exists(mp):
            continue

        img = np.array(Image.open(p).convert("RGB"))
        msk = np.array(Image.open(mp).convert("L"))
        H, W = img.shape[:2]

        bb = bbox_from_mask(msk)
        if bb is None:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0, x1, y1 = pad_and_square_bbox(*bb, H, W, pad=pad)

        img_crop = Image.fromarray(img[y0:y1, x0:x1]).resize((crop_size, crop_size), Image.LANCZOS)
        img_out = os.path.join(out_rgb_dir, f"{stem}.png")
        img_crop.save(img_out)

        meta[stem] = dict(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), H=int(H), W=int(W), crop_size=int(crop_size))

    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

def select_object_keyframes(scene_dir, num_keyframes: int = 4, min_area_frac: float = 0.004,
                            region: str = "full", lambda_div: float = 0.5, topk: int = 8):
    """Select K keyframes for MV-SAM3D (Sec. 4.1.1, Supp. Eqs. 8-9).

    DA3's saddle-balanced selector with diversity weight ``lambda_div`` runs on
    the DINOv2 [CLS] descriptors of frames where the object is reliably visible.
    ``region`` sets what the descriptor sees: the background-masked frame, the
    full frame, or a tight object crop. Writes ``results/selected_views.txt``.
    """
    scene_dir = str(scene_dir)
    # Lazy import keeps DA3's heavy export stack (pycolmap) out of cached reruns.
    from third_party.DA3.src.depth_anything_3.api import DepthAnything3

    rgb_dir = os.path.join(scene_dir, "rgb")
    mask_dir = os.path.join(scene_dir, "masks")
    results_dir = os.path.join(scene_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    frame_ids, areas = [], []
    for p in rgb_paths:
        stem = Path(p).stem
        if not stem.isdigit():
            continue
        mp = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None or m.size == 0:
            continue
        frame_ids.append(int(stem))
        areas.append(float(object_mask_binary(m).sum()) / float(m.size))
    if len(frame_ids) < 3:
        print("[WARN] select_object_keyframes: too few masked frames; skipping.")
        return None
    areas = np.asarray(areas)
    keep = areas >= float(min_area_frac)
    if int(keep.sum()) < max(num_keyframes, 3):
        order = np.argsort(-areas)
        keep = np.zeros_like(areas, dtype=bool)
        keep[order[: max(num_keyframes * 3, 8)]] = True
    gated_ids = [fid for fid, k in zip(frame_ids, keep) if k]

    sel_paths, used_ids = [], []
    if region == "crop":
        build_object_crops(scene_dir, crop_size=504, pad=0.40)
        crop_dir = os.path.join(scene_dir, "obj_crops")
        for fid in gated_ids:
            cp = os.path.join(crop_dir, f"{fid:06d}.png")
            if os.path.exists(cp):
                sel_paths.append(cp)
                used_ids.append(fid)
    elif region == "masked":
        masked_dir = os.path.join(scene_dir, "obj_masked")
        os.makedirs(masked_dir, exist_ok=True)
        for fid in gated_ids:
            ip = os.path.join(rgb_dir, f"{fid:06d}.png")
            mp = os.path.join(mask_dir, f"{fid:06d}.png")
            if not (os.path.exists(ip) and os.path.exists(mp)):
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            msk = object_mask_binary(np.array(Image.open(mp).convert("L")))
            img[~msk] = 0
            op = os.path.join(masked_dir, f"{fid:06d}.png")
            Image.fromarray(img).save(op)
            sel_paths.append(op)
            used_ids.append(fid)
    else:
        for fid in gated_ids:
            ip = os.path.join(rgb_dir, f"{fid:06d}.png")
            if os.path.exists(ip):
                sel_paths.append(ip)
                used_ids.append(fid)
    if len(sel_paths) < 3:
        print(f"[WARN] select_object_keyframes: too few candidate frames (region={region}); skipping.")
        return None

    positions = None
    with tempfile.TemporaryDirectory() as tmp:
        prev_out = os.environ.get("DA3_OUTPUT_DIR")
        os.environ["DA3_OUTPUT_DIR"] = tmp
        os.environ["DA3_KF_TOPK"] = str(max(int(topk), int(num_keyframes)))
        os.environ["DA3_KF_LAMBDA_DIV"] = str(float(lambda_div))
        model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(DEVICE)
        try:
            model.inference(
                sel_paths,
                infer_gs=False,
                export_dir=None,
                ref_view_strategy="saddle_balanced",
            )
        except Exception as e:
            print(f"[WARN] select_object_keyframes: DA3 crop inference failed: {e}")
        finally:
            del model
            torch.cuda.empty_cache()
            if prev_out is None:
                os.environ.pop("DA3_OUTPUT_DIR", None)
            else:
                os.environ["DA3_OUTPUT_DIR"] = prev_out
        sel_file = os.path.join(tmp, "selected_views.txt")
        if os.path.exists(sel_file):
            with open(sel_file) as f:
                raw = f.read().strip()
            try:
                parsed = ast.literal_eval(raw if raw.startswith("[") else f"[{raw}]")
                flat = []
                for el in parsed:
                    if isinstance(el, (list, tuple)):
                        flat.extend(int(v) for v in el)
                    else:
                        flat.append(int(el))
                positions = flat
            except Exception:
                positions = None

    sel_ids = []
    if positions:
        seen = set()
        for pos in positions:
            if 0 <= pos < len(used_ids):
                fid = used_ids[pos]
                if fid not in seen:
                    seen.add(fid)
                    sel_ids.append(fid)

    if len(sel_ids) < num_keyframes and used_ids:
        idx = np.linspace(0, len(used_ids) - 1, num=min(num_keyframes, len(used_ids)), dtype=int)
        for i in idx:
            fid = used_ids[int(i)]
            if fid not in sel_ids:
                sel_ids.append(fid)

    sel_ids = sel_ids[:num_keyframes]
    with open(os.path.join(results_dir, "selected_views.txt"), "w") as f:
        f.write(str(sel_ids))
    with open(os.path.join(results_dir, "obj_keyframes.txt"), "w") as f:
        f.write(str(sel_ids))
    print(f"[INFO] Saddle-balanced keyframes (paper Eq. 8-9, region={region}): {sel_ids}")
    return sel_ids

def _da3_depth_to_world(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Back-project one DA3 depth map into its predicted world frame."""
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float64)
    rays = pixels @ np.linalg.inv(np.asarray(intrinsic, dtype=np.float64)).T
    points_camera = rays * np.asarray(depth, dtype=np.float64)[..., None]
    points_h = np.concatenate(
        [points_camera, np.ones((height, width, 1), dtype=np.float64)], axis=-1
    )
    w2c = np.asarray(extrinsic, dtype=np.float64)
    if w2c.shape == (3, 4):
        w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    return (points_h @ np.linalg.inv(w2c).T)[..., :3]


def _fit_similarity_correspondences(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * rotation @ source + translation (Umeyama)."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] < 32:
        raise RuntimeError("Too few valid overlap points to align DA3 chunks.")

    def solve(src: np.ndarray, dst: np.ndarray):
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        covariance = src_centered.T @ dst_centered / src.shape[0]
        u, singular_values, vt = np.linalg.svd(covariance)
        correction = np.eye(3)
        if np.linalg.det(vt.T @ u.T) < 0:
            correction[-1, -1] = -1.0
        rotation = vt.T @ correction @ u.T
        variance = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = float(np.sum(singular_values * np.diag(correction)) / max(variance, 1e-12))
        translation = dst_mean - scale * (rotation @ src_mean)
        return scale, rotation, translation

    scale, rotation, translation = solve(source, target)
    residual = np.linalg.norm(
        scale * (source @ rotation.T) + translation - target, axis=1
    )
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    threshold = median + 3.0 * max(mad, 1e-6)
    inliers = residual <= threshold
    if inliers.sum() >= 32:
        scale, rotation, translation = solve(source[inliers], target[inliers])
    if not np.isfinite(scale) or not (0.25 <= scale <= 4.0):
        raise RuntimeError(f"Invalid DA3 inter-chunk scale {scale}.")
    return scale, rotation, translation


def align_da3_prediction_chunks(chunks: list) -> list:
    """Align overlapping DA3 chunks into the first chunk's world frame."""
    for chunk_index in range(1, len(chunks)):
        previous = chunks[chunk_index - 1]
        current = chunks[chunk_index]
        target_world = _da3_depth_to_world(
            previous.depth[-1], previous.intrinsics[-1], previous.extrinsics[-1]
        )[::4, ::4]
        source_world = _da3_depth_to_world(
            current.depth[0], current.intrinsics[0], current.extrinsics[0]
        )[::4, ::4]
        scale, rotation, translation = _fit_similarity_correspondences(
            source_world, target_world
        )

        current.depth = np.asarray(current.depth) * scale
        transformed_extrinsics = []
        for extrinsic in np.asarray(current.extrinsics):
            original_shape = extrinsic.shape
            w2c = np.asarray(extrinsic, dtype=np.float64)
            if w2c.shape == (3, 4):
                w2c = np.concatenate(
                    [w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
                )
            c2w = np.linalg.inv(w2c)
            aligned_c2w = np.eye(4, dtype=np.float64)
            aligned_c2w[:3, :3] = rotation @ c2w[:3, :3]
            aligned_c2w[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
            aligned_w2c = np.linalg.inv(aligned_c2w)
            if original_shape == (3, 4):
                aligned_w2c = aligned_w2c[:3]
            transformed_extrinsics.append(aligned_w2c.astype(extrinsic.dtype, copy=False))
        current.extrinsics = np.stack(transformed_extrinsics)
        print(
            f"[INFO] Aligned DA3 chunk {chunk_index + 1}/{len(chunks)} "
            f"to overlap frame (scale={scale:.6f})."
        )
    return chunks


def get_pointmap_da3(scene_dir, interval: int = 1):
    """Run DA3 on scene/rgb and return per-frame world pointmaps, intrinsics and extrinsics.

    Returns (T,H,W,3) PyTorch3D-frame points, (T,3,3) intrinsics and (T,4,4) OpenCV w2c.
    """
    from third_party.DA3.src.depth_anything_3.api import DepthAnything3
    from third_party.DA3.src.depth_anything_3.specs import Prediction
    from third_party.DA3.da3_main import depths_to_world_points_per_frame, _as_homogeneous44

    da3_output_dir = f'{scene_dir}/results'
    os.makedirs(da3_output_dir, exist_ok=True)
    os.environ['DA3_OUTPUT_DIR'] = da3_output_dir
    input_image_dir = f'{scene_dir}/rgb'
    images = sorted(glob.glob(os.path.join(input_image_dir, "*.png")))
    images = images[::interval]

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(DEVICE)

    chunk_size = int(os.environ.get("GRAG_DA3_CHUNK_SIZE", "0"))
    if chunk_size <= 0 or len(images) <= chunk_size:
        prediction = model.inference(images, infer_gs=False, export_format="npz", export_dir=da3_output_dir)
    else:
        # Each DA3 call has its own similarity frame, so adjacent chunks share
        # one frame and are aligned on it before concatenation.
        starts = list(range(0, len(images), chunk_size - 1))
        chunks = [
            model.inference(images[start:start + chunk_size], infer_gs=False)
            for start in starts
        ]
        chunks = align_da3_prediction_chunks(chunks)
        prediction = Prediction(
            depth=np.concatenate([p.depth if i == 0 else p.depth[1:] for i, p in enumerate(chunks)]),
            is_metric=chunks[0].is_metric,
            sky=np.concatenate([p.sky if i == 0 else p.sky[1:] for i, p in enumerate(chunks)]) if chunks[0].sky is not None else None,
            conf=np.concatenate([p.conf if i == 0 else p.conf[1:] for i, p in enumerate(chunks)]) if chunks[0].conf is not None else None,
            extrinsics=np.concatenate([p.extrinsics if i == 0 else p.extrinsics[1:] for i, p in enumerate(chunks)]) if chunks[0].extrinsics is not None else None,
            intrinsics=np.concatenate([p.intrinsics if i == 0 else p.intrinsics[1:] for i, p in enumerate(chunks)]) if chunks[0].intrinsics is not None else None,
            processed_images=np.concatenate([p.processed_images if i == 0 else p.processed_images[1:] for i, p in enumerate(chunks)]) if chunks[0].processed_images is not None else None,
            scale_factor=chunks[0].scale_factor,
        )
    pts_world, _ = depths_to_world_points_per_frame(prediction)
    intrinsics = prediction.intrinsics
    extrinsics = np.array([_as_homogeneous44(ext) for ext in prediction.extrinsics])

    pts_world = torch.from_numpy(pts_world)
    intrinsics = torch.from_numpy(intrinsics)
    extrinsics = torch.from_numpy(extrinsics)

    del model
    torch.cuda.empty_cache()
    return pts_world, intrinsics, extrinsics

def generate_3d_and_poses(
        scene_dir, sam3d_path: str,
        sam3d_pose_inference_steps: int, ss_cfg_strength: float, ss_cfg_strength_pm: float, temporal_guidance_lambda: float,
        pointmap: np.ndarray, selected_view_idx: int,
        shape_mode: str = 'fixed', use_prev_pose: bool = True,
        scale_mode: str = 'video_median',
        lock_scale_during_pose: bool = False,
        calibrate_scale_with_selected_view: bool = False,
        seed: int = -1,
        frame_indices: Optional[List[int]] = None,
        decode_formats: Optional[List[str]] = None,
        pose_only_models: bool = False,
        inference: Optional[Inference] = None,
        ) -> Tuple[str, str]:
    """Run SAM3D-Pose on the MV-SAM3D shape with DA3 pointmaps; returns poses keyed by frame id."""
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)

    imagedir = f'{scene_dir}/rgb'
    maskdir = f'{scene_dir}/masks'
    image_files = {int(Path(f).stem): f for f in glob.glob(os.path.join(imagedir, "*.png"))}
    mask_files = {int(Path(f).stem): f for f in glob.glob(os.path.join(maskdir, "*.png"))}
    pointmap_indices = set(range(int(pointmap.shape[0])))
    common_frame_indices = sorted(set(image_files) & set(mask_files) & pointmap_indices)
    if len(common_frame_indices) == 0:
        raise RuntimeError(f"No RGB frames found in {imagedir}.")

    missing_rgb = sorted((set(mask_files) & pointmap_indices) - set(image_files))
    missing_masks = sorted((set(image_files) & pointmap_indices) - set(mask_files))
    missing_pointmaps = sorted((set(image_files) & set(mask_files)) - pointmap_indices)
    if missing_rgb or missing_masks or missing_pointmaps:
        print(
            f"[WARN] Frame id mismatch in {scene_dir}: "
            f"rgb={len(image_files)}, masks={len(mask_files)}, pointmaps={len(pointmap_indices)}. "
            f"Using {len(common_frame_indices)} frame-id-aligned triples. "
            f"missing_masks={missing_masks[:20]}"
            + ("..." if len(missing_masks) > 20 else "")
        )

    if frame_indices is None:
        frame_indices = common_frame_indices
    else:
        requested_frame_indices = [int(i) for i in frame_indices]
        common_set = set(common_frame_indices)
        frame_indices = [i for i in requested_frame_indices if i in common_set]
        dropped = [i for i in requested_frame_indices if i not in common_set]
        if dropped:
            print(
                f"[WARN] Dropping {len(dropped)} SAM3D-Pose frames without aligned RGB/mask/pointmap: "
                f"{dropped[:20]}" + ("..." if len(dropped) > 20 else "")
            )
    if len(frame_indices) == 0:
        raise ValueError("frame_indices is empty.")

    if selected_view_idx not in frame_indices:
        print(
            f"[WARN] selected_view_idx={selected_view_idx} not in frame_indices; "
            "using the nearest aligned frame as selected view."
        )
        selected_view_idx = min(frame_indices, key=lambda i: abs(i - int(selected_view_idx)))
    selected_view_idx_local = frame_indices.index(selected_view_idx)

    images = [load_image(image_files[i]) for i in frame_indices]
    masks = [load_mask(mask_files[i]) for i in frame_indices]
    pointmap = pointmap[frame_indices]

    mv_sam3d_ss_path = str(sam3d_path) + '/ss_return_dict.pth'
    mv_sam3d_slat_path = str(sam3d_path) + '/slat.pth'

    if inference is None:
        inference = Inference(str(SAM3DO_CKPT), compile=False, pose_only=pose_only_models)
    os.makedirs(sam3d_path, exist_ok=True)
    output = inference(
        images, masks, seed=seed, pointmaps=pointmap,
        selected_view_idx=selected_view_idx_local,
        sam3d_pose_inference_steps=sam3d_pose_inference_steps,
        ss_cfg_strength=ss_cfg_strength,
        ss_cfg_strength_pm=ss_cfg_strength_pm,
        temporal_guidance_lambda=temporal_guidance_lambda,
        shape_mode=shape_mode,
        use_prev_pose=use_prev_pose,
        scale_mode=scale_mode,
        lock_scale_during_pose=lock_scale_during_pose,
        calibrate_scale_with_selected_view=calibrate_scale_with_selected_view,
        mv_sam3d_ss_path=mv_sam3d_ss_path,
        mv_sam3d_slat_path=mv_sam3d_slat_path,
        decode_formats=decode_formats,
        )
    output = list(output)
    poses_local = output[0]
    poses_global = {}
    for local_idx, global_idx in enumerate(frame_indices):
        local_key = str(local_idx)
        if local_key in poses_local:
            poses_global[str(global_idx)] = poses_local[local_key]
    output[0] = densify_pose_dict_by_frame_ids(poses_global, frame_indices)
    return tuple(output)

def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    q0 = q0.clone()
    q1 = q1.clone()
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    alpha_t = torch.as_tensor(float(alpha), dtype=q0.dtype, device=q0.device)
    linear = q0 + alpha_t * (q1 - q0)
    linear = linear / linear.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * alpha_t
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    spherical = s0 * q0 + s1 * q1
    spherical = spherical / spherical.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(dot > 0.9995, linear, spherical)

def clone_pose(pose):
    if isinstance(pose, dict):
        return {k: clone_pose(v) for k, v in pose.items()}
    if torch.is_tensor(pose):
        return pose.clone()
    return deepcopy(pose)

def interpolate_pose_dict(pose0: dict, pose1: dict, alpha: float) -> dict:
    out = clone_pose(pose0)
    for key in set(pose0.keys()) & set(pose1.keys()):
        v0 = pose0[key]
        v1 = pose1[key]
        if not (torch.is_tensor(v0) and torch.is_tensor(v1)):
            continue
        if v0.shape != v1.shape or not torch.is_floating_point(v0) or not torch.is_floating_point(v1):
            continue
        v1 = v1.to(device=v0.device, dtype=v0.dtype)
        if key == "rotation" and v0.shape[-1] == 4:
            out[key] = quat_slerp(v0, v1, alpha)
        else:
            alpha_t = torch.as_tensor(float(alpha), dtype=v0.dtype, device=v0.device)
            out[key] = v0 + alpha_t * (v1 - v0)
    return out

def densify_pose_dict_by_frame_ids(poses: dict, target_frame_indices: list[int]) -> dict[str, dict]:
    if not poses:
        return poses

    key_by_id: dict[int, object] = {}
    for key in poses.keys():
        try:
            frame_id = int(key)
        except (TypeError, ValueError):
            continue
        key_by_id[frame_id] = key
    pose_ids = sorted(key_by_id)
    if not pose_ids:
        return poses

    out: dict[str, dict] = {}
    interpolated: list[int] = []
    pose_ids_np = np.asarray(pose_ids, dtype=np.int64)
    for frame_idx in sorted(set(int(i) for i in target_frame_indices)):
        if frame_idx in key_by_id:
            out[str(frame_idx)] = poses[key_by_id[frame_idx]]
            continue

        right_pos = int(np.searchsorted(pose_ids_np, frame_idx, side="left"))
        if right_pos == 0:
            out[str(frame_idx)] = clone_pose(poses[key_by_id[pose_ids[0]]])
        elif right_pos >= len(pose_ids):
            out[str(frame_idx)] = clone_pose(poses[key_by_id[pose_ids[-1]]])
        else:
            left_id = pose_ids[right_pos - 1]
            right_id = pose_ids[right_pos]
            alpha = float(frame_idx - left_id) / float(max(1, right_id - left_id))
            out[str(frame_idx)] = interpolate_pose_dict(
                poses[key_by_id[left_id]],
                poses[key_by_id[right_id]],
                alpha,
            )
        interpolated.append(frame_idx)

    if interpolated:
        preview = ", ".join(str(i) for i in interpolated[:12])
        suffix = "..." if len(interpolated) > 12 else ""
        print(
            f"[INFO] Interpolated {len(interpolated)} missing SAM3D poses "
            f"for dense frame ids: {preview}{suffix}"
        )
    return out

def merge_bidirectional_pose_outputs(
    forward_output: Any,
    backward_output: Any,
    num_frames: int,
    anchor_idx: int,
) -> Any:
    poses_f = forward_output[0]
    poses_b = backward_output[0]
    merged = {}
    for i in range(num_frames):
        k = str(i)
        if i >= anchor_idx and k in poses_f:
            merged[k] = poses_f[k]
        elif i < anchor_idx and k in poses_b:
            merged[k] = poses_b[k]
        elif k in poses_f:
            merged[k] = poses_f[k]
        elif k in poses_b:
            merged[k] = poses_b[k]

    missing = [i for i in range(num_frames) if str(i) not in merged]
    if missing:
        print(f"[WARN] Missing merged poses for frames: {missing[:20]}")

    out = list(forward_output)
    out[0] = merged
    return tuple(out)

def save_sam3d_output(output, sam3d_dir, save_model: bool = True):
    sam3d_dir = Path(sam3d_dir)
    sam3d_dir.mkdir(parents=True, exist_ok=True)
    poses = output[0]
    geo_outputs = output[1]
    slat = output[2]
    coords = output[3]
    pose_dir = sam3d_dir / "sam3d_poses_wo_postprocess"
    pose_dir.mkdir(parents=True, exist_ok=True)
    torch.save(poses, pose_dir / "poses_dict.pt")

    if save_model:
        gs4 = geo_outputs['gaussian_4'][0]
        gs = geo_outputs['gaussian'][0]
        if hasattr(gs4, "convert_to_fp32"):
            gs4.convert_to_fp32()
        if hasattr(gs, "convert_to_fp32"):
            gs.convert_to_fp32()
        if 'mesh' in geo_outputs:
            mesh = geo_outputs['mesh'][0]
        else:
            mesh_path = sam3d_dir / "result.ply"
            if not mesh_path.exists():
                mesh_path = sam3d_dir / "result.glb"
            if not mesh_path.exists():
                raise FileNotFoundError(
                    "SAM3D mesh decoder was skipped, but no MV-SAM3D result.ply/result.glb "
                    f"was found under {sam3d_dir}."
                )
            mesh = trimesh.load(mesh_path, force="mesh")
            print(f"[INFO] Using existing MV-SAM3D mesh for object_slat_coords.pt: {mesh_path}")

        model_dir = sam3d_dir / "sam3d_model_wo_postprocess"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'slat': slat,
            'coords': coords,
            'gs_4': gs4,
            'gs': gs,
            'mesh': mesh,
        }, model_dir / "object_slat_coords.pt")
    else:
        print("[INFO] Saved SAM3D poses only; preserving existing sam3d_model/object_slat_coords.pt.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialization w/ foundation models.")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to the input video file or rgb dir.")
    parser.add_argument("--fps", type=float, default=VIDEO_FPS, help="Target FPS used for RGB extraction and contact flag timestamps.")
    parser.add_argument("--interval", type=int, default=1, help="Frame interval for pointmap/camera prediction. Use 1 for per-frame processing.")
    parser.add_argument("-m", "--match_shape_trajectory", action='store_true', help="Whether to match trajectory in MV-SAM3D and SAM3D-Pose.")
    parser.add_argument("--shape_mode", choices=["fixed", "trajectory", "sam3d"], default=None, help="Override shape mode (default: based on --match_shape_trajectory).")
    use_prev_pose_group = parser.add_mutually_exclusive_group()
    use_prev_pose_group.add_argument("--use_prev_pose", action="store_true", help="Use previous pose when running SAM3D-Pose.")
    use_prev_pose_group.add_argument("--no-use_prev_pose", dest="use_prev_pose", action="store_false", help="Disable using previous pose when running SAM3D-Pose.")
    parser.set_defaults(use_prev_pose=True)
    parser.add_argument("--sam3d_pose_steps", type=int, default=20, help="Number of denoising steps for SAM3D-Pose per frame.")
    parser.add_argument("--sam3d_ss_cfg_strength", type=float, default=7.0, help="Generic classifier-free guidance strength for SAM3D-Pose.")
    parser.add_argument("--sam3d_ss_cfg_strength_pm", type=float, default=0.0, help="Pointmap-only guidance strength for SAM3D-Pose.")
    parser.add_argument("--sam3d_temporal_guidance_lambda", type=float, default=0.1, help="Temporal guidance strength on SAM3D pose latents.")
    parser.add_argument("--sam3d_pose_overwrite", action="store_true", help="Rerun SAM3D-Pose even if pose/model outputs already exist; reuses MV-SAM3D shape outputs.")
    parser.add_argument("--sam3d_bidirectional_pose", action="store_true", help="Run SAM3D-Pose bidirectionally from an anchor frame (forward and backward) and merge poses.")
    parser.add_argument("--sam3d_anchor_frame_idx", type=int, default=None, help="Anchor frame index for bidirectional SAM3D-Pose. Defaults to selected_view_idx.")
    parser.add_argument("-s", "--seed", type=int, default=2, help="Random seed for initialization.")
    infer_contact_group = parser.add_mutually_exclusive_group()
    infer_contact_group.add_argument("--infer_contact", dest="infer_contact", action="store_true", help="Infer per-frame hand-object contact flags with Gemini.")
    infer_contact_group.add_argument("--no_infer_contact", dest="infer_contact", action="store_false", help="Disable contact flag inference.")
    parser.set_defaults(infer_contact=True)
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview", help="Gemini model for contact inference from video.")
    parser.add_argument("--gemini_api_key", type=str, default=None, help="Gemini API key. If unset, reads GEMINI_API_KEY or GOOGLE_API_KEY.")
    parser.add_argument("--contact_overwrite", action="store_true", help="Overwrite existing contact_flags_gemini outputs.")
    parser.add_argument("--num_keyframes", type=int, default=4, help="Number of object keyframes used for MV-SAM3D reconstruction (paper uses V=4).")
    parser.add_argument("--skip_object_keyframes", action="store_true", help="Skip object-centric saddle-balanced keyframe selection; keep whole-frame/temporal defaults.")
    parser.add_argument("--object_keyframes_overwrite", action="store_true", help="Recompute object-centric keyframes even if results/obj_keyframes.txt exists.")
    parser.add_argument("--keyframe_region", choices=["full", "crop", "masked"], default="masked", help="What the [CLS] descriptor sees for keyframe selection: background-masked frame (default; object kept in place), full frame, or tight object crop.")
    parser.add_argument("--keyframe_lambda_div", type=float, default=0.5, help="Diversity weight lambda_div of the saddle-balanced keyframe selection (Supp. Eq. 8).")
    parser.add_argument("--overwrite_rgb", action="store_true", help="Regenerate scene/rgb from the input video and invalidate cached masks/pointmaps derived from the old RGB frames.")
    parser.add_argument("--pointmap_overwrite", action="store_true", help="Regenerate the DA3 pointmap/camera cache while preserving RGB and masks.")
    parser.add_argument("--skip_hand_masks", action="store_true", help="Skip single-hand SAM3 mask generation.")
    parser.add_argument("--dynhamr_dir_name", type=str, default="dynhamr", help="Scene subdirectory for DynHaMR outputs.")
    parser.add_argument("--dynhamr_yolo_det_conf", type=float, default=0.15, help="YOLO hand detector confidence for DynHaMR/HaMeR. Lower can recover missed hands.")
    parser.add_argument("--dynhamr_hamer_batch_size", type=int, default=8, help="HaMeR batch size inside DynHaMR.")
    parser.add_argument("--dynhamr_hamer_render", action="store_true", help="Enable HaMeR render visualizations inside DynHaMR.")
    parser.add_argument("--dynhamr_overwrite", action="store_true", help="Rerun DynHaMR preprocessing even if its output directory is non-empty.")
    args = parser.parse_args()

    seed = args.seed
    video_path = args.video_path
    target_fps = float(args.fps)
    pointmap_interval = max(1, int(args.interval))
    if pointmap_interval != 1:
        raise ValueError("--interval must be 1 in preprocess.py because SAM3D expects per-frame pointmaps.")
    match_shape_trajectory = args.match_shape_trajectory
    if args.shape_mode is not None:
        shape_mode = args.shape_mode
    elif match_shape_trajectory:
        shape_mode = 'trajectory'
    else:
        shape_mode = 'fixed'
    use_prev_pose = args.use_prev_pose
    sam3d_pose_steps = max(1, int(args.sam3d_pose_steps))
    sam3d_ss_cfg_strength = float(args.sam3d_ss_cfg_strength)
    sam3d_ss_cfg_strength_pm = float(args.sam3d_ss_cfg_strength_pm)
    sam3d_temporal_guidance_lambda = float(args.sam3d_temporal_guidance_lambda)
    sam3d_bidirectional_pose = args.sam3d_bidirectional_pose
    sam3d_anchor_frame_idx = args.sam3d_anchor_frame_idx

    if 'mp4' not in video_path and 'avi' not in video_path:
        rgb_dir = Path(video_path)  # rgb frame directory
        scene_dir = rgb_dir.parent
        scene_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = Path(f'{scene_dir}/masks')
        nickname = str(Path(video_path).parent.stem)
        hand_mask_dir = Path(f'{scene_dir}/hand_masks')
        if not args.skip_hand_masks:
            hand_mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        nickname = str(Path(video_path).stem)
        scene_dir = WORK_DIR / nickname
        rgb_dir = Path(f'{scene_dir}/rgb')
        scene_dir = Path(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite_rgb and any(rgb_dir.iterdir()):
            print(f"[INFO] --overwrite_rgb set; regenerating raw RGB frames in {rgb_dir}.")
            shutil.rmtree(rgb_dir)
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for stale_dir_name in ["masks", "hand_masks"]:
                stale_dir = scene_dir / stale_dir_name
                if stale_dir.exists():
                    shutil.rmtree(stale_dir)
            for stale_file_name in [
                DA3_CACHE_FILENAME,
                "selected_view_idx.txt",
            ]:
                stale_file = scene_dir / stale_file_name
                if stale_file.exists():
                    stale_file.unlink()
            selected_views_file = scene_dir / "results" / "selected_views.txt"
            if selected_views_file.exists():
                selected_views_file.unlink()
            for stale_sam3d_name in [
                "sam3d",
                "sam3d_no_fixed_shape",
                "sam3d_no_prev_pose",
                "sam3d_no_fixed_shape_no_prev_pose",
            ]:
                stale_sam3d_dir = scene_dir / stale_sam3d_name
                if stale_sam3d_dir.exists():
                    shutil.rmtree(stale_sam3d_dir)
        if not any(rgb_dir.iterdir()):
            rgb_dir = process_and_save_rgb(video_path, rgb_dir, target_fps)
        mask_dir = Path(f'{scene_dir}/masks')
        hand_mask_dir = Path(f'{scene_dir}/hand_masks')
        if not args.skip_hand_masks:
            hand_mask_dir.mkdir(parents=True, exist_ok=True)

    # DynHaMR resolves ``data.seq`` relative to GRAG_WORK_DIR.
    os.environ["GRAG_WORK_DIR"] = str(scene_dir.parent)
    print(f'Scene directory: {scene_dir}')
    print(f"Processing video: {video_path}")

    da3_path = scene_dir / DA3_CACHE_FILENAME
    if da3_path.exists() and not args.pointmap_overwrite:
        pointmap_payload = torch.load(da3_path, map_location="cpu")
    else:
        if args.pointmap_overwrite:
            print(f"[INFO] --pointmap_overwrite set; regenerating {da3_path}.")
        pointmaps, intrinsics, extrinsics = get_pointmap_da3(
            scene_dir, interval=pointmap_interval
        )
        pointmap_payload = {
            'pointmaps': pointmaps,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'extrinsics_format': 'opencv_w2c',
            'pointmap_model': 'da3',
            'chunk_alignment': 'overlap_similarity_v1',
        }
        torch.save(pointmap_payload, da3_path)
    pointmaps = pointmap_payload['pointmaps']
    intrinsics = pointmap_payload['intrinsics']
    extrinsics = pointmap_payload['extrinsics']

    views_file_path = os.path.join(f'{scene_dir}/results', "selected_views.txt")
    if os.path.exists(views_file_path):
        with open(views_file_path, "r") as f:
            selected_views_ids = ast.literal_eval(f.read())
            selected_views_ids = [int(v) for v in selected_views_ids]
        selected_view_idx = selected_views_ids[0]
        selected_view_path = str(rgb_dir / f"{selected_view_idx:06d}.png")
        print(f"Using cached selected views indices: {selected_views_ids}")
    else:
        try:
            selected_view_path = select_view(rgb_dir, mask_dir)
            selected_view_idx = int(float(os.path.splitext(os.path.basename(selected_view_path))[0].lstrip('0')))
        except Exception:
            print("View selection failed, using the first frame as selected view.")
            selected_view_idx = int(0)
            selected_view_path = str(rgb_dir / f"{selected_view_idx:06d}.png")
        num_frames = int(pointmaps.shape[0])
        sampled_views = np.linspace(0, max(0, num_frames - 1), num=min(4, num_frames), dtype=int).tolist() if num_frames > 0 else [0]
        selected_views_ids = [selected_view_idx] + [idx for idx in sampled_views if idx != selected_view_idx]
        with open(os.path.join(scene_dir, 'selected_view_idx.txt'), 'w') as f:
            f.write(str(selected_view_idx))
        print(f"Selected view index: {selected_view_idx}, path: {selected_view_path}")

    if (not args.skip_hand_masks) and (not any(hand_mask_dir.iterdir())):
        get_hand_masks(rgb_dir, scene_dir)

    obj_name = None
    masks_complete = mask_dir_complete_for_frames(mask_dir, rgb_dir)
    if not masks_complete:
        mask_dir = Path(f'{scene_dir}/masks')
        mask_dir.mkdir(parents=True, exist_ok=True)
        existing_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
        print(
            f"[WARN] Object mask directory is incomplete "
            f"({existing_count}/{len(sorted_frame_paths(rgb_dir))} masks); rerunning SAM3 object masks."
        )
        obj_name = get_all_masks(rgb_dir, scene_dir, selected_view_idx)
        with open(os.path.join(scene_dir, 'obj_name.txt'), 'w') as f:
            f.write(obj_name)
    else:
        obj_name_file = os.path.join(scene_dir, 'obj_name.txt')
        if os.path.exists(obj_name_file):
            with open(obj_name_file, 'r') as f:
                obj_name = f.read().strip()


    contact_cached = (scene_dir / "contact_flags_gemini.json").exists() and not args.contact_overwrite
    if args.infer_contact and not contact_cached:
        with tempfile.TemporaryDirectory() as tmp:
            contact_video = Path(video_path)
            if contact_video.is_dir():
                # Gemini reads videos: encode the frame folder at the working frame rate.
                contact_video = Path(tmp) / f"{nickname}.mp4"
                imageio.mimwrite(contact_video, [imageio.imread(p) for p in sorted_frame_paths(rgb_dir)],
                                 fps=target_fps, macro_block_size=1)
            infer_contact_flags_from_video(
                video_path=contact_video,
                frame_dir=rgb_dir,
                scene_dir=scene_dir,
                fps=target_fps,
                gemini_model=args.gemini_model,
                overwrite=args.contact_overwrite,
                gemini_api_key=args.gemini_api_key,
            )

    new_W, new_H = pointmaps.shape[1], pointmaps.shape[2]
    target_size = (new_H, new_W)
    for img_path in rgb_dir.glob("*.png"):
        resize_image_file_atomic(img_path, target_size, Image.LANCZOS)

    for mask_path in mask_dir.glob("*.png"):
        resize_image_file_atomic(mask_path, target_size, Image.NEAREST)

    for hand_mask_path in Path(f'{scene_dir}/hand_masks').glob("*.png"):
        resize_image_file_atomic(hand_mask_path, target_size, Image.NEAREST)
    mask_dir = Path(f'{scene_dir}/masks')
    rgb_dir = Path(f'{scene_dir}/rgb')

    # Select object keyframes with supplement Eq. 8-9.
    if not args.skip_object_keyframes:
        obj_kf_cache = os.path.join(scene_dir, "results", "obj_keyframes.txt")
        if os.path.exists(obj_kf_cache) and not args.object_keyframes_overwrite:
            try:
                with open(obj_kf_cache) as f:
                    cached = [int(v) for v in ast.literal_eval(f.read())]
                if cached:
                    selected_views_ids = cached
                    selected_view_idx = cached[0]
                    print(f"[INFO] Using cached object-centric keyframes: {cached}")
            except Exception:
                pass
        else:
            obj_kf = select_object_keyframes(scene_dir, num_keyframes=args.num_keyframes,
                                             region=args.keyframe_region,
                                             lambda_div=args.keyframe_lambda_div)
            if obj_kf:
                selected_views_ids = obj_kf
                selected_view_idx = obj_kf[0]  # best object view -> SAM3D-Pose anchor
                with open(os.path.join(scene_dir, 'selected_view_idx.txt'), 'w') as f:
                    f.write(str(selected_view_idx))

    sam3d_suffix_parts = []
    if shape_mode == "sam3d":
        sam3d_suffix_parts.append("no_fixed_shape")
    if not use_prev_pose:
        sam3d_suffix_parts.append("no_prev_pose")
    sam3d_suffix = f"_{'_'.join(sam3d_suffix_parts)}" if sam3d_suffix_parts else ""
    sam3d_path = f"{scene_dir}/sam3d{sam3d_suffix}"
    os.makedirs(sam3d_path, exist_ok=True)
    sam3d_dir_nonempty = Path(sam3d_path).exists() and any(Path(sam3d_path).iterdir())

    mv_sam3d_inference_steps = 50
    image_names = ','.join([str(idx) for idx in selected_views_ids[:4]])
    mv_sam3d_dir = THIRD_PARTY_ROOT / "MV-SAM3D"
    script_path = "run_inference_weighted.py"

    if not Path(f'{sam3d_path}/ss_return_dict.pth').exists() and not sam3d_dir_nonempty:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Running MV-SAM3D inference with selected images and steps: {mv_sam3d_inference_steps}...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--input_path", str(scene_dir),
                    "--output_path", str(sam3d_path),
                    "--mask_prompt", "masks",
                    "--stage1_steps", str(mv_sam3d_inference_steps),
                    "--image_names", image_names,
                    "--pointmaps_path", str(da3_path),
                ],
                cwd=str(mv_sam3d_dir),
                check=True,
                text=True
            )
            print("MV-SAM3D inference completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running MV-SAM3D: {e}")
        except FileNotFoundError:
            print(f"Could not find the directory or script at: {mv_sam3d_dir}")
    elif not Path(f'{sam3d_path}/ss_return_dict.pth').exists() and sam3d_dir_nonempty:
        print(f"[INFO] Existing non-empty SAM3D directory found at {sam3d_path}; not rerunning MV-SAM3D.")

    model_ready = Path(f'{sam3d_path}/sam3d_model/object_slat_coords.pt').exists()
    poses_ready = Path(f'{sam3d_path}/sam3d_poses/poses_dict.pt').exists()
    if not Path(f'{sam3d_path}/ss_return_dict.pth').exists() and not (model_ready and poses_ready):
        raise RuntimeError("MV-SAM3D inference did not complete successfully.")

    torch.cuda.empty_cache()

    dynhamr_dir = Path(scene_dir) / str(args.dynhamr_dir_name)
    dynhamr_dir.mkdir(parents=True, exist_ok=True)
    os.environ['DYNHAMR_OUTPUT_DIR'] = str(dynhamr_dir)
    old_dynhamr_yolo_det_conf = os.environ.get("DYNHAMR_YOLO_DET_CONF")
    old_dynhamr_hamer_render = os.environ.get("DYNHAMR_HAMER_RENDER")
    old_dynhamr_hamer_batch_size = os.environ.get("DYNHAMR_HAMER_BATCH_SIZE")
    if args.dynhamr_yolo_det_conf is not None:
        os.environ["DYNHAMR_YOLO_DET_CONF"] = str(float(args.dynhamr_yolo_det_conf))
    os.environ["DYNHAMR_HAMER_RENDER"] = "1" if args.dynhamr_hamer_render else "0"
    os.environ["DYNHAMR_HAMER_BATCH_SIZE"] = str(int(args.dynhamr_hamer_batch_size))

    cameras_npz = dynhamr_dir / "cameras" / nickname / "shot-0" / "cameras.npz"
    cameras_npz.parent.mkdir(parents=True, exist_ok=True)
    camera_extrinsics = np.asarray(pointmap_payload["extrinsics"], dtype=np.float32)
    camera_intrinsics = np.asarray(pointmap_payload["intrinsics"], dtype=np.float32)
    camera_intrinsics_4 = camera_intrinsics[:, (0, 1, 0, 1), (0, 1, 2, 2)]
    pointmap_height, pointmap_width = pointmap_payload["pointmaps"].shape[1:3]
    cameras_stale = not cameras_npz.exists()
    if not cameras_stale:
        with np.load(cameras_npz) as cached:
            cameras_stale = cached["w2c"].shape != camera_extrinsics.shape
            if not cameras_stale:
                cameras_stale = not np.allclose(
                    cached["w2c"], camera_extrinsics, atol=1e-5, rtol=1e-5
                )
            intrinsics_stale = (
                "intrins" not in cached
                or cached["intrins"].shape != camera_intrinsics_4.shape
            )
            if not intrinsics_stale:
                intrinsics_stale = not np.allclose(
                    cached["intrins"], camera_intrinsics_4, atol=1e-5, rtol=1e-5
                )
            cameras_stale = cameras_stale or intrinsics_stale
    if cameras_stale:
        cameras_tmp = cameras_npz.with_name(f".{cameras_npz.name}.tmp")
        with open(cameras_tmp, "wb") as camera_file:
            np.savez(
                camera_file,
                height=np.float32(pointmap_height),
                width=np.float32(pointmap_width),
                focal=np.float32(camera_intrinsics_4[:, :2].mean()),
                intrins=camera_intrinsics_4,
                w2c=camera_extrinsics,
            )
        os.replace(cameras_tmp, cameras_npz)
        print(f"[dynhamr] Exported current DA3 cameras to {cameras_npz}.")
    dynhamr_outputs = dynhamr_dir / "outputs"
    dynhamr_has_result = (
        dynhamr_dir.exists()
        and bool(sorted(dynhamr_outputs.glob(f"*/{nickname}*/smooth_fit/*_world_results.npz")))
    )
    dynhamr_incomplete = (
        dynhamr_dir.exists() and any(dynhamr_dir.iterdir()) and not dynhamr_has_result
    )
    if dynhamr_has_result:
        paths = data_loading.ExperimentPaths(nickname, root=Path(scene_dir).parent)
        with data_loading.load_dynhamr_results(
            nickname, dynhamr_dir_name=args.dynhamr_dir_name, root=paths.root
        ) as hand_fit:
            frame_ids = data_loading.infer_dynhamr_frame_indices(paths, hand_fit, dynhamr_dir)
            try:
                data_loading.validate_dynhamr_cameras(hand_fit, pointmap_payload, frame_ids)
            except ValueError:
                cameras_stale = True
    if (cameras_stale or dynhamr_incomplete) and not args.dynhamr_overwrite:
        reason = (
            f"cached cameras or saved hand fit disagree with {da3_path.name}"
            if cameras_stale else "previous run left no smooth_fit output"
        )
        print(f"[dynhamr] Re-fitting with DA3 cameras ({reason}); previous fits retained.")

    try:
        if (args.dynhamr_overwrite or cameras_stale or dynhamr_incomplete
                or not any(dynhamr_dir.iterdir())):
            print('Getting hands from Dyn-HaMR using cached pointmap cameras...')
            dynhamr_python = os.environ.get("GRAG_DYNHAMR_PYTHON", sys.executable)
            dynhamr_cmd = [
                dynhamr_python,
                "dyn_hamr/run_opt.py",
                "data=video_vipe",
                "run_opt=True",
                "run_vis=False",
                f"data.seq={nickname}",
                "is_static=False",
                f"log_dir={dynhamr_dir}/outputs",
                f"data.sources.cameras={dynhamr_dir}/cameras/{nickname}/shot-0",
                f"data.sources.tracks={dynhamr_dir}/track_preds/{nickname}",
                f"data.sources.shots={dynhamr_dir}/shot_idcs/{nickname}.json",
            ]
            if args.dynhamr_overwrite:
                dynhamr_cmd.append("overwrite=True")
            if args.dynhamr_overwrite or cameras_stale or dynhamr_incomplete:
                dynhamr_cmd.append(f"exp_name={time.strftime('%Y-%m-%d_%H-%M-%S')}")
            subprocess.run(
                dynhamr_cmd,
                cwd=str(THIRD_PARTY_ROOT / "Dyn_HaMR"),
                check=True,
                text=True
            )
    finally:
        if args.dynhamr_yolo_det_conf is not None:
            if old_dynhamr_yolo_det_conf is None:
                os.environ.pop("DYNHAMR_YOLO_DET_CONF", None)
            else:
                os.environ["DYNHAMR_YOLO_DET_CONF"] = old_dynhamr_yolo_det_conf
        if old_dynhamr_hamer_render is None:
            os.environ.pop("DYNHAMR_HAMER_RENDER", None)
        else:
            os.environ["DYNHAMR_HAMER_RENDER"] = old_dynhamr_hamer_render
        if old_dynhamr_hamer_batch_size is None:
            os.environ.pop("DYNHAMR_HAMER_BATCH_SIZE", None)
        else:
            os.environ["DYNHAMR_HAMER_BATCH_SIZE"] = old_dynhamr_hamer_batch_size

    model_ready = Path(f'{sam3d_path}/sam3d_model/object_slat_coords.pt').exists()
    poses_ready = Path(f'{sam3d_path}/sam3d_poses/poses_dict.pt').exists()
    pose_only_overwrite = bool(args.sam3d_pose_overwrite and model_ready)
    if args.sam3d_pose_overwrite:
        print("[INFO] --sam3d_pose_overwrite set; rerunning SAM3D-Pose with existing MV-SAM3D shape outputs.")
        if pose_only_overwrite:
            print("[INFO] Existing SAM3D model found; rerunning poses only and preserving object_slat_coords.pt.")
        else:
            model_ready = False
        poses_ready = False
    mv_shape_ready = Path(f'{sam3d_path}/ss_return_dict.pth').exists()
    if sam3d_dir_nonempty and not mv_shape_ready and ((not model_ready) or (not poses_ready)) and not args.sam3d_pose_overwrite:
        raise RuntimeError(
            f"Existing non-empty SAM3D directory at {sam3d_path} is incomplete "
            "(missing sam3d_model/object_slat_coords.pt or sam3d_poses/poses_dict.pt). "
            "Not rerunning SAM3D over an existing non-empty directory."
        )
    if (not model_ready) or (not poses_ready):
        ss_cfg_strength = sam3d_ss_cfg_strength
        ss_cfg_strength_pm = sam3d_ss_cfg_strength_pm
        sam3d_pose_inference_steps = sam3d_pose_steps
        temporal_guidance_lambda = sam3d_temporal_guidance_lambda
        scale_mode = 'mv_sam3d'
        lock_scale_during_pose = True
        calibrate_scale_with_selected_view = True
        sam3d_decode_formats = ["gaussian_4"] if pose_only_overwrite else ["gaussian_4", "gaussian"]
        print(
            "Running SAM3D-Pose with "
            f"steps={sam3d_pose_inference_steps}, "
            f"ss_cfg_strength={ss_cfg_strength:.3f}, "
            f"ss_cfg_strength_pm={ss_cfg_strength_pm:.3f}, "
            f"temporal_guidance_lambda={temporal_guidance_lambda:.3f}, "
            f"use_prev_pose={use_prev_pose}, shape_mode={shape_mode}."
        )
        if sam3d_bidirectional_pose:
            pose_inference = Inference(
                str(SAM3DO_CKPT), compile=False, pose_only=pose_only_overwrite
            )
            num_frames_total = int(pointmaps.shape[0])
            anchor_idx = selected_view_idx if sam3d_anchor_frame_idx is None else int(sam3d_anchor_frame_idx)
            anchor_idx = int(np.clip(anchor_idx, 0, max(0, num_frames_total - 1)))
            fwd_indices = list(range(anchor_idx, num_frames_total))
            bwd_indices = list(range(anchor_idx, -1, -1))
            print(
                f"Running bidirectional SAM3D-Pose from anchor frame {anchor_idx} "
                f"(forward {len(fwd_indices)} frames, backward {len(bwd_indices)} frames)."
            )

            forward_output = generate_3d_and_poses(
                scene_dir=scene_dir, sam3d_path=sam3d_path,
                sam3d_pose_inference_steps=sam3d_pose_inference_steps,
                ss_cfg_strength=ss_cfg_strength,
                ss_cfg_strength_pm=ss_cfg_strength_pm,
                temporal_guidance_lambda=temporal_guidance_lambda,
                pointmap=pointmaps, selected_view_idx=anchor_idx,
                shape_mode=shape_mode, use_prev_pose=use_prev_pose,
                scale_mode=scale_mode,
                lock_scale_during_pose=lock_scale_during_pose,
                calibrate_scale_with_selected_view=calibrate_scale_with_selected_view,
                seed=seed,
                frame_indices=fwd_indices,
                decode_formats=sam3d_decode_formats,
                pose_only_models=pose_only_overwrite,
                inference=pose_inference,
            )
            backward_output = generate_3d_and_poses(
                scene_dir=scene_dir, sam3d_path=sam3d_path,
                sam3d_pose_inference_steps=sam3d_pose_inference_steps,
                ss_cfg_strength=ss_cfg_strength,
                ss_cfg_strength_pm=ss_cfg_strength_pm,
                temporal_guidance_lambda=temporal_guidance_lambda,
                pointmap=pointmaps, selected_view_idx=anchor_idx,
                shape_mode=shape_mode, use_prev_pose=use_prev_pose,
                scale_mode=scale_mode,
                lock_scale_during_pose=lock_scale_during_pose,
                calibrate_scale_with_selected_view=calibrate_scale_with_selected_view,
                seed=seed,
                frame_indices=bwd_indices,
                decode_formats=sam3d_decode_formats,
                pose_only_models=pose_only_overwrite,
                inference=pose_inference,
            )
            output = merge_bidirectional_pose_outputs(
                forward_output=forward_output,
                backward_output=backward_output,
                num_frames=num_frames_total,
                anchor_idx=anchor_idx,
            )
        else:
            output = generate_3d_and_poses(
                scene_dir=scene_dir, sam3d_path=sam3d_path,
                sam3d_pose_inference_steps=sam3d_pose_inference_steps,
                ss_cfg_strength=ss_cfg_strength,
                ss_cfg_strength_pm=ss_cfg_strength_pm,
                temporal_guidance_lambda=temporal_guidance_lambda,
                pointmap=pointmaps, selected_view_idx=selected_view_idx,
                shape_mode=shape_mode, use_prev_pose=use_prev_pose,
                scale_mode=scale_mode,
                lock_scale_during_pose=lock_scale_during_pose,
                calibrate_scale_with_selected_view=calibrate_scale_with_selected_view,
                seed=seed,
                decode_formats=sam3d_decode_formats,
                pose_only_models=pose_only_overwrite,
                )
        save_sam3d_output(output, sam3d_path, save_model=not pose_only_overwrite)

    matrix = intrinsics[0].reshape(3,3)
    np.savetxt(f"{scene_dir}/cam_K.txt", matrix, fmt='%.6f')

    postprocess_sequence(scene_dir, preserve_existing_model=pose_only_overwrite)
    print("Post-processing completed.")
