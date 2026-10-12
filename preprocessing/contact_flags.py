"""Gemini-based hand-object contact-flag inference from a video or frame folder."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from preprocessing.common import VIDEO_FPS, sorted_frame_paths


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    if raw_text is None:
        return {}
    text = raw_text.strip()
    # Remove optional markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def parse_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Accept "12.34" and "HH:MM:SS(.sss)".
        try:
            return float(s)
        except Exception:
            pass
        # Accept common suffixes like "0.2s" or "2 sec".
        m_unit = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\s*$", s, flags=re.IGNORECASE)
        if m_unit:
            return float(m_unit.group(1))
        m = re.match(r"^(?:(\d+):)?([0-5]?\d):([0-5]?\d(?:\.\d+)?)$", s)
        if m:
            h = float(m.group(1) or 0.0)
            mm = float(m.group(2))
            ss = float(m.group(3))
            return h * 3600.0 + mm * 60.0 + ss
    return None


def probe_video_time_span_sec(video_path: Union[str, Path]) -> Optional[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if src_fps <= 0.0 or src_frame_count <= 1.0:
        return None
    # Use first-to-last-frame span so timestamp 0 aligns to frame 0 and max aligns to the last frame.
    return float((src_frame_count - 1.0) / src_fps)


def parse_binary_flag(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) > 0.5)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["1", "true", "yes", "y"]:
            return 1
        if v in ["0", "false", "no", "n"]:
            return 0
    return None


def sanitize_contact_intervals(
    intervals: Any,
    duration_sec: float,
    no_contact_before_sec: Optional[float],
    merge_gap_sec: float = 0.0,
) -> List[Dict[str, float]]:
    if not isinstance(intervals, list):
        return []

    start_floor = max(0.0, float(no_contact_before_sec or 0.0))
    max_t = max(0.0, float(duration_sec))
    cleaned: List[Tuple[float, float]] = []
    for it in intervals:
        if not isinstance(it, dict):
            continue
        start_sec = parse_seconds(it.get("start_sec"))
        end_sec = parse_seconds(it.get("end_sec"))
        if start_sec is None or end_sec is None:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        start_sec = max(start_floor, min(max_t, start_sec))
        end_sec = max(start_floor, min(max_t, end_sec))
        if end_sec >= start_sec:
            cleaned.append((start_sec, end_sec))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[0])
    merged: List[List[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for s, e in cleaned[1:]:
        if s - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [{"start_sec": float(s), "end_sec": float(e)} for s, e in merged]


def flags_to_intervals(flags: np.ndarray, fps: float) -> List[Dict[str, float]]:
    intervals: List[Dict[str, float]] = []
    n = int(flags.shape[0])
    if n == 0:
        return intervals
    fps = float(max(fps, 1e-6))
    i = 0
    while i < n:
        if int(flags[i]) == 1:
            j = i
            while j < n and int(flags[j]) == 1:
                j += 1
            intervals.append({"start_sec": float(i / fps), "end_sec": float((j - 1) / fps)})
            i = j
        else:
            i += 1
    return intervals


def intervals_to_frame_flags(
    intervals: List[Dict[str, Any]],
    num_frames: int,
    fps: float,
) -> np.ndarray:
    flags = np.zeros((num_frames,), dtype=np.uint8)
    fps = float(max(fps, 1e-6))
    for it in intervals:
        if not isinstance(it, dict):
            continue
        start_sec = parse_seconds(it.get("start_sec"))
        end_sec = parse_seconds(it.get("end_sec"))
        if start_sec is None or end_sec is None:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        start_sec = max(0.0, start_sec)
        end_sec = max(0.0, end_sec)
        start_idx = max(0, min(num_frames - 1, int(round(start_sec * fps))))
        end_idx = max(0, min(num_frames - 1, int(round(end_sec * fps))))
        if end_idx >= start_idx:
            flags[start_idx:end_idx + 1] = 1
    return flags


def parse_holding_by_frame(
    records: Any,
    num_frames: int,
) -> Optional[np.ndarray]:
    if not isinstance(records, list):
        return None
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.uint8)

    flags = np.zeros((num_frames,), dtype=np.uint8)
    seen_idx = set()
    sequential_fill_idx = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        holding = parse_binary_flag(rec.get("holding"))
        if holding is None:
            continue

        idx_val = rec.get("frame_index")
        idx: Optional[int] = None
        if isinstance(idx_val, (int, float)):
            idx = int(idx_val)
        elif isinstance(idx_val, str):
            s = idx_val.strip()
            if s.lstrip("-").isdigit():
                idx = int(s)

        if idx is None:
            while sequential_fill_idx in seen_idx and sequential_fill_idx < num_frames:
                sequential_fill_idx += 1
            if sequential_fill_idx >= num_frames:
                continue
            idx = sequential_fill_idx
            sequential_fill_idx += 1

        if idx < 0 or idx >= num_frames:
            continue
        seen_idx.add(idx)
        flags[idx] = int(holding)

    if len(seen_idx) == 0:
        return None
    return flags


def is_payload_too_large_error(err_text: str) -> bool:
    s = str(err_text).lower()
    keys = [
        "payload too large",
        "request entity too large",
        "413",
        "too large",
        "maximum context length",
        "context length exceeded",
        "request body too large",
    ]
    return any(k in s for k in keys)


def resize_video_half_ffmpeg(input_video: Path, output_video: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        "scale=iw/2:ih/2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(chunks).strip()
    return str(content)


def infer_gemini_with_video_data(
    gemini_model: str,
    prompt: str,
    video_path: Path,
    api_key: str,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(f"google.genai import failed: {e}") from e

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    ext = video_path.suffix.lower()
    mime = "video/mp4"
    if ext == ".mov":
        mime = "video/quicktime"
    elif ext == ".webm":
        mime = "video/webm"
    elif ext == ".avi":
        mime = "video/x-msvideo"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type=mime)),
                    types.Part(text=prompt),
                ]
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Gemini SDK request failed: {e}") from e

    out = str(getattr(response, "text", "") or "").strip()
    if not out:
        # Fallback for SDK variants where .text may be empty.
        try:
            parts = response.candidates[0].content.parts
            text_chunks = [str(getattr(p, "text", "") or "") for p in parts]
            out = "\n".join([t for t in text_chunks if t.strip()]).strip()
        except Exception:
            out = ""
    if not out:
        raise RuntimeError("Gemini SDK response missing text.")
    return out


def infer_gemini_with_sampled_frames_data(
    gemini_model: str,
    prompt: str,
    frame_paths: List[Path],
    fps: float,
    api_key: str,
    max_frames_candidates: Optional[List[int]] = None,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(f"google.genai import failed: {e}") from e

    if len(frame_paths) == 0:
        raise RuntimeError("No frames available for Gemini frame fallback.")

    client = genai.Client(api_key=api_key)
    max_frames_candidates = max_frames_candidates or [48, 24, 12]
    last_error: Optional[Exception] = None

    for max_frames in max_frames_candidates:
        stride = max(1, int(math.ceil(len(frame_paths) / float(max_frames))))
        sampled_paths = frame_paths[::stride]
        sampled_fps = float(fps) / float(stride)
        parts: List[Any] = []
        for idx, frame_path in enumerate(sampled_paths):
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            ext = frame_path.suffix.lower()
            mime = "image/jpeg"
            if ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            timestamp_sec = float((idx * stride) / max(fps, 1e-6))
            parts.append(types.Part(text=f"Sampled frame {idx}, timestamp_sec={timestamp_sec:.6f}."))
            parts.append(types.Part(inline_data=types.Blob(data=frame_bytes, mime_type=mime)))
        parts.append(
            types.Part(
                text=(
                    f"{prompt}\n"
                    f"These are uniformly sampled frames from the video at sampled_fps={sampled_fps:.6f}. "
                    "Base your timestamps on the original sampled video timeline."
                )
            )
        )
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            out = str(getattr(response, "text", "") or "").strip()
            if not out:
                try:
                    response_parts = response.candidates[0].content.parts
                    text_chunks = [str(getattr(p, "text", "") or "") for p in response_parts]
                    out = "\n".join([t for t in text_chunks if t.strip()]).strip()
                except Exception:
                    out = ""
            if not out:
                raise RuntimeError("Gemini sampled-frames response missing text.")
            return out
        except Exception as e:
            last_error = RuntimeError(f"Gemini sampled-frames request failed: {e}")
            if max_frames != max_frames_candidates[-1]:
                print(f"[WARN] Gemini sampled-frames fallback failed with {len(sampled_paths)} frames; retrying with fewer frames.")

    raise last_error if last_error is not None else RuntimeError("Gemini sampled-frames fallback failed for unknown reason.")


def infer_contact_flags_from_video(
    video_path: Union[str, Path],
    frame_dir: Union[str, Path],
    scene_dir: Union[str, Path],
    fps: float = VIDEO_FPS,
    gemini_model: str = "gemini-3-flash-preview",
    gemini_api_key: Optional[str] = None,
    overwrite: bool = False,
    output_stem: str = "contact_flags_gemini",
    contact_subject: str = "the hand",
) -> Optional[str]:
    """Infer per-frame hand-holding flags from the video with Gemini and save them."""
    scene_dir = Path(scene_dir)
    output_stem = str(output_stem).strip() or "contact_flags_gemini"
    contact_subject = str(contact_subject).strip() or "the hand"
    out_json = scene_dir / f"{output_stem}.json"
    out_txt = scene_dir / f"{output_stem}.txt"
    out_npy = scene_dir / f"{output_stem}.npy"
    if out_json.exists() and not overwrite:
        print(f"[INFO] Existing contact flags found at {out_json}, skipping.")
        return str(out_json)

    frame_paths = sorted_frame_paths(frame_dir)
    if len(frame_paths) == 0:
        print(f"[WARN] No frames in {frame_dir}; cannot build per-frame flags.")
        return None
    num_frames = len(frame_paths)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[WARN] Video not found at {video_path}. Skipping contact inference.")
        return None

    nominal_fps = max(float(fps), 1e-6)
    nominal_duration_sec = max(0.0, (num_frames - 1) / nominal_fps)
    video_time_span_sec = probe_video_time_span_sec(video_path)
    if video_time_span_sec is not None and num_frames > 1 and video_time_span_sec > 0.0:
        flags_fps = float((num_frames - 1) / video_time_span_sec)
        duration_sec = float(video_time_span_sec)
    else:
        flags_fps = float(nominal_fps)
        duration_sec = float(nominal_duration_sec)

    prompt = (
        f"Analyze this video frame-by-frame and determine EXACTLY when {contact_subject} "
        "transitions from NOT holding (0) to HOLDING (1).\n\n"

        "CRITICAL RULES:\n"
        "1. DO NOT assume grasping at 0.0s unless the very first visible frame clearly shows "
        "the fingers already fully wrapped around the object.\n"
        "2. Touching the object is NOT grasping.\n"
        "3. Reaching toward the object is NOT grasping.\n"
        "4. Grasping begins ONLY at the FIRST frame where the fingers visibly enclose "
        "the object (clear curvature of fingers around it).\n"
        "5. If uncertain, DELAY the start time until enclosure is visually unambiguous.\n"
        "6. The start time must correspond to a specific visible frame change.\n\n"

        "Definitions:\n"
        f"- REACHING (0): {contact_subject} moving toward object or touching without enclosure.\n"
        f"- GRASPING (1): The fingers of {contact_subject} clearly wrapped around object OR object is clearly "
        "being lifted while enclosed.\n\n"

        "Step 1: Briefly describe the timeline of states in order.\n"
        "Step 2: Identify the FIRST exact timestamp where grasping (1) begins.\n"
        "Step 3: Identify when grasping ends (if it ends).\n\n"

        "Then return ONLY a valid JSON object inside a ```json block.\n"
        "Schema:\n"
        "{\n"
        '  "timeline_reasoning": "string",\n'
        '  "contact_intervals": [\n'
        '    {"start_sec": number, "end_sec": number}\n'
        "  ],\n"
        '  "no_contact_before_sec": number\n'
        "}\n\n"

        f"Video metadata: duration_sec={duration_sec:.6f}, sampled_num_frames={num_frames}, "
        f"sampled_fps={flags_fps:.6f}.\n"

        "Output constraints:\n"
        "- Use exact timestamps.\n"
        "- Do NOT round aggressively.\n"
        "- Do NOT assume holding from 0.0s unless clearly visible.\n"
        "- If grasping never occurs, return an empty list for contact_intervals.\n"
    )

    parsed: Dict[str, Any] = {}
    response_text = ""
    source_name = "gemini_video_inline_data"
    resolved_gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # Primary: Gemini on video input.
    if resolved_gemini_key:
        try:
            response_text = infer_gemini_with_video_data(
                gemini_model=gemini_model,
                prompt=prompt,
                video_path=video_path,
                api_key=resolved_gemini_key,
            )
            parsed = extract_json_object(response_text)
        except Exception as e:
            if is_payload_too_large_error(str(e)):
                print("[WARN] Gemini request too large; retrying with 0.5x resized video.")
                try:
                    with tempfile.TemporaryDirectory(prefix="gemini_half_", dir="/tmp") as td:
                        small_video = Path(td) / f"{video_path.stem}_half.mp4"
                        resize_video_half_ffmpeg(video_path, small_video)
                        response_text = infer_gemini_with_video_data(
                            gemini_model=gemini_model,
                            prompt=prompt,
                            video_path=small_video,
                            api_key=resolved_gemini_key,
                        )
                    parsed = extract_json_object(response_text)
                    source_name = "gemini_video_inline_data_half_resized"
                except Exception as e2:
                    print(f"[WARN] Gemini retry with half-size video failed: {e2}")
            else:
                print(f"[WARN] Gemini contact inference failed: {e}")
    else:
        print("[WARN] GEMINI_API_KEY/GOOGLE_API_KEY not set; cannot infer contact flags.")

    # Secondary: Gemini on sampled frames, avoids whole-video upload and keeps the same API key.
    if (not isinstance(parsed, dict) or len(parsed) == 0) and resolved_gemini_key:
        try:
            response_text = infer_gemini_with_sampled_frames_data(
                gemini_model=gemini_model,
                prompt=prompt,
                frame_paths=frame_paths,
                fps=flags_fps,
                api_key=resolved_gemini_key,
            )
            parsed = extract_json_object(response_text)
            source_name = "gemini_sampled_frames_inline_data"
        except Exception as e:
            print(f"[WARN] Gemini sampled-frames fallback failed: {e}")

    if not isinstance(parsed, dict) or len(parsed) == 0:
        print("[WARN] Gemini contact inference produced no parsable output.")
        return None

    flags = parse_holding_by_frame(parsed.get("holding_by_frame"), num_frames=num_frames)
    if flags is None:
        no_contact_before_sec = parse_seconds(parsed.get("no_contact_before_sec"))
        contact_whole_video = parse_binary_flag(parsed.get("contact_whole_video"))
        contact_at_first_frame = parse_binary_flag(parsed.get("contact_at_first_frame"))
        contact_at_last_frame = parse_binary_flag(parsed.get("contact_at_last_frame"))
        intervals = sanitize_contact_intervals(
            parsed.get("contact_intervals", []),
            duration_sec=duration_sec,
            no_contact_before_sec=no_contact_before_sec,
            merge_gap_sec=0.0,
        )
        flags = intervals_to_frame_flags(intervals, num_frames=num_frames, fps=flags_fps)
        if contact_whole_video == 1:
            flags[:] = 1
        else:
            if contact_at_first_frame == 1:
                flags[0] = 1
            if contact_at_last_frame == 1:
                flags[-1] = 1

    # num_frames > 0 is guaranteed by the early return above.
    contact_whole_video = int(np.all(flags == 1))
    contact_at_first_frame = int(flags[0])
    contact_at_last_frame = int(flags[-1])
    first_contact = np.where(flags == 1)[0]
    if len(first_contact) == 0:
        no_contact_before_sec = float(duration_sec)
    elif int(first_contact[0]) == 0:
        no_contact_before_sec = 0.0
    else:
        no_contact_before_sec = float((int(first_contact[0]) - 1) / max(flags_fps, 1e-6))

    intervals = flags_to_intervals(flags, fps=flags_fps)

    per_frame = []
    for idx, frame_path in enumerate(frame_paths):
        per_frame.append(
            {
                "frame_index": int(idx),
                "frame_name": frame_path.name,
                "timestamp_sec": float(idx / max(flags_fps, 1e-6)),
                "contact": int(flags[idx]),
            }
        )

    payload = {
        "source": source_name,
        "model": gemini_model,
        "contact_subject": contact_subject,
        "output_stem": output_stem,
        "video_path": str(video_path),
        "fps_for_flags": float(flags_fps),
        "num_frames": int(num_frames),
        "duration_sec": float(duration_sec),
        "requested_fps": float(fps),
        "video_time_span_sec": float(video_time_span_sec) if video_time_span_sec is not None else None,
        "contact_whole_video": contact_whole_video,
        "contact_at_first_frame": contact_at_first_frame,
        "contact_at_last_frame": contact_at_last_frame,
        "no_contact_before_sec": no_contact_before_sec,
        "contact_intervals": intervals,
        "raw_response_text": response_text,
        "frames": per_frame,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    np.save(out_npy, flags)
    with open(out_txt, "w") as f:
        f.write("# frame_index,timestamp_sec,contact\n")
        for row in per_frame:
            f.write(f"{row['frame_index']},{row['timestamp_sec']:.6f},{row['contact']}\n")

    print(f"[INFO] Saved contact flags: {out_json}")
    return str(out_json)

