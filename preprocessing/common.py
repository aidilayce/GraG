"""Small shared constants and helpers for the preprocessing package."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

VIDEO_FPS = 10


def sorted_frame_paths(frame_dir: Union[str, Path]) -> List[Path]:
    frame_dir = Path(frame_dir)
    return sorted(
        list(frame_dir.glob("*.jpg")) +
        list(frame_dir.glob("*.jpeg")) +
        list(frame_dir.glob("*.png"))
    )


def mask_dir_complete_for_frames(mask_dir: Union[str, Path], frame_dir: Union[str, Path]) -> bool:
    frame_paths = sorted_frame_paths(frame_dir)
    if len(frame_paths) == 0:
        return False
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return False
    return all((mask_dir / f"{frame_path.stem}.png").exists() for frame_path in frame_paths)

