"""Shared runtime configuration for the optimization scripts."""

from __future__ import annotations

import random

import numpy as np
import torch

from grag_common.project_paths import WORK_DIR


sparse_structure_sampler_params = {
    "steps": 12,
    "cfg_strength": 7.5,
}
slat_sampler_params = {
    "steps": 12,
    "cfg_strength": 3,
}

MAX_FRAMES_OFFLINE = 50
MAX_FRAMES_ONLINE = 300
VIDEO_FPS = 10
MAX_SEED = np.iinfo(np.int32).max

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
COLORS = [(0, 0, 255), (0, 255, 255)]  # BGR: Red for negative, Yellow for positive
MARKERS = [1, 5]  # Cross for negative, Star for positive
MARKER_SIZE = 8

device = torch.device("cuda")

opencv_to_opengl = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=torch.float32,
    device=device,
)


def seed_everything(seed: int = MAX_SEED) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(int(seed % (2**32)))


# Seed at import time so that all modules share the same deterministic start.
seed_everything()


__all__ = [
    "WORK_DIR",
    "sparse_structure_sampler_params",
    "slat_sampler_params",
    "MAX_FRAMES_OFFLINE",
    "MAX_FRAMES_ONLINE",
    "VIDEO_FPS",
    "MAX_SEED",
    "LIGHT_BLUE",
    "COLORS",
    "MARKERS",
    "MARKER_SIZE",
    "device",
    "opencv_to_opengl",
]
