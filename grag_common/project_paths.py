from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
ASSETS_ROOT = PROJECT_ROOT / "assets"

WORK_DIR = Path(os.environ.get("GRAG_WORK_DIR", PROJECT_ROOT / "data"))
LOG_ROOT = Path(os.environ.get("GRAG_LOG_ROOT", PROJECT_ROOT / "outputs"))

SAM3DO_CKPT = Path(
    os.environ.get(
        "GRAG_SAM3DO_CKPT",
        THIRD_PARTY_ROOT / "SAM3DO" / "checkpoints" / "hf" / "pipeline_video.yaml",
    )
)
MANO_CFG_PATH = Path(
    os.environ.get(
        "GRAG_MANO_CFG_PATH",
        ASSETS_ROOT / "mano_config.yaml",
    )
)
MANO_MODEL_PATH = Path(
    os.environ.get(
        "GRAG_MANO_MODEL_PATH",
        THIRD_PARTY_ROOT / "Dyn_HaMR" / "_DATA" / "mano",
    )
)
MANO_MEAN_PARAMS_PATH = Path(
    os.environ.get(
        "GRAG_MANO_MEAN_PARAMS_PATH",
        THIRD_PARTY_ROOT / "Dyn_HaMR" / "_DATA" / "data" / "mano_mean_params.npz",
    )
)
CONTACT_ZONES_PATH = Path(
    os.environ.get(
        "GRAG_CONTACT_ZONES_PATH",
        ASSETS_ROOT / "contact_zones.pkl",
    )
)


def append_project_paths() -> None:
    for path in (
        PROJECT_ROOT,
        THIRD_PARTY_ROOT,
        THIRD_PARTY_ROOT / "SAM3DO",
        THIRD_PARTY_ROOT / "sam3",
        THIRD_PARTY_ROOT / "DA3" / "src",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
