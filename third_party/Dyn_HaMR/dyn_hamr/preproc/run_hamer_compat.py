"""Launch vendored HaMeR with the compatibility shim required by chumpy."""

from __future__ import annotations

import inspect
import runpy
import sys
from pathlib import Path

import numpy as np


hamer_root = Path(__file__).resolve().parents[2] / "third-party" / "hamer"
sys.path.insert(0, str(hamer_root))

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

for legacy_name, builtin_type in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if legacy_name not in np.__dict__:
        setattr(np, legacy_name, builtin_type)

runpy.run_path(str(hamer_root / "run.py"), run_name="__main__")
