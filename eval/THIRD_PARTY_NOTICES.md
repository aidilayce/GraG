# Third-party notices — evaluation code

The evaluation tools in this directory reuse and adapt code from two external
projects. We keep their original license terms and attribute them here.

## HOLD (HO3D + HOT3D metrics, GT loading)

- Source: https://github.com/zc-alexfan/hold
- License: MIT — see `LICENSE.hold`
- Copyright (c) 2024 Zicong Fan, Maria Parelli, Maria Eleni Kadoglou,
  Muhammed Kocabas, Xu Chen, Michael J. Black, Otmar Hilliges.

The HOLD evaluator closure is vendored under `hold_eval/`, preserving HOLD's
package layout so its internal imports resolve unchanged:

- `hold_eval/hold/common/` — `xdict.py`, `transforms.py`, `body_models.py`,
  `metrics.py`, `mesh.py`, `data_utils.py`, `np_utils.py`, `thing.py`
- `hold_eval/hold/code/evaluate.py` — HO3D evaluator
- `hold_eval/hold/code/evaluate_hot3d.py` — HOT3D evaluator
- `hold_eval/hold/code/src/utils/` — `eval_modules.py`, `const.py`, `icp.py`
- `hold_eval/hold/code/src/utils/io/` — `pred_loader.py` (prediction loader),
  `gt.py` (HO3D GT)
- `hold_eval/hold/code/src/fitting/utils.py`

Modifications: removed machine-specific absolute paths (data locations are now
resolved via CLI args / environment variables), removed the unused `MANOServer`
import so the model-server subtree is not needed, and neutralized in-library
`sys.path` injections. The metric computation is unchanged and produces
bit-identical results to the upstream code.

MANO models are **not** included (non-commercial license); download them from
https://mano.is.tue.mpg.de and pass `--mano_dir` (see `README.md`).

## HOT3D (clip / GT extraction)

- Source: https://github.com/facebookresearch/hot3d
- License: Apache License 2.0 — see `LICENSE.hot3d`
- Copyright (c) Meta Platforms, Inc. and affiliates.

Files vendored from the HOT3D toolkit, with their Apache-2.0 headers retained:

- `hot3d/export_hot3d_clips.py` (from `export_hot3d_single_hand_same_object_hold.py`)
- `hot3d/undistort_and_export.py` (from `undistort_and_export_hold_from_manifest.py`)

`hot3d/clips_table.csv` lists the exact 18 evaluation clips (sequence name, hand
side, start/end timestamp, object uid) used in the paper.

Using the HOT3D tools additionally requires the HOT3D dataset and the
`projectaria_tools` + HOT3D `data_loaders`/`dataset_api` packages, distributed by
Meta under their own terms (see the HOT3D repository).
