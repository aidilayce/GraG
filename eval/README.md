# Evaluation

Reproduce the HO3D and HOT3D numbers from a trained GraG run
(`refiner_final_state.pt`, written by `optimization/opt.py`).

The pipeline has two stages for both datasets:

1. **Export predictions** — turn a `refiner_final_state.pt` into the five
   prediction `.npy` files the evaluators read.
2. **Evaluate** — compare against ground truth and write a metrics JSON.

The evaluation metric code is adapted from **HOLD**
([zc-alexfan/hold](https://github.com/zc-alexfan/hold), MIT) and vendored under
`hold_eval/`; the HOT3D clip tools are from the **HOT3D toolkit**
([facebookresearch/hot3d](https://github.com/facebookresearch/hot3d),
Apache-2.0). See `THIRD_PARTY_NOTICES.md`, `LICENSE.hold`, `LICENSE.hot3d`.

```
eval/
  export_predictions.py            # refiner_final_state.pt -> 5 prediction .npy files
  run_eval.py                      # batch: export + evaluate many runs -> summary.{json,csv}
  summarize.py                     # aggregate metric JSONs into mean ± std per dataset
  hold_eval/                       # vendored HOLD evaluator closure (adapted from HOLD, MIT)
    hold/common/*.py               #   xdict, transforms, body_models, metrics, ...
    hold/code/evaluate.py          #   HO3D evaluator  (--sd_p <pred_dir>)
    hold/code/evaluate_hot3d.py    #   HOT3D evaluator (--pred_path <pred_dir>, viewer-cache GT)
    hold/code/src/...              #   eval_modules, HO3D GT loader (gt.py)
  hot3d/
    clips_table.csv                # the 18 evaluation clips (sequence, hand side, TS range, object uid)
    export_hot3d_clips.py          # raw HOT3D dataset -> single-hand/single-object clips + manifest
    undistort_and_export.py        # manifest -> undistorted, HOLD-format frames + GT
```

Metrics reported (both datasets): `mpjpe_ra_r` (mm), `mrrpe_ho` (mm),
`cd_ra` / `cd_right` (Chamfer distance, cm²) and `f5/f10` F-scores (%);
`cd_icp` with `--include_icp`.

## Two environments

Export runs in the **GraG environment** (this repo's `.venv`; see the top-level
`INSTALL.md`). The HOLD evaluator was written for an older stack, so it runs in a
**separate HOLD environment** — create it from HOLD's own setup
(https://github.com/zc-alexfan/hold). `run_eval.py` takes both interpreters via
`--python_exec` (GraG) and `--hold_python_exec` (HOLD).

## MANO models (download yourself)

MANO is distributed under a non-commercial license and is **not** redistributed
here. Download `MANO_LEFT.pkl` / `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de
(the same models HOLD uses: see
https://github.com/zc-alexfan/hold/blob/master/docs/setup.md#download-data-and-models)
and pass the directory as `--mano_dir`.

---

## HO3D

Ground truth comes from HOLD's HO3D data generation.

**1. Ground truth (one-time).** Clone HOLD and build the HO3D GT following its
docs. This produces, per sequence, a processed GT tensor
(`hold/generator/assets/ho3d_v3/processed/<seq>.pt`) plus the per-sequence
`code/data/<seq>/build/` assets (`corres.txt`, `boxes.npy`) and object meshes.
The evaluator reads these **relative to the HOLD `code/` dir**, so it is run with
`cwd = <hold>/code`; pass that as `--hold_code_dir`. To point at a processed-`.pt`
dir elsewhere, set `GRAG_HO3D_PROCESSED_DIR` (default `../generator/assets/ho3d_v3/processed`).

**2. Export predictions** (GraG env):

```bash
python eval/export_predictions.py \
    --seq_name hold_ABF12_ho3d \
    --weights_path /path/to/<seq>/refiner_final_state.pt \
    --out_dir /path/to/predictions/HO3D/hold_ABF12_ho3d \
    --mano_dir /path/to/mano \
    --demo_root /path/to/demo_data
```

**3. Evaluate** (HOLD env, from the HOLD `code/` dir):

```bash
cd /path/to/hold/code
HOLD_SKIP_ICP=1 python /path/to/eval/hold_eval/hold/code/evaluate.py \
    --sd_p /path/to/predictions/HO3D/hold_ABF12_ho3d
```

Writes `<pred_dir>.metric.json`. Drop `HOLD_SKIP_ICP` to also compute the ICP
metric (needs `open3d`). The prediction directory **name** must be the sequence
name (e.g. `hold_ABF12_ho3d`) — the GT loader parses the sequence from it.

---

## HOT3D

HOT3D uses a moving (head-mounted) camera. The evaluation clip set is fixed in
`hot3d/clips_table.csv` (sequence name, hand side, start/end timestamp in ns,
object uid) — the 18 clips used in the paper.

**1. Dataset + toolkit (one-time).** Download the HOT3D dataset and install the
HOT3D toolkit (`projectaria_tools` and HOT3D `data_loaders`/`dataset_api`):
https://github.com/facebookresearch/hot3d .

**2. Extract clips.** Use `hot3d/export_hot3d_clips.py` to cut single-hand /
single-object clips and write a clip manifest CSV, then
`hot3d/undistort_and_export.py` to produce the undistorted, rotated frames used as
GraG input. `clips_table.csv` lists the 18 clips the paper uses.

**3. Build the ground-truth cache.** Because the HOT3D camera moves per frame, the
GT hand+object is taken per-frame in the camera frame. Build one `<clip>.npz` per
clip with `hot3d/export_hot3d_gt_viewer_cache.py` (in the HOT3D toolkit env —
needs `projectaria-tools` + the HOT3D SDK):

```bash
python eval/hot3d/export_hot3d_gt_viewer_cache.py \
    --seq_name <clip_token> \
    --out_path <gt_cache_dir>/<clip_token>.npz \
    --dataset_root /path/to/hot3d_dataset \
    --mano_model_dir /path/to/mano
```

Point the evaluator at that cache with `HOT3D_VIEWER_CACHE_DIR` (or
`--gt_viewer_cache_dir`); object meshes are found under the clip-export roots in
`$GRAG_HOT3D_EXPORT_ROOTS` (`:`-separated).

**4. Export predictions** — add `--apply_camera_transform` (the HOT3D camera moves
per frame):

```bash
python eval/export_predictions.py \
    --seq_name <clip_token> \
    --weights_path /path/to/<clip>/refiner_final_state.pt \
    --out_dir /path/to/predictions/HOT3D/<clip_token> \
    --mano_dir /path/to/mano --demo_root /path/to/demo_data \
    --apply_camera_transform
```

**5. Evaluate** (HOLD env):

```bash
HOT3D_VIEWER_CACHE_DIR=/path/to/gt_cache \
python /path/to/eval/hold_eval/hold/code/evaluate_hot3d.py \
    --pred_path /path/to/predictions/HOT3D/<clip_token> \
    --include_icp
```

Writes `<pred_dir>.metric.hot3d_compact.json`.

---

## Batch runs

`run_eval.py` runs export + evaluation over a list of runs and collects a
`summary.{json,csv}`. The runs file has one `dataset,seq,weights_path` per line:

```
# dataset,seq_name,weights_path
HO3D,hold_ABF12_ho3d,/path/to/ABF12/refiner_final_state.pt
HOT3D,H001P0001c...,/path/to/H001.../refiner_final_state.pt
```

```bash
python eval/run_eval.py \
    --runs_file runs.txt \
    --out_root /path/to/eval_out \
    --mano_dir /path/to/mano \
    --demo_root /path/to/demo_data \
    --hold_code_dir /path/to/hold/code \
    --hot3d_gt_cache_dir /path/to/gt_cache \
    --python_exec /path/to/grag/.venv/bin/python \
    --hold_python_exec /path/to/hold_env/bin/python

python eval/summarize.py --pred_root /path/to/eval_out/predictions
```
