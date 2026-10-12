#!/bin/bash
# GraG — single reconstruction entry point.
#
# Usage:
#   bash recon.sh VIDEO_PATH                      # in-the-wild video: preprocess + optimize
#   bash recon.sh VIDEO_PATH ho3d|hot3d           # benchmark video (dataset-tagged)
#   bash recon.sh VIDEO_PATH ho3d|hot3d --eval    # ... then export + evaluate against GT
#
# VIDEO_PATH may be a video file (e.g. clip.mp4) or an rgb frame directory
# (e.g. some_seq/rgb) — the experiment name is derived the same way preprocess.py
# does (video stem, or the rgb dir's parent name).
#
# Configuration via environment variables (defaults in parentheses):
#   GRAG_WORK_DIR   dataset root that holds <exp>/ preprocessing outputs   (./data)
#   GRAG_LOG_ROOT   where opt.py writes runs + checkpoints                 (./outputs)
#   MANO_DIR        directory with MANO_LEFT.pkl / MANO_RIGHT.pkl          (required for --eval)
#   HOLD_CODE_DIR   HOLD checkout code/ dir, provides HO3D GT              (required for ho3d --eval)
#   HOT3D_GT_CACHE_DIR  dir of HOT3D <clip>.npz GT caches                     (required for hot3d --eval)
#   HOLD_PYTHON     python interpreter for the eval step                   (same env by default)
#   GEMINI_API_KEY  used by preprocessing for contact-flag inference
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the 'grag' uv environment if present — unless a Python environment is
# already active (respect an env the caller activated, e.g. conda), so we don't
# clobber it.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Expose the CUDA runtime shipped inside the venv to the compiled extensions.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    _grag_nvlib="$(echo "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/cuda_runtime/lib)"
    [ -d "$_grag_nvlib" ] && export LD_LIBRARY_PATH="$_grag_nvlib:${LD_LIBRARY_PATH:-}"
fi

VIDEO="${1:-}"
if [ -z "$VIDEO" ]; then
    echo "usage: bash recon.sh VIDEO_PATH [ho3d|hot3d] [--eval]" >&2
    exit 2
fi
shift

DATASET=""
DO_EVAL=0
for arg in "$@"; do
    case "$arg" in
        ho3d|hot3d) DATASET="$arg" ;;
        --eval)     DO_EVAL=1 ;;
        *) echo "recon.sh: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

# Derive the experiment name exactly like preprocess.py.
if [ -d "$VIDEO" ]; then
    EXP="$(basename "$(dirname "$(readlink -f "$VIDEO")")")"   # rgb dir -> parent name
else
    base="$(basename "$VIDEO")"
    EXP="${base%.*}"                                           # video file -> stem
fi

PY="$(command -v python)"
export GRAG_WORK_DIR="${GRAG_WORK_DIR:-$REPO_ROOT/data}"
DEMO_ROOT="$GRAG_WORK_DIR"
export GRAG_LOG_ROOT="${GRAG_LOG_ROOT:-$REPO_ROOT/outputs}"

echo "[recon] exp=$EXP  dataset=${DATASET:-in-the-wild}  eval=$DO_EVAL"
echo "[recon] work_dir=$DEMO_ROOT  log_root=$GRAG_LOG_ROOT"

# 1) Preprocess (masks, depth, contact flags, object reconstruction, keyframes).
echo "[recon] === preprocess ==="
"$PY" preprocessing/preprocess.py -v "$VIDEO"

# 2) Joint hand-object optimization.
echo "[recon] === optimize ==="
"$PY" optimization/opt.py --exp "$EXP" --logdir "$GRAG_LOG_ROOT"

# Locate the checkpoint opt.py just wrote (latest under this experiment).
WEIGHTS="$(find "$GRAG_LOG_ROOT/$EXP" -name refiner_final_state.pt 2>/dev/null | sort | tail -1 || true)"

# 3) Evaluate against ground truth.
if [ "$DO_EVAL" = 1 ]; then
    if [ -z "$DATASET" ]; then
        echo "[recon] --eval requires a dataset argument (ho3d or hot3d)" >&2; exit 2
    fi
    if [ -z "$WEIGHTS" ]; then
        echo "[recon] no refiner_final_state.pt under $GRAG_LOG_ROOT/$EXP" >&2; exit 2
    fi
    : "${MANO_DIR:?set MANO_DIR to the folder containing MANO_LEFT.pkl / MANO_RIGHT.pkl}"
    echo "[recon] === evaluate ($DATASET) ==="
    RUNS="$(mktemp)"
    DS_UPPER="$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')"
    echo "$DS_UPPER,$EXP,$WEIGHTS" > "$RUNS"
    "$PY" eval/run_eval.py \
        --runs_file "$RUNS" \
        --out_root "$GRAG_LOG_ROOT/eval/$EXP" \
        --mano_dir "$MANO_DIR" \
        --demo_root "$DEMO_ROOT" \
        --python_exec "$PY" \
        --hold_python_exec "${HOLD_PYTHON:-$PY}" \
        ${HOLD_CODE_DIR:+--hold_code_dir "$HOLD_CODE_DIR"} \
        ${HOT3D_GT_CACHE_DIR:+--hot3d_gt_cache_dir "$HOT3D_GT_CACHE_DIR"} \
        --include_icp
    rm -f "$RUNS"
    "$PY" eval/summarize.py --pred_root "$GRAG_LOG_ROOT/eval/$EXP/predictions" || true
fi

echo "[recon] done. checkpoint: ${WEIGHTS:-<none>}"
