# Installation

GraG targets **Python 3.12** with **CUDA 12.1** and **PyTorch 2.4.1**. We use
[uv](https://docs.astral.sh/uv/) for the environment.

Installation is two steps: (1) the pip-resolvable core via `uv`, then (2) the
CUDA extensions that must be built from source against your local toolchain.

## 1. Core environment (`grag`)

```bash
# from the repo root

# Put the uv cache wherever you have room (optional; this is portable):
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

uv venv --python 3.12 .venv          # creates the 'grag' environment (.venv)
source .venv/bin/activate
uv sync --extra eval                 # core + evaluation deps (incl. cu121 torch)
```

`uv sync` installs everything in `pyproject.toml` and makes the stage packages
(`grag_common` / `preprocessing` / `optimization` / `eval`) importable. The
`--extra eval` group adds `open3d` + `loguru` for the vendored HOLD evaluator.

## 2. CUDA extensions (build from source)

These build against the torch you just installed and need a CUDA toolkit whose
**major** version matches torch's (CUDA **12**.x for the cu121 wheels — a 12.x
minor mismatch only warns, but a `nvcc` from CUDA 11 hard-errors). The uv venv
also needs `pip` + `setuptools` + `wheel` for the legacy `setup.py` builds:

```bash
uv pip install pip setuptools wheel

# CUDA_HOME must point at a CUDA 12.x toolkit with a full `nvcc`. If you don't
# have a system CUDA 12 toolkit, get just the compiler via conda (self-contained;
# the pip `nvidia-cuda-nvcc-cu12` wheels are missing the nvcc frontend on some
# platforms):
#   conda create -y -p /path/cuda121 -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1
export CUDA_HOME=/path/cuda121
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"   # your GPU's arch (8.9 = RTX 4090); speeds up the build

# If UV_CACHE_DIR is on a different filesystem than the venv, add --no-cache to
# these (uv can't hardlink/rename the built wheel across filesystems).

# PyTorch3D 0.7.8 (mesh ops, rendering)
uv pip install --no-cache --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# gsplat 1.5.2 (Gaussian renderer)
uv pip install --no-cache --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf"

# nvdiffrast (some renderers)
uv pip install --no-cache --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# chumpy (MANO dependency; py3.12-compatible fork)
uv pip install --no-cache --no-build-isolation "git+https://github.com/zerchen/chumpy@15479f8"
```

Verify: `python -c "import torch, pytorch3d, gsplat, nvdiffrast.torch, chumpy; print('ok')"`.
Exact commits used in development are recorded in `requirements-lock.txt`.

Two more packages are needed by the vendored SAM3DO import chain
(`grag_common.optimization_utils` → `sam3d_objects`) and are not plain-PyPI
installs:

```bash
# kaolin (build matching torch 2.4.1 / cu121)
uv pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html

# utils3d (pinned commit used in development)
uv pip install "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
```

Verify the optimizer import chain: `python -c "import grag_common.optimization_utils; print('ok')"`.

## 3. Vendored components under `third_party/`

The pipeline reuses several upstream projects (SAM3D-Objects, DepthAnything3,
DynHaMR, MV-SAM3D, SAM3). Each ships its own weights/setup; follow the download
instructions in the top-level `README.md` (models are gated and must be
downloaded by the user under their own licenses).

## 4. Models you must download yourself

- **MANO** (hand model) — register at https://mano.is.tue.mpg.de and place the
  models where `assets/mano_config.yaml` / `GRAG_MANO_MODEL_PATH` expect them.
  Dyn-HaMR also imports the `mano` Python package from the MANO download; put its
  folder on `PYTHONPATH` or copy it into `.venv/lib/python3.12/site-packages/`.
  We follow HOLD's setup; see their
  [setup guide](https://github.com/zc-alexfan/hold/blob/master/docs/setup.md).
- Stage-specific checkpoints (SAM3D-Objects, DepthAnything3, etc.) — see the
  main `README.md`.

## 5. HOT3D clip extraction (separate environment)

Extracting the HOT3D evaluation clips (`eval/hot3d/`) needs `projectaria-tools`
plus the HOT3D repo's `data_loaders` / `dataset_api` modules. `projectaria-tools`
requires `numpy>=2`, which conflicts with GraG's pinned `numpy==1.26.4`, so it
**cannot** live in the `grag` env. Run clip extraction in the HOT3D toolkit's own
environment (see https://github.com/facebookresearch/hot3d), then point the
evaluator at its output via `--manifest_csv` / the `HOT3D_*` env vars
(see `eval/README.md`). The `grag` env still runs the HOT3D *evaluator* itself.

## Running

Everything is wrapped by `recon.sh` at the repo root:

```bash
bash recon.sh path/to/clip.mp4                 # in-the-wild: preprocess + optimize
bash recon.sh path/to/clip.mp4 ho3d  --eval    # + export & evaluate vs HO3D GT
```

See the header of `recon.sh` for the configuration env vars
(`MANO_DIR`, `HOLD_CODE_DIR`, `HOT3D_GT_CACHE_DIR`, `GRAG_LOG_ROOT`, …).

## Notes

- `numpy` is pinned to `1.26.4`: several CUDA extensions and `pytorch3d 0.7.8`
  are built against the NumPy 1.x ABI.
- If `uv sync` cannot find the cu121 torch wheels, confirm the
  `pytorch-cu121` index in `pyproject.toml` is reachable from your network.
