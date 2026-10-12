#!/usr/bin/env python3
"""Export GraG predictions and evaluate them against HO3D / HOT3D ground truth.

For every listed run this script:
  1. exports the five prediction ``.npy`` files with ``export_predictions.py``
     (run in the GraG environment), then
  2. evaluates them with the vendored HOLD evaluator (run in a separate HOLD
     environment): ``hold_eval/hold/code/evaluate.py`` for HO3D or
     ``evaluate_hot3d.py`` for HOT3D,
and finally writes a ``summary.{json,csv}`` over all runs.

Runs are listed in a plain text file, one per line:

    # dataset,seq_name,weights_path
    HO3D,hold_ABF12_ho3d,/path/to/ABF12/refiner_final_state.pt
    HOT3D,H001P0001...,/path/to/H001.../refiner_final_state.pt

See eval/README.md for the required GT setup, MANO download and env vars.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
EXPORT_SCRIPT = EVAL_DIR / "export_predictions.py"
HO3D_EVAL_SCRIPT = EVAL_DIR / "hold_eval" / "hold" / "code" / "evaluate.py"
HOT3D_EVAL_SCRIPT = EVAL_DIR / "hold_eval" / "hold" / "code" / "evaluate_hot3d.py"

PRED_FILES = [
    "pred_v3d_c_object.npy",
    "pred_v3d_c_right.npy",
    "pred_j3d_c_right.npy",
    "pred_faces_object.npy",
    "pred_faces_right.npy",
]


@dataclass(frozen=True)
class Run:
    dataset: str
    seq: str
    weights_path: Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_file", type=Path, required=True,
                   help="Text file listing 'dataset,seq_name,weights_path' per line.")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Output directory for predictions, metrics and summary.")
    p.add_argument("--mano_dir", type=Path, required=True,
                   help="Directory with MANO_LEFT.pkl / MANO_RIGHT.pkl (download yourself).")
    p.add_argument("--demo_root", type=Path, default=REPO_ROOT / "data",
                   help="Root holding <seq>/ preprocessing + reconstruction outputs.")
    p.add_argument("--hold_code_dir", type=Path, default=None,
                   help="HOLD checkout code/ dir; provides HO3D GT data (cwd for evaluate.py).")
    p.add_argument("--hot3d_gt_cache_dir", type=Path, default=None,
                   help="Dir of HOT3D <clip>.npz GT caches (from hot3d/export_hot3d_gt_viewer_cache.py); required for HOT3D runs.")
    p.add_argument("--python_exec", type=Path, default=Path(sys.executable),
                   help="Python interpreter for the GraG export step.")
    p.add_argument("--hold_python_exec", type=Path, default=Path(sys.executable),
                   help="Python interpreter for the HOLD evaluation step.")
    p.add_argument("--only_dataset", choices=["HO3D", "HOT3D"], default=None)
    p.add_argument("--seqs", nargs="*", default=None, help="Restrict to these seq names.")
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--force_export", action="store_true")
    p.add_argument("--include_icp", action="store_true",
                   help="Include the (slow, open3d) ICP metric.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def parse_runs(path: Path) -> list[Run]:
    runs: list[Run] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad runs line (want 'dataset,seq,weights_path'): {raw!r}")
        dataset, seq, weights = parts
        if dataset not in {"HO3D", "HOT3D"}:
            raise ValueError(f"Unknown dataset {dataset!r} in {raw!r}")
        runs.append(Run(dataset=dataset, seq=seq, weights_path=Path(weights)))
    return runs


def run_logged(cmd, cwd, log_path, env, verbose):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"# {datetime.now().isoformat(timespec='seconds')}\n# cwd: {cwd}\n# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            if verbose:
                print(line, end="")
        ret = proc.wait()
        log.write(f"\n# rc={ret}\n")
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def collect_metric(pred_dir: Path, dataset: str):
    suffix = ".metric.json" if dataset == "HO3D" else ".metric.hot3d_compact.json"
    metric_path = Path(str(pred_dir) + suffix)
    if not metric_path.is_file():
        return None
    try:
        return json.loads(metric_path.read_text())
    except json.JSONDecodeError:
        return None


def write_summary(out_root: Path, rows: list[dict]):
    saved = list((out_root / "per_sequence").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in saved]
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("seq", "")))
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    keys = sorted({k for row in rows for k in row.keys()})
    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_sequence_result(out_root: Path, row: dict):
    path = out_root / "per_sequence" / f"{row['dataset']}_{row['seq']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")


def main():
    args = parse_args()
    runs = parse_runs(args.runs_file)
    if args.only_dataset:
        runs = [r for r in runs if r.dataset == args.only_dataset]
    if args.seqs:
        wanted = set(args.seqs)
        runs = [r for r in runs if r.seq in wanted]
    if not runs:
        raise SystemExit("No runs selected.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    log_root = args.out_root / "logs"
    pred_root = args.out_root / "predictions"

    rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.dataset} {run.seq}", flush=True)
        row = {"dataset": run.dataset, "seq": run.seq, "weights_path": str(run.weights_path), "status": "started"}
        pred_dir = pred_root / run.dataset / run.seq
        try:
            if not run.weights_path.is_file():
                raise FileNotFoundError(f"Missing weights: {run.weights_path}")

            # 1. export (GraG env)
            if not args.skip_export:
                have_all = all((pred_dir / n).is_file() for n in PRED_FILES)
                if args.force_export or not have_all:
                    cmd = [str(args.python_exec), str(EXPORT_SCRIPT),
                           "--seq_name", run.seq, "--weights_path", str(run.weights_path),
                           "--out_dir", str(pred_dir), "--mano_dir", str(args.mano_dir),
                           "--demo_root", str(args.demo_root)]
                    if args.hold_code_dir is not None:
                        cmd += ["--hold_code_dir", str(args.hold_code_dir)]
                    if run.dataset == "HOT3D":
                        cmd.append("--apply_camera_transform")
                    run_logged(cmd, cwd=REPO_ROOT, log_path=log_root / run.dataset / f"{run.seq}.export.log",
                               env=dict(os.environ), verbose=args.verbose)
                missing = [n for n in PRED_FILES if not (pred_dir / n).is_file()]
                if missing:
                    raise FileNotFoundError(f"Export left missing files: {missing}")

            # 2. evaluate (HOLD env)
            if not args.skip_eval:
                env = dict(os.environ)
                if run.dataset == "HO3D":
                    cmd = [str(args.hold_python_exec), str(HO3D_EVAL_SCRIPT), "--sd_p", str(pred_dir)]
                    if not args.include_icp:
                        env["HOLD_SKIP_ICP"] = "1"
                    cwd = args.hold_code_dir if args.hold_code_dir else REPO_ROOT
                    # evaluate.py imports `hold.common.metrics` (needs the hold
                    # checkout's parent on the path) and `common.xdict` (needs the
                    # hold/code dir itself).
                    if args.hold_code_dir is not None:
                        hold_code = Path(args.hold_code_dir).resolve()
                        # hold repo root provides `common.*`; its parent provides `hold.*`
                        extra = os.pathsep.join([str(hold_code.parents[1]), str(hold_code.parents[0])])
                        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
                else:
                    if args.hot3d_gt_cache_dir is None:
                        raise ValueError("HOT3D runs need --hot3d_gt_cache_dir")
                    cmd = [str(args.hold_python_exec), str(HOT3D_EVAL_SCRIPT),
                           "--pred_path", str(pred_dir),
                           "--gt_viewer_cache_dir", str(args.hot3d_gt_cache_dir)]
                    if args.include_icp:
                        cmd.append("--include_icp")
                    cwd = args.hold_code_dir if args.hold_code_dir else REPO_ROOT
                run_logged(cmd, cwd=cwd, log_path=log_root / run.dataset / f"{run.seq}.eval.log",
                           env=env, verbose=args.verbose)

            metric = collect_metric(pred_dir, run.dataset)
            if metric:
                for k, v in metric.items():
                    if isinstance(v, (int, float, str)):
                        row[k] = v
            row["prediction_dir"] = str(pred_dir)
            row["status"] = "ok"
            print(f"[OK] {run.seq}", flush=True)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {run.seq}: {row['error']}", flush=True)
        rows.append(row)
        write_sequence_result(args.out_root, row)
        write_summary(args.out_root, rows)

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"[SUMMARY] total={len(rows)} ok={ok} failed={len(rows) - ok} out={args.out_root}", flush=True)


if __name__ == "__main__":
    main()
