#!/usr/bin/env python3
"""
Summarize per-sequence eval metrics (from run_eval.py predictions) as
mean +- std per dataset. HO3Dv3 = hold_* (.metric.json), HOT3D = H0* (.metric.hot3d_compact.json).

  python3 scripts/summarize_eval.py --pred_root <out_root>/predictions [--tag NAME]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

METRICS = ["cd_right", "cd_ra", "mpjpe_ra_r", "mrrpe_ho", "f5_right", "f10_right", "cd_icp"]


def collect(pred_root):
    rows = {"HO3D": [], "HOT3D": []}
    for js in sorted(glob.glob(str(pred_root) + "/**/*.metric*.json", recursive=True)):
        name = os.path.basename(js)
        seq = name.split(".metric")[0]
        ds = "HOT3D" if seq.startswith("H0") else "HO3D"
        try:
            d = json.loads(Path(js).read_text())
        except Exception:
            continue
        d["_seq"] = seq
        d["_json"] = js
        rows[ds].append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = collect(args.pred_root)

    print(f"\n===== {args.tag or args.pred_root} =====")
    for ds in ("HO3D", "HOT3D"):
        rs = rows[ds]
        if not rs:
            continue
        print(f"\n[{ds}]  n={len(rs)} seqs")
        for m in METRICS:
            vals = [float(r[m]) for r in rs if m in r and r[m] is not None
                    and isinstance(r[m], (int, float)) and np.isfinite(r[m])]
            if not vals:
                continue
            print(f"   {m:12s}: {np.mean(vals):8.3f} +- {np.std(vals):6.3f}   (n={len(vals)})")
        # per-seq cd_right for sanity
        cds = [(r["_seq"], r.get("cd_right")) for r in rs]
        print("   cd_right per-seq: " + ", ".join(f"{s.replace('hold_','').replace('_ho3d','')[:10]}={v:.2f}"
              for s, v in cds if isinstance(v, (int, float))))


if __name__ == "__main__":
    main()
