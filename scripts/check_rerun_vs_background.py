"""Compare re-run checkpoints (pod/bgsub scripts/synth_v2_1.py) with the published background checkpoints
(~/pod_data/background/bgsub_v2_1/checkpoints/, READ-ONLY): same peaks, same per-method numbers.

Usage: .venv/bin/python scripts/check_rerun_vs_background.py --set test [--out synth_v2_1_rerun]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from bgsub import config


def compare(a, b):
    A = pd.DataFrame(a["rows"]); B = pd.DataFrame(b["rows"])
    num = [c for c in A.columns if c in B.columns and A[c].dtype.kind in "fiub" and not c.endswith("_s") and "t_fit" not in c and "t_plan" not in c]
    if len(A) != len(B):
        return False, f"rows {len(A)} vs {len(B)}"
    d = {}
    for c in num:
        x, y = A[c].to_numpy(float), B[c].to_numpy(float)
        both = np.isnan(x) & np.isnan(y)
        diff = np.where(both, 0.0, np.abs(x - y))          # NaN in one only -> NaN -> counted as a difference
        d[c] = float(np.inf if np.isnan(diff).any() else (diff.max() if len(diff) else 0.0))
    worst = max(d, key=d.get)
    return d[worst] <= 1e-9, f"max |diff| {d[worst]:.3g} in {worst} over {len(num)} numeric columns"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="test"); ap.add_argument("--out", default="synth_v2_1_rerun")
    a = ap.parse_args()
    mine = config.DATA_ROOT / a.out / "checkpoints" / f"synth_{a.set}"
    ref = config.BG_INPUTS["bgsub_v2_1"] / "checkpoints" / f"synth_{a.set}"
    n_ok = 0; n = 0
    for p in sorted(mine.glob("*.pkl")):
        q = ref / p.name
        if not q.exists():
            print(p.name, "no published checkpoint"); continue
        ok, msg = compare(pickle.load(open(p, "rb")), pickle.load(open(q, "rb")))
        n += 1; n_ok += ok
        print(("OK  " if ok else "DIFF"), p.name, msg)
    print(f"{n_ok} / {n} re-run frames equal the published checkpoints")


if __name__ == "__main__":
    main()
