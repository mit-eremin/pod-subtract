"""Freeze the bgsub v1 outputs (code at commit 88941cb) on two fixed frames, for the v2 regression test
(tests/test_bgsub_v2.py::test_v1_regression). Run ONCE before any v2 change to src/podbg/bgsub/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from podbg import bgsub as BG

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bgsub_v1_make_v0_reference as REFM  # noqa: E402

OUT = ROOT / "tests" / "data" / "bgsub_v1_reference.npz"
METHODS = ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G-ML", "AUTO-ML", "G-LC", "AUTO-LC", "G_PF")


def run(version="v1"):
    c, m, spots, K4, K8 = REFM.frames()
    out = {}
    for name, K, law in (("p", K4, BG.CountLaw.poisson()), ("a8", K8, BG.CountLaw.apollo(8, 0.35, dead_time_tau=2e-4))):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex, version=version)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0))
        out[f"{name}_bg"] = np.nan_to_num(res.bg[::7, ::7].astype(np.float64))
        out[f"{name}_disp"] = np.array(res.dispersion)
        out[f"{name}_flags"] = np.array(res.flags.sum())
        for meth in METHODS:
            out[f"{name}_{meth}_I"] = o[meth]["I"]; out[f"{name}_{meth}_sigma"] = o[meth]["sigma"]
        out[f"{name}_AUTO-ML_lam_min5"] = o["AUTO-ML"]["lam_min5"]
        out[f"{name}_AUTO_use"] = o["AUTO"]["auto_use_GL"]
    return out


if __name__ == "__main__":
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/podbg/bgsub"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert not dirty, "bgsub must be at the committed v1 state"
    out = run(); out["commit"] = np.array(rev)
    np.savez(OUT, **out)
    print("wrote", OUT, rev, {k: np.ravel(v)[:3] for k, v in out.items() if k.endswith("_I")})
