"""Freeze the bgsub v2 outputs (code at commit 5e4a90c, before any v2.1 change) on the two fixed regression frames,
for tests/test_bgsub_v2_1.py::test_v2_regression.  Run with the package of that commit first on the path:

    git archive 5e4a90c src/podbg | tar -x -C /tmp/v2ref && PYTHONPATH=/tmp/v2ref/src .venv/bin/python scripts/bgsub_v2_1_make_v2_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from podbg import bgsub as BG

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bgsub_v1_make_v0_reference as REFM  # noqa: E402

OUT = ROOT / "tests" / "data" / "bgsub_v2_reference.npz"
METHODS = ("G", "GL", "AUTO", "G-ML", "AUTO-ML", "G-LC", "AUTO-LC", "G_PF")


def run(version="v2"):
    c, m, spots, K4, K8 = REFM.frames()
    out = {}
    for name, K, law in (("p", K4, BG.CountLaw.poisson()), ("a8", K8, BG.CountLaw.apollo(8, 0.35, dead_time_tau=2e-4))):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex, version=version)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0), methods=("G", "GL", "AUTO"),
                               lam_eval=np.full(len(spots), 50.0))
        out[f"{name}_bg"] = np.nan_to_num(res.bg[::7, ::7].astype(np.float64))
        out[f"{name}_disp"] = np.array(res.dispersion)
        out[f"{name}_flags"] = np.array(res.flags.sum())
        for meth in METHODS:
            for key in ("I", "sigma", "ci_lo", "ci_hi", "sigma_at"):
                if key in o[meth]:
                    out[f"{name}_{meth}_{key}"] = o[meth][key]
        out[f"{name}_AUTO_w"] = o["AUTO"]["auto_w"]
    return out


if __name__ == "__main__":
    import podbg
    print("podbg from", podbg.__file__)
    out = run()
    out["commit"] = np.array("5e4a90c")
    np.savez(OUT, **out)
    print("wrote", OUT, len(out))
