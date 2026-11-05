"""Freeze the bgsub v0 outputs (code at commit 07a06b9) on two fixed frames, for the v1 regression test
(tests/test_bgsub_v1.py::test_v0_regression). Run ONCE before any v1 change to src/podbg/bgsub/."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from podbg import bgsub as BG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "bgsub_v0_reference.npz"


def frames():
    N = 256; CY, CX = 131.3, 122.6
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - CY, xx - CX); phi = np.arctan2(yy - CY, xx - CX)
    rr = np.clip(r, 12, None)
    b = 3.0 * (rr / 30.0) ** -2.0 + 0.4 + 0.3 * np.exp(-0.5 * ((r - 70) / 8.0) ** 2) + 0.05 * (r / 100) * np.cos(phi - 0.7) + 0.03 * np.sin(2 * phi)
    m = (r < 14) | ((np.abs(phi - np.pi) < 0.15) & (r < 60))
    rng = np.random.default_rng(20261104)
    spots = np.array([[CY + 40.3, CX + 10.1], [CY - 55.2, CX + 30.7], [CY + 5.5, CX - 70.2], [CY + 80.4, CX - 20.9]])
    lam = np.zeros_like(b)
    for cy, cx in spots:
        lam += 60.0 * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 1.0) / (2 * np.pi)
    K4 = rng.poisson(2.0 * b + lam)
    K8 = rng.poisson(1.5 * b + 0.5 * lam)            # used with the Apollo bin-8 law (values as stored counts)
    return (CY, CX), m, spots, K4, K8


def run():
    c, m, spots, K4, K8 = frames()
    out = {}
    for name, K, law in (("p", K4, BG.CountLaw.poisson()), ("a8", K8, BG.CountLaw.apollo(8, 0.35, dead_time_tau=2e-4))):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0))
        out[f"{name}_bg"] = np.nan_to_num(res.bg[::7, ::7].astype(np.float64))
        out[f"{name}_disp"] = np.array(res.dispersion)
        out[f"{name}_flags"] = np.array(res.flags.sum())
        for meth in ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G-ML", "G-LC", "G_PF"):
            out[f"{name}_{meth}_I"] = o[meth]["I"]; out[f"{name}_{meth}_sigma"] = o[meth]["sigma"]
    return out


if __name__ == "__main__":
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/podbg/bgsub"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert not dirty, "bgsub must be at the committed v0 state"
    out = run(); out["commit"] = np.array(rev)
    np.savez(OUT, **out)
    print("wrote", OUT, rev, {k: np.ravel(v)[:3] for k, v in out.items() if k.endswith("_I")})
