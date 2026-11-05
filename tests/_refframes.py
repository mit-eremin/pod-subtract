"""The two fixed regression frames and the reference runners behind tests/data/bgsub_v{0,1,2}_reference.npz.

Merged from pod/background scripts/bgsub_v1_make_v0_reference.py (frames, v0 run; frozen at 07a06b9),
scripts/bgsub_v2_make_v1_reference.py (v1 run; frozen at 88941cb/092ece9) and scripts/bgsub_v2_1_make_v2_reference.py
(v2 run; frozen at 5e4a90c), background commit f1e93f5. The .npz files were produced in background by those scripts and
are copied unchanged; do NOT regenerate them here (they pin the historical outputs). Freeze a new reference (e.g. for
v2.1 before v2.2 work) with a new runner and a new file, from a clean, committed state.
"""
from __future__ import annotations

import numpy as np

import bgsub as BG

V1_METHODS = ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G-ML", "AUTO-ML", "G-LC", "AUTO-LC", "G_PF")
V2_METHODS = ("G", "GL", "AUTO", "G-ML", "AUTO-ML", "G-LC", "AUTO-LC", "G_PF")


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


def _laws():
    return (("p", BG.CountLaw.poisson()), ("a8", BG.CountLaw.apollo(8, 0.35, dead_time_tau=2e-4)))


def run_v1(version="v1"):
    c, m, spots, K4, K8 = frames()
    out = {}
    for (name, law), K in zip(_laws(), (K4, K8)):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex, version=version)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0))
        out[f"{name}_bg"] = np.nan_to_num(res.bg[::7, ::7].astype(np.float64))
        out[f"{name}_disp"] = np.array(res.dispersion)
        out[f"{name}_flags"] = np.array(res.flags.sum())
        for meth in V1_METHODS:
            out[f"{name}_{meth}_I"] = o[meth]["I"]; out[f"{name}_{meth}_sigma"] = o[meth]["sigma"]
        out[f"{name}_AUTO-ML_lam_min5"] = o["AUTO-ML"]["lam_min5"]
        out[f"{name}_AUTO_use"] = o["AUTO"]["auto_use_GL"]
    return out


def run_v2(version="v2"):
    c, m, spots, K4, K8 = frames()
    out = {}
    for (name, law), K in zip(_laws(), (K4, K8)):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex, version=version)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0), methods=("G", "GL", "AUTO"),
                               lam_eval=np.full(len(spots), 50.0))
        out[f"{name}_bg"] = np.nan_to_num(res.bg[::7, ::7].astype(np.float64))
        out[f"{name}_disp"] = np.array(res.dispersion)
        out[f"{name}_flags"] = np.array(res.flags.sum())
        for meth in V2_METHODS:
            for key in ("I", "sigma", "ci_lo", "ci_hi", "sigma_at"):
                if key in o[meth]:
                    out[f"{name}_{meth}_{key}"] = o[meth][key]
        out[f"{name}_AUTO_w"] = o["AUTO"]["auto_w"]
    return out
