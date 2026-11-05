"""Tests of bgsub v1 (design note 2026-11-04_bgsub_algorithm_v1.md): v0 regression, near-stop block."""
import sys
from pathlib import Path

import numpy as np
import pytest

import bgsub as BG

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def test_v0_regression():
    """version='v0' reproduces the frozen v0 outputs (commit 07a06b9) to floating-point summation order."""
    import _refframes as REFM
    ref = np.load(ROOT / "tests" / "data" / "bgsub_v0_reference.npz")
    c, m, spots, K4, K8 = REFM.frames()
    for name, K, law in (("p", K4, BG.CountLaw.poisson()), ("a8", K8, BG.CountLaw.apollo(8, 0.35, dead_time_tau=2e-4))):
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry(c), law, exclude=ex, version="v0")
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.0), version="v0")
        assert np.allclose(np.nan_to_num(res.bg[::7, ::7].astype(np.float64)), ref[f"{name}_bg"], rtol=1e-6, atol=1e-7)
        assert abs(res.dispersion - float(ref[f"{name}_disp"])) < 1e-9
        assert int(res.flags.sum()) == int(ref[f"{name}_flags"])
        for meth in ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G-ML", "G-LC", "G_PF"):
            assert np.allclose(o[meth]["I"], ref[f"{name}_{meth}_I"], rtol=1e-6, atol=1e-6), meth
            assert np.allclose(o[meth]["sigma"], ref[f"{name}_{meth}_sigma"], rtol=1e-6, atol=1e-6), meth


# ----------------------------------------------------------------------------- analytic frame with a stop + arm
N = 512
CY, CX = 260.3, 240.6


def _frame(shadow=0.5):
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - CY, xx - CX)
    stop = (np.hypot(yy - CY - 3, xx - CX + 5) < 30) | ((np.abs(yy - CY) < 9) & (xx < CX))
    d = ndi.distance_transform_edt(~stop)
    b = 50 * (np.clip(r, 20, None) / 40.0) ** -2 + 2
    T = np.clip(1 - shadow * np.exp(-(d - 1) / 1.0) * (np.abs(yy - CY) < 30) * (xx < CX - 20), 0, 1)
    return stop, d, b * T, r


def test_penumbra_removes_shadow_bias():
    stop, d, bt, r = _frame(0.5)
    K = np.random.default_rng(0).poisson(bt)
    out = {}
    for ver in ("v0", "v1"):
        res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), version=ver)
        s = ~stop & np.isfinite(res.bg) & (d >= 1) & (d < 2) & (r < 200)
        out[ver] = np.mean(res.bg[s] - bt[s]) / np.mean(bt[s])
    assert out["v0"] > 0.1 and abs(out["v1"]) < 0.05


def test_v1_unbiased_without_shadow():
    stop, d, bt, r = _frame(0.0)
    rng = np.random.default_rng(1)
    plan = BG.make_plan(bt.shape, stop, BG.Geometry((CY, CX)))
    acc = []
    for _ in range(25):
        acc.append(BG.estimate_background(rng.poisson(bt), stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), plan=plan).bg - bt)
    A = np.mean(acc, 0)
    for lo, hi in [(3, 8), (8, 30)]:
        s = ~stop & (d >= lo) & (d < hi) & (r < 230)
        assert abs(A[s].mean() / bt[s].mean()) < 0.006
    for lo, hi in [(30, 100), (100, 230)]:
        s = ~stop & (r >= lo) & (r < hi)
        assert abs(A[s].mean() / bt[s].mean()) < 0.003


def test_auto_v1_switch_follows_expected_error():
    """No local structure: AUTO v1 keeps G (D <= 0 in most zones); a strong local bump field: it switches."""
    stop, d, bt, r = _frame(0.0)
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:N, 0:N]
    spots = np.array([[CY + 120.2, CX + 30.4], [CY - 90.7, CX + 100.1], [CY + 60.3, CX + 150.6]])
    for bumpy, expect in ((False, False), (True, True)):
        b = bt.copy()
        if bumpy:     # local structure smooth at the annulus scale (correlation ~12 px) that the global model cannot follow
            from scipy import ndimage as ndi
            fld = ndi.gaussian_filter(np.random.default_rng(9).normal(size=b.shape), 12.0)
            b = b * np.clip(1 + 0.3 * fld / fld.std(), 0.05, None)
        K = rng.poisson(b)
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), exclude=ex)
        use, info = BG.integrate.local_switch_mse(K.astype(float), res, spots, 3.0, 5.0, 9.0, 9.0)
        frac = np.mean(info["use_zone"][info["n"] >= 10])
        assert (frac > 0.5) == expect, (bumpy, info["D"], info["n"])


def test_lam_min_and_pooled_ml():
    from bgsub import spotlaw as SL
    law = BG.CountLaw.apollo(8, 0.35)
    yy, xx = np.mgrid[-3:4, -3:4]; box = np.hypot(yy, xx) <= 3
    f = np.exp(-0.5 * (yy ** 2 + xx ** 2) / 1.2 ** 2)[box]; f = f / f.sum()
    lm = [SL.lam_min(np.full((1, box.sum()), m0), f[None], np.ones((1, box.sum()), bool), law)[0] for m0 in (0.15, 0.6, 2.4)]
    pb = [l_ / (box.sum() * m0) for l_, m0 in zip(lm, (0.15, 0.6, 2.4))]       # the limit in P/B units falls with dose
    assert pb[0] > pb[1] > pb[2] > 0 and 2.0 < pb[0] < 4.5 and 0.5 < pb[1] < 1.2
    # pooled ML of 10 equivalents is close to the truth at mu0 0.6 e, P/B 3
    rng = np.random.default_rng(4)
    from bgsub.vendor import binning as B
    mu0 = 0.6; lam = 3 * box.sum() * mu0
    Ks = [B.quantise(rng.poisson(mu0 + lam * f), 8, 0.35, rng=rng) for _ in range(10)]
    l, s = SL.pooled_ml(Ks, [np.full(box.sum(), mu0)] * 10, [f] * 10, law)
    assert abs(l - lam) < 4 * s


def test_v1_numba_matches_numpy():
    stop, d, bt, r = _frame(0.5)
    K = np.random.default_rng(5).poisson(bt)
    a = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), use_numba=True)
    c = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), use_numba=False)
    ok = ~stop & np.isfinite(a.bg)
    assert np.array_equal(a.flags, c.flags)
    assert np.allclose(a.bg[ok], c.bg[ok], rtol=1e-3, atol=1e-4)
