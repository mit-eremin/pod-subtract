"""Tests of bgsub v2 (design note 2026-11-05_bgsub_algorithm_v2.md): v1 regression, re-centring, ring refinement,
continuous local weight, test-inversion intervals."""
import sys
from pathlib import Path

import numpy as np

import bgsub as BG
from bgsub import _kernels as KN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def test_v1_regression():
    """version='v1' reproduces the frozen v1 outputs (bgsub code at the v1 state, 092ece9)."""
    import _refframes as REFM
    ref = np.load(ROOT / "tests" / "data" / "bgsub_v1_reference.npz")
    out = REFM.run_v1("v1")
    for k, v in out.items():
        assert np.allclose(v, ref[k], rtol=1e-6, atol=1e-6, equal_nan=True), k


N = 384
CY, CX = 200.3, 180.6


def _frame(ring_amp=0.0, ring_w=1.0):
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - CY, xx - CX)
    stop = (r < 24) | ((np.abs(yy - CY) < 6) & (xx < CX))
    b = 40 * (np.clip(r, 15, None) / 30.0) ** -2 + 3 + ring_amp * np.exp(-0.5 * ((r - 90) / ring_w) ** 2)
    return stop, b, r


def _sharp_frame():
    """A 2k-like geometry in miniature: the fine spline step is set to 8 px (as at N = 2048), the ring is 1.2 px wide."""
    return _frame(ring_amp=12.0, ring_w=1.2)


def test_ring_refinement_follows_sub_knot_ring():
    stop, b, r = _sharp_frame()
    rng = np.random.default_rng(0)
    err = {}
    for ver in ("v1", "v2"):
        acc = []
        for _ in range(4):
            K = rng.poisson(b)
            res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), version=ver, h_r=8.0)
            s = ~stop & (np.abs(r - 90) < 1.5)
            acc.append(np.mean(res.bg[s] - b[s]) / np.mean(b[s]))
        err[ver] = np.mean(acc)
    assert abs(err["v1"]) > 0.08 and abs(err["v2"]) < 0.03, err


def test_ring_refinement_unbiased_on_smooth_frame():
    stop, b, r = _frame()
    rng = np.random.default_rng(1)
    plan = BG.make_plan(b.shape, stop, BG.Geometry((CY, CX)), version="v2")
    acc = []
    for _ in range(20):
        acc.append(BG.estimate_background(rng.poisson(b), stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), plan=plan).bg - b)
    A = np.mean(acc, 0)
    for lo, hi in [(30, 80), (80, 150), (150, 260)]:
        s = ~stop & (r >= lo) & (r < hi)
        assert abs(A[s].mean() / b[s].mean()) < 0.003


def test_recentring_recovers_gross_centre_error():
    stop, b, r = _frame()
    K = np.random.default_rng(2).poisson(b)
    res = BG.estimate_background(K, stop, BG.Geometry((CY + 5.0, CX - 7.0)), BG.CountLaw.poisson(), version="v2")
    assert res.centre_info["recentred"]
    cy, cx = res.plan.geometry.centre
    assert np.hypot(cy - CY, cx - CX) < 1.0
    res1 = BG.estimate_background(K, stop, BG.Geometry((CY + 0.3, CX - 0.2)), BG.CountLaw.poisson(), version="v2")
    assert not res1.centre_info["recentred"]


def test_ring_kernels_numba_numpy():
    stop, b, r = _frame(ring_amp=5.0)
    K = np.random.default_rng(3).poisson(b).astype(float)
    bg = (b * 0.98).astype(np.float32)
    used = ~stop
    A = KN._ring_acc_np(K, bg, used, CY + 0.4, CX - 0.3, 0.0, 300)
    if KN.HAVE_NUMBA:
        B = KN.ring_acc(K, bg, used, CY + 0.4, CX - 0.3, 0.0, 300)
        assert np.allclose(A, B)
        c = np.linspace(-1, 1, 300)
        assert np.allclose(KN.ring_apply(bg, c, CY, CX, 0.0, True), KN.ring_apply(bg, c, CY, CX, 0.0, False), atol=1e-5)


def test_local_weight_small_on_clean_large_on_local_structure():
    from scipy import ndimage as ndi
    stop, b, r = _frame()
    spots = np.array([[CY + 100.2, CX + 30.4], [CY - 90.7, CX + 100.1], [CY + 60.3, CX + 120.6], [CY - 40.2, CX - 120.3]])
    rng = np.random.default_rng(4)
    ws = {}
    for bumpy in (False, True):
        bb = b.copy()
        if bumpy:
            fld = ndi.gaussian_filter(np.random.default_rng(9).normal(size=b.shape), 12.0)
            bb = bb * np.clip(1 + 0.3 * fld / fld.std(), 0.05, None)
        K = rng.poisson(bb)
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), exclude=ex, version="v2")
        w, info = BG.integrate.local_weight(K.astype(float), res, spots, 3.0, 5.0, 9.0, 9.0)
        ws[bumpy] = w
    assert ws[False].mean() < 0.3 and ws[True].mean() > 0.6, ws


def test_v2_intervals_cover():
    """Poisson spots on a flat background: the inverted 68.3 % intervals of the summation and the ML cover ~68 %, and
    the pull with sigma at the true lambda has mean ~0 and SD ~1."""
    n = 64; yy, xx = np.mgrid[0:n, 0:n]
    cov = {"G": [], "G-ML": []}; pulls = {"G": [], "G-ML": []}
    rng = np.random.default_rng(5)
    spots = np.array([[16.3, 16.6], [16.2, 47.4], [47.7, 16.1], [47.5, 47.8]])
    lam = 40.0
    prof = sum(np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / 1.0) for y, x in spots) / (2 * np.pi)
    m = np.zeros((n, n), bool)
    for _ in range(60):
        K = rng.poisson(2.0 + lam * prof)
        ex = BG.disc_mask(K.shape, spots, 9.0)
        res = BG.estimate_background(K, m, BG.Geometry((31.5, 31.5)), BG.CountLaw.poisson(), exclude=ex, version="v2", recentre=False)
        o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(4, 1.0), methods=("G",), lam_eval=np.full(4, lam * 0.989))
        for mth in ("G", "G-ML"):
            d = o[mth]; tr = lam * 0.989
            cov[mth] += list((d["ci_lo"] <= tr) & (tr <= d["ci_hi"]))
            pulls[mth] += list((d["I"] - tr) / d["sigma_at"])
    for mth in cov:
        assert 0.58 < np.mean(cov[mth]) < 0.78, (mth, np.mean(cov[mth]))
        assert abs(np.mean(pulls[mth])) < 0.25 and 0.8 < np.std(pulls[mth]) < 1.25, (mth, np.mean(pulls[mth]), np.std(pulls[mth]))
