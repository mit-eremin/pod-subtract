"""Tests of bgsub v2.1 (design note 2026-11-05_bgsub_algorithm_v2_1.md): v2 regression (bit for bit), elliptical
ring term, fast probes, gated local weight with the spill correction, v2.1 intervals."""
import sys
from pathlib import Path

import numpy as np

import bgsub as BG  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def test_v2_regression():
    """version='v2' reproduces the frozen v2 outputs (bgsub code at the v2 state, 5e4a90c), bit for bit."""
    import _refframes as REFM
    ref = np.load(ROOT / "tests" / "data" / "bgsub_v2_reference.npz")
    out = REFM.run_v2("v2")
    assert set(out) == set(ref.files) - {"commit"}
    for k, v in out.items():
        assert np.array_equal(np.asarray(v), ref[k], equal_nan=True), k


N = 384
CY, CX = 200.3, 180.6


def _frame(ring_amp=0.0, ring_w=1.0, e=0.0):
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - CY, xx - CX); phi = np.arctan2(yy - CY, xx - CX)
    stop = (r < 24) | ((np.abs(yy - CY) < 6) & (xx < CX))
    re = r * (1 + e * np.cos(2 * (phi - 0.4)))
    b = 40 * (np.clip(r, 15, None) / 30.0) ** -2 + 3 + ring_amp * np.exp(-0.5 * ((re - 90) / ring_w) ** 2)
    return stop, b, r


def test_elliptical_ring_term_follows_elliptical_sharp_ring():
    """A 1.2-px ring with 0.8 % ellipticity (the d2 columns see it through the smooth profile): v2's radial correction
    leaves an m = 2 residual, v2.1's elliptical rings do not."""
    stop, b, r = _frame(ring_amp=12.0, ring_w=1.2, e=0.008)
    disp = {}
    for ver in ("v2", "v2.1"):
        rng = np.random.default_rng(10)
        acc = []
        for _ in range(4):
            K = rng.poisson(b)
            res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), version=ver, h_r=8.0)
            s = ~stop & (np.abs(r - 90) < 3)
            acc.append(np.mean((K[s] - res.bg[s]) ** 2 / b[s]))
        disp[ver] = np.mean(acc)
    assert disp["v2"] > 1.25 and disp["v2.1"] < 1.15, disp


def test_v21_ring_term_unbiased_on_smooth_frame():
    stop, b, r = _frame()
    rng = np.random.default_rng(1)
    plan = BG.make_plan(b.shape, stop, BG.Geometry((CY, CX)), version="v2.1")
    acc = [BG.estimate_background(rng.poisson(b), stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), plan=plan).bg - b for _ in range(20)]
    A = np.mean(acc, 0)
    for lo, hi in [(30, 80), (80, 150), (150, 260)]:
        s = ~stop & (r >= lo) & (r < hi)
        assert abs(A[s].mean() / b[s].mean()) < 0.003


def test_kernels_numba_numpy_v21():
    from bgsub import _kernels as KN
    stop, b, r = _frame(ring_amp=5.0)
    K = np.random.default_rng(3).poisson(b).astype(float)
    bg = (b * 0.98).astype(np.float32)
    A = KN._ring_acc_ell_np(K, bg, ~stop, CY + 0.4, CX - 0.3, 0.004, -0.002, 0.0, 300)
    c = np.linspace(-1, 1, 300)
    oy, ox = np.mgrid[-9:10, -9:10]; rr = np.hypot(oy, ox)
    sb, sa = rr <= 3, (rr > 5) & (rr <= 9)
    CYs = np.arange(20, 360, 7); CXs = np.arange(20, 360, 7)[::-1]
    p_np = KN._probe_sums_np(K - bg, stop, CYs, CXs, oy[sb], ox[sb], oy[sa], ox[sa])
    if KN.HAVE_NUMBA:
        assert np.allclose(A, KN.ring_acc_ell(K, bg, ~stop, CY + 0.4, CX - 0.3, 0.004, -0.002, 0.0, 300))
        assert np.allclose(KN.ring_apply_ell(bg, c, CY, CX, 0.004, -0.002, 0.0, True), KN.ring_apply_ell(bg, c, CY, CX, 0.004, -0.002, 0.0, False), atol=1e-5)
        p_nb = KN.probe_sums(K - bg, stop, CYs, CXs, oy[sb], ox[sb], oy[sa], ox[sa])
        assert np.array_equal(p_np[0], p_nb[0])
        assert np.allclose(p_np[1][p_np[0]], p_nb[1][p_nb[0]]) and np.allclose(p_np[2][p_np[0]], p_nb[2][p_nb[0]])


def test_fast_probes_equal_v2_probes():
    from bgsub import integrate as IG
    stop, b, r = _frame()
    spots = np.array([[CY + 100.2, CX + 30.4], [CY - 90.7, CX + 100.1], [CY + 60.3, CX + 120.6], [CY - 40.2, CX - 120.3]])
    K = np.random.default_rng(4).poisson(b).astype(float)
    res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), exclude=BG.disc_mask(K.shape, spots, 9.0), version="v2.1")
    a = IG.lw_probes(K, res, spots, 3.0, 5.0, 9.0, 9.0); f = IG.lw_probes_fast(K, res, spots, 3.0, 5.0, 9.0, 9.0)
    assert np.array_equal(a["CY"], f["CY"]) and np.array_equal(a["CX"], f["CX"])
    assert np.allclose(a["La"], f["La"], atol=1e-12) and np.allclose(a["Lb"], f["Lb"], atol=1e-12)


def test_gated_local_weight_off_on_clean_on_with_structure():
    from scipy import ndimage as ndi
    from bgsub import integrate as IG
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
        res = BG.estimate_background(K, stop, BG.Geometry((CY, CX)), BG.CountLaw.poisson(), exclude=BG.disc_mask(K.shape, spots, 9.0), version="v2.1")
        ws[bumpy] = IG.local_weight_v21(K.astype(float), res, spots, 3.0, 5.0, 9.0, 9.0)["w"]
    assert ws[False].mean() < 0.15 and ws[True].mean() > 0.6, ws


def test_spill_correction_removes_spot_light_from_annulus():
    """Wide bright spots on a flat background: the local correction without the spill term is biased high by the
    spot's tail in the annulus; v2.1's GL (w = 1, spill-corrected) is not."""
    n = 96; yy, xx = np.mgrid[0:n, 0:n]
    spots = np.array([[24.3, 24.6], [24.2, 71.4], [71.7, 24.1], [71.5, 71.8]])
    sig = 2.2; lam = 4000.0; b0 = 5.0
    prof = sum(np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / sig ** 2) for y, x in spots) / (2 * np.pi * sig ** 2)
    m = np.zeros((n, n), bool)
    rng = np.random.default_rng(6)
    e21, e2 = [], []
    for _ in range(30):
        K = rng.poisson(b0 + lam * prof)
        ex = BG.disc_mask(K.shape, spots, 9.0)
        for ver, acc in (("v2.1", e21), ("v2", e2)):
            res = BG.estimate_background(K, m, BG.Geometry((47.5, 47.5)), BG.CountLaw.poisson(), exclude=ex, version=ver, recentre=False)
            o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(4, sig), methods=("G", "GL"))
            acc += list(o["GL"]["B_box"] / (b0 * o["GL"]["n_box"]) - 1)
    assert np.mean(e2) > 0.1 and abs(np.mean(e21)) < 0.02, (np.mean(e2), np.mean(e21))


def test_v21_intervals_cover_on_dead_time_branch():
    """Poisson spots (pixel-integrated Gaussians) with paralyzable dead time, the brightest past the maximum of
    L(lambda): the v2.1 lambda-interval of the summation and the ML interval cover the true box intensity at ~68 %."""
    from bgsub import integrate as IG
    n = 64; yy, xx = np.mgrid[0:n, 0:n]
    spots = np.array([[16.3, 16.6], [16.2, 47.4], [47.7, 16.1], [47.5, 47.8]])
    tau = 0.004
    law = BG.CountLaw.poisson(); law.tau = tau
    m = np.zeros((n, n), bool)
    rng = np.random.default_rng(5)
    profs = [IG._profile(y, x, 1.0, yy.astype(float), xx.astype(float)) for y, x in spots]
    bf = np.array([p[np.hypot(yy - y, xx - x) <= 3.0].sum() for p, (y, x) in zip(profs, spots)])
    cov = {"G": [], "G-ML": []}
    for lam in (40.0, 400.0, 3000.0):
        prof = sum(profs)
        for _ in range(30):
            mu = 2.0 + lam * prof
            K = rng.poisson(mu * np.exp(-mu * tau))
            ex = BG.disc_mask(K.shape, spots, 9.0)
            res = BG.estimate_background(K, m, BG.Geometry((31.5, 31.5)), law, exclude=ex, version="v2.1", recentre=False)
            o = BG.integrate_spots(K, res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(4, 1.0), methods=("G",))
            tr = lam * bf
            cov["G"] += list((o["G"]["ci_lam_lo"] <= tr) & (tr <= o["G"]["ci_lam_hi"]))
            cov["G-ML"] += list((o["G-ML"]["ci_lo"] <= tr) & (tr <= o["G-ML"]["ci_hi"]))
    for k, v in cov.items():
        assert 0.6 < np.mean(v) < 0.78, (k, np.mean(v))
