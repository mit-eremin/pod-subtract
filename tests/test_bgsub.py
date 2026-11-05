"""Unit tests of bgsub v0 on analytic cases (design note 2026-11-04_bgsub_algorithm_v0.md); pinned to version="v0"
(the v1 default is tested in test_bgsub_v1.py)."""
import numpy as np
import pytest

from bgsub import CountLaw, Geometry, estimate_background, integrate_spots, make_plan, disc_mask
from bgsub import _kernels as KN

N = 256
CY, CX = 131.3, 122.6


def _grid():
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - CY, xx - CX); phi = np.arctan2(yy - CY, xx - CX)
    return r, phi


def _truth(r, phi):
    # steep halo + ring + smooth m = 1 and m = 2 structure (all inside the model span)
    rr = np.clip(r, 12, None)
    return 3.0 * (rr / 30.0) ** -2.0 + 0.4 + 0.3 * np.exp(-0.5 * ((r - 70) / 8.0) ** 2) + 0.05 * (r / 100) * np.cos(phi - 0.7) \
        + 0.03 * np.sin(2 * phi)


def _mask(r, phi):
    return (r < 14) | ((np.abs(phi - np.pi) < 0.15) & (r < 60))      # stop + arm


def test_noiseless_exact_in_span():
    r, phi = _grid(); b = _truth(r, phi); m = _mask(r, phi)
    res = estimate_background(b, m, Geometry((CY, CX)), CountLaw.poisson(), flagging=False, lam_rel=1e-8, version="v0")
    ok = ~m & (r < N / 2 - 2)
    rel = np.abs(res.bg[ok] - b[ok]) / b[ok]
    assert np.median(rel) < 1e-3 and np.percentile(rel, 99) < 5e-3


def test_unbiased_poisson_over_frames():
    rng = np.random.default_rng(3)
    r, phi = _grid(); b = 0.3 * _truth(r, phi); m = _mask(r, phi)
    plan = make_plan(b.shape, m, Geometry((CY, CX)), version="v0")
    acc = []; disp = []
    for _ in range(30):
        res = estimate_background(rng.poisson(b), m, Geometry((CY, CX)), CountLaw.poisson(), plan=plan)
        acc.append(res.bg - b); disp.append(res.dispersion)
    d = np.mean(acc, 0); ok = ~m & (r < N / 2 - 2)
    for lo, hi in [(14, 40), (40, 80), (80, 126)]:
        s = ok & (r >= lo) & (r < hi)
        assert abs(d[s].mean() / b[s].mean()) < 0.006
    assert 0.9 < np.mean(disp) < 1.1


def test_declared_offset():
    rng = np.random.default_rng(1)
    r, phi = _grid(); b = _truth(r, phi); m = _mask(r, phi)
    K = rng.poisson(b)
    law = CountLaw.poisson()
    r0 = estimate_background(K, m, Geometry((CY, CX)), law, version="v0")
    r1 = estimate_background(K + 1, m, Geometry((CY, CX)), law.with_offset(1), version="v0")
    r2 = estimate_background(K + 1, m, Geometry((CY, CX)), law, version="v0")
    ok = ~m & (r < 120)
    assert np.allclose(r0.bg[ok], r1.bg[ok], atol=1e-9)
    assert np.mean(r2.bg[ok] - r0.bg[ok]) > 0.9              # an undeclared pedestal is NOT absorbed silently
    sp = np.array([[CY + 40, CX + 10]])
    i0 = integrate_spots(K, r0, sp, 3.0); i1 = integrate_spots(K + 1, r1, sp, 3.0)
    assert np.allclose(i0["G"]["I"], i1["G"]["I"]) and np.allclose(i0["G"]["sigma"], i1["G"]["sigma"])


def test_spot_recovery_and_flat_curvature_bias():
    r, phi = _grid(); b = 20 * _truth(r, phi); m = _mask(r, phi)
    cy, cx = CY + 22.0, CX + 0.0           # in the steep halo, r = 22
    spots = np.array([[cy, cx]])
    yy, xx = np.mgrid[0:N, 0:N]
    lam = 500.0 * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 0.8 ** 2) / (2 * np.pi * 0.8 ** 2)
    K = b + lam
    ex = disc_mask(K.shape, spots, 9.0)
    res = estimate_background(K, m, Geometry((CY, CX)), CountLaw.poisson(), exclude=ex, flagging=False, lam_rel=1e-8, version="v0")
    out = integrate_spots(K, res, spots, 3.0, oracle=b)
    box = np.hypot(yy - cy, xx - cx) <= 3.0
    lam_box = lam[box].sum()
    assert abs(out["G"]["I"][0] / lam_box - 1) < 0.01
    assert abs(out["O"]["I"][0] / lam_box - 1) < 1e-9
    # flat annulus: predicted bias (1/4) lap(b) (<rho^2>_ann - <rho^2>_box), measured numerically on b
    beta_flat = (out["FLAT"]["B_box"][0] - b[box].sum()) / b[box].sum()
    ann = (np.hypot(yy - cy, xx - cx) > 5) & (np.hypot(yy - cy, xx - cx) <= 9)
    ann &= ~m
    pred = (b[ann].mean() - b[box].mean()) / b[box].mean()
    assert abs(beta_flat - pred) < 1e-9 and beta_flat > 0.01      # flat overestimates in the convex halo
    beta_G = (out["G"]["B_box"][0] - b[box].sum()) / b[box].sum()
    assert abs(beta_G) < 0.1 * beta_flat


def test_zingers_flagged_without_bias():
    rng = np.random.default_rng(5)
    r, phi = _grid(); b = 0.5 * _truth(r, phi); m = _mask(r, phi)
    K = rng.poisson(b).astype(float)
    clean = estimate_background(K, m, Geometry((CY, CX)), CountLaw.poisson(), version="v0")
    Kz = K.copy(); iy = rng.integers(0, N, 40); ix = rng.integers(0, N, 40)
    Kz[iy, ix] += 500
    res = estimate_background(Kz, m, Geometry((CY, CX)), CountLaw.poisson(), version="v0")
    ok = ~m & (r < 120)
    assert res.flags[iy, ix][~m[iy, ix]].all()
    assert np.max(np.abs(res.bg[ok] - clean.bg[ok]) / b[ok]) < 0.08
    assert abs(np.mean(res.bg[ok] - clean.bg[ok]) / np.mean(b[ok])) < 2e-3


def test_apollo_law():
    law = CountLaw.apollo(8, 0.35)
    assert law.epc == 4.0
    b = np.array([1e-4, 0.05, 0.4, 2.0, 20.0])
    g = law.g(b)
    assert abs(g[0] / b[0] - 1) < 0.01 and g[-1] / b[-1] < 0.3          # Bernoulli limit -> b/m + 1/12
    k1 = law.kstar(b, 1e-6, 1)
    assert np.all(np.diff(k1) >= 0)
    p, T = law.tail_mass(0.05, float(law.kstar(np.array([0.05]), 1e-6, 1)[0]))
    assert p <= 1e-6 and T / 0.05 < 1e-4
    assert CountLaw.apollo(4, 0.35).epc == 1.0 and np.allclose(CountLaw.apollo(4).g(b), b)


@pytest.mark.skipif(not KN.HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy():
    rng = np.random.default_rng(2)
    r, phi = _grid(); b = _truth(r, phi); m = _mask(r, phi)
    K = rng.poisson(b); K[50, 60] += 300
    a = estimate_background(K, m, Geometry((CY, CX)), CountLaw.poisson(), use_numba=True, version="v0")
    c = estimate_background(K, m, Geometry((CY, CX)), CountLaw.poisson(), use_numba=False, version="v0")
    ok = ~m & (r < 120)
    assert np.array_equal(a.flags, c.flags)
    assert np.allclose(a.bg[ok], c.bg[ok], rtol=1e-5, atol=1e-6)


def test_box_variance_matches_replicates():
    rng = np.random.default_rng(7)
    r, phi = _grid(); b = _truth(r, phi); m = _mask(r, phi)
    plan = make_plan(b.shape, m, Geometry((CY, CX)), version="v0")
    sp = np.array([[CY + 30, CX + 5], [CY - 60, CX + 40]])
    ex = disc_mask(b.shape, sp, 9.0)
    Bs, Vs = [], []
    for _ in range(60):
        res = estimate_background(rng.poisson(b), m, Geometry((CY, CX)), CountLaw.poisson(), exclude=ex, plan=plan)
        o = integrate_spots(rng.poisson(b), res, sp, 3.0, methods=("G",))
        Bs.append(o["G"]["B_box"]); Vs.append(o["G"]["var_B"])
    ratio = np.var(np.array(Bs), 0) / np.mean(np.array(Vs), 0)
    assert np.all((ratio > 0.6) & (ratio < 1.6))


def test_profile_fit_and_references_on_flat_field():
    rng = np.random.default_rng(11)
    b = np.full((N, N), 2.0); m = np.zeros((N, N), bool)
    spots = np.array([[60.3, 70.6], [180.2, 150.9], [100.0, 200.4]])
    yy, xx = np.mgrid[0:N, 0:N]
    lam = np.zeros((N, N)); sig = 1.0
    for cy, cx in spots:
        lam += 300 * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sig ** 2) / (2 * np.pi * sig ** 2)
    ex = disc_mask(b.shape, spots, 9.0)
    res = estimate_background(b + lam, m, Geometry((128.0, 128.0)), CountLaw.poisson(), exclude=ex, flagging=False, version="v0")
    out = integrate_spots(b + lam, res, spots, 3.0, sigmas=np.full(3, sig))
    lam_box = np.array([lam[np.hypot(yy - cy, xx - cx) <= 3].sum() for cy, cx in spots])
    for meth in ("G", "GL", "FLAT", "PLANE"):
        assert np.allclose(out[meth]["I"] / lam_box, 1, atol=0.02), meth
    # the test spot is sampled at pixel centres, the fit profile is pixel-integrated: ~3 % profile mismatch
    assert np.allclose(out["G_PF"]["I"] / lam_box, 1, atol=0.05)
    # robust (Huber, Poisson-consistent) is unbiased on a Poisson constant field
    from bgsub import reference as REF
    vals = [REF.robust(rng.poisson(2.0, 200).astype(float), CountLaw.poisson())[0] for _ in range(300)]
    assert abs(np.mean(vals) / 2.0 - 1) < 0.01


def test_apollo_bin8_unbiased_low_counts():
    from bgsub.vendor import binning as B
    rng = np.random.default_rng(4)
    r, phi = _grid(); mu = 1.2 * _truth(r, phi); m = _mask(r, phi)        # electrons per px; b ~ 0.05-1 counts
    law = CountLaw.apollo(8, 0.35)
    btrue = np.interp(mu, law._mu, law.b_grid)
    plan = make_plan(mu.shape, m, Geometry((CY, CX)), version="v0")
    acc = []
    for _ in range(30):
        K = B.quantise(rng.poisson(mu), 8, 0.35, rng=rng)
        acc.append(estimate_background(K, m, Geometry((CY, CX)), law, plan=plan).bg - btrue)
    d = np.mean(acc, 0); ok = ~m & (r < N / 2 - 2)
    for lo, hi in [(14, 40), (40, 80), (80, 126)]:
        s = ok & (r >= lo) & (r < hi)
        assert abs(d[s].mean() / btrue[s].mean()) < 0.01


def test_counting_law_aware_intensity_bin8():
    """Summation loses intensity to the bin-8 rounding; X-LC and X-ML recover lambda_box (Amendment 3 a)."""
    from bgsub.vendor import binning as B
    from bgsub.validation import peaks as PK
    rng = np.random.default_rng(21)
    law = CountLaw.apollo(8, 0.35)
    n = 256; mu = np.full((n, n), 0.8)                           # e per px: b ~ 0.07 stored counts
    m = np.zeros((n, n), bool)
    spots = np.array([[cy, cx] for cy in range(20, 240, 24) for cx in range(20, 240, 24)], float) + 0.3
    sig = 0.8; lam_box_true = 40.0
    ratios = {"G": [], "G-LC": [], "G-ML": []}; pulls = {"G-LC": [], "G-ML": []}
    for rep in range(6):
        N0 = rng.poisson(mu)
        N = N0.copy()
        for cy, cx in spots:
            y0, x0, lam = PK.spot_pixels(cy, cx, sig, 1.0, 5)
            rp = PK.patch_radius(y0, x0, 11, cy, cx)
            lam = lam * lam_box_true / lam[rp <= 3.0].sum()
            N[y0:y0 + 11, x0:x0 + 11] += rng.poisson(lam)
        K = B.quantise(N, 8, 0.35, rng=rng)
        ex = disc_mask(K.shape, spots, 9.0)
        res = estimate_background(K, m, Geometry((128.0, 128.0)), law, exclude=ex, version="v0")
        out = integrate_spots(K, res, spots, 3.0, sigmas=np.full(len(spots), sig))
        for k in ratios:
            ratios[k] += list(out[k]["I"] / lam_box_true)
        for k in pulls:
            pulls[k] += list((out[k]["I"] - lam_box_true) / out[k]["sigma"])
    print({k: (np.mean(v), np.std(pulls.get(k, [np.nan]))) for k, v in ratios.items()})
    for k in ("G-LC", "G-ML"):
        assert abs(np.mean(ratios[k]) - 1) < 0.05, (k, np.mean(ratios[k]))
        assert 0.8 < np.std(pulls[k]) < 1.25, (k, np.std(pulls[k]))
