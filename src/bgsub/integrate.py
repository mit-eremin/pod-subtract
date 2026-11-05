# MOVED from pod/background/src/podbg/bgsub/integrate.py at background commit f1e93f5 (file last changed in fd2d10f; sha256 82260ff01bcb7fd1872c265580ad12513be4e318f1af93201afc8efb233883f3).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""bgsub integration: summation and counting-law-aware intensities with the bgsub background, plus the reference local
methods (design note Sec. 1.3.2, 1.4, 1.7 and Amendments 1-3). Works on stored values minus the DECLARED offset; never
on modified images.

Box: pixel centres within r_box of the spot centre (statically valid). Annulus: r_in < rho <= r_out, statically valid,
outside the discs (radius r_excl) of the OTHER spots.  For G the spots' discs must have been passed to
estimate_background as `exclude` (an integrator knows its predicted positions; `disc_mask`).

Methods (summation, I in electrons with expectation lambda_eff): O (oracle, needs `oracle`), G, GL, AUTO, FLAT, PLANE,
ROBUST.  With `sigmas` (spot Gaussian sigma per spot, px): <X>-LC for every summation method (lambda_box estimate by the
law-expected excess, Amendment 3 a), G-ML and AUTO-ML (profile likelihood of the stored counts), and G_PF (linear
profile fit, Amendment 1 a).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import reference as REF
from . import spotlaw as SL

AUTO_P = 0.01


def disc_mask(shape, spots, radius):
    """True inside discs of `radius` about each (cy, cx)."""
    m = np.zeros(shape, bool)
    R = int(np.ceil(radius)) + 1
    for cy, cx in np.asarray(spots, float).reshape(-1, 2):
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue
        iy, ix = int(round(cy)), int(round(cx))
        y0, y1 = max(iy - R, 0), min(iy + R + 1, shape[0]); x0, x1 = max(ix - R, 0), min(ix + R + 1, shape[1])
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= np.hypot(yy - cy, xx - cx) <= radius
    return m


def _vk(law, Bp, Kp, disp):
    """First-order unbiased estimate of sum_box Var(K_p) (Sec. 1.3.2), floored at half the background term."""
    gB = law.g(np.maximum(Bp, 0.0))
    v = np.sum(gB + law.dg(np.maximum(Bp, 0.0)) * (Kp - Bp))
    return disp * max(float(v), 0.5 * float(np.sum(gB)))


def _profile(cy, cx, sigma, ys, xs):
    from scipy.special import erf
    sq = np.sqrt(2.0) * sigma
    py = 0.5 * (erf((ys + 0.5 - cy) / sq) - erf((ys - 0.5 - cy) / sq))
    px = 0.5 * (erf((xs + 0.5 - cx) / sq) - erf((xs - 0.5 - cx) / sq))
    return py * px


def local_structure_test(K, res, spots, r_in, r_out, r_excl, stride=None):
    """AUTO switch (Amendment 3 b): chi^2 of the local residual mean(K - G) on probe annuli away from spots, mask and
    flags. Returns (Q, n_probe, p_value)."""
    plan = res.plan
    ny, nx = K.shape
    stride = int(np.ceil(2 * r_out)) if stride is None else stride
    R = int(np.ceil(r_out))
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(oy, ox)
    sel = (rr > r_in) & (rr <= r_out)
    oy, ox = oy[sel], ox[sel]
    cy = np.arange(R + 1, ny - R - 1, stride); cx = np.arange(R + 1, nx - R - 1, stride)
    CY, CX = np.meshgrid(cy, cx, indexing="ij"); CY = CY.ravel(); CX = CX.ravel()
    bad = ~plan.valid | res.flags | ~np.isfinite(res.bg)
    if len(spots):
        bad = bad | disc_mask(K.shape, spots, r_excl + r_out)
    Y = CY[:, None] + oy[None, :]; X = CX[:, None] + ox[None, :]
    ok = ~bad[Y, X].any(axis=1) & ~bad[CY, CX]
    Y, X = Y[ok], X[ok]
    if len(Y) < 5:
        return np.nan, 0, np.nan
    G = res.bg[Y, X].astype(float)
    L = (K[Y, X] - G).mean(axis=1)
    s2 = res.dispersion * res.law.g(np.maximum(G, 0)).sum(axis=1) / G.shape[1] ** 2
    good = s2 > 0
    Q = float(np.sum(L[good] ** 2 / s2[good])); n = int(good.sum())
    return Q, n, float(stats.chi2.sf(Q, n))


AUTO_ZONE_RATIO = 1.5
AUTO_MIN_PROBES = 10
LAM_MIN_CR = 0.45           # information limit (design note v1 Sec. 1.2 / 2.3): CR relative SD of the box intensity


def local_switch_mse(K, res, spots, r_box, r_in, r_out, r_excl, stride=None, zone_ratio=AUTO_ZONE_RATIO,
                     min_probes=AUTO_MIN_PROBES):
    """AUTO v1 (design note v1 Sec. 2.2): per radial zone, GL replaces G iff the estimated expected squared error of the
    box-mean background is smaller with GL:  D = mean_j Lb_j^2 - mean_j (La_j - Lb_j)^2 > 0, Lb / La the mean of
    K - G over a probe's box / annulus (exact: MSE_GL - MSE_G = E(La - Lb)^2 - E Lb^2; no p value, no variance model).
    Returns (use_GL per spot, info dict with the zone edges, D, n per zone)."""
    plan = res.plan
    ny, nx = K.shape
    stride = int(np.ceil(r_out)) if stride is None else int(stride)
    R = int(np.ceil(r_out))
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(oy, ox)
    sb = rr <= r_box; sa = (rr > r_in) & (rr <= r_out)
    by, bx = oy[sb], ox[sb]; ay, ax = oy[sa], ox[sa]
    cy = np.arange(R + 1, ny - R - 1, stride); cx = np.arange(R + 1, nx - R - 1, stride)
    CY, CX = np.meshgrid(cy, cx, indexing="ij"); CY = CY.ravel(); CX = CX.ravel()
    bad = ~plan.valid | res.flags | ~np.isfinite(res.bg)
    if len(spots):
        bad = bad | disc_mask(K.shape, spots, r_excl + r_out)
    YB = CY[:, None] + by[None, :]; XB = CX[:, None] + bx[None, :]
    YA = CY[:, None] + ay[None, :]; XA = CX[:, None] + ax[None, :]
    ok = ~bad[YB, XB].any(1) & ~bad[YA, XA].any(1)
    edges = [plan.r_lo]
    while edges[-1] < plan.r_hi:
        edges.append(edges[-1] * zone_ratio if edges[-1] > 0 else 1.0)
    edges = np.array(edges)
    nz = len(edges) - 1
    D = np.full(nz, np.nan); n = np.zeros(nz, int); mse_g = np.full(nz, np.nan); mse_gl = np.full(nz, np.nan)
    if ok.any():
        G = res.bg.astype(float)
        Lb = (K[YB[ok], XB[ok]] - G[YB[ok], XB[ok]]).mean(1)
        La = (K[YA[ok], XA[ok]] - G[YA[ok], XA[ok]]).mean(1)
        zp = np.clip(np.searchsorted(edges, plan.rmap[CY[ok], CX[ok]], side="right") - 1, 0, nz - 1)
        n = np.bincount(zp, minlength=nz)
        a = np.bincount(zp, Lb ** 2, nz); c = np.bincount(zp, (La - Lb) ** 2, nz)
        with np.errstate(invalid="ignore", divide="ignore"):
            mse_g = a / np.maximum(n, 1); mse_gl = c / np.maximum(n, 1)
            D = np.where(n > 0, mse_g - mse_gl, np.nan)
    use_zone = (n >= min_probes) & (np.nan_to_num(D, nan=-1.0) > 0)
    if len(spots):
        iy = np.clip(np.round(spots[:, 0]).astype(int), 0, ny - 1); ix = np.clip(np.round(spots[:, 1]).astype(int), 0, nx - 1)
        zs = np.clip(np.searchsorted(edges, plan.rmap[iy, ix], side="right") - 1, 0, nz - 1)
        use = use_zone[zs]
    else:
        zs = np.zeros(0, int); use = np.zeros(0, bool)
    return use, {"edges": edges, "D": D, "n": n, "mse_G": mse_g, "mse_GL_plus_sb": mse_gl, "use_zone": use_zone, "spot_zone": zs}


LW_RNB = 12.0              # v2 local weight: probe neighbourhood radius (arc length) in units of r_out
LW_NMIN = 20               # at least this many (nearest) probes
LW_STRIDE = 0.5            # probe stride in units of r_out
LW_MODE = "ring"           # neighbourhood: "disc" (within R_nb) or "ring" (radius-matched, |dr| <= LW_DR r_out, arc <= R_nb)
LW_DR = 2.0


def lw_probes(K, res, spots, r_box, r_in, r_out, r_excl, R_nb=None, stride=None):
    """Spot-free probes of the v2 / v2.1 local weight: centres on a grid of stride LW_STRIDE r_out near the spots, box
    (r_box) and annulus (r_in - r_out) fully valid, unflagged and finite, more than r_excl + r_out from every spot.
    Returns a dict with CY, CX, La, Lb (mean of K - G over annulus / box), rp, pp (radius, azimuth), stride, R_nb."""
    plan = res.plan
    ny, nx = K.shape
    stride = max(1, int(round(LW_STRIDE * r_out))) if stride is None else int(stride)
    R_nb = LW_RNB * r_out if R_nb is None else float(R_nb)
    R = int(np.ceil(r_out))
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(oy, ox)
    sb = rr <= r_box; sa = (rr > r_in) & (rr <= r_out)
    by, bx = oy[sb], ox[sb]; ay, ax = oy[sa], ox[sa]
    empty = {"CY": np.zeros(0, int), "CX": np.zeros(0, int), "La": np.zeros(0), "Lb": np.zeros(0), "rp": np.zeros(0),
             "pp": np.zeros(0), "stride": stride, "R_nb": R_nb, "na": int(len(ay)), "nb": int(len(by))}
    if len(spots) == 0:
        return empty
    cy = np.arange(R + 1, ny - R - 1, stride); cx = np.arange(R + 1, nx - R - 1, stride)
    CY, CX = np.meshgrid(cy, cx, indexing="ij"); CY = CY.ravel(); CX = CX.ravel()
    # keep probe centres near some spot (all others are irrelevant), then the usual validity
    near = np.zeros(len(CY), bool)
    reach = (R_nb + 4 * r_out) ** 2
    for (sy, sx) in spots:
        near |= (CY - sy) ** 2 + (CX - sx) ** 2 <= reach
    CY, CX = CY[near], CX[near]
    bad = ~plan.valid | res.flags | ~np.isfinite(res.bg)
    bad = bad | disc_mask(K.shape, spots, r_excl + r_out)
    keep = ~bad[CY, CX]
    CY, CX = CY[keep], CX[keep]
    if len(CY) == 0:
        return empty
    YB = CY[:, None] + by[None, :]; XB = CX[:, None] + bx[None, :]
    YA = CY[:, None] + ay[None, :]; XA = CX[:, None] + ax[None, :]
    ok = ~bad[YB, XB].any(1) & ~bad[YA, XA].any(1)
    CY, CX, YB, XB, YA, XA = CY[ok], CX[ok], YB[ok], XB[ok], YA[ok], XA[ok]
    if len(CY) == 0:
        return empty
    G = res.bg.astype(float)
    Lb = (K[YB, XB] - G[YB, XB]).mean(1)
    La = (K[YA, XA] - G[YA, XA]).mean(1)
    rp = plan.rmap[CY, CX].astype(float); pp = plan.phimap[CY, CX].astype(float)
    return {"CY": CY, "CX": CX, "La": La, "Lb": Lb, "rp": rp, "pp": pp, "stride": stride, "R_nb": R_nb,
            "na": int(len(ay)), "nb": int(len(by))}


def lw_select(pr, spot, plan, shape, r_out, n_min=LW_NMIN, mode=None):
    """The v2 probe neighbourhood of one spot (bool over the probes of `pr`): radius-matched (|r_j - r_k| <= LW_DR r_out,
    arc <= R_nb), widened x2, x4, x8 until n_min probes, else the n_min nearest ('disc': within R_nb)."""
    ny, nx = shape
    sy, sx = spot
    CY, CX, rp, pp, R_nb = pr["CY"], pr["CX"], pr["rp"], pr["pp"], pr["R_nb"]
    d2 = (CY - sy) ** 2 + (CX - sx) ** 2
    mode = LW_MODE if mode is None else mode
    if mode == "ring":
        # radius-matched neighbourhood: |r_j - r_k| <= LW_DR r_out and arc distance <= R_nb (structure next to the
        # stop and the rings is organised by radius); widened in steps until n_min probes
        iy = int(np.clip(round(sy), 0, ny - 1)); ix = int(np.clip(round(sx), 0, nx - 1))
        rk = float(plan.rmap[iy, ix]); pk = float(plan.phimap[iy, ix])
        dphi = np.abs(np.angle(np.exp(1j * (pp - pk))))
        sel = np.zeros(len(d2), bool)
        for fac in (1.0, 2.0, 4.0, 8.0):
            sel = (np.abs(rp - rk) <= fac * LW_DR * r_out) & (rk * dphi <= fac * R_nb)
            if sel.sum() >= n_min:
                break
    else:
        sel = d2 <= R_nb ** 2
    if sel.sum() < n_min:
        sel = np.zeros(len(d2), bool); sel[np.argsort(d2)[:n_min]] = True
    return sel


def local_weight(K, res, spots, r_box, r_in, r_out, r_excl, R_nb=None, stride=None, n_min=LW_NMIN, mode=None):
    """AUTO v2 (design note v2 Sec. 2.2): per spot, the continuous weight w* = E[La Lb] / E[La^2] of the local residual,
    estimated from the spot-free probes within R_nb (default: radius-matched, |r_j - r_k| <= LW_DR r_out and arc distance <= LW_RNB r_out; at least the n_min nearest).
    Lb / La: mean of K - G over a probe's box / annulus.  The background B = G + w nb La has MSE
    E(delta_b - w delta_a)^2 + w^2 s_a^2, minimised by w* (no variance model: the noises of La and Lb are independent).
    Returns (w per spot clipped to [0, 1], info)."""
    nsp = len(spots)
    w = np.zeros(nsp); npr = np.zeros(nsp, int)
    pr = lw_probes(K, res, spots, r_box, r_in, r_out, r_excl, R_nb, stride)
    if nsp == 0 or len(pr["CY"]) == 0:
        return w, {"n_probe": npr}
    ab = pr["La"] * pr["Lb"]; aa = pr["La"] * pr["La"]
    for k, sp in enumerate(spots):
        sel = lw_select(pr, sp, res.plan, K.shape, r_out, n_min, mode)
        den = aa[sel].sum()
        w[k] = float(np.clip(ab[sel].sum() / den, 0.0, 1.0)) if den > 0 else 0.0
        npr[k] = int(sel.sum())
    return w, {"n_probe": npr, "R_nb": pr["R_nb"], "stride": pr["stride"]}


LW21_Z = 2.0               # v2.1 gate: the local weight is used only where S_ab > LW21_Z SE_block(S_ab)
LW21_BLOCK = 2.0           # v2.1 block size (units of r_out) for the block standard error of the probe sums


def lw_probes_fast(K, res, spots, r_box, r_in, r_out, r_excl, R_nb=None, stride=None):
    """lw_probes with the box / annulus sums in one compiled pass (v2.1; same probes, same pixel sets)."""
    from . import _kernels as KN
    plan = res.plan
    ny, nx = K.shape
    stride = max(1, int(round(LW_STRIDE * r_out))) if stride is None else int(stride)
    R_nb = LW_RNB * r_out if R_nb is None else float(R_nb)
    R = int(np.ceil(r_out))
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(oy, ox)
    sb = rr <= r_box; sa = (rr > r_in) & (rr <= r_out)
    by, bx = oy[sb], ox[sb]; ay, ax = oy[sa], ox[sa]
    empty = {"CY": np.zeros(0, int), "CX": np.zeros(0, int), "La": np.zeros(0), "Lb": np.zeros(0), "rp": np.zeros(0),
             "pp": np.zeros(0), "stride": stride, "R_nb": R_nb, "na": int(len(ay)), "nb": int(len(by))}
    if len(spots) == 0:
        return empty
    cy = np.arange(R + 1, ny - R - 1, stride); cx = np.arange(R + 1, nx - R - 1, stride)
    CY, CX = np.meshgrid(cy, cx, indexing="ij"); CY = CY.ravel(); CX = CX.ravel()
    near = np.zeros(len(CY), bool)
    reach = (R_nb + 4 * r_out) ** 2
    for (sy, sx) in spots:
        near |= (CY - sy) ** 2 + (CX - sx) ** 2 <= reach
    CY, CX = CY[near], CX[near]
    bad = ~plan.valid | res.flags | ~np.isfinite(res.bg)
    bad = bad | disc_mask(K.shape, spots, r_excl + r_out)
    keep = ~bad[CY, CX]
    CY, CX = CY[keep], CX[keep]
    if len(CY) == 0:
        return empty
    Rm = np.where(bad, 0.0, K - np.nan_to_num(res.bg.astype(float)))
    ok, Lb, La = KN.probe_sums(Rm, bad, CY, CX, by, bx, ay, ax, plan.use_numba)
    CY, CX, Lb, La = CY[ok], CX[ok], Lb[ok], La[ok]
    if len(CY) == 0:
        return empty
    rp = plan.rmap[CY, CX].astype(float); pp = plan.phimap[CY, CX].astype(float)
    return {"CY": CY, "CX": CX, "La": La, "Lb": Lb, "rp": rp, "pp": pp, "stride": stride, "R_nb": R_nb,
            "na": int(len(ay)), "nb": int(len(by))}


def _block_se(x, inv, nB):
    """SE of sum(x) from block sums (blocks are independent; probes inside a block overlap)."""
    if nB < 2:
        return np.inf
    T = np.bincount(inv, x, nB); m = np.bincount(inv, minlength=nB)
    return float(np.sqrt(nB / (nB - 1.0) * np.sum((T - m * x.mean()) ** 2)))


def local_weight_v21(K, res, spots, r_box, r_in, r_out, r_excl, n_min=LW_NMIN, z=LW21_Z):
    """AUTO v2.1 (design note v2.1 Sec. 2.1): G2.1 by default; the v2 continuous weight w* = clip(S_ab / S_aa, 0, 1) of
    the local annulus residual only where its signal is significant, T = S_ab / SE_B(S_ab) > z (S_ab = sum_j a_j b_j,
    S_aa = sum_j a_j^2 over the v2 probe neighbourhood N_k, lw_select).  SE_B from block sums over blocks of
    LW21_BLOCK r_out (overlapping probes are not independent).  Returns dict w, T, n, nB per spot."""
    nsp = len(spots)
    out = {"w": np.zeros(nsp), "T": np.zeros(nsp), "n": np.zeros(nsp, int), "nB": np.zeros(nsp, int)}
    pr = lw_probes_fast(K, res, spots, r_box, r_in, r_out, r_excl)
    if nsp == 0 or len(pr["CY"]) == 0:
        return out
    bs = LW21_BLOCK * r_out
    bid_all = np.floor(pr["CY"] / bs).astype(np.int64) * 1_000_003 + np.floor(pr["CX"] / bs).astype(np.int64)
    ab_all = pr["La"] * pr["Lb"]; aa_all = pr["La"] * pr["La"]
    for i, sp in enumerate(spots):
        sel = lw_select(pr, sp, res.plan, K.shape, r_out, n_min)
        n = int(sel.sum())
        out["n"][i] = n
        if n < 2:
            continue
        _, inv = np.unique(bid_all[sel], return_inverse=True)
        nB = int(inv.max()) + 1
        ab = ab_all[sel]
        Sab, Saa = float(ab.sum()), float(aa_all[sel].sum())
        se = _block_se(ab, inv, nB)
        T = Sab / se if (np.isfinite(se) and se > 0) else 0.0
        out["T"][i] = T; out["nB"][i] = nB
        if T > z and Saa > 0:
            out["w"][i] = float(np.clip(Sab / Saa, 0.0, 1.0))
    return out


def _spill_prep(K, res, boxes, spots, Bsum, VB, sigmas, law, disp):
    """v2.1 spill correction (design note v2.1 Sec. 2.1): the box intensity lambda_G (electrons) from the G background
    (G excludes the spot discs, so lambda_G does not depend on the annulus) by the stored-count ML, and per spot the
    box-normalised profile at its annulus pixels."""
    nsp = len(boxes)
    nmax = max(len(b[0]) for b in boxes)
    Kpad = np.zeros((nsp, nmax)); pv = np.zeros((nsp, nmax), bool); mu = np.zeros((nsp, nmax)); f = np.zeros((nsp, nmax))
    I = np.zeros(nsp); sg = np.zeros(nsp); fa = []
    for k, (bys, bxs, ays, axs) in enumerate(boxes):
        nb = len(bys)
        Kb = K[bys, bxs]; Gb = res.bg[bys, bxs].astype(float)
        Gb = Gb * (Bsum[k] / Gb.sum()) if Gb.sum() > 0 else np.full(nb, Bsum[k] / max(nb, 1))
        Kpad[k, :nb] = Kb; pv[k, :nb] = True; mu[k, :nb] = law.E_inv(np.maximum(Gb, 0.0))
        fk = _profile(spots[k][0], spots[k][1], float(sigmas[k]), bys.astype(float), bxs.astype(float))
        tot = max(fk.sum(), 1e-12)
        f[k, :nb] = fk / tot
        fa.append(_profile(spots[k][0], spots[k][1], float(sigmas[k]), ays.astype(float), axs.astype(float)) / tot)
        I[k] = law.epc * (float(Kb.sum()) - float(Bsum[k]))
        sg[k] = law.epc * np.sqrt(max(disp * float(np.sum(law.g(np.maximum(Gb, 0.0)))) + float(VB[k]), 1e-12))
    l0, s0, _sat, _dl = SL.law_correct(I, sg, mu, f, pv, law)
    lam, _s = SL.ml_fit(Kpad, mu, f, pv, law, l0, s0, I_sum=I)
    return np.where(np.isfinite(lam), np.maximum(lam, 0.0), 0.0), fa


def _spill(law, Ga, fa, lam):
    """Mean stored excess over annulus pixels from the spot's own light (profile tail): E(mu_eff(mu, lam f)) - E(mu)."""
    if not len(Ga) or not np.isfinite(lam) or lam <= 0:
        return 0.0
    mu = law.E_inv(np.maximum(Ga, 0.0))
    return float(np.mean(law.E(SL.mu_eff(mu, lam * fa, law.tau)) - law.E(mu)))


def _sig_sum_grid(mu, f, pv, law, dsp, var_B, g):
    """Summation sigma (electrons) at hypothesised box intensities g (n_spots, n_grid)."""
    me = SL.mu_eff(mu[:, None, :], g[:, :, None] * f[:, None, :], law.tau)
    v = dsp * np.sum(np.where(pv[:, None, :], law.g(np.maximum(law.E(me), 0.0)), 0.0), axis=2)
    return law.epc * np.sqrt(np.maximum(v + np.nan_to_num(var_B)[:, None], 0.0))


def _L_grid(mu, f, pv, law, g):
    me = SL.mu_eff(mu[:, None, :], g[:, :, None] * f[:, None, :], law.tau)
    d = law.E(me) - law.E(mu)[:, None, :]
    return law.epc * np.sum(np.where(pv[:, None, :], d, 0.0), axis=2)


def _invert(est, center, scale, lo, fun_mean, fun_sig, n_grid=161):
    """68.3 % interval {E(lambda): |est - E(lambda)| <= sigma(lambda)} on a lambda grid about `center` +- 10 scale."""
    ns = len(est)
    lo_g = np.maximum(center - 10 * scale, lo)
    hi_g = np.maximum(center + 10 * scale, lo_g + 1e-6)
    g = lo_g[:, None] + (hi_g - lo_g)[:, None] * np.linspace(0, 1, n_grid)[None, :]
    E = fun_mean(g); S = fun_sig(g)
    h = S - np.abs(est[:, None] - E)                  # >= 0 inside the interval
    inside = h >= 0
    any_ = inside.any(1)
    ar = np.arange(ns)
    i0 = np.argmax(inside, axis=1); i1 = n_grid - 1 - np.argmax(inside[:, ::-1], axis=1)

    def edge(i_in, i_out):
        ok = (i_out >= 0) & (i_out < n_grid)
        io = np.clip(i_out, 0, n_grid - 1)
        hi_, ho = h[ar, i_in], h[ar, io]
        t = np.where(ok & (hi_ - ho > 0), hi_ / np.maximum(hi_ - ho, 1e-300), 0.0)
        return E[ar, i_in] + np.clip(t, 0, 1) * (E[ar, io] - E[ar, i_in])
    a = np.where(any_, edge(i0, i0 - 1), np.nan); b = np.where(any_, edge(i1, i1 + 1), np.nan)
    return np.minimum(a, b), np.maximum(a, b)


def _v2_intervals(res_out, summ, ml, Bpix, f, pv, Kpad, law, disp, lam_eval):
    nsp = f.shape[0]
    for m in summ:
        d = res_out.get(m)
        if d is None or not np.isfinite(d["I"]).any() or m == "O":
            continue
        dsp = 1.0 if m in ("FLAT", "PLANE", "ROBUST") else disp
        mu = np.where(pv, law.E_inv(np.maximum(np.nan_to_num(Bpix[m]), 0.0)), 0.0)
        vB = np.nan_to_num(d["var_B"])
        ok = np.isfinite(d["I"]) & np.isfinite(d["sigma"])
        lc = res_out.get(m + "-LC")
        cen = np.nan_to_num(lc["I"]) if lc is not None else np.nan_to_num(d["I"])
        sc = np.maximum(np.nan_to_num(lc["sigma"]) if lc is not None else np.nan_to_num(d["sigma"]), 1e-6)
        lo = -np.min(np.where(pv & (f > 0), mu / np.maximum(f, 1e-12), np.inf), axis=1)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        a = np.full(nsp, np.nan); b = np.full(nsp, np.nan)
        if ok.any():
            a[ok], b[ok] = _invert(d["I"][ok], cen[ok], sc[ok], lo[ok],
                                   lambda g: _L_grid(mu[ok], f[ok], pv[ok], law, g),
                                   lambda g: _sig_sum_grid(mu[ok], f[ok], pv[ok], law, dsp, vB[ok], g))
        d["ci_lo"], d["ci_hi"] = a, b
        if lam_eval is not None:
            sa = np.full(nsp, np.nan)
            le = np.asarray(lam_eval, float)
            sa[ok] = _sig_sum_grid(mu[ok], f[ok], pv[ok], law, dsp, vB[ok], le[ok][:, None])[:, 0]
            d["sigma_at"] = sa
        if m in ml and (m + "-ML") in res_out:
            e = res_out[m + "-ML"]
            sb = np.sqrt(np.maximum(np.nan_to_num(e["sigma"]) ** 2 - np.nan_to_num(e["sigma_stat"]) ** 2, 0.0))
            okm = np.isfinite(e["I"]) & np.isfinite(e["sigma_stat"])

            def sig_ml(g, okm=okm, sb=sb):
                info = SL._info_grid(mu[okm], f[okm], pv[okm], law, g)
                return np.sqrt(1.0 / np.maximum(info, 1e-300) + sb[okm][:, None] ** 2)
            a = np.full(nsp, np.nan); b = np.full(nsp, np.nan)
            if okm.any():
                lo_m = lo[okm] * 0.999
                a[okm], b[okm] = _invert(e["I"][okm], e["I"][okm], np.maximum(e["sigma"][okm], 1e-6), lo_m, lambda g: g, sig_ml)
            e["ci_lo"], e["ci_hi"] = a, b
            if lam_eval is not None:
                sa = np.full(nsp, np.nan)
                if okm.any():
                    sa[okm] = sig_ml(np.asarray(lam_eval, float)[okm][:, None])[:, 0]
                e["sigma_at"] = sa


def _invert21(est, center, scale, lo, hi_ext, fun_mean, fun_sig, n_lin=41, n_ext=24):
    """v2.1 test inversion (fix 3): the confidence set C = {lambda : |est - E(lambda)| <= sigma(lambda)} on a grid of
    n_lin points over center +- 8 scale (clipped at lo) plus n_ext geometric points up to hi_ext (the dead-time branch,
    where E is not monotone).  Returns the image interval [min, max] of E over C (crossings linearly interpolated: the
    image of C, not the E values at C's end points), the lambda interval [min C, max C], and `open` when C reaches the
    grid's upper end."""
    ns = len(est)
    a0 = np.maximum(center - 8 * scale, lo)
    a1 = np.maximum(center + 8 * scale, a0 + 1e-6)
    t = np.linspace(0, 1, n_lin)
    g = a0[:, None] + (a1 - a0)[:, None] * t[None, :]
    he = np.maximum(hi_ext, a1 * (1 + 1e-6))
    if n_ext:
        q = np.linspace(0, 1, n_ext + 1)[None, 1:]
        ge = a1[:, None] * (he / a1)[:, None] ** q
        g = np.concatenate([g, ge], axis=1)
    E = fun_mean(g); S = fun_sig(g)
    h = S - np.abs(est[:, None] - E)
    inside = h >= 0
    # crossings between neighbours with a sign change of h
    h0, h1 = h[:, :-1], h[:, 1:]
    cr = (h0 >= 0) != (h1 >= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = np.clip(np.where(cr, h0 / (h0 - h1), np.nan), 0, 1)
    Ec = E[:, :-1] + tt * (E[:, 1:] - E[:, :-1]); gc = g[:, :-1] + tt * (g[:, 1:] - g[:, :-1])
    Ein = np.concatenate([np.where(inside, E, np.nan), np.where(cr, Ec, np.nan)], axis=1)
    gin = np.concatenate([np.where(inside, g, np.nan), np.where(cr, gc, np.nan)], axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo_E, hi_E = np.nanmin(Ein, 1), np.nanmax(Ein, 1)
        lo_g, hi_g = np.nanmin(gin, 1), np.nanmax(gin, 1)
    return lo_E, hi_E, lo_g, hi_g, inside[:, -1].copy()


def _v21_intervals(res_out, summ, ml, Bpix, f, pv, law, disp, lam_eval):
    nsp = f.shape[0]
    for m in summ:
        d = res_out.get(m)
        if d is None or not np.isfinite(d["I"]).any() or m == "O":
            continue
        dsp = 1.0 if m in ("FLAT", "PLANE", "ROBUST") else disp
        mu = np.where(pv, law.E_inv(np.maximum(np.nan_to_num(Bpix[m]), 0.0)), 0.0)
        vB = np.nan_to_num(d["var_B"])
        ok = np.isfinite(d["I"]) & np.isfinite(d["sigma"])
        lc = res_out.get(m + "-LC")
        cen = np.nan_to_num(lc["I"]) if lc is not None else np.nan_to_num(d["I"])
        sc = np.maximum(np.nan_to_num(lc["sigma"]) if lc is not None else np.nan_to_num(d["sigma"]), 1e-6)
        lo = -np.min(np.where(pv & (f > 0), mu / np.maximum(f, 1e-12), np.inf), axis=1)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        fmax = np.max(np.where(pv, f, 0.0), axis=1)
        hi = 4.0 / (law.tau * np.maximum(fmax, 1e-12)) if law.tau > 0 else np.zeros(nsp)
        cen = np.maximum(cen, lo)
        a = np.full(nsp, np.nan); b = np.full(nsp, np.nan); la = np.full(nsp, np.nan); lb = np.full(nsp, np.nan)
        op = np.zeros(nsp, bool)
        if ok.any():
            a[ok], b[ok], la[ok], lb[ok], op[ok] = _invert21(
                d["I"][ok], cen[ok], sc[ok], lo[ok], hi[ok],
                lambda g: _L_grid(mu[ok], f[ok], pv[ok], law, g),
                lambda g: _sig_sum_grid(mu[ok], f[ok], pv[ok], law, dsp, vB[ok], g), n_ext=24 if law.tau > 0 else 0)
        d["ci_lo"], d["ci_hi"], d["ci_lam_lo"], d["ci_lam_hi"], d["ci_open"] = a, b, la, lb, op.astype(float)
        if lam_eval is not None:
            sa = np.full(nsp, np.nan)
            le = np.asarray(lam_eval, float)
            sa[ok] = _sig_sum_grid(mu[ok], f[ok], pv[ok], law, dsp, vB[ok], le[ok][:, None])[:, 0]
            d["sigma_at"] = sa
        if m in ml and (m + "-ML") in res_out:
            e = res_out[m + "-ML"]
            sb = np.sqrt(np.maximum(np.nan_to_num(e["sigma"]) ** 2 - np.nan_to_num(e["sigma_stat"]) ** 2, 0.0))
            okm = np.isfinite(e["I"]) & np.isfinite(e["sigma_stat"])

            def sig_ml(g, okm=okm, sb=sb):
                info = SL._info_grid(mu[okm], f[okm], pv[okm], law, g)
                return np.sqrt(1.0 / np.maximum(info, 1e-300) + sb[okm][:, None] ** 2)
            a = np.full(nsp, np.nan); b = np.full(nsp, np.nan); op = np.zeros(nsp, bool)
            if okm.any():
                lo_m = lo[okm] * 0.999
                a[okm], b[okm], _l, _h, op[okm] = _invert21(e["I"][okm], np.maximum(e["I"][okm], lo_m), np.maximum(e["sigma"][okm], 1e-6),
                                                            lo_m, np.zeros(int(okm.sum())), lambda g: g, sig_ml, n_ext=0)
            e["ci_lo"], e["ci_hi"], e["ci_open"] = a, b, op.astype(float)
            if lam_eval is not None:
                sa = np.full(nsp, np.nan)
                if okm.any():
                    sa[okm] = sig_ml(np.asarray(lam_eval, float)[okm][:, None])[:, 0]
                e["sigma_at"] = sa


def integrate_spots(frame, res, spots, r_box, r_in=None, r_out=None, r_excl=None, sigmas=None,
                    methods=("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST"), oracle=None, law_methods=True, ml=("G", "AUTO"),
                    version=None, lam_eval=None):
    """Integrate spots (n, 2) = (cy, cx). Returns {method: dict of arrays I, sigma, ...} with I in electrons.
    version: None = the background result's version ("v2.1": AUTO = G with the v2 weight only where significant
    (local_weight_v21) and the spot's own light removed from the annulus (needs `sigmas`), v2.1 intervals (image of the
    confidence set, lambda interval ci_lam_lo / ci_lam_hi, ci_open); "v2": continuous local weight AUTO, test-inversion
    intervals; "v1": effect-size AUTO switch and the v1 ML; "v0": v0 exactly).
    v2 with `sigmas`: every summation and -ML output carries ci_lo / ci_hi, the 68.3 % interval obtained by inverting
    |estimate - E(lambda)| <= sigma(lambda) with sigma evaluated at the HYPOTHESISED lambda (no data of the spot in the
    sigma).  lam_eval (evaluation only, box intensities in electrons per spot): adds sigma_at = sigma(lam_eval)."""
    plan = res.plan; law = res.law
    version = getattr(plan, "version", "v0") if version is None else version
    K = np.asarray(frame, float) - law.offset
    spots = np.asarray(spots, float).reshape(-1, 2)
    r_in = 5.0 / 3.0 * r_box if r_in is None else r_in
    r_out = 3.0 * r_box if r_out is None else r_out
    r_excl = r_out if r_excl is None else r_excl
    epc = law.epc; disp = res.dispersion
    ny, nx = K.shape
    R = int(np.ceil(max(r_out, r_box))) + 1
    others = disc_mask(K.shape, spots, r_excl)
    methods = list(methods)
    auto_info = {}
    use_gl_spot = None
    if "AUTO" in methods and version == "v0":
        Q, nprobe, p = local_structure_test(K, res, spots, r_in, r_out, r_excl)
        use_gl = bool(np.isfinite(p) and p < AUTO_P)
        auto_info = {"Q": Q, "n_probe": nprobe, "p": p, "use_GL": use_gl}
    elif "AUTO" in methods and version == "v1":
        use_gl_spot, zinfo = local_switch_mse(K, res, spots, r_box, r_in, r_out, r_excl)
        auto_info = {"use_GL": bool(use_gl_spot.any()), "zones": zinfo, "p": np.nan}
    elif "AUTO" in methods and version == "v2.1":
        lw21 = local_weight_v21(K, res, spots, r_box, r_in, r_out, r_excl)
        use_gl_spot = lw21["w"]
        auto_info = {"use_GL": bool((lw21["w"] > 0).any()), "w": lw21["w"], "lw21": lw21, "p": np.nan}
    elif "AUTO" in methods:
        w_spot, winfo = local_weight(K, res, spots, r_box, r_in, r_out, r_excl)
        use_gl_spot = w_spot
        auto_info = {"use_GL": bool((w_spot > 0).any()), "w": w_spot, "winfo": winfo, "p": np.nan}
    if "AUTO" in methods:
        if "G" not in methods:
            methods.insert(0, "G")
        if "GL" not in methods:
            methods.insert(1, "GL")
    summ = list(methods) + (["O"] if oracle is not None else [])
    keys = ("I", "sigma", "sigma_plugin", "B_box", "var_B", "n_box", "n_ann")
    out = {m: {k: [] for k in keys} for m in summ}
    if oracle is not None:
        K0 = np.asarray(oracle, float) - law.offset
    boxes = []
    for k, (cy, cx) in enumerate(spots):
        iy, ix = int(round(cy)), int(round(cx))
        y0, y1 = max(iy - R, 0), min(iy + R + 1, ny); x0, x1 = max(ix - R, 0), min(ix + R + 1, nx)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rho = np.hypot(yy - cy, xx - cx)
        v = plan.valid[y0:y1, x0:x1]
        own = disc_mask((y1 - y0, x1 - x0), [(cy - y0, cx - x0)], r_excl)
        oth = others[y0:y1, x0:x1] & ~own
        box = (rho <= r_box) & v
        ann = (rho > r_in) & (rho <= r_out) & v & ~oth
        boxes.append((yy[box], xx[box], yy[ann], xx[ann]))
    nsp = len(boxes)
    if nsp == 0:
        return {}
    rr = np.concatenate([plan.rmap[b[0], b[1]] for b in boxes]).astype(float)
    pp = np.concatenate([plan.phimap[b[0], b[1]] for b in boxes]).astype(float)
    offs = np.concatenate([[0], np.cumsum([len(b[0]) for b in boxes])])
    groups = [np.arange(offs[i], offs[i + 1]) for i in range(nsp)]
    pix = None
    if getattr(plan, "zone", None) is not None:
        pix = np.concatenate([b[0].astype(np.int64) * nx + b[1] for b in boxes])
    Bsum, VB = res.box_sums(rr, pp, groups, pix=pix)
    spill_lam = None
    if version == "v2.1" and sigmas is not None and any(m in methods for m in ("AUTO", "GL")):
        spill_lam, spill_fa = _spill_prep(K, res, boxes, spots, Bsum, VB, sigmas, law, disp)
    nmax = max(len(b[0]) for b in boxes)
    Bpix = {m: np.zeros((nsp, nmax)) for m in summ}
    Kpad = np.zeros((nsp, nmax)); pv = np.zeros((nsp, nmax), bool)
    for k, (bys, bxs, ays, axs) in enumerate(boxes):
        Kb = K[bys, bxs]; Ka = K[ays, axs]
        nb, na = len(Kb), len(Ka)
        Kpad[k, :nb] = Kb; pv[k, :nb] = True
        Gb = res.bg[bys, bxs].astype(float)
        Gb = Gb * (Bsum[k] / Gb.sum()) if Gb.sum() > 0 else np.full(nb, Bsum[k] / max(nb, 1))
        cy, cx = spots[k]
        for m in summ:
            o = out[m]; sig_pl = np.nan
            auto_gl = (bool(use_gl_spot[k]) if use_gl_spot is not None else bool(auto_info.get("use_GL"))) if m == "AUTO" else False
            if version == "v2.1" and (m == "AUTO" or m == "GL"):
                wk = float(lw21["w"][k]) if m == "AUTO" else 1.0
                Ga = res.bg[ays, axs].astype(float)
                ok = ~res.flags[ays, axs]
                resid = (Ka - Ga)[ok]; nok = int(ok.sum())
                corr = float(resid.mean()) if nok else 0.0
                if spill_lam is not None and nok and wk > 0:
                    corr -= _spill(law, Ga[ok], spill_fa[k][ok], spill_lam[k])      # the spot's own light in the annulus
                vgl = nb ** 2 * disp * float(np.sum(law.g(np.maximum(Ga[ok], 0.0)))) / max(nok, 1) ** 2
                B = float(Bsum[k]) + wk * nb * corr
                vB = (1.0 - wk) ** 2 * float(VB[k]) + wk ** 2 * vgl
                Bp = Gb + wk * corr; vk = _vk(law, Bp, Kb, disp); na = nok
            elif m == "AUTO" and version == "v2":
                wk = float(use_gl_spot[k])
                Ga = res.bg[ays, axs].astype(float)
                ok = ~res.flags[ays, axs]
                resid = (Ka - Ga)[ok]; nok = int(ok.sum())
                corr = float(resid.mean()) if nok else 0.0
                vgl = nb ** 2 * disp * float(np.sum(law.g(np.maximum(Ga[ok], 0.0)))) / max(nok, 1) ** 2
                B = float(Bsum[k]) + wk * nb * corr
                vB = (1.0 - wk) ** 2 * float(VB[k]) + wk ** 2 * vgl
                Bp = Gb + wk * corr; vk = _vk(law, Bp, Kb, disp); na = nok
            elif m == "G" or (m == "AUTO" and not auto_gl):
                B = float(Bsum[k]); vB = float(VB[k]); Bp = Gb
                vk = _vk(law, Bp, Kb, disp)
            elif m in ("GL", "AUTO"):
                Ga = res.bg[ays, axs].astype(float)
                ok = ~res.flags[ays, axs]
                resid = (Ka - Ga)[ok]; nok = int(ok.sum())
                corr = float(resid.mean()) if nok else 0.0
                B = float(Bsum[k]) + nb * corr
                vB = nb ** 2 * disp * float(np.sum(law.g(np.maximum(Ga[ok], 0.0)))) / max(nok, 1) ** 2
                Bp = Gb + corr; vk = _vk(law, Bp, Kb, disp); na = nok
            elif m == "O":
                Bp = K0[bys, bxs]; B = float(Bp.sum()); vB = 0.0
                vk = max(float(np.sum(law.g(np.maximum(Bp, 0)) + law.dg(np.maximum(Bp, 0)) * (Kb - Bp))), 0.5 * float(np.sum(law.g(np.maximum(Bp, 0)))))
                Bp = np.full(nb, B / max(nb, 1))
            else:
                dy = ays - cy; dx = axs - cx
                fn = {"FLAT": REF.flat, "PLANE": REF.plane, "ROBUST": REF.robust}[m]
                if na < 4:
                    B = np.nan; vB = np.nan; vk = np.nan; Bp = np.full(nb, np.nan)
                else:
                    b0, gy, gx, info = fn(Ka, law, dy, dx)
                    sy = float(np.sum(bys - cy)); sx = float(np.sum(bxs - cx))
                    B = nb * b0 + gy * sy + gx * sx
                    Bp = b0 + gy * (bys - cy) + gx * (bxs - cx)
                    if info.get("cov") is not None:
                        a = np.array([nb, sy, sx]); vB = float(a @ info["cov"] @ a)
                    else:
                        vB = nb ** 2 * info["var_b0_model"]
                    vk = _vk(law, Bp, Kb, 1.0)
                    if m == "FLAT":   # the plug-in "sigma_law" form of peak recovery v0 (pre-registered Sec. 1.7)
                        vpl = float(np.sum(law.g(np.maximum(Kb, b0)))) + nb ** 2 * info["var_b0_plugin"]
                        sig_pl = epc * np.sqrt(max(vpl, 0.0))
            Bpix[m][k, :nb] = Bp
            I = epc * (float(Kb.sum()) - B)
            o["I"].append(I); o["sigma"].append(epc * np.sqrt(max(vk + vB, 0.0)) if np.isfinite(vk) else np.nan)
            o["sigma_plugin"].append(sig_pl); o["B_box"].append(B); o["var_B"].append(vB); o["n_box"].append(nb); o["n_ann"].append(na)
    res_out = {m: {k: np.asarray(v, float) for k, v in d.items()} for m, d in out.items()}
    if auto_info:
        res_out["AUTO"]["auto_p"] = np.full(nsp, auto_info["p"])
        res_out["AUTO"]["auto_use_GL"] = (use_gl_spot.astype(float) if use_gl_spot is not None else np.full(nsp, float(auto_info["use_GL"])))
        if version == "v2.1":
            res_out["AUTO"]["auto_w"] = np.asarray(lw21["w"], float); res_out["AUTO"]["auto_T"] = np.asarray(lw21["T"], float)
            res_out["AUTO"]["auto_n"] = lw21["n"].astype(float)
            if spill_lam is not None:
                res_out["AUTO"]["spill_lam"] = np.asarray(spill_lam, float)
        elif version == "v2":
            res_out["AUTO"]["auto_w"] = np.asarray(use_gl_spot, float)
            res_out["AUTO"]["auto_n"] = auto_info["winfo"]["n_probe"].astype(float)
        elif use_gl_spot is not None:
            zi = auto_info["zones"]
            res_out["AUTO"]["auto_D"] = zi["D"][zi["spot_zone"]]; res_out["AUTO"]["auto_n"] = zi["n"][zi["spot_zone"]].astype(float)
    if sigmas is None:
        return res_out
    # ---- spot profiles on the box (box-normalised)
    f = np.zeros((nsp, nmax))
    for k, (bys, bxs, _a, _b) in enumerate(boxes):
        fk = _profile(spots[k][0], spots[k][1], float(sigmas[k]), bys.astype(float), bxs.astype(float))
        f[k, :len(fk)] = fk / max(fk.sum(), 1e-12)
    # ---- counting-law-aware intensities: ML first (it fixes the dead-time branch), then law-corrected summation
    if law_methods:
        for m in summ:
            d = res_out[m]
            mu = np.where(pv, law.E_inv(np.maximum(Bpix[m], 0.0)), 0.0)
            okm = np.isfinite(d["I"])
            lam = np.full(nsp, np.nan); sg = np.full(nsp, np.nan); bey = np.zeros(nsp); dL = np.full(nsp, np.nan)
            lam_ml = np.full(nsp, np.nan); s_ml = np.full(nsp, np.nan)
            if okm.any():
                l0, s0, _sa, _dl = SL.law_correct(d["I"][okm], d["sigma"][okm], mu[okm], f[okm], pv[okm], law)
                lam_ml[okm], s_ml[okm] = SL.ml_fit(Kpad[okm], mu[okm], f[okm], pv[okm], law, l0, s0, I_sum=d["I"][okm])
                l_, s_, b_, dl_ = SL.law_correct_branch(d["I"][okm], d["sigma"][okm], mu[okm], f[okm], pv[okm], law, lam_ml[okm])
                lam[okm] = l_; sg[okm] = s_; bey[okm] = b_; dL[okm] = dl_
            res_out[m + "-LC"] = {"I": lam, "sigma": sg, "beyond_max": bey, "dL": dL, "B_box": d["B_box"], "var_B": d["var_B"],
                                  "n_box": d["n_box"], "n_ann": d["n_ann"], "sigma_plugin": np.full(nsp, np.nan)}
            if m in ml:
                # background term: d lambda / d B_box by re-solving the ML with the background shifted (post-hoc fix,
                # report Sec. 3: the summation sensitivity 1 / L' diverges past the dead-time maximum)
                sb = np.full(nsp, np.nan)
                if okm.any():
                    hpx = np.where(pv, 0.02 * np.maximum(Bpix[m], 0.0) + 1e-3, 0.0)
                    mu2 = np.where(pv, law.E_inv(np.maximum(Bpix[m] + hpx, 0.0)), 0.0)
                    l2, _ = SL.ml_fit(Kpad[okm], mu2[okm], f[okm], pv[okm], law, lam_ml[okm], s_ml[okm], I_sum=d["I"][okm])
                    sb[okm] = np.abs(l2 - lam_ml[okm]) / hpx[okm].sum(1) * np.sqrt(np.maximum(d["var_B"][okm], 0))
                res_out[m + "-ML"] = {"I": lam_ml, "sigma": np.sqrt(s_ml ** 2 + sb ** 2), "sigma_stat": s_ml, "B_box": d["B_box"],
                                      "var_B": d["var_B"], "n_box": d["n_box"], "n_ann": d["n_ann"], "sigma_plugin": np.full(nsp, np.nan)}
                if version != "v0":
                    lm = np.full(nsp, np.nan)
                    if okm.any():
                        lm[okm] = SL.lam_min(mu[okm], f[okm], pv[okm], law, LAM_MIN_CR)
                    res_out[m + "-ML"]["lam_min5"] = lm
        # v1 (design note v1 Sec. 2.4, A196): summation variance at the profile model's expected stored values
        if version != "v0" and "G-ML" in res_out:
            lref = np.maximum(np.nan_to_num(res_out["G-ML"]["I"]), 0.0)
            for m in summ:
                d = res_out[m]
                if not np.isfinite(d["I"]).any():
                    continue
                mu = np.where(pv, law.E_inv(np.maximum(Bpix[m], 0.0)), 0.0)
                Ep = law.E(SL.mu_eff(mu, lref[:, None] * f, law.tau))
                dsp = 1.0 if m in ("FLAT", "PLANE", "ROBUST") else disp
                vk = dsp * np.sum(np.where(pv, law.g(np.maximum(Ep, 0.0)), 0.0), axis=1)
                new_sig = epc * np.sqrt(np.maximum(vk + np.nan_to_num(d["var_B"]), 0.0))
                new_sig = np.where(np.isfinite(d["sigma"]), new_sig, np.nan)
                d["sigma_v0form"] = d["sigma"].copy(); d["sigma"] = new_sig
                lc = res_out.get(m + "-LC")
                if lc is not None:
                    lc["sigma_v0form"] = lc["sigma"].copy()
                    lc["sigma"] = new_sig / np.maximum(np.abs(lc["dL"]), 1e-12)
        # v2 (design note v2 Sec. 2.3, fixes 3-4): intervals by test inversion, sigma at the hypothesised lambda
        if version == "v2":
            _v2_intervals(res_out, summ, ml, Bpix, f, pv, Kpad, law, disp, lam_eval)
        elif version == "v2.1":
            _v21_intervals(res_out, summ, ml, Bpix, f, pv, law, disp, lam_eval)
    # ---- linear profile fit on G (Amendment 1 a)
    o = {k: [] for k in keys}
    cws, info = [], []
    for k, (bys, bxs, _a, _b) in enumerate(boxes):
        nb = len(bys); fk = f[k, :nb]; Kb = Kpad[k, :nb]; Gb = Bpix["G"][k, :nb]
        I0 = max(float(Kb.sum() - Gb.sum()), 0.0)
        w = 1.0 / np.maximum(disp * law.g(np.maximum(Gb + I0 * fk, 0.0)), 1e-12)
        c = w * fk / np.sum(w * fk * fk)
        cws.append(c); info.append((float(np.sum(c * (Kb - Gb))), float(np.sum(c * c / w)), nb))
    _, VBc = res.box_sums(rr, pp, groups, weights=np.concatenate(cws), pix=pix)
    for k, (Ist, vK, nb) in enumerate(info):
        o["I"].append(epc * Ist); o["sigma"].append(epc * np.sqrt(max(vK + float(VBc[k]), 0.0)))
        o["sigma_plugin"].append(np.nan); o["B_box"].append(np.nan); o["var_B"].append(float(VBc[k])); o["n_box"].append(nb); o["n_ann"].append(0)
    res_out["G_PF"] = {k: np.asarray(v, float) for k, v in o.items()}
    return res_out
