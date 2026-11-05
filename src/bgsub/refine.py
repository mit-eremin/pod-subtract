# MOVED from pod/background/src/podbg/bgsub/refine.py at background commit f1e93f5 (file last changed in fd2d10f; sha256 b8b7a5ce525659213fe55c7d2506059ac6bec8ed53a414c289a301c91c9f81c4).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""bgsub v2: re-centring and the ring refinement (design note 2026-11-05_bgsub_algorithm_v2.md Sec. 2.1).

Both act AFTER the v1 global fit G1 (core._estimate_core) and keep its linear structure.

RE-CENTRING. The v1 design carries centre-error columns R'(r) cos phi, R'(r) sin phi (the derivative of the OTHER
half's radial profile), so a centre error (dy, dx) shows up as their coefficients (-dx, -dy) to first order. When the
implied error exceeds RECENTRE_THR x N / 1024 px the plan is rebuilt at the corrected centre and the frame refitted
(Gauss-Newton, at most RECENTRE_MAX rounds, accelerated when the linearised step under-shoots; a round is kept only if
the Pearson dispersion does not rise). Below the threshold the first-order columns already absorb the error in G; the
residual sub-threshold shift is still used for the ring refinement below.

RING REFINEMENT. The fine radial spline (step max(3, N / 256) px) cannot follow radial structure narrower than its knot
spacing (the sharp 4.6-A ordered-hydrocarbon ring of some films, A195 / A236). With the residual K - G of the USED
pixels averaged on 1-px rings of r' = |p - c'| (c' = the refined centre) separately in the two checkerboard halves,
rho_A(i), rho_B(i) have the same expectation delta_i (the radial misfit of G in ring i) and independent noises, so

    E[rho_A rho_B] = delta_i^2,      E[rhobar^2] = delta_i^2 + v_i,      rhobar = (rho_A + rho_B) / 2,
    v_i = Var(rhobar) = E[(rho_A - rho_B)^2] / 4                                     (no variance model)

and the MSE-optimal shrinkage of rhobar towards zero is w_i = delta_i^2 / (delta_i^2 + v_i).  The correction is
c_i = w^_i rhobar_i with w^_i = clip(sum_W rho_A rho_B / sum_W rhobar^2, 0, 1) over a window of 2 RING_W + 1 rings
(positive-part James-Stein / Wiener weight).  It is applied only on rings with >= RING_MIN_COUNTS expected stored counts
and >= RING_MIN_PX used pixels in each half (Gaussian ring means; at sparse counts the noise is skewed and the shrinkage
would not be mean-zero).  Var(c_i) ~ w_i^2 v^_i (window mean) enters the box variances (box_sums).  The pixels of a
strong sub-knot ring are flagged as outliers against G, which would bias the ring means low; when |c_i| exceeds
RING_REFLAG pixel SDs somewhere, the count-law flags are recomputed against G + c and the refinement repeated (<= 2x).  On a frame without
sub-knot radial structure w^ is near zero and the correction costs < 1 % of the box variance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import _kernels as KN

RECENTRE_THR = 2.0          # px x N / 1024: re-plan when the implied centre error exceeds this
RECENTRE_TOL = 0.5          # px x N / 1024: stop when the remaining implied error is below this
RECENTRE_MAX = 5
RING_W = 2
RING_MIN_COUNTS = 50.0
RING_MIN_PX = 20
RING_REFLAG = 1.0           # re-flag when |c| exceeds this many pixel SDs in some ring
PLAN_KEYS = ("h_r", "H_factor", "M", "n_sect", "eval_dr", "n_phi_eval", "stop", "zone_kw")


def implied_shift(res):
    """(dy, dx) of the beam-centre error implied by the derivative columns (mean of the two halves); (0, 0) without."""
    plan = res.plan
    v = []
    for e in (1, 2):
        c = []
        for h in res.halves:
            d = h.design
            if not d.with_d:
                return 0.0, 0.0
            c.append(h.theta[d.off[e] + plan.Ka])
        v.append(float(np.mean(c)))
    return -v[1], -v[0]


@dataclass
class RingRefinement:
    centre: tuple               # (cy', cx'): refined centre the rings are taken about
    r_lo: float
    c: np.ndarray               # (n,) correction per 1-px ring (stored counts per pixel)
    var: np.ndarray             # (n,) variance of c
    w: np.ndarray               # (n,) shrinkage weight
    gated: np.ndarray           # (n,) bool: ring eligible

    def ring_index(self, y, x):
        r = np.hypot(np.asarray(y, float) - self.centre[0], np.asarray(x, float) - self.centre[1])
        i = np.floor(r - self.r_lo).astype(np.int64)
        return np.where((i >= 0) & (i < len(self.c)), i, -1)

    def box_terms(self, plan, r, phi, groups, weights=None):
        """Sum over each pixel group of (weights x) c and its variance; pixels given by (r, phi) about the plan centre."""
        cy, cx = plan.geometry.centre
        y = cy + np.asarray(r, float) * np.sin(phi); x = cx + np.asarray(r, float) * np.cos(phi)
        i = self.ring_index(y, x)
        wt = np.ones(len(i)) if weights is None else np.asarray(weights, float)
        cc = np.where(i >= 0, self.c[np.maximum(i, 0)], 0.0) * wt
        dB = np.array([cc[g].sum() for g in groups]) if len(groups) else np.zeros(0)
        dV = np.zeros(len(groups))
        for k, g in enumerate(groups):
            ig = i[g]; ok = ig >= 0
            if ok.any():
                s = np.bincount(ig[ok], weights=wt[g][ok], minlength=len(self.c))
                dV[k] = float(np.sum(s ** 2 * self.var))
        return dB, dV


def ring_refinement(K, res, centre, use_numba=True):
    """The v2 ring refinement of a fitted result (module docstring). K: frame minus the declared offset (finite)."""
    plan = res.plan
    n = plan.n_r + 8
    r_lo = plan.r_lo
    A = KN.ring_acc(K, res.bg, res.used, centre[0], centre[1], r_lo, n, use_numba)     # (n, 2, [resid, bg, npx])
    npx = A[:, :, 2]; S = A[:, :, 0]; Bs = A[:, :, 1]
    gated = (npx.min(1) >= RING_MIN_PX) & (Bs.min(1) >= RING_MIN_COUNTS)
    rho = np.where(npx > 0, S / np.maximum(npx, 1), 0.0)
    rA, rB = rho[:, 0], rho[:, 1]
    T = np.where(gated, rA * rB, 0.0)
    rb = 0.5 * (rA + rB)
    Q = np.where(gated, rb ** 2, 0.0)
    V = np.where(gated, 0.25 * (rA - rB) ** 2, 0.0)
    ker = np.ones(2 * RING_W + 1)
    sT = np.convolve(T, ker, "same"); sQ = np.convolve(Q, ker, "same")
    sV = np.convolve(V, ker, "same"); sn = np.convolve(gated.astype(float), ker, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gated & (sQ > 0), np.clip(sT / sQ, 0.0, 1.0), 0.0)
        vbar = np.where(sn > 0, sV / np.maximum(sn, 1), 0.0)
    c = w * np.where(gated, rb, 0.0)
    return RingRefinement(centre=(float(centre[0]), float(centre[1])), r_lo=float(r_lo), c=c, var=w ** 2 * vbar, w=w,
                          gated=gated)


def _apply_ring(bg, ring: RingRefinement, use_numba=True):
    if not np.any(ring.c != 0):
        return bg
    return KN.ring_apply(bg, ring.c, ring.centre[0], ring.centre[1], ring.r_lo, use_numba)


def _reflag(K, res, ring, law, exclude, kw):
    """Re-run the core's count-law flagging against the cell model + the ring correction (by cell ring)."""
    from .law import ALPHA_PX, ALPHA_WIN
    plan = res.plan
    un = plan.use_numba
    n = plan.n_r
    c = np.zeros(n); m = min(n, len(ring.c)); c[:m] = ring.c[:m]
    bc = (np.maximum(res.b_cells, 0.0) + c[:, None]).ravel()
    kpx = law.kstar(bc, kw.get("alpha_px", ALPHA_PX), 1); tw = law.kstar(bc, kw.get("alpha_win", ALPHA_WIN), 9)
    base = plan.valid.copy()
    if exclude is not None:
        base &= ~np.asarray(exclude, bool)
    N = max(K.shape)
    r_dil = kw.get("r_dil") or 2.0 * N / 1024.0
    f0 = KN.flag(K, plan.cellmap2, base, kpx, tw, un)
    res.flags = KN.dilate(f0, r_dil, un)
    res.used = base & ~res.flags


def estimate_v2(frame, mask, geometry, detector, *, exclude=None, plan=None, recentre=True, ring_refine=True, version="v2", **kw):
    from .core import Geometry, _estimate_core, make_plan
    from .law import CountLaw
    t0 = time.perf_counter()
    est_kw = {k: v for k, v in kw.items() if k not in PLAN_KEYS}
    res = _estimate_core(frame, mask, geometry, detector, exclude=exclude, plan=plan, version=version, **kw)
    plan = res.plan; geo = plan.geometry
    N = max(plan.shape)
    thr = RECENTRE_THR * N / 1024.0; tol = RECENTRE_TOL * N / 1024.0
    info = {"recentred": False, "n_iter": 0, "centre_in": tuple(float(v) for v in geo.centre)}
    simple = geo.r_map is None and geo.ellipse is None
    dy, dx = implied_shift(res)
    info["shift_in"] = (dy, dx)
    if recentre and simple and np.hypot(dy, dx) > thr:
        msk = mask if mask is not None else ~np.isfinite(np.asarray(frame, float))
        best = res; c = np.array(geo.centre, float); prev = None
        for it in range(RECENTRE_MAX):
            step = np.array([dy, dx])
            d = float(np.hypot(dy, dx))
            if prev is not None and d > 0.5 * prev[0] and float(np.dot(step, prev[1])) > 0:
                q = min(d / prev[0], 0.95)
                step = step * min(1.0 / (1.0 - q), 4.0)          # under-shooting linear step (gross error): accelerate
            cn = c + step
            pkw = {k: v for k, v in plan.kw.items()}
            p2 = make_plan(plan.shape, msk, Geometry((float(cn[0]), float(cn[1]))), version=version, **pkw)
            r2 = _estimate_core(frame, mask, Geometry((float(cn[0]), float(cn[1]))), detector, exclude=exclude, plan=p2,
                                version=version, **est_kw)
            info["n_iter"] = it + 1
            if not (r2.dispersion <= best.dispersion * (1 + 1e-3)):
                break
            best = r2; prev = (d, np.array([dy, dx])); c = cn
            dy, dx = implied_shift(r2)
            if np.hypot(dy, dx) < tol:
                break
        if best is not res:
            info["recentred"] = True
        res = best
        info["centre_out"] = tuple(float(v) for v in res.plan.geometry.centre)
        dy, dx = implied_shift(res)
    info["shift_out"] = (dy, dx)
    res.centre_info = info
    if ring_refine and simple and res.bg is not None:
        law = res.law if res.law is not None else (detector or CountLaw.poisson(1.0))
        K = np.asarray(frame, float)
        if law.offset:
            K = K - law.offset
        K = np.where(np.isfinite(K), K, 0.0)
        cg = res.plan.geometry.centre
        sh = (dy, dx) if np.hypot(dy, dx) <= thr else (0.0, 0.0)
        cc = (cg[0] + sh[0], cg[1] + sh[1])
        if version == "v2.1":
            ell = implied_ellipticity(res)
            info["ellipticity"] = ell
            ring = ring_refinement_v21(K, res, cc, ell, use_numba=res.plan.use_numba)
            # skipped ring means re-centred on G + c: where the correction exceeds RING_REFLAG pixel SDs the count-law
            # flags are recomputed against G + c and the ring re-estimated, until it changes by < RING21_CONV SDs
            for _ in range(RING21_REFLAG_MAX if kw.get("flagging", True) else 0):
                n = min(len(ring.c), res.plan.n_r)
                bc = np.maximum(res.b_cells, 0.0)[:n].mean(1)
                sd = np.sqrt(np.maximum(law.g(bc), 1e-12))
                if not np.any(np.abs(ring.c[:n]) > RING_REFLAG * sd):
                    break
                _reflag21(K, res, ring, law, exclude, kw)
                new = ring_refinement_v21(K, res, cc, ell, use_numba=res.plan.use_numba)
                info["ring_reflag"] = info.get("ring_reflag", 0) + 1
                done = not np.any(np.abs(new.c[:n] - ring.c[:n]) > RING21_CONV * sd)
                ring = new
                if done:
                    break
            res.ring = ring
            res.bg = KN.ring_apply_ell(res.bg, ring.c, ring.centre[0], ring.centre[1], ell[0], ell[1], ring.r_lo, res.plan.use_numba)
        else:
            ring = ring_refinement(K, res, cc, use_numba=res.plan.use_numba)
            # pixels of a strong sub-knot ring were flagged as outliers against G (selection would bias the ring means low):
            # when the correction exceeds RING_REFLAG pixel SDs somewhere, re-flag against G + c and recompute (<= 2 rounds)
            for _ in range(2 if kw.get("flagging", True) else 0):
                n = min(len(ring.c), res.plan.n_r)
                bc = np.maximum(res.b_cells, 0.0)[:n].mean(1)
                sd = np.sqrt(np.maximum(law.g(bc), 1e-12))
                if not np.any(np.abs(ring.c[:n]) > RING_REFLAG * sd):
                    break
                _reflag(K, res, ring, law, exclude, kw)
                ring = ring_refinement(K, res, cc, use_numba=res.plan.use_numba)
                info["ring_reflag"] = info.get("ring_reflag", 0) + 1
            res.ring = ring
            res.bg = _apply_ring(res.bg, ring, res.plan.use_numba)
    res.timings["v2_total_s"] = time.perf_counter() - t0
    return res



# ============================================================================= v2.1: the elliptical ring term
RING21_REFLAG_MAX = 3       # re-flag rounds (skipped means re-centred on G + c)
RING21_CONV = 0.1           # stop when the correction changes by less than this many pixel SDs


def implied_ellipticity(res):
    """(ea, eb) of the global pattern from the v1 ellipticity columns r R'(r) cos 2phi, r R'(r) sin 2phi (mean of the two
    halves): b(r, phi) ~ R(r (1 + ea cos 2phi + eb sin 2phi)) to first order; (0, 0) without derivative columns."""
    plan = res.plan
    v = []
    for e in (3, 4):
        c = []
        for h in res.halves:
            d = h.design
            if not d.with_d:
                return 0.0, 0.0
            c.append(h.theta[d.off[e] + plan.Ka])
        v.append(float(np.mean(c)))
    return v[0], v[1]


@dataclass
class RingRefinement21(RingRefinement):
    """v2.1: the v2 ring refinement on 1-px rings of the ELLIPTICAL radius r' = |p - c'| (1 + ea cos 2phi' + eb sin 2phi'),
    (ea, eb) = implied_ellipticity of the global fit (design note v2.1 Sec. 2.2)."""
    ell: tuple = (0.0, 0.0)

    def ring_index(self, y, x):
        dy = np.asarray(y, float) - self.centre[0]; dx = np.asarray(x, float) - self.centre[1]
        r = KN._ell_r(dy, dx, self.ell[0], self.ell[1])
        i = np.floor(r - self.r_lo).astype(np.int64)
        return np.where((i >= 0) & (i < len(self.c)), i, -1)


def ring_refinement_v21(K, res, centre, ell, use_numba=True):
    """The v2 ring refinement (module docstring) on elliptical rings; same gate, same cross-half shrinkage."""
    plan = res.plan
    n = plan.n_r + 8
    r_lo = plan.r_lo
    A = KN.ring_acc_ell(K, res.bg, res.used, centre[0], centre[1], ell[0], ell[1], r_lo, n, use_numba)
    npx = A[:, :, 2]; S = A[:, :, 0]; Bs = A[:, :, 1]
    gated = (npx.min(1) >= RING_MIN_PX) & (Bs.min(1) >= RING_MIN_COUNTS)
    rho = np.where(npx > 0, S / np.maximum(npx, 1), 0.0)
    rA, rB = rho[:, 0], rho[:, 1]
    T = np.where(gated, rA * rB, 0.0)
    rb = 0.5 * (rA + rB)
    Q = np.where(gated, rb ** 2, 0.0)
    V = np.where(gated, 0.25 * (rA - rB) ** 2, 0.0)
    ker = np.ones(2 * RING_W + 1)
    sT = np.convolve(T, ker, "same"); sQ = np.convolve(Q, ker, "same")
    sV = np.convolve(V, ker, "same"); sn = np.convolve(gated.astype(float), ker, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gated & (sQ > 0), np.clip(sT / sQ, 0.0, 1.0), 0.0)
        vbar = np.where(sn > 0, sV / np.maximum(sn, 1), 0.0)
    c = w * np.where(gated, rb, 0.0)
    return RingRefinement21(centre=(float(centre[0]), float(centre[1])), r_lo=float(r_lo), c=c, var=w ** 2 * vbar, w=w,
                            gated=gated, ell=(float(ell[0]), float(ell[1])))


def _reflag21(K, res, ring, law, exclude, kw):
    """Re-run the count-law flagging against the cell model + the elliptical ring correction at each cell's centre."""
    from .law import ALPHA_PX, ALPHA_WIN
    plan = res.plan
    cy, cx = plan.geometry.centre
    ph = (np.arange(plan.n_sect) + 0.5) / plan.n_sect * 2 * np.pi - np.pi
    rr = plan.ring_r[:, None]
    y = cy + rr * np.sin(ph)[None, :]; x = cx + rr * np.cos(ph)[None, :]
    i = ring.ring_index(y, x)
    add = np.where(i >= 0, ring.c[np.maximum(i, 0)], 0.0)
    bc = (np.maximum(res.b_cells, 0.0) + add).ravel()
    kpx = law.kstar(bc, kw.get("alpha_px", ALPHA_PX), 1); tw = law.kstar(bc, kw.get("alpha_win", ALPHA_WIN), 9)
    base = plan.valid.copy()
    if exclude is not None:
        base &= ~np.asarray(exclude, bool)
    N = max(K.shape)
    r_dil = kw.get("r_dil") or 2.0 * N / 1024.0
    f0 = KN.flag(K, plan.cellmap2, base, kpx, tw, plan.use_numba)
    res.flags = KN.dilate(f0, r_dil, plan.use_numba)
    res.used = base & ~res.flags
