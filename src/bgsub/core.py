# MOVED from pod/background/src/podbg/bgsub/core.py at background commit f1e93f5 (file last changed in fd2d10f; sha256 7de8485f926a0af25e4d57e7d62f5213d85f55f12f930d437b75c335f8eeb25c).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""bgsub v0 core: the global linear smooth background model G (design note 2026-11-04_bgsub_algorithm_v0.md Sec. 1.3-1.6).

b(r, phi) = sum_k a_k B_k(r)                                  fine radial cubic B-spline (step h_r)
          + sum_{m<=M} sum_j A_j(r) [alpha cos m phi + beta sin m phi]   harmonics on a coarse radial spline (step H)
          + d1c R'(r) cos phi + d1s R'(r) sin phi              centre-error columns
          + d2c r R'(r) cos 2phi + d2s r R'(r) sin 2phi         ellipticity columns

fitted in STORED counts by weighted linear least squares (weights 1 / g(b) of the count law, from the other
checkerboard half: cross-fitting), on sufficient statistics of polar cells (1-px rings x n_sect sectors x 2 halves),
with iterative one-sided exclusion of spots, hot pixels and zingers under the count law.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline

from . import _kernels as KN
from .law import ALPHA_PX, ALPHA_WIN, CountLaw


# ============================================================================= geometry
@dataclass
class Geometry:
    """Beam centre (y, x) in pixels; optional ellipticity (e, phi0): r = rho (1 + e cos 2 (phi - phi0)); optional
    r_map (any radial coordinate map in px, e.g. s / k) which then replaces rho."""
    centre: tuple
    ellipse: tuple | None = None
    r_map: np.ndarray | None = None


def _bspline_design(x, knots, deriv=0):
    nb = len(knots) - 4
    sp = BSpline(knots, np.eye(nb), 3, extrapolate=False)
    D = sp(x, nu=deriv) if deriv else sp(x)
    return np.nan_to_num(np.asarray(D, float))


def _uniform_knots(lo, hi, h):
    n_int = max(int(np.ceil((hi - lo) / h)), 1)
    return lo + h * np.arange(-3, n_int + 4), n_int + 3


@dataclass
class Plan:
    """Everything that depends only on the frame shape, the static mask and the geometry (cache and reuse it)."""
    shape: tuple
    geometry: Geometry
    h_r: float
    H: float
    M: int
    n_sect: int
    r_lo: float
    r_hi: float
    n_r: int
    cellmap2: np.ndarray            # int32 (ny, nx): (ring * n_sect + sector) * 2 + parity, -1 where invalid
    valid: np.ndarray               # static valid pixels
    rmap: np.ndarray                # float32 radius of every pixel
    phimap: np.ndarray              # float32 azimuth of every pixel
    ring_r: np.ndarray              # mean radius of each 1-px ring (static valid pixels)
    E: np.ndarray                   # (n_r, n_sect, 2M + 1): cell means of [1, cos phi, sin phi, ..., cos M phi, sin M phi]
    knots_r: np.ndarray
    knots_a: np.ndarray
    keep_r: np.ndarray              # kept fine radial basis indices
    keep_a: np.ndarray              # kept coarse radial basis indices
    Br: np.ndarray                  # (n_r, Kr) fine radial design at ring radii
    dBr: np.ndarray                 # derivative
    Ar: np.ndarray                  # (n_r, Ka) coarse radial design
    eval_dr: float
    n_phi_eval: int
    r_eval: np.ndarray
    Bre: np.ndarray
    dBre: np.ndarray
    Are: np.ndarray
    trig_eval: np.ndarray           # (2M + 1, n_phi_eval)
    use_numba: bool = True
    timings: dict = field(default_factory=dict)
    AA: np.ndarray | None = None
    version: str = "v0"
    zone: object = None             # v1: zone.ZonePlan (penumbra + near-zone harmonics), None in v0
    kw: dict = field(default_factory=dict)   # make_plan keywords (v2 re-planning after re-centring)

    @property
    def n_cells(self):
        return self.n_r * self.n_sect

    @property
    def Kr(self):
        return self.Br.shape[1]

    @property
    def Ka(self):
        return self.Ar.shape[1]


def make_plan(shape, mask, geometry: Geometry, h_r: float | None = None, H_factor: float = 12.0, M: int = 6,
              n_sect: int = 64, eval_dr: float = 0.5, n_phi_eval: int = 256, use_numba: bool = True, version: str = "v1",
              stop=None, zone_kw: dict | None = None) -> Plan:
    """mask: True = bad (beam stop, arm, seams, dead pixels, ...).  version "v1" (default) adds the near-stop block
    (zone.py: penumbra columns about `stop`, derived from the mask if None, and near-zone harmonics); "v0" is the v0
    plan exactly; "v2" and "v2.1" are the v1 plan (their steps, refine.py, act after the fit)."""
    if version not in ("v0", "v1", "v2", "v2.1"):
        raise ValueError(f"version {version!r}: 'v0', 'v1', 'v2' or 'v2.1'")
    t0 = time.perf_counter()
    ny, nx = shape
    N = max(ny, nx)
    h_r = float(max(3.0, N / 256.0)) if h_r is None else float(h_r)
    H = H_factor * h_r
    cy, cx = float(geometry.centre[0]), float(geometry.centre[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    dy = yy - cy; dx = xx - cx
    phi = np.arctan2(dy, dx)
    if geometry.r_map is not None:
        r = np.asarray(geometry.r_map, float)
    else:
        r = np.hypot(dy, dx)
        if geometry.ellipse is not None:
            e, ph0 = geometry.ellipse
            r = r * (1.0 + e * np.cos(2.0 * (phi - ph0)))
    valid = ~np.asarray(mask, bool) & np.isfinite(r)
    rv = r[valid]
    r_lo = float(np.floor(rv.min())); r_hi = float(rv.max()) + 1e-6
    n_r = int(np.ceil(r_hi - r_lo))
    ri = np.clip(np.floor(r - r_lo).astype(np.int64), 0, n_r - 1)
    si = np.clip(np.floor((phi + np.pi) / (2 * np.pi) * n_sect).astype(np.int64), 0, n_sect - 1)
    par = (yy + xx) & 1
    cell = ri * n_sect + si
    cellmap2 = np.where(valid, cell * 2 + par, -1).astype(np.int32)
    # cell means of the trig functions and ring mean radii over static valid pixels
    cv = cell[valid]
    ncell = np.bincount(cv, minlength=n_r * n_sect).astype(float)
    E = np.zeros((n_r * n_sect, 2 * M + 1))
    E[:, 0] = 1.0
    pv = phi[valid]
    sector_centre = ((np.arange(n_r * n_sect) % n_sect) + 0.5) / n_sect * 2 * np.pi - np.pi
    for m in range(1, M + 1):
        c = np.bincount(cv, weights=np.cos(m * pv), minlength=n_r * n_sect)
        s = np.bincount(cv, weights=np.sin(m * pv), minlength=n_r * n_sect)
        E[:, 2 * m - 1] = np.where(ncell > 0, c / np.maximum(ncell, 1), np.cos(m * sector_centre))
        E[:, 2 * m] = np.where(ncell > 0, s / np.maximum(ncell, 1), np.sin(m * sector_centre))
    E = E.reshape(n_r, n_sect, 2 * M + 1)
    rr = ri[valid]
    nring = np.bincount(rr, minlength=n_r).astype(float)
    ring_r = np.where(nring > 0, np.bincount(rr, weights=rv, minlength=n_r) / np.maximum(nring, 1), r_lo + np.arange(n_r) + 0.5)
    # radial bases
    kn_r, _ = _uniform_knots(r_lo, r_hi, h_r)
    kn_a, _ = _uniform_knots(r_lo, r_hi, H)
    Br = _bspline_design(ring_r, kn_r); dBr = _bspline_design(ring_r, kn_r, 1); Ar = _bspline_design(ring_r, kn_a)
    supp_r = (nring[:, None] * (Br > 0)).sum(0); supp_a = (nring[:, None] * (Ar > 0)).sum(0)
    keep_r = np.flatnonzero(supp_r >= 20); keep_a = np.flatnonzero(supp_a >= 20 * (2 * M))
    r_eval = r_lo + eval_dr * np.arange(int(np.ceil((r_hi - r_lo) / eval_dr)) + 2)
    Bre = _bspline_design(r_eval, kn_r)[:, keep_r]; dBre = _bspline_design(r_eval, kn_r, 1)[:, keep_r]; Are = _bspline_design(r_eval, kn_a)[:, keep_a]
    ph_e = (np.arange(n_phi_eval)) / n_phi_eval * 2 * np.pi - np.pi
    trig = [np.ones(n_phi_eval)]
    for m in range(1, M + 1):
        trig += [np.cos(m * ph_e), np.sin(m * ph_e)]
    plan = Plan(shape=(ny, nx), geometry=geometry, h_r=h_r, H=H, M=M, n_sect=n_sect, r_lo=r_lo, r_hi=r_hi, n_r=n_r,
                cellmap2=cellmap2, valid=valid, rmap=r.astype(np.float32), phimap=phi.astype(np.float32), ring_r=ring_r, E=E,
                knots_r=kn_r, knots_a=kn_a, keep_r=keep_r, keep_a=keep_a, Br=Br[:, keep_r], dBr=dBr[:, keep_r], Ar=Ar[:, keep_a],
                eval_dr=eval_dr, n_phi_eval=n_phi_eval, r_eval=r_eval, Bre=Bre, dBre=dBre, Are=Are, trig_eval=np.array(trig),
                use_numba=use_numba)
    plan.AA = (plan.Ar[:, :, None] * plan.Ar[:, None, :]).reshape(plan.n_r, -1)
    plan.version = version
    plan.kw = {"h_r": h_r, "H_factor": H_factor, "M": M, "n_sect": n_sect, "eval_dr": eval_dr, "n_phi_eval": n_phi_eval,
               "use_numba": use_numba, "stop": stop, "zone_kw": zone_kw}
    if version in ("v1", "v2", "v2.1"):
        from .zone import make_zone
        plan.zone = make_zone(plan, mask, stop=stop, **(zone_kw or {}))
    plan.timings["plan_s"] = time.perf_counter() - t0
    return plan


# ============================================================================= the linear model on cells
RIDGE_HARM = 1e-3
DISP_MIN_COUNTS = 5.0
DISP_MIN_CELLS = 200
D_E = (1, 2, 3, 4)          # e-index of the derivative columns: cos phi, sin phi, cos 2phi, sin 2phi


class _Design:
    """Column bookkeeping: group e = 0 (fine radial B), e = 1..2M (coarse A [+ one derivative column for e <= 4])."""

    def __init__(self, plan: Plan, Rp_ring: np.ndarray | None):
        self.plan = plan
        self.with_d = Rp_ring is not None
        ne = 2 * plan.M + 1
        self.G = [plan.Br]
        for e in range(1, ne):
            g = plan.Ar
            if self.with_d and e in D_E:
                f = Rp_ring if e in (1, 2) else plan.ring_r * Rp_ring
                g = np.column_stack([g, f])
            self.G.append(g)
        self.sizes = [g.shape[1] for g in self.G]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)])
        self.p = int(self.off[-1])

    def ext(self, delta):
        """Cell channels [1, trig..., delta]: delta = (used-pixel mean radius of the cell) - ring radius, so that the fine
        radial basis is evaluated at each cell's own mean radius to first order (B(r_i) + B'(r_i) delta)."""
        return np.concatenate([self.plan.E, delta[..., None]], axis=2)

    def normal(self, w, u, delta):
        """F = X^T diag(w) X and r = X^T u for cell weights w (n_r, n_sect) and u (n_r, n_sect)."""
        E = self.ext(delta)
        WEE = np.matmul((E * w[..., None]).transpose(0, 2, 1), E)          # (n_r, ne, ne)
        UE = np.einsum("is,ise->ie", u, E)
        return self._assemble(WEE), self._rhs(UE)

    def _rhs(self, UE):
        rhs = np.empty(self.p)
        for a in range(UE.shape[1] - 1):
            rhs[self.off[a]:self.off[a + 1]] = self.G[a].T @ UE[:, a]
        rhs[:self.plan.Kr] += self.plan.dBr.T @ UE[:, -1]
        return rhs

    def _assemble(self, WEE):
        """Blocks of X^T W X: (0,0) fine radial, (0,e) and (e,f) through two BLAS products over the shared coarse
        basis, then the derivative columns (one vector each)."""
        plan = self.plan
        Br, Ar = plan.Br, plan.Ar
        dBr = plan.dBr
        n_r, ne1, _ = WEE.shape
        ne = ne1 - 1; dl = ne                      # index of the delta channel
        Kr, Ka = Br.shape[1], Ar.shape[1]
        F = np.zeros((self.p, self.p))
        X0 = Br.T @ (WEE[:, dl, 0][:, None] * dBr)
        F[:Kr, :Kr] = Br.T @ (WEE[:, 0, 0][:, None] * Br) + X0 + X0.T + dBr.T @ (WEE[:, dl, dl][:, None] * dBr)
        # (0, e): Br^T diag(WEE[:,0,e]) Ar for e = 1..ne-1
        T = (WEE[:, 0, 1:ne, None] * Ar[:, None, :]).reshape(n_r, (ne - 1) * Ka)
        Td = (WEE[:, dl, 1:ne, None] * Ar[:, None, :]).reshape(n_r, (ne - 1) * Ka)
        C0 = (Br.T @ T + dBr.T @ Td).reshape(Kr, ne - 1, Ka)
        # (e, f): Ar^T diag(WEE[:,e,f]) Ar
        AA = plan.AA if getattr(plan, "AA", None) is not None else (Ar[:, :, None] * Ar[:, None, :]).reshape(n_r, Ka * Ka)
        Cef = (np.ascontiguousarray(WEE[:, 1:ne, 1:ne]).reshape(n_r, -1).T @ AA).reshape(ne - 1, ne - 1, Ka, Ka)
        for e in range(1, ne):
            se = slice(self.off[e], self.off[e] + Ka)
            F[:Kr, se] = C0[:, e - 1]; F[se, :Kr] = C0[:, e - 1].T
            for f in range(1, ne):
                F[se, self.off[f]:self.off[f] + Ka] = Cef[e - 1, f - 1]
        if self.with_d:
            # derivative columns: column vector d_e (n_r,) attached to group e
            for e in D_E:
                j = self.off[e] + Ka
                de = self.G[e][:, Ka]
                for f in range(ne):
                    Gf = self.G[f]
                    v = Gf.T @ (WEE[:, e, f] * de)
                    if f == 0:
                        v = v + dBr.T @ (WEE[:, e, dl] * de)
                    F[j, self.off[f]:self.off[f + 1]] = v
                    F[self.off[f]:self.off[f + 1], j] = v
        return F

    def meat(self, w2, delta):
        E = self.ext(delta)
        return self._assemble(np.matmul((E * w2[..., None]).transpose(0, 2, 1), E))

    def penalty(self, F, lam_rel):
        P = np.zeros_like(F)
        for e, g in enumerate(self.G):
            k = self.plan.Kr if e == 0 else self.plan.Ka
            if k < 3:
                continue
            D2 = np.diff(np.eye(k), 2, axis=0)
            sl = slice(self.off[e], self.off[e] + k)
            d = np.diag(F)[sl]
            scale = float(np.mean(d[d > 0])) if np.any(d > 0) else 1.0
            P[sl, sl] += lam_rel * scale * (D2.T @ D2)
            if e > 0:
                # ridge on the harmonic coefficients (Amendment 2): shrinks harmonics to zero only where the local
                # information is far below the block mean (partly covered outer rings, frame corners)
                P[sl, sl] += RIDGE_HARM * scale * np.eye(k)
        # numerical safety ridge
        dd = np.diag(F)
        P[np.diag_indices_from(P)] += 1e-10 * (float(np.mean(dd[dd > 0])) if np.any(dd > 0) else 1.0) + (dd <= 0) * 1e-6
        return P

    def coef_functions(self, theta, which="ring", Rp=None):
        """Q[i, e]: the radial function multiplying E_e, on rings (which='ring') or the fine eval grid ('eval')."""
        plan = self.plan
        ne = 2 * plan.M + 1
        if which == "ring":
            Bm, Am, rr = plan.Br, plan.Ar, plan.ring_r
        else:
            Bm, Am, rr = plan.Bre, plan.Are, plan.r_eval
        Q = np.zeros((len(rr), ne))
        Q[:, 0] = Bm @ theta[self.off[0]:self.off[1]]
        Ka = plan.Ka
        for e in range(1, ne):
            th = theta[self.off[e]:self.off[e + 1]]
            Q[:, e] = Am @ th[:Ka]
            if self.with_d and e in D_E:
                f = Rp if e in (1, 2) else rr * Rp
                Q[:, e] += th[Ka] * f
        return Q

    def rows_cells(self, ring, sect, delta_c):
        """Exact design rows of the polar CELLS (ring, sect) as used by normal(): (n, p)."""
        plan = self.plan
        X = np.zeros((len(ring), self.p))
        X[:, self.off[0]:self.off[1]] = plan.Br[ring] + plan.dBr[ring] * delta_c[:, None]
        Ka = plan.Ka
        Ec = plan.E[ring, sect]
        for e in range(1, 2 * plan.M + 1):
            X[:, self.off[e]:self.off[e] + Ka] = plan.Ar[ring] * Ec[:, e:e + 1]
            if self.with_d and e in D_E:
                X[:, self.off[e] + Ka] = self.G[e][ring, Ka] * Ec[:, e]
        return X

    def cross(self, ring, sect, delta_c, w_c, Z):
        """X0^T diag(w) Z over the cells (ring, sect), without forming X0 (ring aggregation of the cell channels)."""
        plan = self.plan
        ne = 2 * plan.M + 1
        q = Z.shape[1]
        Ec = plan.E[ring, sect]                                           # (n, ne)
        wz = w_c[:, None] * Z                                             # (n, q)
        T = np.concatenate([Ec[:, :, None] * wz[:, None, :], (delta_c[:, None] * wz)[:, None, :]], axis=1)   # (n, ne + 1, q)
        S = getattr(plan.zone, "_ring_agg", None) if plan.zone is not None else None
        if S is None or S.shape[1] != len(ring):
            from scipy import sparse
            S = sparse.csr_matrix((np.ones(len(ring)), (ring, np.arange(len(ring)))), shape=(plan.n_r, len(ring)))
            if plan.zone is not None:
                plan.zone._ring_agg = S
        A = np.asarray(S @ T.reshape(len(ring), -1)).reshape(plan.n_r, ne + 1, q)
        out = np.zeros((self.p, q))
        out[self.off[0]:self.off[1]] = plan.Br.T @ A[:, 0] + plan.dBr.T @ A[:, ne]
        Ka = plan.Ka
        for e in range(1, ne):
            out[self.off[e]:self.off[e] + Ka] = plan.Ar.T @ A[:, e]
            if self.with_d and e in D_E:
                out[self.off[e] + Ka] = self.G[e][:, Ka] @ A[:, e]
        return out

    def rows(self, r, phi, Rp_at_r):
        """Exact design rows for arbitrary pixels (r, phi): (n, p)."""
        plan = self.plan
        Bm = _bspline_design(r, plan.knots_r)[:, plan.keep_r]
        Am = _bspline_design(r, plan.knots_a)[:, plan.keep_a]
        X = np.zeros((len(r), self.p))
        X[:, self.off[0]:self.off[1]] = Bm
        Ka = plan.Ka
        for e in range(1, 2 * plan.M + 1):
            m = (e + 1) // 2
            t = np.cos(m * phi) if e % 2 == 1 else np.sin(m * phi)
            X[:, self.off[e]:self.off[e] + Ka] = Am * t[:, None]
            if self.with_d and e in D_E:
                f = Rp_at_r if e in (1, 2) else r * Rp_at_r
                X[:, self.off[e] + Ka] = f * t
        return X


@dataclass
class _HalfFit:
    design: _Design
    theta: np.ndarray
    F: np.ndarray
    chol: tuple
    Rp_coef: np.ndarray | None      # fine-radial coefficients of the OTHER half whose R' feeds the derivative columns
    w: np.ndarray                   # weights used (n_r, n_sect)
    gw: np.ndarray                  # g used in the weights
    delta: np.ndarray = None        # cell mean radius offsets of the used pixels
    Z: np.ndarray | None = None     # v1: (n_cz, q) near-stop columns on the cz cells (this half)
    X0z: tuple | None = None        # v1: (ring, sect, delta) of the cz cells (the v0 rows there, implicitly)
    Rpen_coef: np.ndarray | None = None   # v1: fine-radial coefficients of R(r) in the penumbra columns (other half)
    p0: int = 0                     # number of v0 columns (theta[:p0])


def _cells_model(design: _Design, theta, plan: Plan, Rp_ring, hf=None):
    Q = design.coef_functions(theta[:design.p], "ring", Rp_ring)
    b = np.einsum("ie,ise->is", Q, plan.E)
    if hf is not None and hf.Z is not None and plan.zone is not None and plan.zone.q:
        from .zone import z_cells
        zp = plan.zone
        Rr = plan.Br @ hf.Rpen_coef if hf.Rpen_coef is not None else np.zeros(plan.n_r)
        Zs = z_cells(zp, Rr, zp.pen_static)
        bf = b.ravel().copy()
        bf[zp.cz] += Zs @ theta[design.p:]
        b = bf.reshape(b.shape)
    return b


def _fit_half(plan, S, n, delta, g_w, Rp_coef, lam_rel, Z=None, Rpen_coef=None):
    Rp_ring = None if Rp_coef is None else plan.dBr @ Rp_coef
    des = _Design(plan, Rp_ring)
    w = n / g_w
    u = S / g_w
    F, rhs = des.normal(w, u, delta)
    P = des.penalty(F, lam_rel)
    X0z = None
    if Z is not None and Z.shape[1]:
        from .zone import z_penalty
        zp = plan.zone
        dz = delta.ravel()[zp.cz]
        X0z = (zp.cz_ring, zp.cz_sect, dz)                       # cell keys; X0^T W Z by ring aggregation (_Design.cross)
        wz = w.ravel()[zp.cz]; uz = u.ravel()[zp.cz]
        F0z = des.cross(zp.cz_ring, zp.cz_sect, dz, wz, Z); Fzz = Z.T @ (wz[:, None] * Z)
        p0 = des.p
        Ff = np.zeros((p0 + Z.shape[1],) * 2)
        Ff[:p0, :p0] = F; Ff[:p0, p0:] = F0z; Ff[p0:, :p0] = F0z.T; Ff[p0:, p0:] = Fzz
        Pf = np.zeros_like(Ff); Pf[:p0, :p0] = P; Pf[p0:, p0:] = z_penalty(zp, Fzz, lam_rel, RIDGE_HARM)
        F, P = Ff, Pf
        rhs = np.concatenate([rhs, Z.T @ uz])
    A = F + P
    try:
        ch = linalg.cho_factor(A, lower=True, check_finite=False)
        theta = linalg.cho_solve(ch, rhs, check_finite=False)
    except linalg.LinAlgError:
        A = A + np.eye(len(A)) * 1e-6 * np.mean(np.abs(np.diag(A)))
        ch = linalg.cho_factor(A, lower=True, check_finite=False)
        theta = linalg.cho_solve(ch, rhs, check_finite=False)
    return _HalfFit(des, theta, A, ch, Rp_coef, w, g_w, delta, Z=Z if X0z is not None else None, X0z=X0z,
                    Rpen_coef=Rpen_coef, p0=des.p)


# ============================================================================= result
@dataclass
class BackgroundResult:
    plan: Plan
    law: CountLaw
    halves: list                    # [_HalfFit A, _HalfFit B]
    b_cells: np.ndarray             # (n_r, n_sect) final model on cells (average of halves)
    bg: np.ndarray                  # float32 (ny, nx) stored counts, NaN outside [r_lo, r_hi]
    used: np.ndarray                # pixels used in the final fit
    flags: np.ndarray               # pixels flagged as outliers (after dilation)
    dispersion: float               # Pearson phi on cells
    timings: dict
    _cov: list | None = None
    ring: object = None             # v2: refine.RingRefinement (fine radial correction about the refined centre)
    centre_info: dict | None = None  # v2: re-centring record

    def __iter__(self):             # bg, var = estimate_background(...)
        yield self.bg
        yield self.var_map()

    # -------------------------------------------------------------- covariance (sandwich, lazy)
    def covariances(self):
        if self._cov is None:
            out = []
            for hf, (S, n, _d) in zip(self.halves, self._final_stats):
                gfin = np.maximum(self.law.g(np.maximum(self.b_cells, 0.0)), self._gfloor)
                w2 = n * gfin / hf.gw ** 2
                Mt = hf.design.meat(w2, hf.delta)
                if hf.Z is not None:
                    zp = self.plan.zone
                    w2z = w2.ravel()[zp.cz]
                    M0z = hf.design.cross(hf.X0z[0], hf.X0z[1], hf.X0z[2], w2z, hf.Z); Mzz = hf.Z.T @ (w2z[:, None] * hf.Z)
                    p0 = hf.p0
                    Mf = np.zeros((p0 + hf.Z.shape[1],) * 2)
                    Mf[:p0, :p0] = Mt; Mf[:p0, p0:] = M0z; Mf[p0:, :p0] = M0z.T; Mf[p0:, p0:] = Mzz
                    Mt = Mf
                Finv_M = linalg.cho_solve(hf.chol, Mt, check_finite=False)
                C = linalg.cho_solve(hf.chol, Finv_M.T, check_finite=False)
                out.append(self.dispersion * 0.5 * (C + C.T))
            self._cov = out
        return self._cov

    def _Rp_at(self, hf, r):
        if hf.Rp_coef is None:
            return np.zeros_like(r)
        return _bspline_design(r, self.plan.knots_r, 1)[:, self.plan.keep_r] @ hf.Rp_coef

    def _rows(self, hf, r, phi, pix=None):
        X = hf.design.rows(r, phi, self._Rp_at(hf, r))
        if hf.Z is not None:
            from .zone import z_rows
            Rpen = (_bspline_design(r, self.plan.knots_r)[:, self.plan.keep_r] @ hf.Rpen_coef) if hf.Rpen_coef is not None else np.zeros_like(r)
            X = np.concatenate([X, z_rows(self.plan.zone, self.plan, r, phi, pix, Rpen)], axis=1)
        return X

    def design_sum(self, r, phi, groups, weights=None, pix=None):
        """For pixel groups (list of index arrays into r/phi) return per half the (weighted) summed design rows.
        pix: flat pixel indices of r/phi (v1: needed for the penumbra columns; without them the penumbra is ignored)."""
        out = []
        for hf in self.halves:
            X = self._rows(hf, r, phi, pix)
            if weights is not None:
                X = X * np.asarray(weights, float)[:, None]
            Sg = np.stack([X[g].sum(0) for g in groups]) if len(groups) else np.zeros((0, X.shape[1]))
            out.append(Sg)
        return out

    def box_sums(self, r, phi, groups, weights=None, pix=None):
        """(sum of (weights x) b^ over each pixel group, its variance) from the exact pixel-level model."""
        sums = self.design_sum(r, phi, groups, weights, pix)
        B = 0.5 * (sums[0] @ self.halves[0].theta + sums[1] @ self.halves[1].theta)
        C = self.covariances()
        V = 0.25 * (np.einsum("gi,ij,gj->g", sums[0], C[0], sums[0]) + np.einsum("gi,ij,gj->g", sums[1], C[1], sums[1]))
        if self.ring is not None:
            dB, dV = self.ring.box_terms(self.plan, r, phi, groups, weights)
            B = B + dB; V = V + dV
        return B, V

    def var_map(self, dr_px: float = 8.0, n_phi: int = 32):
        """Variance of the background ESTIMATE per pixel (stored counts^2), bilinear from a coarse polar grid."""
        plan = self.plan
        rg = np.arange(plan.r_lo, plan.r_hi + dr_px, dr_px)
        pg = np.arange(n_phi) / n_phi * 2 * np.pi - np.pi
        R, PH = np.meshgrid(rg, pg, indexing="ij")
        C = self.covariances()
        V = np.zeros(R.size)
        for hf, Ck in zip(self.halves, C):
            X = self._rows(hf, R.ravel(), PH.ravel(), None)
            V += 0.25 * np.einsum("ni,ij,nj->n", X, Ck, X)
        V = V.reshape(R.shape)
        return KN.eval_map(plan.rmap, plan.phimap, V, plan.r_lo, dr_px, n_phi, plan.use_numba)

    def pixel_var(self, b=None):
        """Variance of a single stored value at the model mean (dispersion included)."""
        b = self.bg if b is None else b
        return self.dispersion * self.law.g(np.maximum(b, 0.0))


# ============================================================================= the estimator
def _ring_pilot_thresholds(plan, S, n):
    Sr = S.reshape(plan.n_r, plan.n_sect, 2).sum((1, 2)); nr = n.reshape(plan.n_r, plan.n_sect, 2).sum((1, 2))
    m = np.where(nr > 0, Sr / np.maximum(nr, 1), 0.0)
    return np.repeat(m + 10 * np.sqrt(np.maximum(m, 0)) + 10, plan.n_sect)


DEFAULT_VERSION = "v1"


def estimate_background(frame, mask=None, geometry: Geometry | tuple | None = None, detector: CountLaw | None = None, *,
                        exclude=None, plan: Plan | None = None, version: str | None = None, recentre: bool = True,
                        ring_refine: bool = True, **kw) -> BackgroundResult:
    """Background of one frame in stored counts (see _estimate_core for the keywords).

    version "v2" (design note 2026-11-05_bgsub_algorithm_v2.md): the v1 fit, then (i) RE-CENTRING when the derivative
    columns imply a beam-centre error above 2 px x N / 1024 (Gauss-Newton on the plan centre, re-planned), and (ii) the
    RING REFINEMENT (a shrunken 1-px radial correction about the refined centre, refine.py). "v2.1" (design note
    2026-11-05_bgsub_algorithm_v2_1.md): as v2 with the ring refinement taken on ELLIPTICAL rings (ellipticity from the
    fit's own d2 columns) and re-centred skipped ring means (refine.ring_refinement_v21). "v1" / "v0": unchanged."""
    if plan is not None and version is not None and version != plan.version:
        raise ValueError(f"version {version!r} differs from the plan's {plan.version!r}")
    ver = plan.version if plan is not None else (version or DEFAULT_VERSION)
    if ver in ("v2", "v2.1"):
        from .refine import estimate_v2
        return estimate_v2(frame, mask, geometry, detector, exclude=exclude, plan=plan, recentre=recentre,
                           ring_refine=ring_refine, version=ver, **kw)
    return _estimate_core(frame, mask, geometry, detector, exclude=exclude, plan=plan, version=ver, **kw)


def _estimate_core(frame, mask=None, geometry: Geometry | tuple | None = None, detector: CountLaw | None = None, *,
                   exclude=None, plan: Plan | None = None, n_flag_rounds: int = 2, alpha_px: float = ALPHA_PX,
                   alpha_win: float = ALPHA_WIN, r_dil: float | None = None, lam_rel: float = 1e-4,
                   crossfit: bool = True, flagging: bool = True, with_derivative_columns: bool = True,
                   make_map: bool = True, use_numba: bool = True, version: str | None = None, **plan_kw) -> BackgroundResult:
    """Background of one frame in stored counts.

    frame: 2-D stored values (minus detector.offset, the DECLARED zero level; bg is returned WITHOUT the offset). mask: True = bad (stop, arm, seams, dead). geometry: Geometry or (cy, cx).
    detector: CountLaw (default Poisson gain 1). exclude: True = pixels to leave out of the fit without flagging (the
    discs of spots to be integrated). plan: a cached Plan for this shape / mask / geometry (built if None).
    version: "v1" (default; near-stop block, design note v1 Sec. 1) or "v0" (bit-identical v0). With a plan, the plan's
    version is used (a conflicting `version` raises).
    Returns a BackgroundResult (iterable as (bg, var)).
    """
    t0 = time.perf_counter()
    if plan is not None and version is not None and version != plan.version:
        raise ValueError(f"version {version!r} differs from the plan's {plan.version!r}")
    K = np.asarray(frame)
    if not np.issubdtype(K.dtype, np.floating):
        K = K.astype(np.float64)
    law = CountLaw.poisson(1.0) if detector is None else detector
    if law.offset:
        K = K - law.offset                                   # declared zero level (Amendment 1 b)
    if geometry is None:
        geometry = Geometry((K.shape[0] / 2.0, K.shape[1] / 2.0))
    elif not isinstance(geometry, Geometry):
        geometry = Geometry(tuple(geometry))
    if mask is None:
        mask = ~np.isfinite(K)
    if plan is None:
        plan = make_plan(K.shape, mask, geometry, use_numba=use_numba, version=version or "v1", **plan_kw)
    zp = plan.zone if plan.version in ("v1", "v2", "v2.1") else None
    if zp is not None and zp.q == 0:
        zp = None
    un = plan.use_numba and use_numba
    N = max(K.shape)
    r_dil = 2.0 * N / 1024.0 if r_dil is None else r_dil
    base = plan.valid.copy()
    if exclude is not None:
        base &= ~np.asarray(exclude, bool)
    Kf = np.where(np.isfinite(K), K, 0.0)
    nc2 = plan.n_cells * 2
    tim = {}

    def stats(used):
        S, n, Rs = KN.accumulate(Kf, plan.cellmap2, used, nc2, un, plan.rmap)
        S = S.reshape(plan.n_r, plan.n_sect, 2); n = n.reshape(plan.n_r, plan.n_sect, 2); Rs = Rs.reshape(plan.n_r, plan.n_sect, 2)
        d = np.where(n > 0, Rs / np.maximum(n, 1) - plan.ring_r[:, None, None], 0.0)
        return [(S[..., 0], n[..., 0], d[..., 0]), (S[..., 1], n[..., 1], d[..., 1])]

    # ---- pilot: ring-wise zinger clip, then OLS (unit weights) per half
    t = time.perf_counter()
    st = stats(base)
    Sall = np.stack([st[0][0], st[1][0]], -1).ravel(); nall = np.stack([st[0][1], st[1][1]], -1).ravel()
    thr = _ring_pilot_thresholds(plan, Sall, nall)
    flags = KN.flag(Kf, plan.cellmap2, base, thr, None, un) if flagging else np.zeros(K.shape, bool)
    flags = KN.dilate(flags, r_dil, un)
    used = base & ~flags
    if zp is not None and zp.n_pen:
        u0 = used.copy(); u0.ravel()[zp.pen_pix] = False        # v1: the pilot (no penumbra columns yet) skips the penumbra
        st = stats(u0)
    else:
        st = stats(used)
    tim["pilot_s"] = time.perf_counter() - t
    one = np.ones((plan.n_r, plan.n_sect))
    halves = [_fit_half(plan, S, n, d, one, None, lam_rel) for (S, n, d) in st]
    b_h = [_cells_model(h.design, h.theta, plan, None) for h in halves]
    b = 0.5 * (b_h[0] + b_h[1])
    bpos = b[b > 0]
    gfloor = float(law.g(np.array([max(1e-3 * (np.median(bpos) if bpos.size else 1.0), 1e-6)]))[0])
    tim["fit0_s"] = time.perf_counter() - t - tim["pilot_s"]
    for it in range(n_flag_rounds):
        t = time.perf_counter()
        if flagging:
            bc = b.ravel()
            kpx = law.kstar(bc, alpha_px, 1); tw = law.kstar(bc, alpha_win, 9)
            f0 = KN.flag(Kf, plan.cellmap2, base, kpx, tw, un)
            flags = KN.dilate(f0, r_dil, un)
            used = base & ~flags
            st = stats(used)
        tim[f"flag{it}_s"] = time.perf_counter() - t
        t = time.perf_counter()
        if not flagging and it == 0 and zp is not None and zp.n_pen:
            st = stats(used)                                     # v1: the penumbra pixels come back with their columns
        Pm = None
        if zp is not None and zp.n_pen:
            from .zone import pen_used_means
            Pm = pen_used_means(zp, used.ravel(), [st[0][1], st[1][1]])
        new = []
        for k, (S, n, d) in enumerate(st):
            other = halves[1 - k] if crossfit else halves[k]
            R0 = plan.Br @ other.theta[other.design.off[0]:other.design.off[1]]      # weights from the OTHER half's m = 0 part
            g_w = np.maximum(law.g(np.maximum(R0, 0.0)), gfloor)[:, None] * np.ones((1, plan.n_sect))
            Rp = other.theta[other.design.off[0]:other.design.off[1]] if with_derivative_columns else None
            if zp is not None:
                from .zone import z_cells
                Rc = other.theta[other.design.off[0]:other.design.off[1]].copy()
                Zk = z_cells(zp, plan.Br @ Rc, Pm[k] if Pm is not None else np.zeros((len(zp.cz), 0)))
                new.append(_fit_half(plan, S, n, d, g_w, Rp, lam_rel, Z=Zk, Rpen_coef=Rc))
            else:
                new.append(_fit_half(plan, S, n, d, g_w, Rp, lam_rel))
        halves = new
        b_h = [_cells_model(h.design, h.theta, plan, None if h.Rp_coef is None else plan.dBr @ h.Rp_coef, h) for h in halves]
        b = 0.5 * (b_h[0] + b_h[1])
        tim[f"fit{it + 1}_s"] = time.perf_counter() - t
    # ---- Pearson dispersion on cells
    num = 0.0; cnt = 0
    R0 = 0.5 * sum(plan.Br @ h.theta[h.design.off[0]:h.design.off[1]] for h in halves)      # radial (m = 0) model
    bring = (0.5 * np.maximum(R0, 0.0))[:, None] * np.ones((1, plan.n_sect))    # floor in b, not in g (g is not monotone at bin 16)
    n_all = 0
    for (S, n, _d), bh in zip(st, b_h):
        n_all += int((n > 0).sum())
        # Amendment 4: only cells with >= 5 expected stored counts (the Pearson mean is heavy-tailed and typically
        # far below 1 when K = 1 events are rare, e.g. bin 16 at archive doses)
        ok = (n > 0) & (n * b >= DISP_MIN_COUNTS)
        gg = np.maximum(law.g(np.maximum(b[ok], bring[ok])), gfloor)
        num += float(np.sum((S[ok] - n[ok] * b[ok]) ** 2 / (n[ok] * gg)))
        cnt += int(ok.sum())
    p = halves[0].design.p
    disp = num / max(cnt * (1.0 - p / max(n_all, 1)), 1) if cnt >= DISP_MIN_CELLS else 1.0
    # ---- map
    t = time.perf_counter()
    bg = None
    if make_map:
        grid = np.zeros((len(plan.r_eval), plan.n_phi_eval))
        for h in halves:
            Rp = None if h.Rp_coef is None else plan.dBre @ h.Rp_coef
            Q = h.design.coef_functions(h.theta[:h.design.p], "eval", Rp)
            grid += 0.5 * (Q @ plan.trig_eval)
            if h.Z is not None and zp.q_near:
                th = h.theta[h.p0 + zp.n_pen:].reshape(2 * zp.M_near, zp.K_near)
                grid += 0.5 * ((zp.Ane @ th.T) @ zp.trig_near_eval)
        bg = KN.eval_map(plan.rmap, plan.phimap, grid, plan.r_lo, plan.eval_dr, plan.n_phi_eval, un)
        if zp is not None and zp.n_pen and halves[0].Z is not None:
            add = np.zeros(len(zp.pen_pix))
            rp = plan.rmap.ravel()[zp.pen_pix].astype(float)
            for h in halves:
                Rr = np.interp(rp, plan.r_eval, plan.Bre @ h.Rpen_coef)      # R on the 0.5-px eval grid, linear in r
                tz = h.theta[h.p0:h.p0 + zp.n_pen]
                v = np.zeros(len(zp.pen_pix))
                for c_ in range(2):
                    ok = zp.pen_col[:, c_] >= 0
                    v[ok] += zp.pen_val[ok, c_] * tz[zp.pen_col[ok, c_]]
                add += 0.5 * Rr * v
            bgf = bg.ravel(); bgf[zp.pen_pix] += add.astype(np.float32)
    tim["map_s"] = time.perf_counter() - t
    tim["total_s"] = time.perf_counter() - t0
    res = BackgroundResult(plan, law, halves, b, bg, used, flags, disp, tim)
    res._final_stats = st
    res._gfloor = gfloor
    return res
