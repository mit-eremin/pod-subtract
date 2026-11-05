# MOVED from pod/background/src/podbg/bgsub/zone.py at background commit f1e93f5 (file last changed in 1f1796f; sha256 2e319759a3519db3a837e979660b851fbba481983c85b4f00aa9a1263836c291).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""bgsub v1: the near-stop block (design note 2026-11-04_bgsub_algorithm_v1.md Sec. 1, fix 1).

Two sets of extra columns are appended to the v0 design, both linear in the counts, so the unbiasedness theorem of
v0 (Sec. 1.3.1) carries over unchanged:

- PENUMBRA columns  R(r) phi_k(d) [pixel azimuth in sector j]:  d = Euclidean distance (px) of the pixel to the beam
  stop (disc + arm), phi_k linear hat functions on knots d = t_1 .. t_K (zero beyond t_K + step), j one of n_pen_sect
  azimuthal sectors about the beam centre (the arm's two sides and the disc rim shadow differently), R(r) the radial
  (m = 0) profile of the OTHER checkerboard half (cross-fitting, as the v0 derivative columns). They represent a
  multiplicative shadow transmission T_j(d) = 1 + sum_k c_jk phi_k(d) to first order.
- NEAR-ZONE harmonics  A_l(r) cos / sin (m phi), m = 1 .. M_near, on a radial cubic B-spline of step H_near that lives
  only inside r < r_zone (finer radial and azimuthal resolution next to the stop than the global m <= 6 on step H).

Cell values of the penumbra columns are means over the USED pixels of the cell (accumulated per half, so excluded
spot discs and flags do not bias them); the near-zone harmonics use static cell means as the v0 harmonics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class ZonePlan:
    stop: np.ndarray            # bool (ny, nx): the beam stop (+ arm) used for the penumbra
    dmap: np.ndarray            # float32 distance to the stop (px), inf without a stop
    # penumbra
    pen_pix: np.ndarray         # flat pixel indices (valid, d < t_K + step)
    pen_cell2: np.ndarray       # their cellmap2 values
    pen_col: np.ndarray         # (n_pix, 2) column indices of the two hats (-1 = none)
    pen_val: np.ndarray         # (n_pix, 2) hat values
    knots_d: np.ndarray
    n_pen_sect: int
    n_pen: int                  # number of kept penumbra columns
    # near-zone harmonics
    M_near: int
    r_zone: float
    zone_rings: np.ndarray      # ring indices inside the zone
    knots_n: np.ndarray
    keep_n: np.ndarray
    An: np.ndarray              # (n_zr, K_near) radial basis on zone rings
    Ane: np.ndarray             # (n_eval, K_near) on the eval grid (0 outside the zone)
    En: np.ndarray              # (n_zr, n_sect, 2 M_near) static cell means of cos m phi, sin m phi
    trig_near_eval: np.ndarray  # (2 M_near, n_phi_eval)
    # cell set of the Z block
    cz: np.ndarray              # cell indices (ring * n_sect + sect) carrying any Z column
    cz_ring: np.ndarray
    cz_sect: np.ndarray
    cz_zone_pos: np.ndarray     # position of each cz cell's ring in zone_rings (-1 outside)
    pen_static: np.ndarray      # (n_cz, n_pen) static-valid cell means of the penumbra hats (both parities)
    pen_cz_index: np.ndarray    # for every penumbra pixel: its row in cz
    pen_px_par: np.ndarray      # its checkerboard parity
    pen_keep_map: np.ndarray    # full column id (class * K + knot) -> kept column (-1 = dropped)
    pen_class_map: np.ndarray   # int8 (ny, nx): penumbra class of every pixel within reach (-1 elsewhere)

    @property
    def K_near(self):
        return self.An.shape[1]

    @property
    def q_near(self):
        return 2 * self.M_near * self.K_near

    @property
    def q(self):
        return self.n_pen + self.q_near


def derive_stop(mask, centre, open_px):
    """The beam stop (+ arm): connected components of the mask, opened with a (2 open_px + 1)^2 square (removes seams and
    1-3 px detector borders), that reach within 0.2 N of the beam centre."""
    m = np.asarray(mask, bool)
    k = 2 * int(open_px) + 1
    op = ndi.binary_opening(m, structure=np.ones((k, k), bool))
    if not op.any():
        return np.zeros_like(m)
    lab, n = ndi.label(op)
    ny, nx = m.shape
    yy, xx = np.indices(m.shape)
    near = np.hypot(yy - centre[0], xx - centre[1]) < 0.2 * max(ny, nx)
    keep = np.unique(lab[near & op])
    keep = keep[keep > 0]
    stop = np.isin(lab, keep)
    # give back the pixels removed by the opening that belong to the stop (its 1-px dilation within the mask)
    return m & ndi.binary_dilation(stop, iterations=int(open_px) + 1)


def pen_part_class(stop, pix, centre, sc):
    """Class of each penumbra pixel by the part of the stop that casts it: 0 = disc, 1 / 2 = the two sides of the arm
    (as seen from the beam centre). The disc is the opening of the stop with a disc of radius 0.6 x the stop's largest
    inscribed radius (removes the narrower arm); a pixel takes the part of its nearest stop pixel."""
    inner = ndi.distance_transform_edt(stop)
    ro = max(2.0, 0.6 * float(inner.max()))
    # opening with a disc of radius ro by two exact Euclidean distance transforms (erosion, then dilation)
    core = inner > ro
    disc = (ndi.distance_transform_edt(~core) <= ro) & stop if core.any() else np.zeros_like(stop)
    _, (iy, ix) = ndi.distance_transform_edt(~stop, return_indices=True)
    py, px = np.unravel_index(pix, stop.shape)
    ny_, nx_ = iy.ravel()[pix], ix.ravel()[pix]
    is_disc = disc[ny_, nx_]
    a_pix = np.arctan2(py - centre[0], px - centre[1]); a_stop = np.arctan2(ny_ - centre[0], nx_ - centre[1])
    side = np.angle(np.exp(1j * (a_pix - a_stop))) >= 0
    return np.where(is_disc, 0, np.where(side, 1, 2))


def make_zone(plan, mask, stop=None, pen_step=None, pen_kmax=None, n_pen_sect="parts", M_near=0, H_near=None,
              r_zone=None, min_px=20):
    from .core import _bspline_design, _uniform_knots
    ny, nx = plan.shape
    N = max(ny, nx)
    sc = N / 1024.0
    cy, cx = plan.geometry.centre
    valid = plan.valid
    if stop is None:
        stop = derive_stop(mask, (cy, cx), max(2, int(round(2 * sc))))
    stop = np.asarray(stop, bool)
    if stop.any():
        dmap = ndi.distance_transform_edt(~stop).astype(np.float32)
    else:
        dmap = np.full((ny, nx), np.inf, np.float32)
    # ---- penumbra hats
    step = max(1.0, sc) if pen_step is None else float(pen_step)
    kmax = 4 if pen_kmax is None else int(pen_kmax)          # Amendment 1 (dev): hats on d = 1..4 px (reach < 5 px)
    knots_d = step * np.arange(1, kmax + 1)
    dlim = knots_d[-1] + step if kmax > 0 else -1.0
    flat_valid = valid.ravel()
    dflat = dmap.ravel()
    pix = np.flatnonzero(flat_valid & (dflat < dlim) & (plan.cellmap2.ravel() >= 0))
    d = np.maximum(dflat[pix].astype(float), knots_d[0] if kmax > 0 else 0.0)
    u = (d - (knots_d[0] if kmax > 0 else 0.0)) / step
    k0 = np.floor(u).astype(int); t = u - k0
    phi = plan.phimap.ravel()[pix].astype(float)
    if n_pen_sect == "parts":
        j = pen_part_class(stop, pix, (cy, cx), sc)
        n_pen_sect = 3
    else:
        j = np.clip(np.floor((phi + np.pi) / (2 * np.pi) * n_pen_sect).astype(int), 0, n_pen_sect - 1)
    col = np.full((len(pix), 2), -1); val = np.zeros((len(pix), 2))
    a = k0 < kmax
    col[a, 0] = j[a] * kmax + k0[a]; val[a, 0] = 1.0 - t[a]
    b = (k0 + 1) < kmax
    col[b, 1] = j[b] * kmax + k0[b] + 1; val[b, 1] = t[b]
    # drop columns with too little support
    sup = np.zeros(n_pen_sect * kmax)
    for c_ in range(2):
        ok = col[:, c_] >= 0
        np.add.at(sup, col[ok, c_], (val[ok, c_] > 0.05).astype(float))
    keep_cols = np.flatnonzero(sup >= min_px)
    remap = np.full(n_pen_sect * kmax, -1); remap[keep_cols] = np.arange(len(keep_cols))
    col = np.where(col >= 0, remap[np.maximum(col, 0)], -1)
    val = np.where(col >= 0, val, 0.0)
    cmap = np.full(ny * nx, -1, np.int8); cmap[pix] = j; cmap = cmap.reshape(ny, nx)
    keep_pix = (col >= 0).any(1)
    pix, col, val = pix[keep_pix], col[keep_pix], val[keep_pix]
    cell2 = plan.cellmap2.ravel()[pix].astype(np.int64)
    n_pen = len(keep_cols)
    # ---- near-zone harmonics
    if r_zone is None:
        rs = plan.rmap[stop] if stop.any() else np.array([plan.r_lo])
        # outermost reach of the stop DISC (arm excluded: the 95th percentile of the stop radius within 0.15 N)
        rs = rs[rs < 0.15 * N] if np.any(rs < 0.15 * N) else rs
        r_zone = float(np.percentile(rs, 99)) + 48.0 * sc
    H_near = 4.0 * plan.h_r if H_near is None else float(H_near)
    zone_rings = np.flatnonzero(plan.ring_r < r_zone)
    if M_near > 0 and len(zone_rings) > 4:
        kn, _ = _uniform_knots(plan.r_lo, r_zone, H_near)
        An = _bspline_design(plan.ring_r[zone_rings], kn)
        # basis functions must vanish at the zone edge (no step at r_zone): drop those non-zero beyond r_zone - H_near
        edge = _bspline_design(np.array([r_zone - 1e-6]), kn)[0] > 0
        nring = np.bincount(np.clip(np.floor(plan.rmap[valid] - plan.r_lo).astype(int), 0, plan.n_r - 1), minlength=plan.n_r)[zone_rings]
        supp = (nring[:, None] * (An > 0)).sum(0)
        keep_n = np.flatnonzero(~edge & (supp >= min_px * 2 * M_near))
        An = An[:, keep_n]
        Ane = _bspline_design(np.clip(plan.r_eval, plan.r_lo, r_zone), kn)[:, keep_n] * (plan.r_eval < r_zone)[:, None]
        # static cell means of cos / sin m phi on zone cells
        ri = np.clip(np.floor(plan.rmap - plan.r_lo).astype(np.int64), 0, plan.n_r - 1)
        si = np.clip(np.floor((plan.phimap.astype(float) + np.pi) / (2 * np.pi) * plan.n_sect).astype(np.int64), 0, plan.n_sect - 1)
        inz = valid & (plan.ring_r[ri] < r_zone)
        zpos = np.full(plan.n_r, -1); zpos[zone_rings] = np.arange(len(zone_rings))
        cz_ = zpos[ri[inz]] * plan.n_sect + si[inz]
        nz = len(zone_rings) * plan.n_sect
        ncell = np.bincount(cz_, minlength=nz).astype(float)
        pv = plan.phimap[inz].astype(float)
        sect_c = ((np.arange(nz) % plan.n_sect) + 0.5) / plan.n_sect * 2 * np.pi - np.pi
        En = np.zeros((nz, 2 * M_near))
        for m in range(1, M_near + 1):
            c = np.bincount(cz_, weights=np.cos(m * pv), minlength=nz); s = np.bincount(cz_, weights=np.sin(m * pv), minlength=nz)
            En[:, 2 * m - 2] = np.where(ncell > 0, c / np.maximum(ncell, 1), np.cos(m * sect_c))
            En[:, 2 * m - 1] = np.where(ncell > 0, s / np.maximum(ncell, 1), np.sin(m * sect_c))
        En = En.reshape(len(zone_rings), plan.n_sect, 2 * M_near)
        ph_e = np.arange(plan.n_phi_eval) / plan.n_phi_eval * 2 * np.pi - np.pi
        trig_n = np.array([f(m * ph_e) for m in range(1, M_near + 1) for f in (np.cos, np.sin)])
        zone_cells = (zone_rings[:, None] * plan.n_sect + np.arange(plan.n_sect)[None, :]).ravel()
    else:
        M_near = 0; kn = np.zeros(0); keep_n = np.zeros(0, int)
        An = np.zeros((len(zone_rings), 0)); Ane = np.zeros((len(plan.r_eval), 0))
        En = np.zeros((len(zone_rings), plan.n_sect, 0)); trig_n = np.zeros((0, plan.n_phi_eval))
        zone_cells = np.zeros(0, np.int64)
    # ---- Z cell set
    pen_cells = cell2 >> 1
    cz = np.unique(np.concatenate([zone_cells, pen_cells])).astype(np.int64)
    cz_ring = cz // plan.n_sect; cz_sect = cz % plan.n_sect
    zpos = np.full(plan.n_r, -1); zpos[zone_rings] = np.arange(len(zone_rings))
    cz_zone_pos = zpos[cz_ring] if M_near > 0 else np.full(len(cz), -1)
    pos = np.searchsorted(cz, pen_cells)
    # static means of the hats per cz cell (all valid pixels, both parities)
    allv = np.flatnonzero(plan.cellmap2.ravel() >= 0)
    allc = plan.cellmap2.ravel()[allv].astype(np.int64) >> 1
    inn = np.isin(allc, cz)
    nstat = np.bincount(np.searchsorted(cz, allc[inn]), minlength=len(cz)).astype(float)
    pen_static = np.zeros((len(cz), n_pen))
    for c_ in range(2):
        ok = col[:, c_] >= 0
        np.add.at(pen_static, (pos[ok], col[ok, c_]), val[ok, c_])
    pen_static /= np.maximum(nstat, 1)[:, None]
    return ZonePlan(stop=stop, dmap=dmap, pen_pix=pix, pen_cell2=cell2, pen_col=col, pen_val=val, knots_d=knots_d,
                    n_pen_sect=n_pen_sect, n_pen=n_pen, M_near=M_near, r_zone=float(r_zone), zone_rings=zone_rings,
                    knots_n=kn, keep_n=keep_n, An=An, Ane=Ane, En=En, trig_near_eval=trig_n, cz=cz, cz_ring=cz_ring,
                    cz_sect=cz_sect, cz_zone_pos=cz_zone_pos, pen_static=pen_static, pen_cz_index=pos,
                    pen_px_par=(cell2 & 1).astype(np.int64), pen_keep_map=remap, pen_class_map=cmap)


# ----------------------------------------------------------------------------- per-fit pieces
def pen_used_means(zp: ZonePlan, used_flat, n_cells_half):
    """(n_cz, n_pen) per half: mean over the USED pixels of each cz cell of the penumbra hats.  n_cells_half: list of
    (n_r, n_sect) used-pixel counts per half."""
    out = []
    u = used_flat[zp.pen_pix]
    ncz = len(zp.cz); q = zp.n_pen
    for h in range(2):
        sel = u & (zp.pen_px_par == h)
        P = np.zeros(ncz * q)
        for c_ in range(2):
            ok = sel & (zp.pen_col[:, c_] >= 0)
            P += np.bincount(zp.pen_cz_index[ok] * q + zp.pen_col[ok, c_], weights=zp.pen_val[ok, c_], minlength=ncz * q)
        P = P.reshape(ncz, q)
        n = n_cells_half[h].ravel()[zp.cz]
        out.append(P / np.maximum(n, 1)[:, None])
    return out


def z_cells(zp: ZonePlan, R_ring, Pm):
    """Z matrix (n_cz, q) for one half: penumbra = R(ring) * Pm, near = A_near(ring) x E_near(ring, sect)."""
    Z = np.zeros((len(zp.cz), zp.q))
    if zp.n_pen:
        Z[:, :zp.n_pen] = R_ring[zp.cz_ring][:, None] * Pm
    if zp.q_near:
        inz = zp.cz_zone_pos >= 0
        A = zp.An[zp.cz_zone_pos[inz]]                                  # (n, Kn)
        E = zp.En[zp.cz_zone_pos[inz], zp.cz_sect[inz]]                 # (n, 2M)
        Z[inz, zp.n_pen:] = (E[:, :, None] * A[:, None, :]).reshape(int(inz.sum()), -1)
    return Z


def z_rows(zp: ZonePlan, plan, r, phi, pix, R_at_r):
    """Z columns at arbitrary pixels: r, phi and the flat pixel indices `pix` (for the distance to the stop and the
    penumbra class; None = no penumbra contribution, e.g. points of a polar grid)."""
    from .core import _bspline_design
    n = len(r)
    Z = np.zeros((n, zp.q))
    if zp.n_pen and pix is not None:
        pix = np.asarray(pix, np.int64)
        kmax = len(zp.knots_d)
        step = zp.knots_d[1] - zp.knots_d[0] if kmax > 1 else 1.0
        dd = np.maximum(zp.dmap.ravel()[pix].astype(float), zp.knots_d[0])
        j = zp.pen_class_map.ravel()[pix].astype(int)
        u = (dd - zp.knots_d[0]) / step
        k0 = np.floor(u).astype(int); t = u - k0
        for kk, vv in ((k0, 1.0 - t), (k0 + 1, t)):
            ok = (kk < kmax) & (kk >= 0) & (j >= 0) & np.isfinite(dd)
            cf = np.where(ok, np.maximum(j, 0) * kmax + np.clip(kk, 0, kmax - 1), 0)
            ck = zp.pen_keep_map[cf]
            ok &= ck >= 0
            np.add.at(Z, (np.flatnonzero(ok), ck[ok]), vv[ok] * R_at_r[ok])
    if zp.q_near:
        inz = r < zp.r_zone
        if inz.any():
            A = _bspline_design(r[inz], zp.knots_n)[:, zp.keep_n]
            T = np.array([f(m * phi[inz]) for m in range(1, zp.M_near + 1) for f in (np.cos, np.sin)]).T
            Z[np.flatnonzero(inz), zp.n_pen:] = (T[:, :, None] * A[:, None, :]).reshape(int(inz.sum()), -1)
    return Z


PEN_RIDGE = 1e-4


def z_penalty(zp: ZonePlan, Fzz, lam_rel, ridge_harm):
    P = np.zeros_like(Fzz)
    d = np.diag(Fzz)
    if zp.n_pen:
        s = slice(0, zp.n_pen)
        dd = d[s]; sc = float(np.mean(dd[dd > 0])) if np.any(dd > 0) else 1.0
        P[s, s] += PEN_RIDGE * sc * np.eye(zp.n_pen)
        # smoothness of the transmission along d within each class (second differences of consecutive kept knots)
        kmax = len(zp.knots_d)
        for jcl in range(zp.n_pen_sect):
            ids = [zp.pen_keep_map[jcl * kmax + k] for k in range(kmax)]
            ids = [i for i in ids if i >= 0]
            if len(ids) >= 3:
                D2 = np.diff(np.eye(len(ids)), 2, axis=0)
                P[np.ix_(ids, ids)] += lam_rel * sc * (D2.T @ D2)
    if zp.q_near:
        Kn = zp.K_near
        for e in range(2 * zp.M_near):
            s = slice(zp.n_pen + e * Kn, zp.n_pen + (e + 1) * Kn)
            dd = d[s]; sc = float(np.mean(dd[dd > 0])) if np.any(dd > 0) else 1.0
            if Kn >= 3:
                D2 = np.diff(np.eye(Kn), 2, axis=0)
                P[s, s] += lam_rel * sc * (D2.T @ D2)
            P[s, s] += ridge_harm * sc * np.eye(Kn)
    P[np.diag_indices_from(P)] += (d <= 0) * 1e-6
    return P
