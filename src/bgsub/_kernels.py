# MOVED from pod/background/src/podbg/bgsub/_kernels.py at background commit f1e93f5 (file last changed in fd2d10f; sha256 0331761531f53beabd23264c022ac489d10c83cd89e08365637c67bc126451d9).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""Per-pixel passes of bgsub: numba when importable, numpy otherwise (identical results up to float summation order).

All kernels take 2-D arrays; ``cellmap2`` holds (cell * 2 + checkerboard parity) for statically valid pixels, -1 else.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on the machine
    import numba as _nb
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    _nb = None
    HAVE_NUMBA = False


# ----------------------------------------------------------------------------- numpy versions
def _accumulate_np(K, cellmap2, used, n_cells2, rmap):
    sel = used & (cellmap2 >= 0)
    c = cellmap2[sel]
    S = np.bincount(c, weights=K[sel].astype(np.float64), minlength=n_cells2)
    n = np.bincount(c, minlength=n_cells2).astype(np.float64)
    R = np.bincount(c, weights=rmap[sel].astype(np.float64), minlength=n_cells2)
    return S, n, R


def _box3_np(K, valid):
    from scipy import ndimage as ndi
    a = np.where(valid, K, 0).astype(np.float64)
    return ndi.uniform_filter(a, 3, mode="constant") * 9.0


def _flag_np(K, cellmap2, valid, kpx, tw):
    cell = np.where(cellmap2 >= 0, cellmap2 >> 1, 0)
    f = valid & (cellmap2 >= 0) & (K >= kpx[cell])
    if tw is not None:
        S9 = _box3_np(K, valid & (cellmap2 >= 0))
        f |= valid & (cellmap2 >= 0) & (S9 >= tw[cell])
    return f


def _dilate_np(flags, r_dil):
    if r_dil <= 0 or not flags.any():
        return flags.copy()
    ys, xs = np.nonzero(flags)
    R = int(np.ceil(r_dil))
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    keep = oy ** 2 + ox ** 2 <= r_dil ** 2 + 1e-9
    oy, ox = oy[keep], ox[keep]
    Y = (ys[:, None] + oy[None, :]).ravel(); X = (xs[:, None] + ox[None, :]).ravel()
    ok = (Y >= 0) & (Y < flags.shape[0]) & (X >= 0) & (X < flags.shape[1])
    out = np.zeros_like(flags)
    out[Y[ok], X[ok]] = True
    return out


def _eval_np(rmap, phimap, grid, r_lo, dr, n_phi):
    nre = grid.shape[0]
    fr = (rmap - r_lo) / dr
    i0 = np.floor(fr).astype(np.int64)
    ok = (i0 >= 0) & (i0 < nre - 1)
    i0c = np.clip(i0, 0, nre - 2)
    tr = np.clip(fr - i0c, 0.0, 1.0)
    fp = (phimap + np.pi) / (2 * np.pi) * n_phi
    j0 = np.floor(fp).astype(np.int64) % n_phi
    tp = fp - np.floor(fp)
    j1 = (j0 + 1) % n_phi
    g = grid
    out = ((1 - tr) * ((1 - tp) * g[i0c, j0] + tp * g[i0c, j1]) + tr * ((1 - tp) * g[i0c + 1, j0] + tp * g[i0c + 1, j1]))
    return np.where(ok, out, np.nan).astype(np.float32)


def _ring_acc_np(K, bg, used, cy, cx, r_lo, n_r):
    """v2 ring refinement: per 1-px ring of r' = |p - (cy, cx)| and checkerboard parity, sums of (K - bg), of bg and
    the pixel count over the used pixels (bg finite)."""
    ny, nx = K.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sel = used & np.isfinite(bg)
    r = np.hypot(yy[sel] - cy, xx[sel] - cx)
    ri = np.floor(r - r_lo).astype(np.int64)
    ok = (ri >= 0) & (ri < n_r)
    par = ((yy[sel] + xx[sel]) & 1)[ok]
    idx = ri[ok] * 2 + par
    res = (K[sel] - bg[sel])[ok].astype(np.float64)
    A = np.zeros((n_r * 2, 3))
    A[:, 0] = np.bincount(idx, weights=res, minlength=2 * n_r)
    A[:, 1] = np.bincount(idx, weights=bg[sel][ok].astype(np.float64), minlength=2 * n_r)
    A[:, 2] = np.bincount(idx, minlength=2 * n_r)
    return A.reshape(n_r, 2, 3)


def _ell_r(dy, dx, ea, eb):
    r2 = dy * dy + dx * dx
    r = np.sqrt(r2)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(r2 > 0, 1.0 + ea * (dx * dx - dy * dy) / np.maximum(r2, 1e-300) + eb * 2.0 * dx * dy / np.maximum(r2, 1e-300), 1.0)
    return r * f


def _ring_acc_ell_np(K, bg, used, cy, cx, ea, eb, r_lo, n_r):
    """v2.1 ring term: as _ring_acc_np on 1-px rings of the ELLIPTICAL radius r' = |p - c| (1 + ea cos 2phi + eb sin 2phi)."""
    ny, nx = K.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sel = used & np.isfinite(bg)
    r = _ell_r(yy[sel] - cy, xx[sel] - cx, ea, eb)
    ri = np.floor(r - r_lo).astype(np.int64)
    ok = (ri >= 0) & (ri < n_r)
    par = ((yy[sel] + xx[sel]) & 1)[ok]
    idx = ri[ok] * 2 + par
    res = (K[sel] - bg[sel])[ok].astype(np.float64)
    A = np.zeros((n_r * 2, 3))
    A[:, 0] = np.bincount(idx, weights=res, minlength=2 * n_r)
    A[:, 1] = np.bincount(idx, weights=bg[sel][ok].astype(np.float64), minlength=2 * n_r)
    A[:, 2] = np.bincount(idx, minlength=2 * n_r)
    return A.reshape(n_r, 2, 3)


def _ring_apply_ell_np(bg, c, cy, cx, ea, eb, r_lo):
    ny, nx = bg.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    i = np.floor(_ell_r(yy - cy, xx - cx, ea, eb) - r_lo).astype(np.int64)
    ok = (i >= 0) & (i < len(c))
    return (bg + np.where(ok, np.asarray(c)[np.where(ok, i, 0)], 0.0)).astype(np.float32)


def _probe_sums_np(R, bad, CY, CX, oyb, oxb, oya, oxa):
    YB = CY[:, None] + oyb[None, :]; XB = CX[:, None] + oxb[None, :]
    YA = CY[:, None] + oya[None, :]; XA = CX[:, None] + oxa[None, :]
    ok = ~bad[YB, XB].any(1) & ~bad[YA, XA].any(1)
    Lb = np.where(ok, R[YB, XB].mean(1), 0.0); La = np.where(ok, R[YA, XA].mean(1), 0.0)
    return ok, Lb, La


# ----------------------------------------------------------------------------- numba versions
if HAVE_NUMBA:  # pragma: no cover
    @_nb.njit(cache=True)
    def _accumulate_nb(K, cellmap2, used, n_cells2, rmap):
        A = np.zeros((n_cells2, 3))
        ny, nx = K.shape
        for y in range(ny):
            for x in range(nx):
                c = cellmap2[y, x]
                if c >= 0 and used[y, x]:
                    A[c, 0] += K[y, x]
                    A[c, 1] += 1.0
                    A[c, 2] += rmap[y, x]
        return A[:, 0].copy(), A[:, 1].copy(), A[:, 2].copy()

    @_nb.njit(cache=True)
    def _flag_nb(K, cellmap2, valid, kpx, tw, use_win):
        ny, nx = K.shape
        f = np.zeros((ny, nx), np.bool_)
        A = np.zeros((ny, nx), np.float32)
        for y in range(ny):
            for x in range(nx):
                if cellmap2[y, x] >= 0 and valid[y, x]:
                    A[y, x] = K[y, x]
        R = np.zeros((ny, nx), np.float32)
        if use_win:
            for y in range(ny):
                for x in range(nx):
                    s = A[y, x]
                    if x > 0:
                        s += A[y, x - 1]
                    if x < nx - 1:
                        s += A[y, x + 1]
                    R[y, x] = s
        for y in range(ny):
            for x in range(nx):
                c2 = cellmap2[y, x]
                if c2 < 0 or not valid[y, x]:
                    continue
                c = c2 >> 1
                if K[y, x] >= kpx[c]:
                    f[y, x] = True
                elif use_win:
                    s = R[y, x]
                    if y > 0:
                        s += R[y - 1, x]
                    if y < ny - 1:
                        s += R[y + 1, x]
                    if s >= tw[c]:
                        f[y, x] = True
        return f

    @_nb.njit(cache=True)
    def _dilate_nb(flags, r_dil):
        ny, nx = flags.shape
        out = np.zeros((ny, nx), np.bool_)
        R = int(np.ceil(r_dil))
        r2 = r_dil * r_dil + 1e-9
        for y in range(ny):
            for x in range(nx):
                if flags[y, x]:
                    for dy in range(-R, R + 1):
                        yy = y + dy
                        if yy < 0 or yy >= ny:
                            continue
                        for dx in range(-R, R + 1):
                            xx = x + dx
                            if xx < 0 or xx >= nx:
                                continue
                            if dy * dy + dx * dx <= r2:
                                out[yy, xx] = True
        return out

    @_nb.njit(cache=True)
    def _eval_nb(rmap, phimap, grid, r_lo, dr, n_phi):
        ny, nx = rmap.shape
        nre = grid.shape[0]
        out = np.empty((ny, nx), np.float32)
        two_pi = 2.0 * np.pi
        for y in range(ny):
            for x in range(nx):
                fr = (rmap[y, x] - r_lo) / dr
                i0 = int(np.floor(fr))
                if i0 < 0 or i0 >= nre - 1:
                    out[y, x] = np.nan
                    continue
                tr = fr - i0
                fp = (phimap[y, x] + np.pi) / two_pi * n_phi
                jf = np.floor(fp)
                tp = fp - jf
                j0 = int(jf) % n_phi
                j1 = (j0 + 1) % n_phi
                out[y, x] = ((1 - tr) * ((1 - tp) * grid[i0, j0] + tp * grid[i0, j1])
                             + tr * ((1 - tp) * grid[i0 + 1, j0] + tp * grid[i0 + 1, j1]))
        return out

    @_nb.njit(cache=True)
    def _ring_acc_nb(K, bg, used, cy, cx, r_lo, n_r):
        ny, nx = K.shape
        A = np.zeros((n_r, 2, 3))
        for y in range(ny):
            dy2 = (y - cy) * (y - cy)
            for x in range(nx):
                if not used[y, x]:
                    continue
                b = bg[y, x]
                if not np.isfinite(b):
                    continue
                r = np.sqrt(dy2 + (x - cx) * (x - cx))
                i = int(np.floor(r - r_lo))
                if i < 0 or i >= n_r:
                    continue
                p = (y + x) & 1
                A[i, p, 0] += K[y, x] - b
                A[i, p, 1] += b
                A[i, p, 2] += 1.0
        return A

    @_nb.njit(cache=True)
    def _ring_apply_nb(bg, c, cy, cx, r_lo):
        ny, nx = bg.shape
        out = bg.copy()
        n = c.shape[0]
        for y in range(ny):
            dy2 = (y - cy) * (y - cy)
            for x in range(nx):
                r = np.sqrt(dy2 + (x - cx) * (x - cx))
                i = int(np.floor(r - r_lo))
                if i >= 0 and i < n:
                    out[y, x] += c[i]
        return out

    @_nb.njit(cache=True)
    def _ring_acc_ell_nb(K, bg, used, cy, cx, ea, eb, r_lo, n_r):
        ny, nx = K.shape
        A = np.zeros((n_r, 2, 3))
        for y in range(ny):
            dy = y - cy
            for x in range(nx):
                if not used[y, x]:
                    continue
                b = bg[y, x]
                if not np.isfinite(b):
                    continue
                dx = x - cx
                r2 = dy * dy + dx * dx
                r = np.sqrt(r2)
                if r2 > 0.0:
                    r = r * (1.0 + ea * (dx * dx - dy * dy) / r2 + eb * 2.0 * dx * dy / r2)
                i = int(np.floor(r - r_lo))
                if i < 0 or i >= n_r:
                    continue
                p = (y + x) & 1
                A[i, p, 0] += K[y, x] - b
                A[i, p, 1] += b
                A[i, p, 2] += 1.0
        return A

    @_nb.njit(cache=True)
    def _ring_apply_ell_nb(bg, c, cy, cx, ea, eb, r_lo):
        ny, nx = bg.shape
        out = bg.copy()
        n = c.shape[0]
        for y in range(ny):
            dy = y - cy
            for x in range(nx):
                dx = x - cx
                r2 = dy * dy + dx * dx
                r = np.sqrt(r2)
                if r2 > 0.0:
                    r = r * (1.0 + ea * (dx * dx - dy * dy) / r2 + eb * 2.0 * dx * dy / r2)
                i = int(np.floor(r - r_lo))
                if i >= 0 and i < n:
                    out[y, x] += c[i]
        return out

    @_nb.njit(cache=True)
    def _probe_sums_nb(R, bad, CY, CX, oyb, oxb, oya, oxa):
        n = CY.shape[0]
        ok = np.zeros(n, np.bool_)
        Lb = np.zeros(n); La = np.zeros(n)
        nbx = oyb.shape[0]; nax = oya.shape[0]
        for k in range(n):
            cy = CY[k]; cx = CX[k]
            good = True
            sb = 0.0
            for j in range(nbx):
                y = cy + oyb[j]; x = cx + oxb[j]
                if bad[y, x]:
                    good = False
                    break
                sb += R[y, x]
            if not good:
                continue
            sa = 0.0
            for j in range(nax):
                y = cy + oya[j]; x = cx + oxa[j]
                if bad[y, x]:
                    good = False
                    break
                sa += R[y, x]
            if good:
                ok[k] = True
                Lb[k] = sb / nbx
                La[k] = sa / nax
        return ok, Lb, La


# ----------------------------------------------------------------------------- dispatch
def accumulate(K, cellmap2, used, n_cells2, use_numba=True, rmap=None):
    if use_numba and HAVE_NUMBA:
        return _accumulate_nb(K, cellmap2, used, n_cells2, rmap)
    return _accumulate_np(K, cellmap2, used, n_cells2, rmap)


def flag(K, cellmap2, valid, kpx, tw, use_numba=True):
    if use_numba and HAVE_NUMBA:
        return _flag_nb(K, cellmap2, valid, kpx.astype(np.float64), (tw if tw is not None else kpx).astype(np.float64), tw is not None)
    return _flag_np(K, cellmap2, valid, kpx, tw)


def dilate(flags, r_dil, use_numba=True):
    if r_dil <= 0:
        return flags.copy()
    if use_numba and HAVE_NUMBA:
        return _dilate_nb(flags, float(r_dil))
    return _dilate_np(flags, r_dil)


def eval_map(rmap, phimap, grid, r_lo, dr, n_phi, use_numba=True):
    if use_numba and HAVE_NUMBA:
        return _eval_nb(rmap, phimap, np.ascontiguousarray(grid, np.float64), float(r_lo), float(dr), int(n_phi))
    return _eval_np(rmap, phimap, grid, r_lo, dr, n_phi)


def ring_acc(K, bg, used, cy, cx, r_lo, n_r, use_numba=True):
    if use_numba and HAVE_NUMBA:
        return _ring_acc_nb(np.ascontiguousarray(K, np.float64), np.ascontiguousarray(bg, np.float32), np.ascontiguousarray(used), float(cy), float(cx), float(r_lo), int(n_r))
    return _ring_acc_np(K, bg, used, cy, cx, r_lo, n_r)


def ring_apply(bg, c, cy, cx, r_lo, use_numba=True):
    """bg + c[floor(|p - (cy, cx)| - r_lo)] (v2 ring refinement); NaN stays NaN."""
    if use_numba and HAVE_NUMBA:
        return _ring_apply_nb(np.ascontiguousarray(bg, np.float32), np.ascontiguousarray(c, np.float32), float(cy), float(cx), float(r_lo))
    ny, nx = bg.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    i = np.floor(np.hypot(yy - cy, xx - cx) - r_lo).astype(np.int64)
    ok = (i >= 0) & (i < len(c))
    return (bg + np.where(ok, np.asarray(c)[np.where(ok, i, 0)], 0.0)).astype(np.float32)


def ring_acc_ell(K, bg, used, cy, cx, ea, eb, r_lo, n_r, use_numba=True):
    """v2.1 ring term accumulation (n_r, 2, [resid, bg, npx]) on 1-px rings of the elliptical radius about (cy, cx)."""
    if use_numba and HAVE_NUMBA:
        return _ring_acc_ell_nb(np.ascontiguousarray(K, np.float64), np.ascontiguousarray(bg, np.float32), np.ascontiguousarray(used),
                                float(cy), float(cx), float(ea), float(eb), float(r_lo), int(n_r))
    return _ring_acc_ell_np(K, bg, used, cy, cx, ea, eb, r_lo, n_r)


def ring_apply_ell(bg, c, cy, cx, ea, eb, r_lo, use_numba=True):
    """bg + c[floor(r' - r_lo)], r' the elliptical radius about (cy, cx) (v2.1 ring term); NaN stays NaN."""
    if use_numba and HAVE_NUMBA:
        return _ring_apply_ell_nb(np.ascontiguousarray(bg, np.float32), np.ascontiguousarray(c, np.float32), float(cy), float(cx),
                                  float(ea), float(eb), float(r_lo))
    return _ring_apply_ell_np(bg, c, cy, cx, ea, eb, r_lo)


def probe_sums(R, bad, CY, CX, oyb, oxb, oya, oxa, use_numba=True):
    """v2.1 probes: (ok, box mean, annulus mean) of R at integer centres; ok = no bad pixel in box or annulus."""
    if use_numba and HAVE_NUMBA:
        i = lambda a: np.ascontiguousarray(a, np.int64)  # noqa: E731
        return _probe_sums_nb(np.ascontiguousarray(R, np.float64), np.ascontiguousarray(bad), i(CY), i(CX), i(oyb), i(oxb), i(oya), i(oxa))
    return _probe_sums_np(R, bad, CY, CX, oyb, oxb, oya, oxa)
