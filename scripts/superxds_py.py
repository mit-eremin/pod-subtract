"""superxds_py: a numpy port of the spot-finding path of superxds (github.com/XMB5/superxds, a line-by-line port of
XDS): INIT (BKGINIT / GAIN tables), COLSPOT (MARKPIX2 screening, Poisson strong-pixel test) and JOINCOLSPOT (strong
pixels joined into spots, SPOT.XDS).  Written from the superxds source (src/initialization.f90, frame_handling.f90,
colspot.f90, joincolspot.f90, image_preprocessing.f90, superxds_init.c), file:line in the comments.  superxds itself does
not build on macOS/arm64; this port is VALIDATED against the installed XDS's own outputs (BKGINIT.cbf, GAIN.cbf,
COLSPOT.LP, SPOT.XDS) by `validate` below.  REAL(4) arithmetic is reproduced with float32 where XDS uses it; NINT is
Fortran's (half away from zero); integer division truncates toward zero as in Fortran.

Conventions: pixels are 1-based (ix, iy) inside; arrays are numpy [row = iy-1, col = ix-1].  Dark current = 0 (BLANK.cbf
is 0 for ADSC/SMV input, as in the XDS runs of this study).  One detector segment (1..NX, 1..NY).

API:  tabs = init_tables(frames, good, min_valid, nbx, nby)              -> dict(bkginit int32 (ny,nx), gain int32 (cny,cnx), ...)
      res  = colspot_frame(frame, tabs, min_valid, nbx, nby, sigma, limit)  -> dict(mask int8, ratio, nclear, nstrong, strong (pos, val))
      spots = joincolspot(per_frame_strong, nx, ny, minpix, maxdev, maxpix)  -> DataFrame(x, y, z, intensity) in SPOT.XDS units
      run(frames, ...) does all three;  validate(run_dir, frames, ...) compares with an XDS run directory.
"""
from __future__ import annotations
import math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

F4 = np.float32


def nint(x):
    """Fortran NINT: nearest integer, halves away from zero."""
    x = np.asarray(x); return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5)).astype(np.int64)


def tdiv(a, b):
    """Fortran integer division (truncation toward zero)."""
    a = np.asarray(a); q = np.abs(a) // b; return np.where(a < 0, -q, q)


# ------------------------------------------------------------------------------------------------ GOODPIX
def goodpix(nx, ny, bx, by, tmin, tmax):
    """image_preprocessing.f90 GOODPIX / superxds_init.c superxds_goodpix_row: good iff inside the border (x > bx, x < nx-bx,
    y > by, y < ny-by) and tmin <= rad <= tmax with rad = sqrt(((x - nx/2) 2/nx)^2 + ((y - ny/2) 2/ny)^2), float32."""
    x = np.arange(1, nx + 1, dtype=F4); y = np.arange(1, ny + 1, dtype=F4)
    nxh = F4(nx) * F4(0.5); sx = F4(2.0) / F4(nx)
    ny2 = (y - F4(ny) / F4(2.0)) * (F4(2.0) / F4(ny)); nyy = ny2 * ny2
    nxv = (x - nxh) * sx
    rad = np.sqrt(nxv[None, :] ** 2 + nyy[:, None]).astype(F4)
    good = (np.arange(1, nx + 1)[None, :] > bx) & (np.arange(1, nx + 1)[None, :] < nx - bx) & \
           (np.arange(1, ny + 1)[:, None] > by) & (np.arange(1, ny + 1)[:, None] < ny - by) & (F4(tmin) <= rad) & (rad <= F4(tmax))
    return good


# ------------------------------------------------------------------------------------------------ box sums
def box_sums(img_valid, img_val, hx, hy, centres_y, centres_x, y_lo_clip=1, x_lo_clip=1):
    """count, sum, sum of squares of the valid pixels of the box rows max(y_lo_clip, cy-hy)..min(ny, cy+hy), cols
    max(x_lo_clip, cx-hx)..min(nx, cx+hx) for the given 1-based centre rows/cols (outer product), by integral images (int64)."""
    ny, nx = img_val.shape
    v = np.where(img_valid, img_val.astype(np.int64), 0)
    I1 = np.zeros((ny + 1, nx + 1), np.int64); I2 = np.zeros_like(I1); I0 = np.zeros_like(I1)
    I0[1:, 1:] = np.cumsum(np.cumsum(img_valid.astype(np.int64), 0), 1)
    I1[1:, 1:] = np.cumsum(np.cumsum(v, 0), 1)
    I2[1:, 1:] = np.cumsum(np.cumsum(v * v, 0), 1)
    cy = np.asarray(centres_y); cx = np.asarray(centres_x)
    y0 = np.maximum(y_lo_clip, cy - hy) - 1; y1 = np.minimum(ny, cy + hy); x0 = np.maximum(x_lo_clip, cx - hx) - 1; x1 = np.minimum(nx, cx + hx)
    def S(I):
        return I[np.ix_(y1, x1)] - I[np.ix_(y0, x1)] - I[np.ix_(y1, x0)] + I[np.ix_(y0, x0)]
    return S(I0), S(I1), S(I2)


# ------------------------------------------------------------------------------------------------ FSMOOTH
def fsmooth(image, rx, ry):
    """image_preprocessing.f90 FSMOOTH: box mean over (2rx+1) x (2ry+1) of the non-negative values (negative = missing);
    where the box holds no valid value the pixel keeps its own value.  Values float32, sums in double."""
    ny, nx = image.shape
    valid = image >= 0
    w = valid.astype(np.float64); v = np.where(valid, image.astype(np.float64), 0.0)
    P0 = np.zeros((ny + 1, nx + 1)); P1 = np.zeros_like(P0)
    P0[1:, 1:] = np.cumsum(np.cumsum(w, 0), 1); P1[1:, 1:] = np.cumsum(np.cumsum(v, 0), 1)
    y = np.arange(1, ny + 1); x = np.arange(1, nx + 1)
    y0 = np.maximum(1, y - ry) - 1; y1 = np.minimum(ny, y + ry); x0 = np.maximum(1, x - rx) - 1; x1 = np.minimum(nx, x + rx)
    def S(P):
        return P[np.ix_(y1, x1)] - P[np.ix_(y0, x1)] - P[np.ix_(y1, x0)] + P[np.ix_(y0, x0)]
    cnt = S(P0); sm = S(P1)
    out = np.where(cnt <= 0.5, image.astype(np.float64), sm / np.maximum(cnt, 1e-300))
    return out.astype(F4)


# ------------------------------------------------------------------------------------------------ INIT
def init_tables(frames, good, min_valid, nbx=3, nby=3, dark=0):
    """initialization.f90 INIT_PROCESS (per frame, lines 969-1110) + INIT_TABLES (540-760), one segment, dark current 0.
    frames: iterable of 2-D integer arrays (the BACKGROUND_RANGE images).  Returns the tables and the per-frame means."""
    ny, nx = good.shape
    cnx = (nx - 2) // 4 + 1; cny = (ny - 2) // 4 + 1
    cell_count = np.zeros((cny, cnx), np.int64); gain_sum = np.zeros((cny, cnx), np.float64); bkg_sum = np.zeros((cny, cnx), np.float64)
    node_y = np.arange(4, ny + 1, 4); node_x = np.arange(4, nx + 1, 4)      # J, I multiples of 4 (segment 1..N)
    per_frame = []
    for fr in frames:
        img = np.where(good & (fr >= min_valid), np.maximum(fr.astype(np.int64) - dark, 0), -3)
        valid = img >= 0
        M, SS, QQ = box_sums(valid, img, nbx, nby, node_y, node_x)
        gain = np.full((cny, cnx), -1.0, F4); bkg = np.full((cny, cnx), -1.0, F4)     # untouched nodes (M < 2) stay out
        iy = (node_y // 4) - 1; ix = (node_x // 4) - 1
        ok = M >= 2
        mean = np.where(ok, SS / np.maximum(M, 1), 0.0)
        var = np.where(ok, ((QQ * M - SS * SS) / np.maximum(M, 1)) / np.maximum(M - 1, 1), 0.0)     # REAL(8)
        g = np.where((mean <= 0) | (var <= 0), F4(-0.01), (var / np.where(mean > 0, mean, 1)).astype(F4))
        G = np.full(ok.shape, -1.0, F4); G[ok] = g[ok]
        Bk = np.full(ok.shape, -1.0, F4); Bk[ok] = mean[ok].astype(F4)
        gain[np.ix_(iy, ix)] = G; bkg[np.ix_(iy, ix)] = Bk
        scratch = g[ok & (mean > 0) & (var > 0)].astype(F4)
        N = scratch.size
        if N > 0:                                                            # robust cutoff (1058-1089)
            K = (N + 1) // 2
            srt = np.sort(scratch); median = srt[K - 1]
            if median > 0:
                mad = np.sort(np.abs(scratch - median).astype(F4))[K - 1]
                clip = F4(median + F4(4.8777537) * mad)
                sel = gain >= -0.5
                low = sel & (gain <= 0); gain[low] = median
                hi = sel & (gain > clip); gain[hi] = median; bkg[hi] = -1.0
        for _ in range(3):                                                   # fill (1093-1110)
            sm = fsmooth(bkg, nbx, nby)
            need = (gain > 0) & (bkg < 0) & (sm >= 0)
            nfill = int(need.sum()); bkg[need] = sm[need]
            if nfill < 10:
                break
        acc = (gain > 0) & (bkg >= 0)                                        # accumulate (1118-1127)
        cell_count[acc] += 1; gain_sum[acc] += gain[acc]; bkg_sum[acc] += bkg[acc]
        ns = int(acc.sum()); per_frame.append({"nsample": ns, "gain_mean": float(gain[acc].mean()) if ns else 0.0, "bkg_mean": float(bkg[acc].mean()) if ns else 0.0})
    gm = np.mean([p["gain_mean"] for p in per_frame if p["nsample"] > 0]) if per_frame else 0.0
    # node means (563-571)
    Gn = np.where(cell_count >= 1, gain_sum / np.maximum(cell_count, 1), gm).astype(F4)
    Bn = np.where(cell_count >= 1, bkg_sum / np.maximum(cell_count, 1), -1.0).astype(F4)
    # background map: nodes, then bilinear interpolation (577-633), float32
    bmap = np.zeros((ny, nx), F4)
    bmap[np.ix_(node_y - 1, node_x - 1)] = Bn[np.ix_((node_y // 4) - 1, (node_x // 4) - 1)]
    Y = np.arange(1, ny + 1); X = np.arange(1, nx + 1)
    Y0 = 4 * (Y // 4); X0 = 4 * (X // 4)
    FY = (F4(0.25) * Y.astype(F4) - (Y // 4).astype(F4)).astype(F4); FX = (F4(0.25) * X.astype(F4) - (X // 4).astype(F4)).astype(F4)
    def node_val(y0, x0):
        """node value at pixel (y0, x0) 1-based if inside the segment and >= 0, else nan (with its validity)."""
        v = np.full((ny, nx), np.nan, F4)
        yy, xx = np.meshgrid(y0, x0, indexing="ij")
        inside = (yy >= 1) & (yy <= ny) & (xx >= 1) & (xx <= nx)
        vv = np.full((ny, nx), -1.0, F4); vv[inside] = bmap[yy[inside] - 1, xx[inside] - 1]
        v[inside & (vv >= 0)] = vv[inside & (vv >= 0)]
        return v
    W = np.zeros((ny, nx), F4); V = np.zeros((ny, nx), F4)
    for (dy, dx, wf) in [(0, 0, lambda: ((F4(1) - FX)[None, :] * (F4(1) - FY)[:, None])), (0, 4, lambda: (FX[None, :] * (F4(1) - FY)[:, None])),
                         (4, 0, lambda: ((F4(1) - FX)[None, :] * FY[:, None])), (4, 4, lambda: (FX[None, :] * FY[:, None]))]:
        nv = node_val(Y0 + dy, X0 + dx); w = wf().astype(F4); has = np.isfinite(nv)
        W = np.where(has, (W + w).astype(F4), W); V = np.where(has, (V + np.nan_to_num(nv) * w).astype(F4), V)
    interp = np.where(W >= F4(0.1), (V / np.where(W > 0, W, 1)).astype(F4), bmap)
    is_node = np.zeros((ny, nx), bool); is_node[np.ix_(node_y - 1, node_x - 1)] = True
    bmap = np.where(is_node, bmap, interp).astype(F4)
    scaled = (F4(1000.0) * bmap).astype(F4)
    table = np.where((bmap >= F4(2147483.8)) | (scaled < 0), -2, nint(scaled)).astype(np.int32)
    table[~good] = -3
    # GAIN.cbf (723-756): smoothed node gains x 1000 (cap 1048576), -3 where the node pixel is untrusted or the value <= 0
    gs = fsmooth(Gn, nbx, nby)
    gscaled = np.minimum(F4(1048576.0), F4(1000.0) * gs).astype(F4)
    code = np.where((gscaled < F4(2.1474836e9)) & (gscaled > 0), nint(gscaled), -3).astype(np.int32)
    cyi = np.arange(1, cny + 1); cxi = np.arange(1, cnx + 1)
    node_pix_good = good[np.ix_(np.minimum(ny, 4 * cyi) - 1, np.minimum(nx, 4 * cxi) - 1)]
    code[~node_pix_good] = -3
    return {"bkginit": table, "gain": code, "gain_f": np.where(code > 0, code / F4(1000.0), F4(-1.0)).astype(F4), "good": good,
            "per_frame": per_frame, "cnx": cnx, "cny": cny}


# ------------------------------------------------------------------------------------------------ COLSPOT
def ncut_table(sigma):
    """frame_handling.f90 MARKPIX_NCUT: smallest count whose Poisson CDF reaches P = Phi(sigma), for means 0.1 ... 30.0."""
    P = min(0.5 * (1.0 + float(F4(math.erf(float(F4(sigma) * F4(0.70710677)))))), 0.9999999)
    out = np.zeros(300, np.int64)
    for I in range(2, 302):
        mean = (I - 1) * 0.1; term = math.exp(-(I - 1) * 0.1); n = 0
        if P >= term:
            t = mean * term; total = term + t; n = 1
            if total <= P:
                n = 127
                for K in range(1, 127):
                    t = t * mean / (K + 1); total += t
                    if P < total:
                        n = K + 1; break
        out[I - 2] = n
    return out


def colspot_frame(frame, tabs, min_valid, nbx, nby, sigma, limit, dark=0):
    """colspot.f90 545-562 (preparation) + frame_handling.f90 MARKPIX2 (411-640) and CLASSIFY_VALUE (813-848), one segment.
    Returns the mask (int8: -3 invalid, 0 unflagged background, 1 candidate cleared ... no: 0 = unflagged OR cleared (M=0),
    1 = candidate above the cut-off but not strong, 2..127 = strong label), RATIO, NCLEAR, NSTRONG, the strong pixels."""
    bk = tabs["bkginit"]; ny, nx = bk.shape; cnx = tabs["cnx"]; gain = tabs["gain_f"]
    img = np.where((bk < 0) | (frame < min_valid), -3, np.maximum(frame.astype(np.int64) - dark, 0))
    valid = img >= 0
    hx, hy = nbx, nby; IX = 2 * hx + 1; IY = 2 * hy + 1
    FAC = F4(F4(math.sqrt(4.0 / (IX * IY - 1))) * F4(sigma) + F4(1.0))
    # candidate screen for centres cx in 1..nx-hx, cy in 1..ny-hy (box clipped at the first row/col only)
    cys = np.arange(1, ny - hy + 1); cxs = np.arange(1, nx - hx + 1)
    N, T2, T3 = box_sums(valid, img, hx, hy, cys, cxs)
    # gain cell of the centre: ((cy-2)/4)*cnx + (cx-2)/4 (Fortran truncating division)
    gy = tdiv(cys - 2, 4); gx = tdiv(cxs - 2, 4)
    G = gain[np.ix_(gy, gx)]
    with np.errstate(divide="ignore", invalid="ignore"):
        VAR = (F4(1) * (T3 * N - T2 * T2).astype(F4)) / ((N - 1) * N).astype(F4)
        lhs = (T2.astype(F4) * G / N.astype(F4) * FAC).astype(F4)
    sub = img[:ny - hy, :nx - hx]
    flag = (sub >= 0) & (N > 2) & (sub > 0) & (G > 0) & (lhs < VAR)
    mask = np.where(valid, 0, -3).astype(np.int8)
    mask[:ny - hy, :nx - hx][flag] = 1
    # RATIO over the unflagged valid pixels (605-616)
    unfl = mask == 0
    SUMV = int(img[unfl].sum()); SUMC = int(bk[unfl].astype(np.int64).sum())
    ratio = F4(SUMV) / F4(SUMC) if SUMC > 0 else F4(0.001)
    ratio = F4(ratio)
    # classify the flagged pixels (CLASSIFY_VALUE)
    fy, fx = np.where(mask == 1)
    IV = img[fy, fx].astype(np.int64); IC = bk[fy, fx].astype(np.int64)
    gc = gain[tdiv(fy + 1 - 2, 4), tdiv(fx + 1 - 2, 4)]
    E = (IC.astype(F4) * ratio).astype(F4)
    okg = (E >= 0) & (IV >= 0) & (gc > 0)
    R = np.where(okg, E / np.where(gc > 0, gc, 1), 0).astype(F4); V = np.where(okg, IV.astype(F4) / np.where(gc > 0, gc, 1), 0).astype(F4)
    ncut = ncut_table(sigma)
    I = nint(R * F4(10.0))
    cut_tab = np.maximum(ncut[np.clip(np.maximum(I, 1), 1, 300) - 1], 1)
    cut_big = nint((np.sqrt(R) * F4(sigma) + R).astype(F4))
    CUT = np.where(I < 301, cut_tab, cut_big)
    above = okg & (CUT < nint(V))
    SCORE = ((IV.astype(F4) - E) / np.where(gc > 0, gc, 1)).astype(F4)
    pos = above & (SCORE > 0) & (V > 0)
    SCORE2 = np.where(pos, SCORE / np.sqrt(np.where(V > 0, V, 1)).astype(F4), 0).astype(F4)
    strong = pos & (SCORE2 > F4(limit))
    M = np.full(len(fy), -3, np.int64)
    M[okg] = 0; M[above] = 1
    M[strong] = nint(np.minimum(SCORE2[strong] * F4(10.0), F4(127.0)))
    nclear = int((okg & ~above).sum()); nstrong = int(strong.sum())
    mask[fy, fx] = M.astype(np.int8)
    sp = strong
    pos_lin = (fy[sp] * nx + fx[sp] + 1).astype(np.int64)            # Fortran linear pixel index (1-based, row-major)
    order = np.argsort(pos_lin, kind="stable")
    return {"mask": mask, "ratio": float(ratio), "nclear": nclear, "nstrong": nstrong, "n_unflagged": int(unfl.sum()), "n_valid": int(valid.sum()),
            "strong_pos": pos_lin[order], "strong_val": M[sp][order].astype(np.int16), "img": img}


# ------------------------------------------------------------------------------------------------ JOINCOLSPOT
def joincolspot(per_frame, nx, ny, minpix, maxdev, maxpix=1500000, bkginit=None):
    """joincolspot.f90: histogram threshold LEVEL (503-527), union-find of strong pixels with the same pixel of the previous
    frame, the previous row and the previous column (562-684), spots = classes with >= minpix pixels whose label-weighted
    centroid lies within maxdev (Chebyshev) of the brightest pixel (834-866); intensity = sum of labels; sorted by
    intensity (descending); spots below 0.1 x the median intensity dropped, spots outside [3, N-2] or on an untrusted
    pixel dropped (segment_at, 91-110); x, y as 1-based pixel indices (no XY corrections), z = frame-weighted - 0.5."""
    frames = sorted(per_frame)
    hist = np.zeros(32768, np.int64)
    for f in frames:
        v = per_frame[f]["strong_val"]; np.add.at(hist, v[v > 1], 1)
    nstrong = hist[32767]; level = 32767
    if nstrong <= maxpix:
        level = 1
        for i in range(32766, 0, -1):
            nstrong += hist[i]
            if nstrong > maxpix:
                level = i; break
    # entries in key order
    keys, vals = [], []
    nxy = nx * ny
    for f in frames:
        v = per_frame[f]["strong_val"]; p = per_frame[f]["strong_pos"]; keep = v >= level
        keys.append(nxy * (f - 1) + p[keep]); vals.append(v[keep].astype(np.int64))
    key = np.concatenate(keys) if keys else np.zeros(0, np.int64); val = np.concatenate(vals) if vals else np.zeros(0, np.int64)
    n = len(key)
    parent = np.arange(n)
    def find(i):
        r = i
        while parent[r] != r:
            r = parent[r]
        while parent[i] != r:
            parent[i], i = r, parent[i]
        return r
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb: parent[rb] = ra
            else: parent[ra] = rb
    idx = {int(k): i for i, k in enumerate(key)}
    fr = (key - 1) // nxy + 1; ipos = key - nxy * (fr - 1)
    frames_set = set(frames)
    for i in range(n):
        k = int(key[i]); f = int(fr[i]); p = int(ipos[i])
        if f > 1 and (f - 1) in frames_set:                   # previous frame processed (prev_valid): same pixel
            j = idx.get(k - nxy)
            if j is not None: union(j, i)
        if p > nx:                                             # previous row
            j = idx.get(k - nx)
            if j is not None: union(j, i)
        if p > 1 and (p - 1) % nx != 0:                        # previous column (same row)
            j = idx.get(k - 1)
            if j is not None: union(j, i)
    roots = np.array([find(i) for i in range(n)]) if n else np.zeros(0, np.int64)
    spots = []
    if n:
        order = np.argsort(roots, kind="stable"); rs = roots[order]
        bounds = np.flatnonzero(np.diff(rs)) + 1; starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [n]])
        iz = fr - 1; iy = (ipos - 1) // nx + 1; ix = ipos - nx * (iy - 1)
        for s, e in zip(starts, ends):
            if e - s < minpix:
                continue
            m = order[s:e]; w = val[m].astype(np.float64)
            sw = w.sum(); xc = (ix[m] * w).sum() / sw; yc = (iy[m] * w).sum() / sw; zc = ((iz[m] + 1) * w).sum() / sw
            jmax = m[np.argmax(w)]                            # first maximum in key order
            if max(abs(xc - ix[jmax]), abs(yc - iy[jmax])) <= maxdev:
                spots.append((xc, yc, zc, sw, e - s))
    df = pd.DataFrame(spots, columns=["x", "y", "z", "intensity", "npix"])
    if len(df):
        df = df.sort_values("intensity", ascending=False, kind="stable").reset_index(drop=True)
        thr = df.intensity.iloc[(len(df) + 1) // 2 - 1] * 0.1
        keep = df.intensity >= thr
        ixr = nint(df.x.to_numpy()); iyr = nint(df.y.to_numpy())
        inside = (ixr >= 3) & (ixr <= nx - 2) & (iyr >= 3) & (iyr <= ny - 2)
        if bkginit is not None:
            inside &= bkginit[np.clip(iyr - 1, 0, ny - 1), np.clip(ixr - 1, 0, nx - 1)] >= 0
        df = df[keep & inside].reset_index(drop=True)
        df["z"] = df.z - 0.5
    return df, {"level": level, "n_strong": n, "hist_nonzero": int((hist > 0).sum())}


# ------------------------------------------------------------------------------------------------ driver
def run(frames, min_valid, nbx=3, nby=3, sigma=2.0, limit=3.0, minpix=6, maxdev=3.0, tmin=0.0, tmax=1.4, bkg_range=None, verbose=True):
    """frames: list of 2-D integer arrays in sweep order (frame numbers 1..n).  bkg_range: frames for INIT (default all)."""
    ny, nx = frames[0].shape
    good = goodpix(nx, ny, nbx, nby, tmin, tmax)
    t0 = time.perf_counter()
    tabs = init_tables([frames[i] for i in (bkg_range or range(len(frames)))], good, min_valid, nbx, nby)
    t_init = time.perf_counter() - t0
    per = {}; t0 = time.perf_counter()
    for i, fr in enumerate(frames, 1):
        per[i] = colspot_frame(fr, tabs, min_valid, nbx, nby, sigma, limit)
        if verbose:
            print(f"  frame {i}: ratio {per[i]['ratio']:.4f} unflagged {per[i]['n_unflagged']} / {per[i]['n_valid']} nclear {per[i]['nclear']} nstrong {per[i]['nstrong']}", flush=True)
    t_col = time.perf_counter() - t0; t0 = time.perf_counter()
    spots, jinfo = joincolspot(per, nx, ny, minpix, maxdev, bkginit=tabs["bkginit"])
    t_join = time.perf_counter() - t0
    return {"tabs": tabs, "per_frame": per, "spots": spots, "join": jinfo, "t_init": t_init, "t_colspot": t_col, "t_join": t_join}


def validate(run_dir, frames, min_valid, **kw):
    """Compare with an XDS run directory (BKGINIT.cbf, GAIN.cbf, COLSPOT.LP, SPOT.XDS)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_v0 import read_cbf_byte_offset, read_spot_xds
    import re
    res = run(frames, min_valid, **kw)
    out = {}
    bk = read_cbf_byte_offset(Path(run_dir) / "BKGINIT.cbf"); mine = res["tabs"]["bkginit"]
    d = (mine.astype(np.int64) - bk.astype(np.int64)); both = (bk >= 0) & (mine >= 0)
    out["bkginit"] = {"status_equal": float((np.sign(bk) == np.sign(mine)).mean()), "exact": float((d[both] == 0).mean()), "within1": float((np.abs(d[both]) <= 1).mean()),
                      "max_abs": int(np.abs(d[both]).max()) if both.any() else None, "rel_rms": float(np.sqrt(np.mean((d[both] / np.maximum(bk[both], 1)) ** 2)))}
    g = read_cbf_byte_offset(Path(run_dir) / "GAIN.cbf"); gm = res["tabs"]["gain"]
    dg = gm.astype(np.int64) - g.astype(np.int64); bg = (g > 0) & (gm > 0)
    out["gain"] = {"status_equal": float((np.sign(g) == np.sign(gm)).mean()), "exact": float((dg[bg] == 0).mean()), "within1": float((np.abs(dg[bg]) <= 1).mean()), "max_abs": int(np.abs(dg[bg]).max())}
    lp = open(Path(run_dir) / "COLSPOT.LP", errors="replace").read()
    tab = {int(a): (int(b), int(c)) for a, b, c, _ in re.findall(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s*$", lp, re.M)}
    rows = [(f, tab[f][0], res["per_frame"][f]["nclear"], tab[f][1], res["per_frame"][f]["nstrong"]) for f in sorted(res["per_frame"]) if f in tab]
    cf = pd.DataFrame(rows, columns=["frame", "nclear_xds", "nclear_py", "nstrong_xds", "nstrong_py"])
    out["colspot"] = {"frames": len(cf), "nclear_exact": int((cf.nclear_xds == cf.nclear_py).sum()), "nstrong_exact": int((cf.nstrong_xds == cf.nstrong_py).sum()),
                      "nstrong_xds_total": int(cf.nstrong_xds.sum()), "nstrong_py_total": int(cf.nstrong_py.sum()), "table": cf}
    spx = read_spot_xds(Path(run_dir) / "SPOT.XDS") if (Path(run_dir) / "SPOT.XDS").exists() else pd.DataFrame()
    sp = res["spots"]
    out["spots"] = {"n_xds": len(spx), "n_py": len(sp)}
    if len(spx) and len(sp):
        from scipy.spatial import cKDTree
        d, j = cKDTree(spx[["x_xds", "y_xds", "z_xds"]].to_numpy()).query(sp[["x", "y", "z"]].to_numpy(), distance_upper_bound=0.51)
        m = np.isfinite(d)
        out["spots"].update({"matched_within_0.5px": int(m.sum()), "intensity_exact": int((np.abs(sp.intensity.to_numpy()[m] - spx.I_xds.to_numpy()[j[m]]) < 0.5).sum()),
                             "pos_max_abs": float(d[m].max()) if m.any() else None})
    return res, out


if __name__ == "__main__":
    import argparse, glob, tifffile
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--frames", required=True, help="glob of SMV .img or TIFF frames in order")
    ap.add_argument("--min-valid", type=int, default=0); ap.add_argument("--sigma", type=float, default=2.0); ap.add_argument("--limit", type=float, default=3.0)
    ap.add_argument("--minpix", type=int, default=6); ap.add_argument("--maxdev", type=float, default=3.0); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.frames)); files = files[:a.n] if a.n else files
    def load(p):
        if p.endswith(".img"):
            b = open(p, "rb").read(); hb = int(b[:512].decode("latin1").split("HEADER_BYTES=")[1].split(";")[0]); hdr = b[:hb].decode("latin1")
            n1 = int(hdr.split("SIZE1=")[1].split(";")[0]); n2 = int(hdr.split("SIZE2=")[1].split(";")[0]); return np.frombuffer(b[hb:], "<u2").reshape(n2, n1).astype(np.int64)
        return tifffile.imread(p).astype(np.int64)
    frames = [load(p) for p in files]
    res, out = validate(a.run_dir, frames, a.min_valid, sigma=a.sigma, limit=a.limit, minpix=a.minpix, maxdev=a.maxdev)
    cf = out["colspot"].pop("table")
    for k, v in out.items():
        print(k, v)
    print(cf.to_string(index=False))
    print(f"times: init {res['t_init']:.1f} s, colspot {res['t_colspot']:.1f} s, join {res['t_join']:.1f} s")
