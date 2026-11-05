# VENDORED COPY. Source: pod/background/src/podbg/masks.py at background commit f1e93f5 (file last changed in 0d32d02);
# sha256 of the source file: 829603d8889c0b8640e291b9b3b21fca9e9aeaa61bd0052100fc759b197edc48. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: none (byte-identical below this header).
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Beam-stop masks by thresholding.

Beam stops differ per session in shape, size and entry side, but they are near
zero: zero or electronic noise. So: sum ~100 film frames of the session (high
scattered signal everywhere except behind the stop), threshold the summed frame
at a small fraction of its median, close morphologically, and keep the
connected components that touch the frame centre. No learned segmentation.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def summed_frame(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float64).sum(axis=0)


def _shell_median(sm: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Median of `sm` in each integer-radius shell."""
    order = np.argsort(r.ravel(), kind="stable")
    rs, vals = r.ravel()[order], sm.ravel()[order]
    bounds = np.searchsorted(rs, np.arange(rs.max() + 2))
    out = np.zeros(rs.max() + 1)
    for k in range(len(out)):
        seg = vals[bounds[k]:bounds[k + 1]]
        out[k] = np.median(seg) if seg.size else 0.0
    return out


def beamstop_mask(summed: np.ndarray, smooth_px: float = 3.0, disc_frac: float = 0.3,
                  inner_frac: float = 0.05, outer_frac: float = 0.2, bright_level: float = 20.0,
                  close_px: int = 5, dilate_px: int = 2, border_px: int = 4) -> tuple[np.ndarray, dict]:
    """True where the beam stop is. Radial, two-zone rule; no hole filling.

    disc:  shells from the centre outward are entirely dark until the stop
           edge; r_disc is the first radius whose shell median exceeds
           disc_frac x the brightest shell median.
    inner: where the shell is bright (median >= bright_level = 20 counts, so a
           Poisson zero has probability ~2e-9), a stop pixel is near zero in
           the RAW sum: raw <= max(1, inner_frac x shell). Raw, not smoothed,
           because the bright rim next to the stop leaks into a smoothed value
           and would shrink the mask.
    outer: where the shell is fainter, zeros are common, so use the smoothed
           sum (sigma 3 px averages ~57 px): sm < outer_frac x shell median of
           the smoothed sum.
    Components are kept only if connected to the disc; the result is dilated
    by dilate_px for safety. A border_px rim is ignored (dark detector border).
    """
    raw = summed.astype(np.float64)
    sm = ndi.gaussian_filter(raw, smooth_px)
    h, w = raw.shape
    yy, xx = np.indices((h, w)); r = np.hypot(yy - h / 2, xx - w / 2).astype(np.int32)
    shell_raw, shell_sm = _shell_median(raw, r), _shell_median(sm, r)
    r_half = min(h, w) / 2
    peak = shell_sm[: int(r_half)].max()
    above = np.nonzero(shell_sm > disc_frac * peak)[0]
    r_disc = int(above[0]) if above.size else 0
    disc = r < r_disc
    bright = shell_raw[r] >= bright_level
    inner = bright & (raw <= np.maximum(1.0, inner_frac * shell_raw[r]))
    outer = ~bright & (sm < outer_frac * shell_sm[r])
    cand = disc | inner | outer
    cand[:border_px, :] = cand[-border_px:, :] = cand[:, :border_px] = cand[:, -border_px:] = False
    cand = ndi.binary_closing(cand, structure=np.ones((close_px, close_px), bool)) | disc
    lab, n = ndi.label(cand)
    keep = np.unique(lab[disc & (lab > 0)])
    mask = np.isin(lab, keep)
    if dilate_px:
        mask = ndi.binary_dilation(mask, iterations=dilate_px)
    ys, xs = np.nonzero(mask)
    info = {"r_disc_px": r_disc, "peak_shell_median": float(peak), "n_components_kept": int(len(keep)),
            "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
            "bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None,
            "centroid": [float(ys.mean()), float(xs.mean())] if len(ys) else None}
    return mask, info


# =============================================================================
# masks v1 (ported 2026-10-26 from ringcal.stopmask, ringcal Stage 2): dark-core
# disc + local shell rule. Works on int16 dark-subtracted sums with a negative
# pedestal (CETA, Spectra), sparse float32 sums (0.2-0.5 counts/px/frame F200C)
# and uint8/uint16 Apollo sums; flags a tapered blade without a disc (F30).
# `beamstop_mask` (v0) above is unchanged.
#
# Rule (no learned segmentation, no hand drawing, no frame-centred disc):
# 1. Smooth the sum (Gaussian, sigma_px; default 3 px per 2048 px).
# 2. Dark core: inside the central `core_box` fraction of the frame, pixels of
#    the smoothed sum below `core_frac` x its 95th percentile there; the
#    connected component holding the minimum is the core; the maximum of its
#    distance transform seeds the disc centre.
# 3. Local shell rule about the seed: shell medians of the smoothed sum over
#    the pixels NOT in the (dilated) current dark set, shells with too few
#    bright pixels filled from the first bright shell outside; dark where
#    sm < `frac` x shell median, in shells whose median exceeds a Poisson
#    noise floor (sparse sums). Two passes.
# 4. Closing (`close_px`), border rows/cols dropped, components touching the
#    core kept, dilation `dilate_px`.
# 5. Disc geometry: circle fit to the undilated boundary outside the arm
#    sector; "no disc" when the radius is below 1.3 x the arm half-width.


def _shell_median_v1(vals: np.ndarray, rbin: np.ndarray, use: np.ndarray, n_shells: int, min_frac: float = 0.3) -> np.ndarray:
    """Median of vals per integer-radius shell over `use` pixels; shells with
    fewer than min_frac of their pixels usable are filled from the first
    usable shell at larger radius (the disc interior), else from smaller radius."""
    n_all = np.bincount(rbin.ravel(), minlength=n_shells)[:n_shells]
    rb = rbin[use]; v = vals[use]
    order = np.lexsort((v, rb))
    rb_s, v_s = rb[order], v[order]
    counts = np.bincount(rb_s, minlength=n_shells)[:n_shells]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    med = np.full(n_shells, np.nan)
    ok = (counts > 0) & (counts >= min_frac * np.maximum(n_all, 1))
    lo = starts[ok] + (counts[ok] - 1) // 2; hi = starts[ok] + counts[ok] // 2
    med[ok] = 0.5 * (v_s[lo] + v_s[hi])
    idx = np.arange(n_shells)
    good = np.isfinite(med)
    if not good.any():
        raise ValueError("no usable shells")
    nxt = np.minimum.accumulate(np.where(good, idx, n_shells)[::-1])[::-1]
    prv = np.maximum.accumulate(np.where(good, idx, -1))
    fill = np.where(nxt < n_shells, nxt, prv)
    return med[fill]


def _circle_fit(y: np.ndarray, x: np.ndarray):
    """Geometric circle fit (LS on radial residuals) seeded by the algebraic
    (Kasa) fit. Returns (cy, cx, R, cov3x3, rms)."""
    from scipy.optimize import least_squares
    A = np.column_stack([2 * y, 2 * x, np.ones_like(y)])
    b = y * y + x * x
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cy0, cx0 = sol[0], sol[1]
    R0 = np.sqrt(max(sol[2] + cy0 ** 2 + cx0 ** 2, 1e-9))
    fit = least_squares(lambda p: np.hypot(y - p[0], x - p[1]) - p[2], [cy0, cx0, R0])
    res = fit.fun
    dof = max(len(y) - 3, 1)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.pinv(fit.jac.T @ fit.jac)
    return fit.x[0], fit.x[1], fit.x[2], cov, np.sqrt(s2)


def disc_from_mask(mask: np.ndarray, arm_sector_deg: float = 70.0, edge_band: float = 1.3) -> dict:
    """Centre and radius of the disc part of a beam-stop mask (port of
    ringcal.centre.centre_from_mask). Seed = maximum of the distance transform;
    arm direction = principal axis of the mask; pixels farther than R along
    the axis are dropped; the remaining boundary within edge_band x R of the
    seed, outside the +-arm_sector_deg/2 sector facing the arm, is circle-fitted."""
    m = np.asarray(mask, dtype=bool)
    if m.sum() < 10:
        raise ValueError("mask too small")
    dt = ndi.distance_transform_edt(m)
    iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
    R_seed = float(dt[iy, ix])
    ys, xs = np.nonzero(m)
    cyc, cxc = ys.mean(), xs.mean()
    evals, evecs = np.linalg.eigh(np.cov(np.vstack([ys - cyc, xs - cxc])))
    axis = evecs[:, np.argmax(evals)]
    if np.array([cyc - iy, cxc - ix]) @ axis < 0:
        axis = -axis
    proj = (ys - iy) * axis[0] + (xs - ix) * axis[1]
    keep = np.abs(proj) <= R_seed
    disc = np.zeros_like(m); disc[ys[keep], xs[keep]] = True
    edge = disc & ~ndi.binary_erosion(disc, iterations=1, border_value=0)
    ey, ex = np.nonzero(edge)
    d = np.hypot(ey - iy, ex - ix)
    arm_ang = np.arctan2(axis[0], axis[1])
    dang = np.angle(np.exp(1j * (np.arctan2(ey - iy, ex - ix) - arm_ang)))
    sel = (d <= edge_band * R_seed) & (d >= 0.5 * R_seed) & (np.abs(dang) > np.deg2rad(arm_sector_deg) / 2)
    if sel.sum() < 10:
        raise ValueError("too few disc boundary pixels")
    cy, cx, R, c3, rms = _circle_fit(ey[sel].astype(float), ex[sel].astype(float))
    return dict(cy=float(cy), cx=float(cx), sigma_y=float(np.sqrt(c3[0, 0])), sigma_x=float(np.sqrt(c3[1, 1])), radius=float(R),
                sigma_radius=float(np.sqrt(c3[2, 2])), rms=float(rms), n_edge=int(sel.sum()), arm_direction_deg=float(np.rad2deg(arm_ang)),
                seed_yx=(int(iy), int(ix)), R_seed=R_seed)


def arm_geometry(mask: np.ndarray, seed_yx: tuple[int, int], border_px: int = 8) -> dict:
    """Arm axis (principal axis of the mask), the frame edge it enters from (if
    it reaches one), its half-width at mid-length, and the tip: the mask pixel
    farthest along the axis on the side of the dark-core seed (for a disc stop
    the far side of the disc, for a tapered blade the point)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        return dict(entry_edge=None, arm_direction_deg=np.nan, arm_half_width=np.nan, tip_yx=None, edge_pixels=0)
    cy, cx = ys.mean(), xs.mean()
    evals, evecs = np.linalg.eigh(np.cov(np.vstack([ys - cy, xs - cx])))
    axis = evecs[:, np.argmax(evals)]
    if (seed_yx[0] - cy) * axis[0] + (seed_yx[1] - cx) * axis[1] < 0:
        axis = -axis                                   # axis points from the centroid towards the stop head
    proj = (ys - cy) * axis[0] + (xs - cx) * axis[1]
    across = -(ys - cy) * axis[1] + (xs - cx) * axis[0]
    i = int(np.argmax(proj))
    lo, hi = np.percentile(proj, [20, 40])             # mid-length of the arm (the head sits at the top percentiles)
    mid = (proj >= lo) & (proj <= hi)
    half_w = float(0.5 * (np.percentile(across[mid], 97.5) - np.percentile(across[mid], 2.5))) if mid.sum() > 10 else np.nan
    touches = {"left": int((xs < border_px).sum()), "right": int((xs >= w - border_px).sum()), "top": int((ys < border_px).sum()), "bottom": int((ys >= h - border_px).sum())}
    edge = max(touches, key=touches.get)
    if touches[edge] == 0:
        edge = None
    return dict(entry_edge=edge, arm_direction_deg=float(np.rad2deg(np.arctan2(-axis[0], -axis[1]))), arm_half_width=half_w,
                tip_yx=(int(ys[i]), int(xs[i])), edge_pixels=touches[edge] if edge else 0)


def beamstop_mask_v1(summed: np.ndarray, sigma_px: float | None = None, core_box: float = 0.3, core_frac: float = 0.05,
                     frac: float = 0.15, close_px: int = 5, dilate_px: int = 3, border_px: int = 4,
                     n_pass: int = 2) -> tuple[np.ndarray, dict]:
    """True where the beam stop is (masks v1, see the block comment above).
    Returns (mask, info); info['disc'] carries the disc fit and 'is_disc',
    info['arm'] the arm geometry (entry edge, half-width, tip)."""
    raw = np.asarray(summed, dtype=np.float64)
    h, w = raw.shape
    if sigma_px is None:
        sigma_px = 3.0 * min(h, w) / 2048.0
    sm = ndi.gaussian_filter(raw, sigma_px)
    # 2. dark core in the central box
    hy, hx = int(h * core_box / 2), int(w * core_box / 2)
    sl = (slice(h // 2 - hy, h // 2 + hy), slice(w // 2 - hx, w // 2 + hx))
    box = sm[sl]
    thr0 = core_frac * float(np.percentile(box, 95))
    dark0 = box < thr0
    lab, n = ndi.label(dark0)
    if n == 0:
        raise ValueError("no dark core in the central box: no beam stop?")
    imin = np.unravel_index(np.argmin(np.where(dark0, box, np.inf)), box.shape)
    core = np.zeros((h, w), bool); core[sl] = lab == lab[imin]
    dt = ndi.distance_transform_edt(core)
    sy, sx = np.unravel_index(np.argmax(dt), dt.shape)
    r_seed = float(dt[sy, sx])
    # 3. shell rule about the seed
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - sy, xx - sx)
    rbin = r.astype(np.int64); n_shells = int(rbin.max()) + 1
    dark = core.copy()
    # noise floor: a shell is thresholded only where a halo pixel falls below the threshold with z >= 3,
    # z = (1 - frac) mu / sigma(sm), sigma(sm) ~ sqrt(mu / (4 pi sigma_px^2)) for Poisson counts smoothed by a Gaussian;
    # below mu = 9 / ((1 - frac)^2 4 pi sigma_px^2) (0.11 counts/px at sigma 3) a sparse sum has sm = 0 over large areas without a stop
    min_level = 9.0 / ((1 - frac) ** 2 * 4 * np.pi * sigma_px ** 2)
    for _ in range(n_pass):
        use = ~ndi.binary_dilation(dark, iterations=int(np.ceil(2 * sigma_px)))
        med = _shell_median_v1(sm, rbin, use, n_shells)
        thr = frac * np.maximum(med[rbin], 0.0)
        dark = ((sm < thr) & (med[rbin] > min_level)) | core
    # 4. closing, border, components touching the core, dilation
    cand = ndi.binary_closing(dark, structure=np.ones((close_px, close_px), bool)) | core
    cand[:border_px, :] = cand[-border_px:, :] = cand[:, :border_px] = cand[:, -border_px:] = False
    lab, n = ndi.label(cand)
    keep = np.unique(lab[core & (lab > 0)])
    mask = np.isin(lab, keep)
    mask_nodil = mask.copy()
    if dilate_px:
        mask = ndi.binary_dilation(mask, iterations=dilate_px)
    # 5. disc geometry
    try:
        disc = disc_from_mask(mask_nodil)
    except ValueError as e:
        disc = dict(error=str(e))
    tip = arm_geometry(mask_nodil, (int(sy), int(sx)))
    if "radius" in disc and np.isfinite(tip["arm_half_width"]):
        disc["is_disc"] = bool(disc["radius"] > 1.3 * tip["arm_half_width"])   # a tapered blade has no disc: the fit rides on the arm
    inside = raw[mask_nodil]
    ys, xs = np.nonzero(mask)
    info = dict(version="v1", sigma_px=float(sigma_px), core_box=core_box, core_frac=core_frac, frac=frac, close_px=close_px, dilate_px=dilate_px,
                border_px=border_px, n_pass=n_pass, min_level=float(min_level), core_threshold=thr0, core_pixels=int(core.sum()), seed_yx=(int(sy), int(sx)),
                r_seed=r_seed, n_components_kept=int(len(keep)), mask_pixels=int(mask.sum()), mask_fraction=float(mask.mean()),
                mask_pixels_before_dilation=int(mask_nodil.sum()),
                bbox=[int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None,
                stop_raw_mean=float(inside.mean()), stop_raw_median=float(np.median(inside)), stop_raw_p95=float(np.percentile(inside, 95)),
                stop_raw_min=float(inside.min()), disc=disc, arm=tip)
    return mask, info
