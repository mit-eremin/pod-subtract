# VENDORED COPY. Source: pod/background/src/podbg/centre.py at background commit f1e93f5 (file last changed in fe1c2b9);
# sha256 of the source file: 9d4773eaf68f7cd2b30de247ccb11ceee1f913234f32eaaa3c340f43b8da5686. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: none (byte-identical below this header; podbg's copy is itself vendored from ringcal.centre).
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Beam-centre estimation by point symmetry (podbg copy).

Vendored copy of ringcal.centre.centre_from_symmetry with its helpers (CentreResult, _prep, _rot_corr, _best_shift,
_paraboloid_peak), from ringcal/src/ringcal/centre.py at ringcal commit 38247ae (file last changed in 7751a9e;
sha256 of the source file 996367216ec282dfa4ecefea3f7231766f75b06f5933ebaaf2324ef6ac45f8e0), taken 2026-10-30 so that
podbg does not import the sibling repo at run time. The code below is unchanged from the source; only the estimators
(a) centre_from_mask and (c) centre_from_ring were left out. Keep the two copies in step when either changes
(tests/test_centre.py checks identity against ringcal when ringcal is importable).

The centre of point symmetry is the point about which the image is most nearly invariant under a 180-degree rotation
(the halo and the rings are centrosymmetric; a one-sided stop arm is excluded by the mask). The score is the mean
squared difference between the Anscombe-transformed image (2 sqrt(I + 3/8), unit Poisson variance) and its rotation
over the pixels valid in both; its noise floor is the same for every candidate domain, so the minimum is unbiased by
the stop (a Pearson correlation on log1p was biased by ~1 px towards the stop centre on synthetic frames, 2026-10-25).
Coarse-to-fine search; sub-pixel position from a paraboloid fit; uncertainty from a block bootstrap of the fine stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

@dataclass
class CentreResult:
    cy: float
    cx: float
    sigma_y: float
    sigma_x: float
    method: str
    info: dict = field(default_factory=dict)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.cy, self.cx)


# ------------------------------------------------------------- (b) point symmetry

def _prep(img: np.ndarray, mask: np.ndarray | None, median_px: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(img, dtype=np.float64)
    valid = np.isfinite(a)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    a = np.where(valid, a, 0.0)
    if median_px > 1:
        a = median_filter(a, size=median_px)
    a = 2.0 * np.sqrt(np.maximum(a, 0.0) + 0.375)      # Anscombe: Var ~ 1 for Poisson counts
    return a, valid


def _rot_corr(a: np.ndarray, valid: np.ndarray, af: np.ndarray, vf: np.ndarray, dy: int, dx: int) -> float:
    """Symmetry score (higher is better): minus the mean squared difference
    between a and its 180-degree rotation about ((N-1+dy)/2, (M-1+dx)/2), i.e.
    the flipped image shifted by (dy, dx), over pixels valid in both."""
    ny, nx = a.shape
    ys = slice(max(dy, 0), ny + min(dy, 0)); yf = slice(max(-dy, 0), ny + min(-dy, 0))
    xs = slice(max(dx, 0), nx + min(dx, 0)); xf = slice(max(-dx, 0), nx + min(-dx, 0))
    v = valid[ys, xs] & vf[yf, xf]
    n = v.sum()
    if n < 100:
        return -np.inf
    d = a[ys, xs][v] - af[yf, xf][v]
    return -float(d @ d) / n


def _best_shift(a, valid, af, vf, dys, dxs):
    C = np.array([[_rot_corr(a, valid, af, vf, int(dy), int(dx)) for dx in dxs] for dy in dys])
    i, j = np.unravel_index(np.argmax(C), C.shape)
    return C, int(dys[i]), int(dxs[j])


def _paraboloid_peak(C: np.ndarray, dys, dxs):
    """Sub-sample peak of a 2-D surface by least-squares paraboloid over all samples."""
    Y, X = np.meshgrid(dys, dxs, indexing="ij")
    A = np.column_stack([np.ones(Y.size), Y.ravel(), X.ravel(), Y.ravel() ** 2, X.ravel() ** 2, (Y * X).ravel()])
    c, *_ = np.linalg.lstsq(A, C.ravel(), rcond=None)
    H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    if np.linalg.det(H) <= 0 or c[3] >= 0:
        i, j = np.unravel_index(np.argmax(C), C.shape)
        return float(dys[i]), float(dxs[j]), False
    sol = -np.linalg.solve(H, np.array([c[1], c[2]]))
    return float(sol[0]), float(sol[1]), True


def centre_from_symmetry(img: np.ndarray, mask: np.ndarray | None = None, init: tuple[float, float] | None = None,
                         search_px: float = 60.0, median_px: int = 1, coarse: int = 4, fine_half: int = 3,
                         n_boot: int = 16, block: int = 64, seed: int = 0) -> CentreResult:
    """Centre of point symmetry of the image (see module docstring).

    init: starting guess (default: frame centre). search_px: half-width of the
    coarse search around init. The fine stage evaluates integer shifts within
    +-fine_half px of the coarse optimum at full resolution and fits a
    paraboloid for the sub-pixel position. Uncertainty: std over n_boot block
    bootstrap replicates (blocks of `block` px) of the fine stage.
    """
    a, valid = _prep(img, mask, median_px)
    ny, nx = a.shape
    if init is None:
        init = ((ny - 1) / 2, (nx - 1) / 2)
    af = a[::-1, ::-1]; vf = valid[::-1, ::-1]
    # shift (dy, dx) of the flipped image <-> centre ((ny-1+dy)/2, (nx-1+dx)/2)
    dy0 = 2 * init[0] - (ny - 1); dx0 = 2 * init[1] - (nx - 1)
    # coarse stage on a block-averaged image (coarse pixel i covers full-res
    # pixels [i*coarse, (i+1)*coarse), centre at i*coarse + (coarse-1)/2)
    nyc, nxc = ny // coarse, nx // coarse
    ac = a[:nyc * coarse, :nxc * coarse].reshape(nyc, coarse, nxc, coarse).mean(axis=(1, 3))
    vc = valid[:nyc * coarse, :nxc * coarse].reshape(nyc, coarse, nxc, coarse).all(axis=(1, 3))
    acf = ac[::-1, ::-1]; vcf = vc[::-1, ::-1]

    def to_coarse_shift(d_full, n_full, n_c):     # centre (n_full-1+d)/2 -> coarse shift
        c_full = (n_full - 1 + d_full) / 2
        c_c = (c_full - (coarse - 1) / 2) / coarse
        return 2 * c_c - (n_c - 1)

    def to_full_shift(d_c, n_full, n_c):
        c_c = (n_c - 1 + d_c) / 2
        c_full = c_c * coarse + (coarse - 1) / 2
        return 2 * c_full - (n_full - 1)

    half = int(np.ceil(2 * search_px / coarse))
    dys = np.arange(round(to_coarse_shift(dy0, ny, nyc)) - half, round(to_coarse_shift(dy0, ny, nyc)) + half + 1)
    dxs = np.arange(round(to_coarse_shift(dx0, nx, nxc)) - half, round(to_coarse_shift(dx0, nx, nxc)) + half + 1)
    Cc, by, bx = _best_shift(ac, vc, acf, vcf, dys, dxs)
    by = int(round(to_full_shift(by, ny, nyc))); bx = int(round(to_full_shift(bx, nx, nxc)))
    dys = np.arange(by - fine_half, by + fine_half + 1); dxs = np.arange(bx - fine_half, bx + fine_half + 1)
    Cf, fy, fx = _best_shift(a, valid, af, vf, dys, dxs)
    if (fy, fx) != (by, bx):   # re-centre the fine window on the discrete optimum once
        dys = np.arange(fy - fine_half, fy + fine_half + 1); dxs = np.arange(fx - fine_half, fx + fine_half + 1)
        Cf, fy, fx = _best_shift(a, valid, af, vf, dys, dxs)
    sy, sx, ok = _paraboloid_peak(Cf, dys, dxs)
    cy = (ny - 1 + sy) / 2; cx = (nx - 1 + sx) / 2
    # block bootstrap of the fine stage
    rng = np.random.default_rng(seed)
    by_ = np.arange(0, ny, block); bx_ = np.arange(0, nx, block)
    blocks = [(y, x) for y in by_ for x in bx_]
    cys, cxs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        wmask = np.zeros_like(valid)
        for k in pick:
            y, x = blocks[k]; wmask[y:y + block, x:x + block] = True
        v2 = valid & wmask
        d2 = np.arange(fy - 2, fy + 3); x2 = np.arange(fx - 2, fx + 3)
        C2, _, _ = _best_shift(a, v2, af, v2[::-1, ::-1], d2, x2)
        s2y, s2x, _ = _paraboloid_peak(C2, d2, x2)
        cys.append((ny - 1 + s2y) / 2); cxs.append((nx - 1 + s2x) / 2)
    return CentreResult(cy, cx, float(np.std(cys, ddof=1)), float(np.std(cxs, ddof=1)), "symmetry",
                        dict(score_max=float(Cf.max()), paraboloid_ok=ok, coarse_score_max=float(Cc.max()),
                             coarse_centre=((ny - 1 + by) / 2, (nx - 1 + bx) / 2)))
