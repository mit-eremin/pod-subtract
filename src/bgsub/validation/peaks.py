# VENDORED COPY. Source: pod/background/src/podbg/peaks.py at background commit f1e93f5 (file last changed in 8715367);
# sha256 of the source file: d9aad0f1ef4cc011ce8828efbd41888402481df3ba3c942ca8faf28b262cb3e0. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: `from . import binning as B` -> `from ..vendor import binning as B`.
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Synthetic Bragg peaks of known intensity injected into real frames, and their integration (peak recovery v0, 2026-11-04).

Pre-registration: notes/reports/2026-11-04_peak_recovery_injection_v0.md Sec. 1.

Physics assumptions, next to the code that uses them:

- Spot shape: an isotropic 2-D Gaussian integrated over each output pixel (erf).  No separate detector PSF: the Apollo's
  super-resolution centroiding places each electron inside its 32-um bin-8 pixel; whatever PSF exists is inside the measured
  width of real spots, from which sigma is drawn.
- Injection happens at the ELECTRON level and then passes the stored-count law, so rounding, ties and dead time act on the
  peak as on real data.  The host frame stores K = Q(N) only; its electrons N are drawn from the posterior
  P(N | K) ~ Poisson(N; mu_h) T(K | N) with mu_h the host's local electron mean and T the quantiser
  (K = rint(N / m), m = b^2 / 16 e per count; an exact tie N / m = j + 1/2 goes to j + 1 with probability tie_up when the
  gain reference is applied to the movie, to the even neighbour (parity) otherwise; `binning.quantise`).
- Dead time (paralyzable, per output pixel per stored frame, tau_b = detector.tau_for_binning): the peak's electrons are
  counted with probability exp(-(mu_h + lambda_px) tau_b), and the host's with the extra survival exp(-lambda_px tau_b).
  This is the mean-field form of the paralyzable law (the same approximation as `forward.render(sample=True)`).
- Integration: I = epc x sum_box (K' - B), epc = electrons per count; B the method's background in stored counts.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf, gammaln

from ..vendor import binning as B


# ----------------------------------------------------------------------------- geometry of a peak
def spot_pixels(cy: float, cx: float, sigma: float, total: float, half: int):
    """Expected electrons per pixel of a pixel-integrated isotropic Gaussian of `total` electrons centred at (cy, cx)
    (pixel centres at integers), on the (2 half + 1)^2 patch about (round(cy), round(cx)).
    Returns (y0, x0, lam) with lam[i, j] at pixel (y0 + i, x0 + j)."""
    y0 = int(round(cy)) - half; x0 = int(round(cx)) - half
    ey = np.arange(y0, y0 + 2 * half + 2) - 0.5; ex = np.arange(x0, x0 + 2 * half + 2) - 0.5
    sq = np.sqrt(2.0) * sigma
    py = np.diff(0.5 * (1 + erf((ey - cy) / sq))); px = np.diff(0.5 * (1 + erf((ex - cx) / sq)))
    return y0, x0, total * np.outer(py, px)


def patch_radius(y0: int, x0: int, n: int, cy: float, cx: float) -> np.ndarray:
    yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n]
    return np.hypot(yy - cy, xx - cx)


def box_and_annulus(rpatch: np.ndarray, r_box: float, r_in: float, r_out: float):
    return rpatch <= r_box, (rpatch > r_in) & (rpatch <= r_out)


# ----------------------------------------------------------------------------- the stored-count quantiser and its posterior
def quantiser_prob(N: np.ndarray, K: np.ndarray, m: float, tie_up: float | None) -> np.ndarray:
    """T(K | N) of `binning.quantise` (rule rint_even with the tie rule), elementwise."""
    N = np.asarray(N); K = np.asarray(K)
    x = N / m; lo = np.floor(x + 1e-9)
    tie = np.isclose(x - lo, 0.5)
    base = np.rint(x)
    p = (base == K).astype(float)
    if tie_up is not None:
        p = np.where(tie, np.where(K == lo + 1, tie_up, np.where(K == lo, 1.0 - tie_up, 0.0)), p)
    return p


def sample_latent(K: np.ndarray, mu: np.ndarray, binning: int, tie_up: float | None, rng) -> np.ndarray:
    """Draw host electrons N per pixel from P(N | K) ~ Poisson(N; mu) T(K | N).  At bin 4 (m = 1) N = K exactly."""
    m = B.electrons_per_count(binning)
    K = np.asarray(K, np.int64)
    if m == 1:
        return K.copy()
    mi = int(round(m))
    off = np.arange(-(mi // 2), mi // 2 + 1)                     # N in m K + off covers every N that can give K
    N = mi * K[..., None] + off
    mu = np.maximum(np.asarray(mu, float), 1e-6)[..., None]
    lp = np.where(N >= 0, N * np.log(mu) - mu - gammaln(np.maximum(N, 0) + 1.0), -np.inf)
    T = quantiser_prob(N, K[..., None], m, tie_up)
    with np.errstate(divide="ignore"):
        lw = lp + np.log(T)
    lw = lw - np.max(lw, axis=-1, keepdims=True)
    w = np.exp(lw); w = w / w.sum(-1, keepdims=True)
    u = rng.random(K.shape)[..., None]
    j = (np.cumsum(w, -1) < u).sum(-1)
    j = np.minimum(j, len(off) - 1)
    return np.take_along_axis(N, j[..., None], -1)[..., 0]


def inject(K: np.ndarray, mu_h: np.ndarray, lam: np.ndarray, binning: int, tie_up: float | None, tau_b: float, rng,
           N_known: np.ndarray | None = None) -> np.ndarray:
    """Stored values after adding peak electrons `lam` (expected, per pixel) to a host patch with stored values K.
    N_known: the host's true electrons (synthetic check); otherwise drawn from the posterior."""
    m = B.electrons_per_count(binning)
    K = np.asarray(K, np.int64)
    Nh0 = np.asarray(N_known, np.int64) if N_known is not None else sample_latent(K, mu_h, binning, tie_up, rng)
    lam = np.asarray(lam, float)
    if tau_b > 0:
        surv_p = np.exp(-(np.asarray(mu_h, float) + lam) * tau_b)
        n = rng.poisson(lam * surv_p)
        Nh = rng.binomial(Nh0, np.exp(-lam * tau_b))
    else:
        n = rng.poisson(lam); Nh = Nh0
    if m == 1:
        return (Nh + n).astype(np.int64)
    out = B.quantise(Nh + n, binning, tie_up, rng=rng).astype(np.int64)
    # a pixel whose electron number did not change keeps its recorded value (its rounding already happened once)
    return np.where((Nh + n) == Nh0, K, out)


# ----------------------------------------------------------------------------- variance of a stored value
class StoredVariance:
    """g(m): variance of a stored value whose mean is m (counts), from the response table at the session's tie rule.
    Bin 4 (1 count = 1 e, no rounding): g(m) = m."""

    def __init__(self, binning: int, tie_up: float | None):
        self.binning = binning
        if B.electrons_per_count(binning) == 1:
            self.e = None
        else:
            _, e, v = B.response_table(binning, tie_up, 0.0, 0.0, mu_min=1e-4, mu_max=2000.0, n=900)
            self.e, self.v = e, v

    def __call__(self, m):
        m = np.maximum(np.asarray(m, float), 0.0)
        if self.e is None:
            return m
        return np.interp(m, self.e, self.v, left=0.0)


# ----------------------------------------------------------------------------- integration
def integrate(Kp: np.ndarray, Bpx: np.ndarray | None, box: np.ndarray, ann: np.ndarray, epc: float, g: StoredVariance):
    """One peak.  Bpx = the method's background (stored counts) on the patch, or None for the local annulus (B = mean of
    Kp over `ann`).  Returns (I, sigma_naive, sigma_law, B_box) with B_box = sum of the background over the box (counts)."""
    Kb = Kp[box].astype(float); nb = box.sum()
    if Bpx is None:
        Ka = Kp[ann].astype(float); na = len(Ka)
        b = Ka.mean(); Bb = np.full(nb, b)
        var_naive = Kb.sum() + nb ** 2 * Ka.sum() / na ** 2
        var_law = g(np.maximum(Kb, b)).sum() + nb ** 2 * g(Ka).sum() / na ** 2
    else:
        Bb = np.asarray(Bpx, float)[box]
        var_naive = Kb.sum()
        var_law = g(np.maximum(Kb, Bb)).sum()
    I = epc * (Kb.sum() - Bb.sum())
    return float(I), float(epc * np.sqrt(max(var_naive, 0.0))), float(epc * np.sqrt(max(var_law, 0.0))), float(Bb.sum())


# ----------------------------------------------------------------------------- radial profiles on patches
def profile_at(profile: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Radial profile (index i = shell [i, i+1) px, mean radius i + 0.5) evaluated at radius r: linear interpolation at
    r - 0.5; NaN shells filled from the nearest finite neighbours (as atlas.render_background)."""
    P = np.asarray(profile, float).copy(); fin = np.isfinite(P)
    if not fin.any():
        return np.full(np.shape(r), np.nan)
    idx = np.arange(len(P))
    P = np.interp(idx, idx[fin], P[fin])
    return np.interp(np.asarray(r, float) - 0.5, idx, P)


def shell_means(img: np.ndarray, valid: np.ndarray, centre, n_shell: int):
    """Stage 0 convention: shell r = floor(hypot), mean over valid pixels.  Returns (mean, n_px)."""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    ri = np.floor(np.hypot(yy - centre[0], xx - centre[1])).astype(int)
    use = valid & (ri < n_shell)
    n = np.bincount(ri[use], minlength=n_shell)[:n_shell].astype(float)
    t = np.bincount(ri[use], weights=img[use].astype(float), minlength=n_shell)[:n_shell]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, t / n, np.nan), n


def window_scale(target: np.ndarray, n_target: np.ndarray, ref: np.ndarray, s: np.ndarray, lo: float = 0.34, hi: float = 0.80) -> float:
    """h / p: pixel-weighted mean of `target` over s in [lo, hi) divided by the same of `ref` (weights n_target)."""
    sel = (s >= lo) & (s < hi) & np.isfinite(target) & np.isfinite(ref) & (n_target > 0)
    w = n_target[sel]
    return float(np.sum(w * target[sel]) / max(np.sum(w * ref[sel]), 1e-12))


# ----------------------------------------------------------------------------- summaries
def robust_sd(x) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def within(medians_by_session, median_pooled, x: float, frac: float = 2.0 / 3.0) -> bool:
    """The pre-registered WITHIN-X rule: |pooled median - 1| <= x AND >= frac of sessions within x."""
    m = np.asarray(medians_by_session, float); m = m[np.isfinite(m)]
    if not np.isfinite(median_pooled) or len(m) == 0:
        return False
    return bool(abs(median_pooled - 1) <= x and np.mean(np.abs(m - 1) <= x) >= frac)


def pb_star(cells: dict, ladder, x: float):
    """Smallest P/B on the ladder such that it and every higher P/B pass `within` (cells[pb] = (session medians, pooled))."""
    ok = [within(*cells[pb], x) if pb in cells else False for pb in ladder]
    star = None
    for i in range(len(ladder) - 1, -1, -1):
        if ok[i]:
            star = ladder[i]
        else:
            break
    return star
