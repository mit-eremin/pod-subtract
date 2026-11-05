# VENDORED COPY. Source: pod/background/src/podbg/binning.py at background commit f1e93f5 (file last changed in e14cb38);
# sha256 of the source file: 03adf83dc6a840d80fb31b90aaf1954772d0f1aabbe0d9823f617da2ef3cd2ed. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: none (byte-identical below this header).
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Apollo hardware-binning response: how counted electrons become stored counts.

Physics and server assumptions, written next to the code that uses them
(measured 2026-10-26, notes/reports/2026-10-26_binning_response_v0.md):

* The DE server counts events in hardware (one event = one counted electron,
  "Feature - Hardware Counting = On", "Centroiding Mode = Super-resolution"),
  places each event on the 8192 x 8192 super-resolution grid with a value of
  16, SUMS the `n_raw` = 3 raw frames of a stored frame, bins by AVERAGING
  ("Binning Method = Average") over the b x b super-resolution block of an
  output pixel at binning b, and stores the result as an unsigned 8-bit
  integer. The autosaved movie frames are NOT divided by 2: at bin 4 the
  stored sum equals the server's own event count ("Event Counting -
  Sparseness" x 4096^2 x 3) to 0.03 % in three sessions with 1 count per
  event, so
      counts_per_electron(b) = 16 / b**2          (bin 4: 1, bin 8: 1/4, bin 16: 1/16)
  i.e. `electrons_per_count` = 1, 4, 16. The mdoc's CountsPerElectron
  (0.5 / 0.125 / 0.03125) is half of this: it describes SerialEM's own
  DividedBy2 copy of the image, not the DE autosave. `podbg.detector` still
  carries the 8 e/count convention of the 2026-10-25 report; all electron
  figures derived with it are 2x too high.

* Quantisation. The averaged value c N (N = counted electrons in the output
  pixel in the stored frame, c = counts per electron) is converted to an
  integer with ROUND-HALF-TO-EVEN (the IEEE default of rint / cvRound /
  __float2int_rn), not floor and not round-half-up. With m = 1/c even, level
  k is reached when c N >= k - 1/2, i.e. N >= t_k = m k - m/2; the exact tie
  N = t_k rounds to k when k is even and to k - 1 when k is odd. The
  thresholds at bin 8 are therefore N = 3, 6, 11, 14, 19, 22, ... and at
  bin 16 N = 9, 24, 41, 56, ... (gaps alternate m - 1, m + 1). The three raw
  frames are summed BEFORE the rounding (a per-raw-frame rounding would need
  N >= 3 electrons in one 16-ms raw frame and is excluded by the data).

* Gain on movie. When the server applies the integrating gain reference to
  the movie frames ("Image Processing - Apply Gain on Movie = Enable", 20 of
  23 sessions) every event carries its super-resolution pixel's gain g_i
  (mean 1.000, p1-p99 0.81-1.18) and the rounded value is c sum g_i. Only the
  exact ties are affected: a tie rounds up with probability `tie_up` ~ P(mean
  g over the tied events > 1) ~ 0.5 instead of the parity rule. This is the
  only difference between "Enable" and "Disable" sessions in this model and
  it is what made the "Disable" sessions look steep in the halo survey.

* Statistics. Per pixel and stored frame N ~ Poisson(mu). Where a population
  of pixels/frames with different mu is averaged (an azimuthal mean over
  frames of different film thickness), mu is taken Gamma-distributed with
  coefficient of variation `cv`, so N is negative binomial; cv = 0 is the
  pure Poisson case. The mean recorded value is convex in mu below the first
  threshold, so a spread in mu raises the recorded mean: inverting with the
  wrong cv biases the result (quantified in the report).

Everything here is the stored-count statistics of the output pixel given the
counted electrons; coincidence loss (`podbg.detector`) acts on the counting
BEFORE this and is not included unless `dead_time_tau` is passed to the
helpers that accept it.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats, optimize

SUPERRES_VALUE_PER_EVENT = 16          # value placed on the 8192 grid per counted event
RAW_FRAMES_PER_STORED = 3
BINNINGS = (4, 8, 16)
RULES = ("rint_even", "floor", "half_up", "rint_even_raw", "floor_raw", "half_up_raw")


def counts_per_electron(binning: int) -> float:
    """Stored counts per counted electron in the DE autosave at `binning` (16 / b^2)."""
    return SUPERRES_VALUE_PER_EVENT / float(binning) ** 2


def electrons_per_count(binning: int) -> float:
    return 1.0 / counts_per_electron(binning)


# ----------------------------------------------------------------------------- electron-number law
def _electron_pmf(mu: float, cv: float, n_max: int) -> np.ndarray:
    """P(N = n), n = 0..n_max, for N ~ Poisson(mu) (cv = 0) or the Gamma-Poisson
    mixture with coefficient of variation cv of mu (negative binomial)."""
    n = np.arange(n_max + 1)
    if mu <= 0:
        p = np.zeros(n_max + 1); p[0] = 1.0; return p
    if cv <= 0:
        return stats.poisson.pmf(n, mu)
    r = 1.0 / cv ** 2
    return stats.nbinom.pmf(n, r, r / (r + mu))


def _n_max(mu: float, cv: float) -> int:
    sd = np.sqrt(mu + (cv * mu) ** 2)
    return int(max(20, mu + 12 * sd + 20))


def thresholds(binning: int, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Electron numbers at which the stored value first reaches k = 1..k_max under
    round-half-to-even of c N, as (t_k, tie_is_counted) with t_k = m k - m/2
    (m = electrons per count). At an exact tie the value is k only when k is
    even. For m = 1 (bin 4) there are no ties (t_k = k - 1/2)."""
    m = electrons_per_count(binning)
    k = np.arange(1, k_max + 1)
    t = m * k - m / 2.0
    return t, (k % 2 == 0)


def recorded_pmf(mu: float, binning: int, tie_up: float | None = None, cv: float = 0.0,
                 rule: str = "rint_even", m: float | None = None, k_max: int | None = None) -> np.ndarray:
    """P(stored value = k), k = 0.., for an output pixel with mean electron number mu.

    rule: 'rint_even' (the measured law; `tie_up` None = parity rule, else the
    probability that an exact tie rounds up), 'floor', 'half_up' (alternatives
    H1/H3 with the same m), and the '_raw' variants that round each of the 3
    raw frames separately and sum (H2). `m` overrides electrons per count
    (default 16 / b^2 ... e.g. m = 8 tests the old 8 e/count convention).
    """
    m = electrons_per_count(binning) if m is None else float(m)
    if rule.endswith("_raw"):
        base = rule[:-4]
        mu_r = mu / RAW_FRAMES_PER_STORED
        p1 = recorded_pmf(mu_r, binning, tie_up, cv, base, m, k_max)
        out = p1.copy()
        for _ in range(RAW_FRAMES_PER_STORED - 1):
            out = np.convolve(out, p1)
        return out
    n_max = _n_max(mu, cv)
    pN = _electron_pmf(mu, cv, n_max)
    n = np.arange(n_max + 1)
    x = n / m
    if rule == "floor":
        val = np.floor(x + 1e-9)
    elif rule == "half_up":
        val = np.floor(x + 0.5 + 1e-9)
    elif rule == "rint_even":
        val = np.rint(x)                                    # numpy rint = ties to even
        if tie_up is not None:
            # exact ties: value x = k - 1/2 -> k with prob tie_up, k - 1 otherwise
            frac = x - np.floor(x)
            tie = np.isclose(frac, 0.5)
            lo = np.floor(x)
            kk = int(val.max()) + 2
            pmf = np.zeros(kk + 1)
            np.add.at(pmf, val[~tie].astype(int), pN[~tie])
            np.add.at(pmf, lo[tie].astype(int), pN[tie] * (1 - tie_up))
            np.add.at(pmf, lo[tie].astype(int) + 1, pN[tie] * tie_up)
            return pmf if k_max is None else _pad(pmf, k_max)
    else:
        raise ValueError(rule)
    kk = int(val.max()) + 1
    pmf = np.zeros(kk + 1)
    np.add.at(pmf, val.astype(int), pN)
    return pmf if k_max is None else _pad(pmf, k_max)


def _pad(p: np.ndarray, k_max: int) -> np.ndarray:
    out = np.zeros(k_max + 1)
    n = min(len(p), k_max + 1)
    out[:n] = p[:n]
    if len(p) > k_max + 1:
        out[k_max] += p[k_max + 1:].sum()
    return out


def _moments(pmf: np.ndarray) -> tuple[float, float]:
    k = np.arange(len(pmf))
    mean = float((k * pmf).sum())
    var = float((k * k * pmf).sum() - mean ** 2)
    return mean, max(var, 0.0)


def recorded_mean(true_mean, binning: int, tie_up: float | None = None, cv: float = 0.0,
                  rule: str = "rint_even", m: float | None = None, dead_time_tau: float = 0.0):
    """E[stored value] per output pixel per stored frame for true mean electron
    number `true_mean` (per output pixel per stored frame). Vectorised in
    true_mean. `dead_time_tau` (per electron per output pixel per stored
    frame) applies the paralyzable counting loss before the quantisation."""
    mu = np.atleast_1d(np.asarray(true_mean, float))
    if dead_time_tau:
        mu = mu * np.exp(-mu * dead_time_tau)
    out = np.array([_moments(recorded_pmf(float(u), binning, tie_up, cv, rule, m))[0] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def recorded_var(true_mean, binning: int, tie_up: float | None = None, cv: float = 0.0,
                 rule: str = "rint_even", m: float | None = None):
    mu = np.atleast_1d(np.asarray(true_mean, float))
    out = np.array([_moments(recorded_pmf(float(u), binning, tie_up, cv, rule, m))[1] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def fano(true_mean, binning: int, **kw):
    """Variance / mean of the stored value (1 for Poisson counts)."""
    mu = np.atleast_1d(np.asarray(true_mean, float))
    mean = recorded_mean(mu, binning, **kw); var = recorded_var(mu, binning, **kw)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(mean > 0, var / mean, np.nan)
    return f if np.ndim(true_mean) else float(f[0])


def linear_mean(true_mean, binning: int):
    """What a linear detector would store: c * mu."""
    return np.asarray(true_mean, float) * counts_per_electron(binning)


def efficiency(true_mean, binning: int, **kw):
    """recorded_mean / linear_mean: the fraction of the signal that survives the quantisation."""
    return recorded_mean(true_mean, binning, **kw) / linear_mean(true_mean, binning)


# ----------------------------------------------------------------------------- inversion
# Tie-constant systematic (audit E6, 2026-10-28).  The tie probabilities (TIE_UP_V1 = 0.35 for gain on; the v2 rate law's
# p_hi 0.403 / p_hi_film_scale_free 0.436) are FITTED on film / bar / vacuum histograms, not derived from the firmware,
# and every inversion depends on them: inverting a p = 0.35 mean with p = 0.40 scales mu by 0.944 / 0.954 / 0.964 /
# 0.976 at 0.2 / 0.5 / 1 / 2 e per bin-8 px per frame (review Sec. 2, #7).  TIE_UP_ERR is the adopted 1-sigma
# uncertainty of the constant: 0.05 covers the v0 value 0.35 against the v2 film value 0.403 and the free-scale 0.436.
# It is ONE number per gain flag, so its effect is CORRELATED across every pixel, radius and frame inverted with it;
# `true_mean*(..., tie_up_err=...)` returns the inverted mean together with that correlated systematic (`sigma_tie`,
# half the spread of the inversion at p +/- TIE_UP_ERR) and its derivative `dmu_dp`; the consumers carry it as one
# correlated nuisance (atlas_v1.residual_z `tie_up_err`, forward.invert_profile / fit_profile) rather than as an
# uncorrelated per-point error.
TIE_UP_ERR = 0.05


@dataclass
class Inversion:
    true_mean: np.ndarray        # electrons per output pixel per stored frame
    sigma: np.ndarray            # 1-sigma from the counting noise of the recorded mean (NaN if n_eff unknown)
    amplification: np.ndarray   # d ln mu / d ln E: relative-error amplification of the inversion
    recorded_mean: np.ndarray
    ok: np.ndarray               # finite and inside the invertible range
    sigma_tie: np.ndarray | None = None   # CORRELATED systematic from the tie probability: |mu(p + err) - mu(p - err)| / 2 (0 for the parity rule)
    dmu_dp: np.ndarray | None = None      # d mu / d p_tie (electrons per unit tie probability), the sign of the correlation


def _mean_scalar(mu, binning, tie_up, cv, dead_time_tau):
    return recorded_mean(mu, binning, tie_up, cv, dead_time_tau=dead_time_tau)


def response_table(binning: int, tie_up: float | None = None, cv: float = 0.0, dead_time_tau: float = 0.0,
                   mu_min: float = 1e-3, mu_max: float = 2000.0, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E[stored | mu] and Var on a log grid of mu (for the fast, vectorised inversion)."""
    mu = np.geomspace(mu_min, mu_max, n)
    e = recorded_mean(mu, binning, tie_up, cv, dead_time_tau=dead_time_tau)
    v = recorded_var(mu * np.exp(-mu * dead_time_tau) if dead_time_tau else mu, binning, tie_up, cv)
    return mu, e, v


_TIE_TABLES: dict = {}


def _table_at(binning, tie_up, cv, dead_time_tau, like=None):
    """A response table for a shifted tie probability, on the grid of `like` (cached)."""
    mu_min, mu_max, n = (1e-3, 2000.0, 600) if like is None else (float(like[0][0]), float(like[0][-1]), int(len(like[0])))
    key = (int(binning), None if tie_up is None else round(float(tie_up), 4), round(float(cv), 4), round(float(dead_time_tau), 8), mu_min, mu_max, n)
    if key not in _TIE_TABLES:
        _TIE_TABLES[key] = response_table(binning, tie_up, cv, dead_time_tau, mu_min=mu_min, mu_max=mu_max, n=n)
    return _TIE_TABLES[key]


def _with_tie_systematic(inv: Inversion, recorded, binning, tie_up, cv, dead_time_tau, tie_up_err, table, fast: bool) -> Inversion:
    """Fill `sigma_tie` / `dmu_dp` of an inversion by re-inverting at p +/- tie_up_err (no change for the parity rule)."""
    rec = np.atleast_1d(np.asarray(recorded, float)); scalar = np.ndim(recorded) == 0
    if tie_up is None or not tie_up_err:
        z = np.zeros(rec.shape)
        inv.sigma_tie = z[0] if scalar else z; inv.dmu_dp = (z[0] if scalar else z).copy()
        return inv
    p_hi = min(float(tie_up) + float(tie_up_err), 1.0); p_lo = max(float(tie_up) - float(tie_up_err), 0.0)
    if fast:
        hi = true_mean_fast(rec, binning, p_hi, cv, None, dead_time_tau, table=_table_at(binning, p_hi, cv, dead_time_tau, table)).true_mean
        lo = true_mean_fast(rec, binning, p_lo, cv, None, dead_time_tau, table=_table_at(binning, p_lo, cv, dead_time_tau, table)).true_mean
    else:
        hi = true_mean(rec, binning, p_hi, cv, None, dead_time_tau).true_mean
        lo = true_mean(rec, binning, p_lo, cv, None, dead_time_tau).true_mean
    hi = np.atleast_1d(np.asarray(hi, float)); lo = np.atleast_1d(np.asarray(lo, float))
    d = (hi - lo) / (p_hi - p_lo)
    st = np.abs(hi - lo) / 2.0
    inv.sigma_tie = st[0] if scalar else st; inv.dmu_dp = d[0] if scalar else d
    return inv


def tie_systematic(recorded, binning: int, tie_up: float | None, cv: float = 0.0, dead_time_tau: float = 0.0,
                   tie_up_err: float = TIE_UP_ERR, table=None) -> Inversion:
    """Inverted mean with the correlated tie-constant systematic (`sigma_tie`, `dmu_dp`), by the fast table inversion."""
    return true_mean_fast(recorded, binning, tie_up, cv, None, dead_time_tau, table=table, tie_up_err=tie_up_err)


def true_mean_fast(recorded, binning: int, tie_up: float | None = None, cv: float = 0.0, n_eff=None,
                   dead_time_tau: float = 0.0, table=None, tie_up_err: float | None = None) -> Inversion:
    """Vectorised inversion by interpolation of `response_table` (log-log, monotonic
    part only; relative accuracy ~1e-4). Same contract as `true_mean`.  With `tie_up_err`
    the inversion is repeated at p +/- tie_up_err and the correlated systematic is returned
    in `sigma_tie` / `dmu_dp` (audit E6; see TIE_UP_ERR)."""
    mu, e, v = response_table(binning, tie_up, cv, dead_time_tau) if table is None else table
    full_table = (mu, e, v)
    # keep the monotonic rising branch (dead time can make e turn over)
    last = int(np.argmax(e)) + 1
    mu, e, v = mu[:last], e[:last], v[:last]
    rec = np.atleast_1d(np.asarray(recorded, float))
    out = np.full(rec.shape, np.nan); sig = np.full(rec.shape, np.nan); amp = np.full(rec.shape, np.nan)
    ok = np.isfinite(rec) & (rec > e[0]) & (rec < e[-1])
    le, lm = np.log(e), np.log(mu)
    out[ok] = np.exp(np.interp(np.log(rec[ok]), le, lm))
    slope = np.gradient(le, lm)                       # d ln E / d ln mu
    amp[ok] = 1.0 / np.interp(np.log(rec[ok]), le, slope)
    if n_eff is not None:
        ne = np.broadcast_to(np.asarray(n_eff, float), rec.shape)
        var = np.interp(np.log(rec[ok]), le, v)
        d = rec[ok] / out[ok] / amp[ok]               # dE/dmu = (E/mu) * dlnE/dlnmu
        sig[ok] = np.sqrt(var / ne[ok]) / d
    inv = Inversion(out[0], sig[0], amp[0], rec[0], ok[0]) if np.ndim(recorded) == 0 else Inversion(out, sig, amp, rec, ok)
    if tie_up_err is not None:
        inv = _with_tie_systematic(inv, recorded, binning, tie_up, cv, dead_time_tau, tie_up_err, full_table, fast=True)
    return inv


def true_mean(recorded, binning: int, tie_up: float | None = None, cv: float = 0.0, n_eff=None,
              dead_time_tau: float = 0.0, mu_max: float = 2000.0, tie_up_err: float | None = None) -> Inversion:
    """Invert E[stored | mu] for the mean stored value `recorded` (counts per
    output pixel per stored frame, an average over `n_eff` independent
    pixel-frames). Returns the true mean electron number per output pixel per
    stored frame with its counting-noise sigma and the error amplification.

    Ill-conditioning: the forward curve is monotonic, so every recorded mean
    in (0, c mu_max) has one solution, but below the first threshold E ~
    P(N >= t_1) is a steep power of mu: the relative error is DAMPED (amplification
    < 1) while any additive contamination (stray floor, seam leakage, a
    mixture of fluxes) is magnified. Values <= 0 return NaN. With dead time the
    recorded mean has a ceiling; values above it return NaN.  With `tie_up_err` the
    correlated tie-constant systematic is returned as well (`sigma_tie`, `dmu_dp`; audit E6).
    """
    rec = np.atleast_1d(np.asarray(recorded, float))
    out = np.full(rec.shape, np.nan); sig = np.full(rec.shape, np.nan); amp = np.full(rec.shape, np.nan)
    lo = 1e-6
    f_hi = _mean_scalar(mu_max, binning, tie_up, cv, dead_time_tau)
    for i, r in enumerate(rec.ravel()):
        if not np.isfinite(r) or r <= 0 or r >= f_hi:
            continue
        g = lambda lm: _mean_scalar(np.exp(lm), binning, tie_up, cv, dead_time_tau) - r
        try:
            lm = optimize.brentq(g, np.log(lo), np.log(mu_max), xtol=1e-10, rtol=1e-10, maxiter=200)
        except ValueError:
            continue
        mu = float(np.exp(lm))
        h = 1e-3 * mu
        d = (_mean_scalar(mu + h, binning, tie_up, cv, dead_time_tau) - _mean_scalar(mu - h, binning, tie_up, cv, dead_time_tau)) / (2 * h)
        out.flat[i] = mu
        amp.flat[i] = r / (mu * d) if d > 0 else np.inf
        if n_eff is not None:
            ne = np.broadcast_to(np.asarray(n_eff, float), rec.shape).flat[i]
            var = recorded_var(mu, binning, tie_up, cv)
            sig.flat[i] = np.sqrt(var / ne) / d if d > 0 else np.inf
    ok = np.isfinite(out)
    inv = Inversion(out[0], sig[0], amp[0], rec[0], ok[0]) if np.ndim(recorded) == 0 else Inversion(out, sig, amp, rec, ok)
    if tie_up_err is not None:
        inv = _with_tie_systematic(inv, recorded, binning, tie_up, cv, dead_time_tau, tie_up_err, None, fast=False)
    return inv


def true_profile(recorded_profile, binning: int, n_pixels, **kw) -> Inversion:
    """Convenience: invert an azimuthal-mean profile given the pixel-frame count per shell."""
    return true_mean(recorded_profile, binning, n_eff=n_pixels, **kw)


# ----------------------------------------------------------------------------- simulation
def quantise(electrons: np.ndarray, binning: int, tie_up: float | None = None, rng=None,
             rule: str = "rint_even", m: float | None = None) -> np.ndarray:
    """Stored values for integer electron numbers `electrons` (per output pixel per
    stored frame; for the '_raw' rules pass shape (..., 3) raw-frame numbers)."""
    m = electrons_per_count(binning) if m is None else float(m)
    N = np.asarray(electrons)
    if rule.endswith("_raw"):
        return quantise(N, binning, tie_up, rng, rule[:-4], m).sum(axis=-1)
    x = N / m
    if rule == "floor":
        return np.floor(x + 1e-9).astype(int)
    if rule == "half_up":
        return np.floor(x + 0.5 + 1e-9).astype(int)
    val = np.rint(x).astype(int)
    if tie_up is not None:
        tie = np.isclose(x - np.floor(x), 0.5)
        rng = np.random.default_rng() if rng is None else rng
        up = rng.random(N.shape) < tie_up
        val = np.where(tie, np.floor(x).astype(int) + up.astype(int), val)
    return val


def simulate(true_mean: float, binning: int, n: int, tie_up: float | None = None, cv: float = 0.0,
             rng=None, rule: str = "rint_even", m: float | None = None) -> np.ndarray:
    """n stored values for a pixel population with mean electron number true_mean."""
    rng = np.random.default_rng() if rng is None else rng
    mu = np.full(n, float(true_mean))
    if cv > 0:
        r = 1.0 / cv ** 2
        mu = rng.gamma(r, true_mean / r, size=n)
    if rule.endswith("_raw"):
        N = rng.poisson(mu[:, None] / RAW_FRAMES_PER_STORED, size=(n, RAW_FRAMES_PER_STORED))
    else:
        N = rng.poisson(mu)
    return quantise(N, binning, tie_up, rng, rule, m)


# ----------------------------------------------------------------------------- hypothesis scoring
def pmf_loglik(hist: np.ndarray, pmf: np.ndarray) -> float:
    """Multinomial log-likelihood of an observed value histogram under a pmf (tail pooled)."""
    k = len(hist)
    p = _pad(pmf, k - 1)
    p = np.clip(p, 1e-300, None)
    return float((hist * np.log(p)).sum())


def fit_scale(hist_by_shell: list[np.ndarray], mu_ref: np.ndarray, binning: int, rule: str = "rint_even",
              tie_up: float | None = None, cv: float = 0.0, m: float | None = None,
              scale_grid=np.linspace(0.6, 1.6, 101)) -> dict:
    """Maximum-likelihood film-scale factor s (true mu = s * mu_ref per shell) for one
    rule, from per-shell value histograms. Returns the best scale, its log-likelihood
    and the log-likelihood at s = 1 (the parameter-free prediction)."""
    ll = []
    for s in scale_grid:
        ll.append(sum(pmf_loglik(h, recorded_pmf(float(s * u), binning, tie_up, cv, rule, m))
                      for h, u in zip(hist_by_shell, mu_ref) if np.isfinite(u) and h.sum() > 0))
    ll = np.asarray(ll)
    i = int(np.argmax(ll))
    ll1 = sum(pmf_loglik(h, recorded_pmf(float(u), binning, tie_up, cv, rule, m))
              for h, u in zip(hist_by_shell, mu_ref) if np.isfinite(u) and h.sum() > 0)
    return {"scale": float(scale_grid[i]), "loglik": float(ll[i]), "loglik_at_1": float(ll1),
            "rule": rule, "m": electrons_per_count(binning) if m is None else m, "tie_up": tie_up}


# ============================================================================= v1: split-pair clusters
# Physics and measurements (2026-10-26, notes/reports/2026-10-26_binning_response_v1.md):
#
# * A small fraction of the counted electrons is recorded by the hardware counter as TWO events
#   inside one bin-8 block: in the same bin-4 pixel (2 x 2 physical 8-um pixels) or in
#   8-neighbouring bin-4 pixels. Measured on bin-4 vacuum and bar frames with the chance
#   adjacency subtracted (ring-2 displacement rate; seams and borders masked): 0.08-0.11 %
#   of the events on bars, 0.19-0.30 % on vacuum frames, the same with the gain flag on and
#   off (bin 4 is linear in events either way). The "1.3 % split pairs" of the vacuum
#   calibration v0 was the chance adjacency of the centrally concentrated stray (plus seam
#   pixels). The project's "electrons" (bin-4 counts = the server's event count) are events;
#   the scattering physics is in the EVENT density mu and the pair excess is left in the unit.
# * Geometry: in a b x b output block (B = b/4 bin-4 pixels per side) a pair displaced by one
#   bin-4 pixel along an axis lies inside one block with probability 1 - 1/B, a diagonal pair
#   with (1 - 1/B)^2, a same-pixel pair always (`in_block_fraction`, `pair_fraction`).
# * Per block and stored frame the events are S + 2 P with S ~ Poisson((1 - phi) mu),
#   P ~ Poisson(phi mu / 2); phi = the fraction of the events that belong to an in-block
#   pair. With a Gamma spread of mu (cv) the pair (S, P) is a negative-binomial total with a
#   binomial split (exact, `_joint_pmf_v1`). The stored value is rint_even((S + 2 P) / m) as
#   in v0; with the gain flag on an exact tie of independent singles rounds up with
#   TIE_UP_V1 = 0.35 (v0) and a block holding exactly one pair with P_PAIR_V1, mixed ties
#   with the event-weighted mean; with the flag off every tie follows the parity rule.
# * Fitted on the matched bin-4 / bin-8 film histograms (scripts/binning_v1_fit.py, 60-480
#   px, film scale 1): gain off phi = 0.0053 (sbee23cf5, 32 M px; consistent with the
#   geometric 0.003-0.006), gain on phi = 0.064 with p_pair = 0.21 (s1cfa6c4d, 36 M px;
#   f_c = p_pair phi / 2 = 0.0067 counts per event at low flux). The gain-on value is an
#   EFFECTIVE parameter: it is 15-20x the measured pair rate, and the matched bar floor
#   (bin-4 bar frames vs the adjacent bin-8 row, 48 pairs) needs every 2-event block to
#   round up with p = 0.61 (IQR 0.47-0.78; vacuum 0.83) against 0.35 on film, which the
#   pairs cannot supply (p_pair would have to be 3-6). What makes 2-event blocks round up
#   more often at bar / vacuum flux than on film is OPEN; the v1 law reproduces the
#   low-flux linear term phenomenologically. PHI_V1 holds the per-flag defaults.
F_SPLIT_V1 = 0.0040          # measured pairs per counted electron (bin-4 geometry; bars 0.0008-0.0011, vacuum 0.0019-0.0030 in-block)
W_EDGE_V1 = 0.90             # fraction of the adjacent pairs displaced along an axis (bars / vacuum: 0.8-1.0)
P_PAIR_V1 = 0.21             # gain on: P(round up) for a block holding one pair only (film ML, effective)
TIE_UP_V1 = 0.35             # gain on: P(round up) for an exact tie of independent singles (v0)
PHI_V1 = {"Enable": 0.0639, "Disable": 0.00529}   # film-ML in-block pair fraction of the events per gain flag
GAIN_TIE_V1 = {"Enable": TIE_UP_V1, "Disable": None, True: TIE_UP_V1, False: None, None: None}


def in_block_fraction(binning: int, w_edge: float = W_EDGE_V1) -> float:
    """Fraction of split pairs (displacement one bin-4 pixel; `w_edge` along an axis, the rest
    diagonal) whose two events fall into the same output block of `binning` (B = b / 4 bin-4
    pixels per side): w_edge (1 - 1/B) + (1 - w_edge) (1 - 1/B)^2. Bin 4: 0."""
    B = float(binning) / 4.0
    if B <= 1:
        return 0.0
    e = 1.0 - 1.0 / B
    return float(w_edge * e + (1.0 - w_edge) * e * e)


def pair_fraction(binning: int, f_split: float = F_SPLIT_V1, f_in: float | None = None, w_edge: float = W_EDGE_V1) -> float:
    """phi: fraction of the counted EVENTS of an output block that belong to an in-block split
    pair, 2 f_split f_in / (1 + f_split); lambda_P = phi mu / 2 pairs per block."""
    fi = in_block_fraction(binning, w_edge) if f_in is None else float(f_in)
    return float(2.0 * f_split * fi / (1.0 + f_split))


def _phi_for(binning: int, tie_up, phi, f_split, f_in, w_edge) -> float:
    """phi used by the v1 helpers: an explicit `phi`; else the geometric value from (f_split, f_in /
    w_edge) when f_split is given; else the per-flag film default PHI_V1 (Enable when tie_up is a
    probability, Disable when None). Bin 4 has no in-block pairs by construction."""
    if binning == 4 and phi is None and f_in is None:
        return 0.0
    if phi is not None:
        return float(phi)
    if f_split is not None:
        return pair_fraction(binning, f_split, f_in, w_edge)
    return PHI_V1["Disable" if tie_up is None else "Enable"]


def _tie_for_flag(gain_flag):
    if isinstance(gain_flag, (float, np.floating)) and not isinstance(gain_flag, bool):
        return float(gain_flag)
    if isinstance(gain_flag, str):
        return GAIN_TIE_V1[gain_flag]
    return GAIN_TIE_V1[bool(gain_flag)] if gain_flag is not None else None


def _joint_pmf_v1(mu: float, cv: float, phi: float) -> tuple[np.ndarray, int, int]:
    """P(S = s, P = p) on a (p_max + 1, s_max + 1) grid for the single/pair law at event mean mu
    with a Gamma spread cv: T = S + P is Poisson (cv = 0) or negative binomial with mean
    mu (1 - phi/2), and S | T is binomial with q = (1 - phi) / (1 - phi/2)."""
    lam_T = mu * (1.0 - phi / 2.0)
    n_max = _n_max(mu, cv)
    pT = _electron_pmf(lam_T, cv, n_max)
    q = (1.0 - phi) / (1.0 - phi / 2.0)
    lam_P = mu * phi / 2.0
    p_max = int(min(n_max, max(6, lam_P * (1 + cv ** 2) + 10 * np.sqrt(lam_P * (1 + (cv * lam_P))) + 10)))
    p = np.arange(p_max + 1)[:, None]; s = np.arange(n_max + 1)[None, :]
    t = s + p
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = (stats.binom.logpmf(p, t, 1.0 - q))
    pTt = np.where(t <= n_max, pT[np.minimum(t, n_max)], 0.0)
    J = np.where(np.isfinite(logc), np.exp(logc), 0.0) * pTt
    return J, p_max, n_max


def recorded_pmf_v1(mu: float, binning: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0,
                    phi: float | None = None, p_pair: float = P_PAIR_V1, f_split: float | None = None,
                    f_in: float | None = None, w_edge: float = W_EDGE_V1, m: float | None = None, k_max: int | None = None,
                    tie_law: str | None = None) -> np.ndarray:
    """P(stored value = k) for an output pixel with mean EVENT number mu under the split-pair
    law (v1). `tie_up` None = gain off (parity rule for every tie); a float = gain on, the
    tie-up probability of independent singles, with `p_pair` for a pure pair block and the
    event-weighted mix in between. The pair fraction is `phi` (default PHI_V1 of the flag), or
    the geometric value from `f_split` (+ `f_in` / `w_edge`) when that is given. phi = 0 (or
    bin 4) reproduces `recorded_pmf` exactly."""
    m = electrons_per_count(binning) if m is None else float(m)
    if tie_law == "v2":
        # v2 option: the rate-dependent tie law replaces tie_up (gain on) / the parity rule (gain off, tie_up None);
        # the pair class is dropped unless phi is given explicitly (the v2 law carries the split pairs in p).
        flag = "Disable" if tie_up is None else "Enable"
        if phi is None or phi <= 0:
            return recorded_pmf_v2(mu, binning, flag, cv, None, m, k_max)
        tie_up = tie_prob(mu, flag) if tie_up is not None else None
        p_pair = tie_prob(mu, flag)
    phi = _phi_for(binning, tie_up, phi, f_split, f_in, w_edge)
    if phi <= 0 or mu <= 0:
        return recorded_pmf(mu, binning, tie_up, cv, "rint_even", m, k_max)
    J, p_max, n_max = _joint_pmf_v1(mu, cv, phi)
    p = np.arange(p_max + 1)[:, None]; s = np.arange(n_max + 1)[None, :]
    E = s + 2 * p
    x = E / m
    val = np.rint(x)
    frac = x - np.floor(x)
    tie = np.isclose(frac, 0.5)
    lo = np.floor(x)
    if tie_up is None:
        up = (lo + 1) % 2 == 0                                    # parity: up when the upper value is even
        up = np.where(tie, up.astype(float), 0.0)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            mix = np.where(E > 0, (s * tie_up + 2 * p * p_pair) / np.maximum(E, 1), tie_up)
        up = np.where(tie, mix, 0.0)
    kk = int(val.max()) + 2
    pmf = np.zeros(kk + 1)
    w_notie = np.where(tie, 0.0, J)
    np.add.at(pmf, val.astype(int).ravel(), w_notie.ravel())
    w_tie = np.where(tie, J, 0.0)
    np.add.at(pmf, lo.astype(int).ravel(), (w_tie * (1 - up)).ravel())
    np.add.at(pmf, (lo.astype(int) + 1).ravel(), (w_tie * up).ravel())
    return pmf if k_max is None else _pad(pmf, k_max)


def recorded_mean_v1(true_mean, binning: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0,
                     phi: float | None = None, p_pair: float = P_PAIR_V1, f_split: float | None = None,
                     f_in: float | None = None, w_edge: float = W_EDGE_V1, dead_time_tau: float = 0.0, m: float | None = None,
                     tie_law: str | None = None):
    """E[stored value] per output pixel per stored frame under the split-pair law (v1) for the
    mean event number `true_mean` (per output pixel per stored frame). Vectorised. `tie_law="v2"`
    (2026-10-27) replaces the constant tie_up / parity rule by the rate-dependent law `tie_prob`."""
    mu = np.atleast_1d(np.asarray(true_mean, float))
    if dead_time_tau:
        mu = mu * np.exp(-mu * dead_time_tau)
    out = np.array([_moments(recorded_pmf_v1(float(u), binning, tie_up, cv, phi, p_pair, f_split, f_in, w_edge, m, tie_law=tie_law))[0] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def recorded_var_v1(true_mean, binning: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0,
                     phi: float | None = None, p_pair: float = P_PAIR_V1, f_split: float | None = None,
                     f_in: float | None = None, w_edge: float = W_EDGE_V1, m: float | None = None, tie_law: str | None = None):
    mu = np.atleast_1d(np.asarray(true_mean, float))
    out = np.array([_moments(recorded_pmf_v1(float(u), binning, tie_up, cv, phi, p_pair, f_split, f_in, w_edge, m, tie_law=tie_law))[1] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def forward_v1(true_mean, binning: int = 8, gain_flag="Enable", cv: float = 0.0, phi: float | None = None,
               p_pair: float = P_PAIR_V1, dead_time_tau: float = 0.0, tie_law: str | None = None, **kw):
    """Recorded mean under the v1 law with the DE gain flag ('Enable' / 'Disable', bool, or a
    tie-up probability) instead of `tie_up`. `tie_law="v2"` uses the rate-dependent tie law (then
    equal to `forward_v2` unless `phi` is given)."""
    return recorded_mean_v1(true_mean, binning, _tie_for_flag(gain_flag), cv, phi, p_pair, dead_time_tau=dead_time_tau, tie_law=tie_law, **kw)


def response_table_v1(binning: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0, phi: float | None = None,
                      p_pair: float = P_PAIR_V1, dead_time_tau: float = 0.0, mu_min: float = 1e-3, mu_max: float = 2000.0,
                      n: int = 600, tie_law: str | None = None, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, E[stored], Var[stored]) on a log grid under the v1 law (for `true_mean_v1`)."""
    mu = np.geomspace(mu_min, mu_max, n)
    e = recorded_mean_v1(mu, binning, tie_up, cv, phi, p_pair, dead_time_tau=dead_time_tau, tie_law=tie_law, **kw)
    mu_c = mu * np.exp(-mu * dead_time_tau) if dead_time_tau else mu
    v = recorded_var_v1(mu_c, binning, tie_up, cv, phi, p_pair, tie_law=tie_law, **kw)
    return mu, e, v


def true_mean_v1(recorded, binning: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0, n_eff=None,
                 phi: float | None = None, p_pair: float = P_PAIR_V1, dead_time_tau: float = 0.0, table=None,
                 tie_law: str | None = None, **kw) -> Inversion:
    """Inversion of the v1 law by table interpolation (same contract and return type as
    `true_mean_fast`): mean EVENT number per output pixel per stored frame for a mean stored
    value `recorded` (counts per output pixel per stored frame over n_eff pixel-frames).
    `gain_flag` style callers pass tie_up = GAIN_TIE_V1[flag]. Pass a cached `table` from
    `response_table_v1` for repeated calls."""
    tab = response_table_v1(binning, tie_up, cv, phi, p_pair, dead_time_tau, tie_law=tie_law, **kw) if table is None else table
    return true_mean_fast(recorded, binning, tie_up, cv, n_eff, dead_time_tau, table=tab)


def quantise_v1(singles: np.ndarray, pairs: np.ndarray, binning: int, tie_up: float | None = TIE_UP_V1,
                p_pair: float = P_PAIR_V1, rng=None, m: float | None = None) -> np.ndarray:
    """Stored values for blocks holding `singles` single events and `pairs` in-block split pairs
    (integer arrays of one shape; E = singles + 2 pairs events)."""
    m = electrons_per_count(binning) if m is None else float(m)
    S = np.asarray(singles); P = np.asarray(pairs); E = S + 2 * P
    x = E / m
    val = np.rint(x).astype(int)
    tie = np.isclose(x - np.floor(x), 0.5)
    if tie_up is None or not tie.any():
        return val
    rng = np.random.default_rng() if rng is None else rng
    with np.errstate(invalid="ignore", divide="ignore"):
        mix = np.where(E > 0, (S * tie_up + 2 * P * p_pair) / np.maximum(E, 1), tie_up)
    up = rng.random(E.shape) < mix
    return np.where(tie, np.floor(x).astype(int) + up.astype(int), val)


def simulate_v1(true_mean: float, binning: int, n: int, tie_up: float | None = TIE_UP_V1, cv: float = 0.0,
                phi: float | None = None, p_pair: float = P_PAIR_V1, rng=None, f_split: float | None = None, **kw) -> np.ndarray:
    """n stored values for a pixel population with mean event number true_mean under the v1 law."""
    rng = np.random.default_rng() if rng is None else rng
    phi = _phi_for(binning, tie_up, phi, f_split, kw.get("f_in"), kw.get("w_edge", W_EDGE_V1))
    mu = np.full(n, float(true_mean))
    if cv > 0:
        r = 1.0 / cv ** 2
        mu = rng.gamma(r, true_mean / r, size=n)
    S = rng.poisson(mu * (1 - phi)); P = rng.poisson(mu * phi / 2)
    return quantise_v1(S, P, binning, tie_up, p_pair, rng)


# ============================================================================= v2: the tie-rule law p_up(rate, gain flag)
# Measured 2026-10-27 (notes/reports/2026-10-27_tie_rule_law_v0.md, scripts/binning_v2_{measure,fetch,matched,law}.py):
#
# * The effective probability that an exact tie (N = 2 electrons in a bin-8 block, N = 6, 10 ... at higher
#   flux) rounds UP is NOT one number per gain flag. Per shell on the matched film sets (0.1-1 e per block)
#   it is 0.40 +/- 0.01 (statistical) +/- 0.03 (film scale) with the gain on and 0.00-0.02 with it off, flat
#   in rate and in radius; on position-matched bar frames (0.007-0.022 e per block, 48 bin-4 / 48 bin-8
#   frames 5 um apart) it is 0.6-0.9 (mean 0.67) with the gain on and 0.010-0.037 (0.020) with it off; on
#   position-matched vacuum frames (0.002-0.034 e) 0.36-0.82 (0.53, gain on). The low-rate excess is where
#   half of the N = 2 blocks are SPLIT PAIRS (two counted events from one electron, 8-connected 2-clusters
#   2-4x chance on bars, 2.3-2.9 % of events at low density; v1's 0.1-0.4 % was too low) rather than
#   coincidences of two electrons: the data are consistent with coincidence ties rounding up with 0.40 and
#   split-pair ties with ~1 (gain on), but the vacuum frames (0.53) sit 0.15 below that reading, so the law
#   is kept phenomenological in the RATE: a logistic step in log rate per flag. The firmware origin (how the
#   DE server rounds a gain-weighted tie) is out of reach from the archive.
# * Position: the recorded bar floor follows N2 (quadratic in the electron rate) with the radius; its centroid
#   is 40-70 px above-left of the beam because the bar-scattered ELECTRONS are (bin-4 centroid), not because
#   the response is; split-half pixel correlation of 900-frame floor sums 0.001-0.007 (a 50/50 fixed pattern
#   would give 0.11-0.12): no gain-map pattern. The gain-off floor equals N(>= 3) + 0.02 N(= 2) on the floor
#   population (<= 40 counts); the earlier "4-7x excess" was the Grid population (bar-edge frames).
# * Domain: bin 8, 49-ms frames, 0.002-1 e per block (measured); 1-2 e degenerate with the film match; above
#   2 e ties are < 15 % of the recorded mean and p is immaterial at the 3 % level; no points between 0.034
#   and 0.11 e, so the step's position (mu_half) and sharpness are bounded, not measured.
TIE_LAW_V2 = {"Enable": {"p_lo": 0.670, "p_hi": 0.403, "mu_half": 0.054, "slope": 6.0},
              "Disable": {"p_lo": 0.020, "p_hi": 0.000, "mu_half": 0.19, "slope": 5.4}}
TIE_LAW_V2_SYST = {"Enable": {"p_hi_film_scale_free": 0.436, "p_lo_se": 0.015, "vacuum_offset": -0.14},
                   "Disable": {"p_hi_film_scale_free": 0.018, "p_lo_se": 0.004}}
PHI_V2 = 0.0                       # the split-pair class is carried by the rate law itself (v2); no separate pair term


def _flag_name(gain_flag) -> str:
    if isinstance(gain_flag, str):
        return gain_flag
    return "Enable" if bool(gain_flag) else "Disable"


def tie_prob(rate, gain_flag="Enable", law: dict | None = None):
    """p_up(rate, flag): the probability that an exact bin-8 tie rounds up, as a function of the true mean
    electron rate per bin-8 block per stored frame (vectorised). Gain on: all ties; gain off: the ties at
    ODD upper levels (N = 2, 10, ...), the even ones (N = 6, 14) follow the parity rule (up). `law`
    overrides TIE_LAW_V2[flag] (keys p_lo, p_hi, mu_half, slope)."""
    th = TIE_LAW_V2[_flag_name(gain_flag)] if law is None else law
    mu = np.asarray(rate, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = th["p_hi"] + (th["p_lo"] - th["p_hi"]) / (1.0 + np.where(mu > 0, (mu / th["mu_half"]) ** th["slope"], 0.0))
    p = np.clip(p, 0.0, 1.0)
    return p if np.ndim(rate) else float(p)


def pmf_tie_levels(mu: float, binning: int = 8, cv: float = 0.0, p_odd: float = 0.0, p_even: float = 1.0, m: float | None = None, k_max: int | None = None) -> np.ndarray:
    """P(stored = k) under round-half-to-even(N / m) with the exact ties at ODD upper levels k (N = 2, 10 ...
    at bin 8) rounding up with p_odd and those at EVEN upper levels (N = 6, 14 ...) with p_even. Parity rule:
    (0, 1); gain-on single p: (p, p). N ~ Poisson(mu) or the Gamma mixture with cv (as `recorded_pmf`)."""
    m = electrons_per_count(binning) if m is None else float(m)
    if mu <= 0:
        p0 = np.zeros(2); p0[0] = 1.0
        return p0 if k_max is None else _pad(p0, k_max)
    n_max = _n_max(mu, cv)
    pN = _electron_pmf(mu, cv, n_max)
    n = np.arange(n_max + 1); x = n / m
    val = np.rint(x); frac = x - np.floor(x); tie = np.isclose(frac, 0.5); lo = np.floor(x)
    p_up = np.where((lo + 1) % 2 == 1, p_odd, p_even)
    kk = int(val.max()) + 2
    pmf = np.zeros(kk + 1)
    np.add.at(pmf, val[~tie].astype(int), pN[~tie])
    np.add.at(pmf, lo[tie].astype(int), pN[tie] * (1 - p_up[tie]))
    np.add.at(pmf, lo[tie].astype(int) + 1, pN[tie] * p_up[tie])
    return pmf if k_max is None else _pad(pmf, k_max)


def _p_levels_v2(mu, gain_flag, law):
    p = tie_prob(mu, gain_flag, law)
    return (p, p) if _flag_name(gain_flag) == "Enable" else (p, 1.0)


def recorded_pmf_v2(mu: float, binning: int = 8, gain_flag="Enable", cv: float = 0.0, law: dict | None = None, m: float | None = None, k_max: int | None = None) -> np.ndarray:
    """P(stored value = k) under the v2 tie-rule law: the v0 rounding law with the tie-up probability
    evaluated at the block's mean rate, `tie_prob(mu, gain_flag)` (gain on: every tie; gain off: the odd
    ties, parity otherwise). With a Gamma spread (cv) the tie probability is taken at the MEAN rate
    (the law is a function of the local mean, not of the realisation)."""
    p_odd, p_even = _p_levels_v2(float(mu), gain_flag, law)
    return pmf_tie_levels(float(mu), binning, cv, p_odd, p_even, m, k_max)


def recorded_mean_v2(true_mean, binning: int = 8, gain_flag="Enable", cv: float = 0.0, law: dict | None = None, dead_time_tau: float = 0.0, m: float | None = None):
    """E[stored value] per output pixel per stored frame under the v2 law. Vectorised in true_mean."""
    mu = np.atleast_1d(np.asarray(true_mean, float))
    if dead_time_tau:
        mu = mu * np.exp(-mu * dead_time_tau)
    out = np.array([_moments(recorded_pmf_v2(float(u), binning, gain_flag, cv, law, m))[0] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def recorded_var_v2(true_mean, binning: int = 8, gain_flag="Enable", cv: float = 0.0, law: dict | None = None, m: float | None = None):
    mu = np.atleast_1d(np.asarray(true_mean, float))
    out = np.array([_moments(recorded_pmf_v2(float(u), binning, gain_flag, cv, law, m))[1] for u in mu])
    return out if np.ndim(true_mean) else float(out[0])


def forward_v2(true_mean, binning: int = 8, gain_flag="Enable", cv: float = 0.0, dead_time_tau: float = 0.0, law: dict | None = None, **kw):
    """Recorded mean under the v2 tie-rule law for the DE gain flag ('Enable' / 'Disable' or bool)."""
    return recorded_mean_v2(true_mean, binning, gain_flag, cv, law, dead_time_tau=dead_time_tau, **kw)


def response_table_v2(binning: int = 8, gain_flag="Enable", cv: float = 0.0, dead_time_tau: float = 0.0, law: dict | None = None,
                      mu_min: float = 1e-3, mu_max: float = 2000.0, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, E[stored], Var[stored]) on a log grid under the v2 law (for `true_mean_v2`)."""
    mu = np.geomspace(mu_min, mu_max, n)
    e = recorded_mean_v2(mu, binning, gain_flag, cv, law, dead_time_tau=dead_time_tau)
    mu_c = mu * np.exp(-mu * dead_time_tau) if dead_time_tau else mu
    v = recorded_var_v2(mu_c, binning, gain_flag, cv, law)
    return mu, e, v


LAW_V2_ERR = {"p_hi": 0.033, "p_lo": 0.015}      # audit E6: p_hi 0.403 vs the free-scale 0.436; p_lo standard error (TIE_LAW_V2_SYST)


def true_mean_v2(recorded, binning: int = 8, gain_flag="Enable", cv: float = 0.0, n_eff=None, dead_time_tau: float = 0.0, law: dict | None = None, table=None,
                 law_err: dict | None = None) -> Inversion:
    """Inversion of the v2 law by table interpolation (same contract and return type as `true_mean_fast`):
    mean electron number per output pixel per stored frame for a mean stored value `recorded` (counts per
    output pixel per stored frame over n_eff pixel-frames). Pass a cached `table` from `response_table_v2`
    for repeated calls. The forward curve is monotonic (the step in p adds at most 0.27 x P(N = 2) at
    0.03-0.1 e, where P(N = 2) itself rises as mu^2), so the inversion is one-valued.  With `law_err`
    ({"p_hi": dp_hi, "p_lo": dp_lo}, e.g. LAW_V2_ERR) the inversion is repeated with the law's plateaus
    shifted together by +/- the errors and the correlated systematic returned in `sigma_tie` / `dmu_dp`
    (per unit shift of p_hi; audit E6)."""
    tab = response_table_v2(binning, gain_flag, cv, dead_time_tau, law) if table is None else table
    tie = TIE_UP_V1 if _flag_name(gain_flag) == "Enable" else None
    inv = true_mean_fast(recorded, binning, tie, cv, n_eff, dead_time_tau, table=tab)
    if law_err is not None:
        base = dict(TIE_LAW_V2[_flag_name(gain_flag)] if law is None else law)
        rec = np.atleast_1d(np.asarray(recorded, float)); scalar = np.ndim(recorded) == 0
        res = []
        for sgn in (+1.0, -1.0):
            lw = dict(base); lw["p_hi"] = float(np.clip(base["p_hi"] + sgn * law_err.get("p_hi", 0.0), 0.0, 1.0)); lw["p_lo"] = float(np.clip(base["p_lo"] + sgn * law_err.get("p_lo", 0.0), 0.0, 1.0))
            key = ("v2", int(binning), _flag_name(gain_flag), round(float(cv), 4), round(float(dead_time_tau), 8), lw["p_hi"], lw["p_lo"])
            if key not in _TIE_TABLES:
                _TIE_TABLES[key] = response_table_v2(binning, gain_flag, cv, dead_time_tau, lw)
            res.append(np.atleast_1d(np.asarray(true_mean_fast(rec, binning, tie, cv, None, dead_time_tau, table=_TIE_TABLES[key]).true_mean, float)))
        hi, lo = res
        st = np.abs(hi - lo) / 2.0; d = (hi - lo) / max(2.0 * law_err.get("p_hi", 0.0), 1e-12)
        inv.sigma_tie = st[0] if scalar else st; inv.dmu_dp = d[0] if scalar else d
    return inv


def quantise_v2(electrons: np.ndarray, binning: int = 8, gain_flag="Enable", rate=None, rng=None, law: dict | None = None, m: float | None = None) -> np.ndarray:
    """Stored values for integer electron numbers `electrons` per block under the v2 law; `rate` is the
    local mean rate per block (same shape or scalar; defaults to the array mean) that sets the tie
    probability of each block."""
    m = electrons_per_count(binning) if m is None else float(m)
    N = np.asarray(electrons); x = N / m
    val = np.rint(x).astype(int)
    tie = np.isclose(x - np.floor(x), 0.5)
    if not tie.any():
        return val
    rng = np.random.default_rng() if rng is None else rng
    rate = np.full(N.shape, float(np.mean(N))) if rate is None else np.broadcast_to(np.asarray(rate, float), N.shape)
    p = tie_prob(rate, gain_flag, law)
    lo = np.floor(x).astype(int)
    if _flag_name(gain_flag) != "Enable":
        p = np.where((lo + 1) % 2 == 1, p, 1.0)
    up = rng.random(N.shape) < p
    return np.where(tie, lo + up.astype(int), val)


def simulate_v2(true_mean: float, binning: int = 8, n: int = 100000, gain_flag="Enable", cv: float = 0.0, rng=None, law: dict | None = None) -> np.ndarray:
    """n stored values for a pixel population with mean electron number true_mean under the v2 law."""
    rng = np.random.default_rng() if rng is None else rng
    mu = np.full(n, float(true_mean))
    if cv > 0:
        r = 1.0 / cv ** 2
        mu = rng.gamma(r, true_mean / r, size=n)
    return quantise_v2(rng.poisson(mu), binning, gain_flag, rate=float(true_mean), rng=rng, law=law)
