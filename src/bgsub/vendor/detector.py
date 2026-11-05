# VENDORED COPY. Source: pod/background/src/podbg/detector.py at background commit f1e93f5 (file last changed in e14cb38);
# sha256 of the source file: 32ed9c9afe61e4c8314880265972b2b7601d8882ff163158d4dfd6a9b5525509. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: the two lazy `from podbg import binning as B` -> `from . import binning as B`.
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Apollo counting response: units, dead-time (coincidence-loss) law, the combined
forward model (dead time -> 3-raw-frame sum -> rounding law -> stored uint8),
its inversion, the dark/stray floor model and the bin-8 fixed-pattern correction.

Physics assumptions written next to the code that uses them:

Units (corrected 2026-10-26, `notes/reports/2026-10-26_binning_response_v0.md`
and `_detector_reconciliation_v0.md`). The DE server counts events in hardware,
places 16 per event on the 8192 x 8192 super-resolution grid, SUMS the 3 raw
frames of a stored frame, AVERAGES over the b x b super-resolution block of an
output pixel and stores round-half-to-even of the result as uint8. So
    counts_per_electron(b) = 16 / b**2          (bin 4: 1, bin 8: 1/4, bin 16: 1/16)
    electrons_per_count(b) = b**2 / 16          (bin 4: 1, bin 8: 4, bin 16: 16)
verified at bin 4 against the server's own event count (sparseness x 4096^2 x 3)
to 0.03 %. The mdoc CountsPerElectron (0.5 / 0.125 / 0.03125 = 8 / b^2) is
SerialEM's DividedBy2 copy and does NOT describe the autosaved frames; every
electron number derived with 8 e/count (the 2026-10-25 detector report) is 2x
too high. The linear conversions below (`counts_to_electron_rate`,
`electron_rate_to_counts`) hold for MEANS well above ~2 counts/px/frame; below
that the rounding law is nonlinear and `podbg.binning.true_mean` or
`invert_mean` here must be used.

Geometry and time. A physical pixel is 8 um (4096 grid); an output pixel at
binning b (relative to the 8192 super-resolution grid) is b/2 physical pixels
wide, i.e. `physical_pixels_per_binned(8)` = 16 (32 um). A stored fly-scan
frame is `RAW_FRAMES_PER_STORED` = 3 raw frames at `FPS` = 61.05 Hz, 49.14 ms.
The sensor's event time resolution is 418 us per pixel block (F8), 39.2
blocks of time per raw frame and 117.6 per stored frame.

Dead time. Two classical laws relate a true event rate n to a counted rate m
through one parameter tau (in the units of 1/n):
    non-paralyzable  m = n / (1 + n tau)          ceiling 1/tau, no turnover
    paralyzable      m = n exp(-n tau)            peak 1/(e tau) at n = 1/tau,
                                                  then m -> 0 (zero core)
For an event-counting MAPS the paralyzable law is the 1-D limit of the spatial
overlap model (F12: Zambon 2024): a second electron inside the cluster window
of the first within one time block spoils both. The loss is LOCAL (a few
physical pixels, one time block), so tau is a property of the physical pixel
and the raw frame; the canonical unit here is
    TAU: per (electron per PHYSICAL pixel per RAW frame)
and `tau_for_binning` converts it to per (e per output pixel per stored frame)
= tau / (physical_pixels_per_binned(b) * n_raw). The product tau x T_raw is the
footprint area (physical px) times the time window; `dead_time_window` gives
the window for an assumed footprint (1 px, 2x2, 3x3). Which footprint applies
cannot be told from azimuthal means (all are the same scaling of mu); only the
product is measured. The counting loss acts per raw frame; because all three
raw frames of a stored frame see the same rate, the law for the stored sum is
the same function of the stored-frame rate (sum of 3 identical terms).

Fitted value (v1, no-stop session, full chain): TAU_PHYS_V1 below, with the
measured ceiling and loss points in the reconciliation report. The paralyzable
inverse is two-valued; `true_rate` returns the lower branch by default and NaN
above the peak; `branch="upper"` gives the saturated branch (zero core).

Forward model. `forward(mu, binning, gain_flag, tau)` returns the expected
stored mean and variance for a true mean of mu electrons per output pixel per
stored frame: mu_c = D(mu; tau_b), then `podbg.binning.recorded_mean/var`
(Poisson N at mu_c, round-half-to-even of N/m, ties up with p = 0.35 when the
gain reference is applied to the movie, parity rule otherwise). Approximation:
the counted number is taken Poisson at the counted mean (dead time makes it
sub-Poisson near the ceiling; the variance is then overestimated by < 30 %).
`invert_mean` chains `binning.true_mean` with the Lambert-W inversion and
reports the validity range (counted mean below the ceiling).

Floor. The bar-frame floor is modelled as a centred inverse-square halo plus a
flat term,   P(r) = A / (r^2 + rc^2) + C,   in whatever unit the profile is
given (recorded counts or inverted electrons); alternatives (1/r, exponential)
are fitted by the floor script for comparison, not provided here as models.
In recorded bin-8 counts the floor is in the strongly nonlinear regime (a
single electron never makes a count at bin 8): the electron floor is ~1e4 e
per frame in every session (bin-4 rows), see the reconciliation report.

Fixed pattern. Sensor-segment seams (bin 8: columns 255-256, 511-512, 767-768
and rows 511-512) and the first row read high. The correction map is
    corrected = (frame - add) / gain
with `gain` (multiplicative, dimensionless) and `add` (additive, in the unit of
the frame). Because of the rounding law a map measured on recorded bin-8 means
is level-dependent; the electron-space terms (gain on the true rate, additive
electrons) are given in the reconciliation report (`seam_response_chain`).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- units
SUPERRES_VALUE_PER_EVENT = 16              # value placed on the 8192 grid per counted event (F8)
SUPERRES_COUNTS_PER_ELECTRON = SUPERRES_VALUE_PER_EVENT   # old name, kept
SUPERRES_GRID = 8192
PHYSICAL_GRID = 4096                       # 8 um pixels
PHYSICAL_PIXEL_UM = 8.0
MDOC_DIVIDED_BY_2 = 2                      # SerialEM's own copy of the image; NOT applied to the DE autosave
RAW_FRAMES_PER_STORED = 3
FPS = 61.05                                # raw frames per second (mdoc FramesPerSecond)
BLOCK_TIME_S = 418e-6                      # event time resolution per pixel block (F8)
TIE_UP_GAIN_ON = 0.35                      # tie-rounding probability with Apply Gain on Movie = Enable (binning report)


def counts_per_electron(binning: int) -> float:
    """Stored counts per counted electron in the DE autosave at `binning` (16 / b^2)."""
    return SUPERRES_VALUE_PER_EVENT / float(binning) ** 2


def electrons_per_count(binning: int) -> float:
    """Electrons per stored count: b^2 / 16 (bin 4: 1, bin 8: 4, bin 16: 16)."""
    return 1.0 / counts_per_electron(binning)


def mdoc_counts_per_electron(binning: int) -> float:
    """The value SerialEM writes to the mdoc (8 / b^2): its DividedBy2 copy, 2x the autosave's. For cross-checks only."""
    return counts_per_electron(binning) / MDOC_DIVIDED_BY_2


def physical_pixels_per_binned(binning: int) -> int:
    """Physical (8 um) pixels inside one binned pixel (binning relative to the 8192 grid): (b/2)^2."""
    return (binning * PHYSICAL_GRID // SUPERRES_GRID) ** 2


def binned_pixel_um(binning: int) -> float:
    return PHYSICAL_PIXEL_UM * binning * PHYSICAL_GRID / SUPERRES_GRID


def stored_frame_time_s(n_raw: int = RAW_FRAMES_PER_STORED, fps: float = FPS) -> float:
    return n_raw / fps


def counts_to_electron_rate(counts_per_px_per_frame, binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED,
                            fps: float = FPS):
    """LINEAR conversion: counts per binned pixel per stored frame -> electrons per PHYSICAL pixel per second
    (1 count = b^2/16 e). Valid for means above ~2 counts/px/frame; otherwise invert the rounding law first."""
    e_per_frame = np.asarray(counts_per_px_per_frame, float) * electrons_per_count(binning)
    return e_per_frame / physical_pixels_per_binned(binning) / stored_frame_time_s(n_raw, fps)


def electron_rate_to_counts(e_per_physpx_per_s, binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED,
                            fps: float = FPS):
    """LINEAR conversion (inverse of counts_to_electron_rate)."""
    e_per_frame = np.asarray(e_per_physpx_per_s, float) * physical_pixels_per_binned(binning) * stored_frame_time_s(n_raw, fps)
    return e_per_frame * counts_per_electron(binning)


def electrons_per_frame_to_rate(e_per_binned_px_per_frame, binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED, fps: float = FPS):
    """Electrons per output pixel per stored frame -> electrons per physical pixel per second."""
    return np.asarray(e_per_binned_px_per_frame, float) / physical_pixels_per_binned(binning) / stored_frame_time_s(n_raw, fps)


# ---------------------------------------------------------------- dead time
MODELS = ("nonparalyzable", "paralyzable")
# v1 fit (reconciliation report): per (electron per physical pixel per raw frame). Filled from
# ~/pod_data/background/detector/dead_time_fit_v1.json; the report carries the interval.
TAU_PHYS_V1 = 0.206
TAU_PHYS_V1_ERR = (0.0005, 0.015)          # (statistical, systematic); tau_b(bin 8) = 0.0043 +/- 0.0003 per e per px per stored frame


def tau_for_binning(tau_phys: float, binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED) -> float:
    """tau per (e per physical px per raw frame) -> per (e per output px per stored frame)."""
    return tau_phys / (physical_pixels_per_binned(binning) * n_raw)


def tau_from_binning(tau_bin: float, binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED) -> float:
    return tau_bin * physical_pixels_per_binned(binning) * n_raw


def dead_time_window(tau_phys: float, footprint_px: float = 4.0, fps: float = FPS) -> float:
    """Time window (s) of the paralysis for an assumed footprint of `footprint_px` physical pixels:
    tau_phys x T_raw = footprint x window."""
    return tau_phys / fps / footprint_px


def recorded_rate(n, tau: float, model: str = "paralyzable"):
    """Forward law: true rate n -> counted rate m (same units; tau in 1/n units)."""
    n = np.asarray(n, float)
    if model == "nonparalyzable":
        return n / (1.0 + n * tau)
    if model == "paralyzable":
        return n * np.exp(-n * tau)
    raise ValueError(model)


def true_rate(m, tau: float, model: str = "paralyzable", branch: str = "lower"):
    """Inverse: counted m -> true n. Paralyzable: `branch` 'lower' (n < 1/tau, the
    usable one) or 'upper' (n > 1/tau, the saturated/zero-core side); NaN above the peak."""
    m = np.asarray(m, float)
    if model == "nonparalyzable":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = m / (1.0 - m * tau)
        return np.where(m * tau < 1, out, np.nan)
    if model == "paralyzable":
        from scipy.special import lambertw
        x = -m * tau
        with np.errstate(invalid="ignore"):
            out = -np.real(lambertw(x, 0 if branch == "lower" else -1)) / tau
        return np.where(x >= -np.exp(-1), out, np.nan)
    raise ValueError(model)


def loss_fraction(n, tau: float, model: str = "paralyzable"):
    """1 - m/n."""
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, 1.0 - recorded_rate(n, tau, model) / n, 0.0)


def ceiling(tau: float, model: str = "paralyzable") -> float:
    """Maximum countable rate: 1/tau (non-paralyzable, asymptote) or 1/(e tau) (paralyzable, peak)."""
    return 1.0 / tau if model == "nonparalyzable" else 1.0 / (np.e * tau)


def rate_at_loss(frac: float, tau: float, model: str = "paralyzable") -> float:
    """True rate at which the loss fraction reaches `frac` (e.g. 0.05, 0.20)."""
    if model == "nonparalyzable":
        return frac / ((1 - frac) * tau)
    return -np.log(1 - frac) / tau


def law_points(tau_phys: float = TAU_PHYS_V1, model: str = "paralyzable", binning: int = 8, n_raw: int = RAW_FRAMES_PER_STORED) -> dict:
    """Ceiling and 5 % / 20 % loss points of the law in e per output px per stored frame,
    e per physical px per s, and the equivalent linear stored counts."""
    tb = tau_for_binning(tau_phys, binning, n_raw)
    out = {}
    for lab, val in (("ceiling", ceiling(tb, model)), ("loss05", rate_at_loss(0.05, tb, model)), ("loss20", rate_at_loss(0.20, tb, model))):
        out[lab] = {"e_per_px_per_frame": float(val), "e_per_physpx_per_s": float(electrons_per_frame_to_rate(val, binning, n_raw)),
                    "counts_linear": float(val * counts_per_electron(binning))}
    out["tau_per_e_px_frame"] = tb
    return out


@dataclass(frozen=True)
class DeadTimeFit:
    model: str
    tau: float                      # in the units of 1/n_true as passed
    tau_err: float
    chi2: float
    ndof: int

    def ceiling_counts(self) -> float:
        return ceiling(self.tau, self.model)


def fit_dead_time(n_true, m_rec, var=None, model: str = "paralyzable", tau0: float = 0.02) -> DeadTimeFit:
    """Least-squares tau for pairs (true, counted) with known true rates (no quantisation; use
    `forward` in the residual for stored bin-8/16 means)."""
    from scipy.optimize import least_squares
    n = np.asarray(n_true, float); m = np.asarray(m_rec, float)
    w = 1.0 / np.sqrt(np.asarray(var, float)) if var is not None else np.ones_like(m)

    def res(p):
        return (recorded_rate(n, p[0], model) - m) * w
    r = least_squares(res, [tau0], bounds=([0.0], [np.inf]))
    J = r.jac; cov = np.linalg.inv(J.T @ J) if J.size else np.full((1, 1), np.nan)
    chi2 = float((r.fun ** 2).sum()); ndof = max(len(m) - 1, 1)
    return DeadTimeFit(model, float(r.x[0]), float(np.sqrt(cov[0, 0] * chi2 / ndof)), chi2, ndof)


# ---------------------------------------------------------------- combined forward model and inversion
@dataclass
class Forward:
    stored_mean: np.ndarray       # expected stored value per output pixel per stored frame
    stored_var: np.ndarray        # its variance (single pixel-frame)
    counted_mean: np.ndarray      # electrons counted per output pixel per stored frame (after dead time)
    true_mean: np.ndarray         # input
    efficiency: np.ndarray        # stored_mean / (true_mean * counts_per_electron): the total nonlinearity


def _tie(gain_flag) -> float | None:
    if gain_flag is None or isinstance(gain_flag, bool):
        return TIE_UP_GAIN_ON if gain_flag else None
    return float(gain_flag)       # an explicit tie probability


def forward(true_flux_e_per_px_per_frame, binning: int = 8, gain_flag: bool | float = True, tau: float = TAU_PHYS_V1,
            model: str = "paralyzable", cv: float = 0.0, n_raw: int = RAW_FRAMES_PER_STORED, fast: bool | None = None) -> Forward:
    """Expected stored mean and variance for a true mean of `true_flux_e_per_px_per_frame`
    electrons per OUTPUT pixel per STORED frame.

    Chain: paralyzable (or non-paralyzable) loss with `tau` (per e per physical px per
    raw frame, converted with `tau_for_binning`) -> counted mean mu_c -> Poisson N(mu_c)
    summed over the raw frames -> round-half-to-even(N / m), m = b^2/16, ties up with
    p = 0.35 when `gain_flag` is True (Apply Gain on Movie = Enable), parity rule when
    False; a float `gain_flag` is used as the tie probability. `cv` is the spread of the
    true mean over the averaged population (Gamma mixture). `fast` uses the tabulated
    law (1e-4 relative) and is the default for > 64 values."""
    from . import binning as B
    mu = np.atleast_1d(np.asarray(true_flux_e_per_px_per_frame, float))
    tb = tau_for_binning(tau, binning, n_raw) if tau else 0.0
    mu_c = recorded_rate(mu, tb, model) if tb else mu
    tie = _tie(gain_flag)
    if fast is None:
        fast = mu.size > 64
    if fast:
        lm, e, v = B.response_table(binning, tie, cv, 0.0, mu_min=1e-4, mu_max=max(5e3, float(np.nanmax(mu_c)) * 1.1 if np.isfinite(np.nanmax(mu_c)) else 5e3), n=1200)
        lm = np.log(lm); x = np.log(np.clip(mu_c, 1e-4, None))
        E = np.where(mu_c > 0, np.interp(x, lm, e), 0.0); V = np.where(mu_c > 0, np.interp(x, lm, v), 0.0)
    else:
        E = B.recorded_mean(mu_c, binning, tie, cv); V = B.recorded_var(mu_c, binning, tie, cv)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(mu > 0, E / (mu * counts_per_electron(binning)), np.nan)
    sq = np.ndim(true_flux_e_per_px_per_frame) == 0
    f = (lambda a: a[0]) if sq else (lambda a: a)
    return Forward(f(E), f(V), f(mu_c), f(mu), f(eff))


@dataclass
class Inverted:
    true_mean: np.ndarray         # electrons per output pixel per stored frame (lower branch)
    counted_mean: np.ndarray      # after the rounding-law inversion, before the dead-time inversion
    sigma: np.ndarray             # counting-noise sigma of true_mean (NaN without n_eff)
    amplification: np.ndarray    # d ln mu / d ln E of the whole chain
    ok: np.ndarray                # inside the validity range
    stored_max: float             # largest stored mean the chain can produce (at the dead-time peak)
    counted_max: float            # ceiling of the counted mean, e per output px per stored frame
    true_at_peak: float           # 1/tau_b: true rate at the peak (the inversion is one-valued below it)
    sigma_tie: np.ndarray | None = None   # correlated systematic of true_mean from the tie probability (with tie_up_err; audit E6)
    dmu_dp: np.ndarray | None = None      # d true_mean / d p_tie


def invert_mean(stored_mean, binning: int = 8, gain_flag: bool | float = True, tau: float = TAU_PHYS_V1, cv: float = 0.0,
                n_eff=None, model: str = "paralyzable", n_raw: int = RAW_FRAMES_PER_STORED, margin: float = 0.95,
                tie_up_err: float | None = None) -> Inverted:
    """Mean stored value (average over `n_eff` pixel-frames) -> true mean electrons per
    output pixel per stored frame, chaining `podbg.binning.true_mean` (rounding law,
    exact Brent) with the dead-time inversion (Lambert W, lower branch).

    Validity: the counted mean must lie below `margin` x the ceiling 1/(e tau_b)
    (paralyzable) or 1/tau_b (non-paralyzable); above it the stored mean is
    two-valued (saturated core) and `ok` is False with NaN. Near the ceiling the
    error amplification diverges as 1/(1 - mu tau_b). The counting sigma is
    propagated through both steps; additive contamination of the stored mean
    (floor, seams) must be removed before the call (see binning.true_mean).  With
    `tie_up_err` (e.g. binning.TIE_UP_ERR) the correlated tie-constant systematic is
    propagated through the dead-time step as well (`sigma_tie`, `dmu_dp`; audit E6)."""
    from . import binning as B
    tie = _tie(gain_flag)
    tb = tau_for_binning(tau, binning, n_raw) if tau else 0.0
    c_max = ceiling(tb, model) if tb else np.inf
    inv = B.true_mean(stored_mean, binning, tie, cv, n_eff=n_eff, tie_up_err=tie_up_err)
    mu_c = np.atleast_1d(np.asarray(inv.true_mean, float)); sig_c = np.atleast_1d(np.asarray(inv.sigma, float))
    amp_c = np.atleast_1d(np.asarray(inv.amplification, float)); ok = np.atleast_1d(np.asarray(inv.ok, bool)).copy()
    st_c = None if inv.sigma_tie is None else np.atleast_1d(np.asarray(inv.sigma_tie, float)); dp_c = None if inv.dmu_dp is None else np.atleast_1d(np.asarray(inv.dmu_dp, float))
    if tb:
        ok &= np.isfinite(mu_c) & (mu_c < margin * c_max)
        mu = np.where(ok, true_rate(np.where(ok, mu_c, 0.0), tb, model), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            if model == "paralyzable":
                dmu_dmuc = 1.0 / ((1.0 - mu * tb) * np.exp(-mu * tb))      # 1 / (dm/dn)
            else:
                dmu_dmuc = (1.0 + mu * tb) ** 2
            sig = sig_c * dmu_dmuc
            amp = amp_c * (mu_c / mu) * dmu_dmuc                             # d ln mu / d ln E
            st = None if st_c is None else st_c * dmu_dmuc; dp = None if dp_c is None else dp_c * dmu_dmuc
        stored_max = float(forward(1.0 / tb if model == "paralyzable" else 50.0 / tb, binning, gain_flag, tau, model, cv, n_raw, fast=False).stored_mean)
        true_peak = 1.0 / tb
    else:
        mu, sig, amp = mu_c, sig_c, amp_c; stored_max = np.inf; true_peak = np.inf; st, dp = st_c, dp_c
    sq = np.ndim(stored_mean) == 0
    f = (lambda a: a[0]) if sq else (lambda a: a)
    return Inverted(f(mu), f(mu_c), f(sig), f(amp), f(ok), stored_max, float(c_max), float(true_peak),
                    None if st is None else f(st), None if dp is None else f(dp))


# ---------------------------------------------------------------- floor
def floor_model(r, A: float, rc: float, C: float):
    """Centred inverse-square halo plus flat term, counts/px/frame."""
    r = np.asarray(r, float)
    return A / (r ** 2 + rc ** 2) + C


def fit_floor(r, P, r_min: float = 40.0, r_max: float = 500.0):
    """Fit floor_model to an azimuthal profile on [r_min, r_max]; returns (A, rc, C), chi2, n."""
    from scipy.optimize import least_squares
    r = np.asarray(r, float); P = np.asarray(P, float)
    ok = (r >= r_min) & (r <= r_max) & np.isfinite(P)
    rr, pp = r[ok], P[ok]
    # Poisson-like weights on a profile: var ~ P / n_px; use relative weights sqrt(P) with a floor
    w = 1.0 / np.sqrt(np.maximum(pp, 1e-6))
    A0 = pp[0] * rr[0] ** 2

    def res(p):
        return (floor_model(rr, *p) - pp) * w
    f = least_squares(res, [A0, 10.0, max(pp.min(), 0.0)], bounds=([0, 0, 0], [np.inf, 300, np.inf]))
    return tuple(float(v) for v in f.x), float((f.fun ** 2).sum()), int(ok.sum())


def fit_floor_centre(image, mask, centre0, r_in: float = 60.0, r_out: float = 400.0, step: float = 0.5,
                     half_width: float = 12.0):
    """Centre of a 1/r^2 halo in a summed image by maximising the profile likelihood on a grid.

    For each trial centre the azimuthal profile on [r_in, r_out] is fitted with
    floor_model and the Poisson deviance of the image pixels against the
    fitted axisymmetric model is recorded; the minimum over a grid of
    (dy, dx) in [-half_width, half_width] with `step` is returned together
    with the deviance surface. Masked pixels (True) are ignored."""
    img = np.asarray(image, float); mask = np.asarray(mask, bool)
    h, w = img.shape; yy, xx = np.indices((h, w))
    offs = np.arange(-half_width, half_width + step / 2, step)
    dev = np.full((len(offs), len(offs)), np.nan)
    for i, dy in enumerate(offs):
        for j, dx in enumerate(offs):
            cy, cx = centre0[0] + dy, centre0[1] + dx
            r = np.hypot(yy - cy, xx - cx)
            sel = (r >= r_in) & (r <= r_out) & ~mask
            ri = r[sel].astype(int); v = img[sel]
            n = np.bincount(ri); s = np.bincount(ri, weights=v)
            ok = n > 0; prof = np.full(len(n), np.nan); prof[ok] = s[ok] / n[ok]
            (A, rc, C), _, _ = fit_floor(np.arange(len(n)), prof, r_in, r_out)
            mu = np.maximum(floor_model(r[sel], A, rc, C), 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = 2 * np.where(v > 0, v * np.log(v / mu), 0.0) - 2 * (v - mu)
            dev[i, j] = d.sum()
    k = np.nanargmin(dev); i, j = np.unravel_index(k, dev.shape)
    return (float(centre0[0] + offs[i]), float(centre0[1] + offs[j])), dev, offs


# ---------------------------------------------------------------- fixed pattern
def seam_lines(binning: int = 8, shape: tuple[int, int] | None = None) -> dict:
    """Row and column indices of the sensor-segment seams at `binning`.

    The 4 x 2 segments of the 4096-pixel sensor meet at physical columns 1024,
    2048, 3072 and physical row 2048; at binning b (relative to 8192) the
    seam straddles the two binned lines on either side of the boundary."""
    n = SUPERRES_GRID // binning
    cols = sorted(set(int(c) for q in (1, 2, 3) for c in (q * n // 4 - 1, q * n // 4)))
    rows = [n // 2 - 1, n // 2]
    return {"cols": cols, "rows": rows, "first_row": 0, "n": n}


def fixed_pattern_mask(binning: int = 8, border: int = 0) -> np.ndarray:
    """Boolean mask (True = exclude) of seams, first row and an optional border."""
    s = seam_lines(binning); n = s["n"]
    m = np.zeros((n, n), bool)
    m[:, s["cols"]] = True; m[s["rows"], :] = True; m[s["first_row"], :] = True
    if border > 0:
        m[:border, :] = m[-border:, :] = True; m[:, :border] = m[:, -border:] = True
    return m


@dataclass(frozen=True)
class FixedPattern:
    gain: np.ndarray        # multiplicative, 1 where no correction
    add: np.ndarray         # additive, counts/px/stored frame, 0 where no correction
    exclude: np.ndarray     # pixels that should be masked rather than corrected
    binning: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """(frame - add) / gain as float; excluded pixels set to NaN."""
        out = (np.asarray(frame, float) - self.add) / self.gain
        out[self.exclude] = np.nan
        return out


def save_fixed_pattern(path, fp: FixedPattern) -> Path:
    path = Path(path)
    np.savez_compressed(path, gain=fp.gain.astype(np.float32), add=fp.add.astype(np.float32),
                        exclude=fp.exclude, binning=fp.binning)
    return path


def load_fixed_pattern(path) -> FixedPattern:
    z = np.load(path)
    return FixedPattern(z["gain"].astype(float), z["add"].astype(float), z["exclude"].astype(bool), int(z["binning"]))


def line_response(level: np.ndarray, line: np.ndarray):
    """Additive vs multiplicative test for a seam line.

    `level` are local reference values (mean of the neighbouring lines) and
    `line` the seam values, over groups of increasing signal. Fits
    line = a + g * level by weighted least squares and returns (a, g, resid rms).
    A pure multiplicative seam has a ~ 0 and g > 1; a pure additive seam has
    g ~ 1 and a > 0."""
    L = np.asarray(level, float); S = np.asarray(line, float)
    X = np.c_[np.ones_like(L), L]
    coef, *_ = np.linalg.lstsq(X, S, rcond=None)
    resid = S - X @ coef
    return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(resid ** 2)))
