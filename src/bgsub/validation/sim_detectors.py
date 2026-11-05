# VENDORED COPY. Source: pod/background/src/podbg/sim/detectors.py at background commit f1e93f5 (file last changed in 95fd1e4);
# sha256 of the source file: 33f8b3284e26ccc2fd4037420634e3bfb1d3f82fdf171144ef6e6d4c70a9cb32. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: `from .. import detector/binning` -> `from ..vendor import detector/binning`.
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Detector chains of the simulator: the Apollo stored-count law and a generic integrating detector.

Apollo (counting; `podbg.detector`, `podbg.binning`, all MEASURED on the Caltech Apollo, 200 kV):
  N ~ Poisson(mu exp(-mu tau_b))        paralyzable dead time, tau 0.206 per e per physical px per raw frame (418 us),
                                        tau_b = tau / (b/2)^2 / 3 per output pixel per stored frame (detector.tau_for_binning)
  stored = round-half-to-even(N / m)    m = b^2 / 16: 1 count = 1 e at bin 4, 4 e at bin 8, 16 e at bin 16
  ties (N / m = k + 1/2)                gain flag on: up with p (default 0.403, the measured film p, constant-p mode;
                                        "v2" = binning.tie_prob rate law; "0.35" = the B0 inversion's value);
                                        gain flag off: the parity rule (to even; measured p_odd 0.02 is ignored)
Not rendered: seams / border offsets (masked in the archive), split 2-clusters (1.44 % of bin-4 events), the
multi-event bar floor of the gain-on flag (it lives on bar frames), the counting ceiling's zero core (the chain is
valid below 0.95 of the ceiling, 85.7 e per bin-8 px per frame; the truth record flags frames that exceed it).

Integrating (generic, ASSUMPTION): ADU = gain x Poisson(mu) + N(0, read_noise); no DQE, MTF or coincidence; the
defaults (1 ADU/e, 1 ADU) are placeholders, not a CETA calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vendor import detector as D
from ..vendor import binning as B

FILM_P = 0.403                       # measured film tie p (gain on), TIE FILM + CONSTANT p v0 / MEASURED-p v0
APOLLO_SUPERRES_UM = 4.0             # Apollo binning is relative to the 8192 super-resolution crop (4 um)
APOLLO_CL420_L_MM = 559.4            # effective camera length of nominal CL 420 (garnet nets, k 0.0022811 A^-1 / bin-8 px)
APOLLO_CL960_L_MM = 1255.0           # windowed ring-1 fit (sdb61cf94); not from a crystal standard
APOLLO_CENTRE_BIN8 = (516.0, 470.0)  # typical measured beam centre (y, x) at bin 8 (extraction v1: medians y 509-523, x 448-482)
APOLLO_STOP_R_BIN8 = 52.0            # beam-stop dark core radius at bin 8 (geometry calibration; v0 mask disc 44 px)
APOLLO_STOP_ARM_HALF_BIN8 = 10.0     # ASSUMPTION: half-width of the stop arm (to the left), bin-8 px


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    kind: str                     # "counting" | "integrating"
    shape: tuple
    pixel_um: float
    distance_mm: float
    binning: int | None = None
    gain_flag: bool = True
    tie: object = FILM_P          # float p, "v2", or None (parity)
    tau_phys: float = D.TAU_PHYS_V1
    gain: float = 1.0             # integrating: ADU per electron
    read_noise: float = 1.0       # integrating: ADU rms
    centre: tuple | None = None
    stop_r_px: float = 0.0
    stop_arm_half_px: float = 0.0

    @property
    def tau_binned(self) -> float:
        return D.tau_for_binning(self.tau_phys, self.binning) if self.kind == "counting" else 0.0

    @property
    def m(self) -> int:
        return int(round(self.binning ** 2 / 16)) if self.kind == "counting" else 1


def apollo(binning: int = 8, gain_flag: bool = True, tie=FILM_P, distance_mm: float = APOLLO_CL420_L_MM, shape=None) -> DetectorSpec:
    if binning not in (4, 8, 16):
        raise ValueError("the Apollo stored-count law is measured at bin 4, 8 and 16 only")
    n = 8192 // binning
    f = 8.0 / binning
    c = (APOLLO_CENTRE_BIN8[0] * f, APOLLO_CENTRE_BIN8[1] * f)
    return DetectorSpec("apollo", "counting", tuple(shape) if shape else (n, n), APOLLO_SUPERRES_UM * binning, float(distance_mm), binning,
                        bool(gain_flag), tie, centre=c, stop_r_px=APOLLO_STOP_R_BIN8 * f, stop_arm_half_px=APOLLO_STOP_ARM_HALF_BIN8 * f)


def ceta(gain: float = 1.0, read_noise: float = 1.0, distance_mm: float = 957.8, shape=(2048, 2048)) -> DetectorSpec:
    """Arctica-CETA at CL 960 as processed by ringcal (QX 0.028 mm, D 957.8 mm); integrating response is an ASSUMPTION."""
    return DetectorSpec("ceta", "integrating", tuple(shape), 28.0, float(distance_mm), None, gain=gain, read_noise=read_noise)


def generic(shape=(4096, 4096), pixel_um: float = 15.0, distance_mm: float = 1000.0, gain: float = 1.0, read_noise: float = 1.0) -> DetectorSpec:
    return DetectorSpec("generic", "integrating", tuple(shape), float(pixel_um), float(distance_mm), None, gain=gain, read_noise=read_noise)


def quantise_fast(N: np.ndarray, m: int, gain_flag: bool, tie, rng, rate=None) -> np.ndarray:
    """round-half-to-even(N / m) with the gain flag's tie rule (integer arithmetic; equal in law to binning.quantise)."""
    if m == 1:
        return N
    q, rem = np.divmod(N, m)
    val = q + (rem * 2 > m)
    if m % 2:
        return val
    tie_m = rem * 2 == m
    if not tie_m.any():
        return val
    if not gain_flag or tie is None:
        up = (q % 2) == 1                                        # parity: round to even
    elif isinstance(tie, str) and tie == "v2":
        p = B.tie_prob(np.broadcast_to(rate, N.shape) if rate is not None else np.full(N.shape, float(N.mean())), "Enable")
        up = rng.random(N.shape) < p
    else:
        up = rng.random(N.shape) < float(tie)
    return np.where(tie_m, q + up, val)


def record(mu: np.ndarray, spec: DetectorSpec, rng, noise: bool = True) -> np.ndarray:
    """Stored frame for expected electrons mu (per output pixel per frame)."""
    if spec.kind == "counting":
        mu_c = mu * np.exp(-mu * spec.tau_binned) if spec.tau_binned else mu
        if not noise:
            tie_arg = False if not spec.gain_flag else (float(spec.tie) if isinstance(spec.tie, (int, float)) else FILM_P)   # "v2": constant-p stand-in
            fw = D.forward(mu.ravel(), spec.binning, tie_arg, tau=spec.tau_phys)
            return np.asarray(fw.stored_mean, np.float32).reshape(mu.shape)
        N = rng.poisson(mu_c)
        out = quantise_fast(N, spec.m, spec.gain_flag, spec.tie, rng, rate=mu)
        return out.astype(np.uint16)
    if not noise:
        return (spec.gain * mu).astype(np.float32)
    N = rng.poisson(mu).astype(np.float32)
    out = spec.gain * N
    if spec.read_noise:
        out += rng.standard_normal(mu.shape, dtype=np.float32) * np.float32(spec.read_noise)
    return out


def beam_stop_mask(shape, centre_yx, r_px: float, arm_half_px: float) -> np.ndarray:
    """True where the stop blocks: a disc plus an arm running to the left frame edge."""
    if r_px <= 0:
        return np.zeros(shape, bool)
    ny, nx = shape
    yy = np.arange(ny)[:, None] - centre_yx[0]; xx = np.arange(nx)[None, :] - centre_yx[1]
    m = (yy ** 2 + xx ** 2) <= r_px ** 2
    if arm_half_px > 0:
        m |= (np.abs(yy) <= arm_half_px) & (xx <= 0)
    return m
