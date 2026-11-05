# VENDORED COPY. Source: pod/background/src/podbg/sim/geometry.py at background commit f1e93f5 (file last changed in 95fd1e4);
# sha256 of the source file: 8e95eb9ccb22c41857e4c9cade621f42afa076b0d2c7b790c6592aff3a19c67b. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: none (byte-identical below this header).
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Exact diffraction geometry of the vanilla simulator (any wavelength, distance and pixel pitch).

A flat detector at distance L (mm) normal to the beam; a pixel at radius R (mm) from the beam centre sees the
scattering angle theta = atan(R / L) and s = 2 sin(theta / 2) / lambda (A^-1, s = 1/d).  The expected electrons in a
pixel are dose x g(s) x a_s, with g the angular distribution per incident electron per unit s-AREA (A^2) and a_s the
area of the pixel mapped into the s plane (A^-2), from the Jacobian of R -> s:

    a_s = p^2 (s / R) (ds / dR),   ds/dR = cos(theta / 2) L / (lambda (L^2 + R^2)),

which tends to (p / (lambda L))^2 = k^2 at small angle (k the s per pixel, 0.0022811 A^-1 per bin-8 px at the Apollo
CL 420, garnet calibration).  Obliquity and the 1/r^2 fall-off of a flat detector are inside the Jacobian.
"""
from __future__ import annotations

import numpy as np

ME_C2_EV = 510998.95                 # electron rest energy (CODATA 2018)
HC_EV_A = 12398.419843320026         # h c in eV A


def gamma(kV: float) -> float:
    return 1.0 + float(kV) * 1e3 / ME_C2_EV


def beta2(kV: float) -> float:
    g = gamma(kV)
    return 1.0 - 1.0 / g ** 2


def wavelength_A(kV: float) -> float:
    """Relativistic electron wavelength in A at accelerating voltage kV."""
    E = float(kV) * 1e3
    return HC_EV_A / np.sqrt(E * (2.0 * ME_C2_EV + E))


def beta_scale(kV: float, kV_ref: float = 200.0) -> float:
    """beta_ref^2 / beta^2: the factor by which every kinematic cross-section per unit s-area at fixed s grows from
    kV_ref to kV (lambda^2 |f|^2 with f ~ gamma and lambda ~ 1/(gamma beta)); thicknesses enter as t x this factor."""
    return beta2(kV_ref) / beta2(kV)


def s_of_R(R_mm, L_mm: float, lam_A: float) -> np.ndarray:
    th = np.arctan(np.asarray(R_mm, float) / float(L_mm))
    return 2.0 * np.sin(th / 2.0) / lam_A


def s_area_per_pixel(R_mm, L_mm: float, lam_A: float, pixel_mm: float) -> np.ndarray:
    """a_s(R): area in the s plane (A^-2) of a pixel of pitch pixel_mm at radius R_mm (exact Jacobian)."""
    R = np.asarray(R_mm, float)
    th = np.arctan(R / L_mm)
    dsdR = np.cos(th / 2.0) * L_mm / (lam_A * (L_mm ** 2 + R ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_over_R = np.where(R > 0, s_of_R(R, L_mm, lam_A) / np.where(R > 0, R, 1.0), 1.0 / (lam_A * L_mm))
    return pixel_mm ** 2 * s_over_R * dsdR


def k_small_angle(L_mm: float, lam_A: float, pixel_mm: float) -> float:
    """s per pixel at the centre (A^-1 per px)."""
    return pixel_mm / (lam_A * L_mm)


def radius_map(shape, centre_yx, ellipticity: float = 1.0) -> np.ndarray:
    """Pixel-centre radius in px about centre (y, x); `ellipticity` stretches x (k_x / k_y; 1.008 measured on the
    Caltech Arctica by garnet nets, not applied by default because S v2 was extracted from azimuthal means)."""
    ny, nx = shape
    y = (np.arange(ny, dtype=np.float32) - np.float32(centre_yx[0]))[:, None]
    x = (np.arange(nx, dtype=np.float32) - np.float32(centre_yx[1]))[None, :] * np.float32(ellipticity)
    return np.sqrt(y * y + x * x)
