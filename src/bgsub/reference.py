# MOVED from pod/background/src/podbg/bgsub/reference.py at background commit f1e93f5 (file last changed in 28fa4c6; sha256 8f03dd2f8ec5971e2af7c7a31b96a9d3a582f2c34849eca40ff3ce42ec577475).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""Generic reference local background estimates (written here; not copied from XDS or DIALS), design note Sec. 1.7.

Each takes the annulus pixels of ONE spot: values Ka (stored counts, offset removed), offsets (dy, dx) from the spot
centre, and returns (b0, gy, gx, var_coefficients) so that the background summed over the box is
B_box = n_box b0 + gy sum(dy_box) + gx sum(dx_box).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

HUBER_C = 1.345


def flat(Ka, law, dy=None, dx=None):
    n = len(Ka)
    b0 = float(np.mean(Ka)) if n else np.nan
    # variance of the mean: plug-in sum g(K) / n^2 (the peak-recovery "sigma_law" form) and model form g(b0) / n
    v_plug = float(np.sum(law.g(np.maximum(Ka, 0.0)))) / max(n, 1) ** 2
    v_mod = float(law.g(np.array([max(b0, 0.0)]))[0]) / max(n, 1)
    return b0, 0.0, 0.0, {"var_b0_plugin": v_plug, "var_b0_model": v_mod, "cov": None}


def plane(Ka, law, dy, dx):
    A = np.column_stack([np.ones(len(Ka)), dy, dx])
    AtA = A.T @ A
    coef = np.linalg.solve(AtA, A.T @ Ka)
    gb = float(law.g(np.array([max(coef[0], 0.0)]))[0])
    cov = np.linalg.inv(AtA) * gb
    return float(coef[0]), float(coef[1]), float(coef[2]), {"cov": cov}


def _psi(z, c=HUBER_C):
    return np.clip(z, -c, c)


def robust(Ka, law, dy=None, dx=None, c=HUBER_C, n_iter=50):
    """Huber M-estimate of a constant with Poisson variance and the POISSON Fisher-consistency correction:
    solve sum psi((K - mu)/sqrt(mu)) - n E_Poisson(mu)[psi((K - mu)/sqrt(mu))] = 0 for mu (bisection, monotone)."""
    Ka = np.asarray(Ka, float); n = len(Ka)
    if n == 0:
        return np.nan, 0.0, 0.0, {"cov": None}
    kmax = int(max(Ka.max(), 1) * 3 + 60)
    kk = np.arange(kmax + 1)

    def f(mu):
        s = np.sqrt(mu)
        lhs = np.sum(_psi((Ka - mu) / s, c))
        p = stats.poisson.pmf(kk, mu)
        return lhs - n * np.sum(p * _psi((kk - mu) / s, c))

    lo, hi = 1e-6, max(float(Ka.mean()) * 4 + 5, 5.0)
    flo = f(lo)
    if flo <= 0:
        mu = lo
    else:
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
    # asymptotic variance of the M-estimate under Poisson: mu E[psi^2] / (E[psi'] ... ) / n
    s = np.sqrt(mu); p = stats.poisson.pmf(kk, mu); z = (kk - mu) / s
    Epsi2 = np.sum(p * _psi(z, c) ** 2); Edpsi = np.sum(p * (np.abs(z) <= c))
    v = mu * Epsi2 / max(Edpsi, 1e-3) ** 2 / n
    return float(mu), 0.0, 0.0, {"var_b0_model": float(v), "cov": None}
