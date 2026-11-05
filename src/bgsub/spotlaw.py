# MOVED from pod/background/src/podbg/bgsub/spotlaw.py at background commit f1e93f5 (file last changed in 98139b6; sha256 52636bcc47740239905c4b3e3b79e3b3f0d52f6a11cf6c4908bf1f5fc8ebbec3).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""Counting-law-aware spot intensities (design note Amendment 3 a): law-corrected summation (X-LC) and the profile
likelihood of the stored counts (X-ML).  Vectorised over spots: arrays are (n_spots, n_max_box_pixels) with a pixel
validity mask `pv`.

mu   : background electrons per box pixel (post dead time; = law.E_inv(B_p))
f    : box-normalised spot profile (sum over the valid box pixels = 1), so lambda is the BOX intensity in electrons
"""
from __future__ import annotations

import numpy as np


def mu_eff(mu, lam_p, tau):
    """Electron mean that reaches the quantiser: background thinned by the spot's dead time plus the counted spot
    electrons (paralyzable mean field, as peaks.inject); for lam_p < 0 the linear continuation mu + lam_p."""
    if tau:
        pos = mu * np.exp(-np.maximum(lam_p, 0) * tau) + np.maximum(lam_p, 0) * np.exp(-(mu + np.maximum(lam_p, 0)) * tau)
    else:
        pos = mu + np.maximum(lam_p, 0)
    return np.where(lam_p >= 0, pos, np.maximum(mu + lam_p, 0.0))


def L_of(lam, mu, f, pv, law):
    """Law-expected stored excess in ELECTRONS (epc x stored), per spot, for box intensities lam (n_spots,)."""
    lp = lam[:, None] * f
    d = law.E(mu_eff(mu, lp, law.tau)) - law.E(mu)
    return law.epc * np.sum(np.where(pv, d, 0.0), axis=1)


def law_correct(I_sum, sig_sum, mu, f, pv, law, n_iter=60):
    """lambda^ = L^-1(I_sum) by bisection (monotone branch below the dead-time maximum); sigma = sig_sum / L'(lambda^).
    Returns (lam, sigma, saturated)."""
    n = len(I_sum)
    mu_box = np.sum(np.where(pv, mu, 0), 1)
    lo = -np.minimum(np.min(np.where(pv & (f > 0), mu / np.maximum(f, 1e-12), np.inf), axis=1), 10 * mu_box + 10)
    hi = np.full(n, 1.0)
    # grow hi until L(hi) >= I_sum or the dead-time maximum is passed
    Lhi = L_of(hi, mu, f, pv, law)
    sat = np.zeros(n, bool)
    for _ in range(60):
        need = (Lhi < I_sum) & ~sat
        if not need.any():
            break
        h2 = np.where(need, hi * 2, hi)
        L2 = L_of(h2, mu, f, pv, law)
        sat |= need & (L2 <= Lhi)                       # past the maximum: saturated
        hi = np.where(need & ~sat, h2, hi); Lhi = np.where(need & ~sat, L2, Lhi)
    a = lo.copy(); b = hi.copy()
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        Lm = L_of(m, mu, f, pv, law)
        up = Lm < I_sum
        a = np.where(up, m, a); b = np.where(up, b, m)
    lam = 0.5 * (a + b)
    h = np.maximum(1e-3 * np.abs(lam), 1e-3 * (mu_box + 1.0))
    dL = (L_of(lam + h, mu, f, pv, law) - L_of(lam - h, mu, f, pv, law)) / (2 * h)
    sig = sig_sum / np.maximum(dL, 1e-12)
    return lam, sig, sat, dL


def _logp(law, mu_e, K):
    lmu, lp, _ = law.pmf_table()
    x = np.log(np.maximum(mu_e, 1e-300))
    j = np.clip(np.searchsorted(lmu, x) - 1, 0, len(lmu) - 2)
    t = np.clip((x - lmu[j]) / (lmu[j + 1] - lmu[j]), 0, 1)
    k = np.clip(K.astype(np.int64), 0, lp.shape[1] - 1)
    v = (1 - t) * lp[j, k] + t * lp[j + 1, k]
    # below the table: Poisson-like limit P(0) -> 1, P(k > 0) -> tiny
    small = mu_e < np.exp(lmu[0])
    return np.where(small, np.where(K <= 0, -mu_e / max(law.epc, 1.0), lp[0, k]), v)


def _fisher(law, mu_e):
    lmu, _, F = law.pmf_table()
    x = np.log(np.maximum(mu_e, np.exp(lmu[0])))
    return np.exp(np.interp(x, lmu, np.log(np.maximum(F, 1e-300))))


def _ll(K, mu, f, pv, law, g):
    me = mu_eff(mu[:, None, :], g[:, :, None] * f[:, None, :], law.tau)
    return np.sum(np.where(pv[:, None, :], _logp(law, me, K[:, None, :]), 0.0), axis=2)


def ml_fit(K, mu, f, pv, law, lam0, sig0, I_sum=None, n_grid=81):
    """Profile ML of the box intensity under the stored-count law. A global log-spaced pass over [lo, lam_max] picks the
    branch (beyond the paralyzable dead-time maximum the stored excess falls again, so the likelihood can be
    bimodal), then two local passes refine. Returns (lam, sigma_stat) with sigma_stat from the EXPECTED Fisher
    information at lam^ (finite for all-zero boxes)."""
    n = len(lam0); ar = np.arange(n)
    mu_box = np.sum(np.where(pv, mu, 0), 1)
    lo = -np.minimum(np.min(np.where(pv & (f > 0), mu / np.maximum(f, 1e-12), np.inf), axis=1), 10 * mu_box + 10) * 0.999
    big = np.maximum.reduce([50 * (mu_box + 1.0), 30 * np.abs(np.nan_to_num(lam0)), 30 * np.abs(np.nan_to_num(I_sum if I_sum is not None else lam0))])
    g0 = np.concatenate([lo[:, None] * np.linspace(1, 0, 12)[None, :-1],
                         np.geomspace(1e-3, 1.0, 140)[None, :] * big[:, None]], axis=1)
    ll0 = _ll(K, mu, f, pv, law, g0)
    i0 = np.argmax(ll0, axis=1)
    c = g0[ar, i0]
    s0 = np.maximum(np.abs(g0[ar, np.clip(i0 + 1, 0, g0.shape[1] - 1)] - g0[ar, np.clip(i0 - 1, 0, g0.shape[1] - 1)]), 1e-3 * (mu_box + 1))
    lam = c
    for width in (1.0, 0.1):
        g = c[:, None] + s0[:, None] * width * np.linspace(-1, 1, n_grid)[None, :]
        g = np.maximum(g, lo[:, None])
        ll = _ll(K, mu, f, pv, law, g)
        i = np.argmax(ll, axis=1)
        ii = np.clip(i, 1, n_grid - 2)
        y0 = ll[ar, ii - 1]; y1 = ll[ar, ii]; y2 = ll[ar, ii + 1]
        den = y0 - 2 * y1 + y2
        step = g[ar, ii + 1] - g[ar, ii]
        off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
        lam = g[ar, ii] + np.clip(off, -1, 1) * step
        lam = np.where((i == 0) | (i == n_grid - 1), g[ar, i], lam)
        c = lam
    h = np.maximum(1e-3 * np.abs(lam), 1e-3 * (mu_box + 1.0))
    dme = (mu_eff(mu, (lam + h)[:, None] * f, law.tau) - mu_eff(mu, (lam - h)[:, None] * f, law.tau)) / (2 * h[:, None])
    me = mu_eff(mu, lam[:, None] * f, law.tau)
    info = np.sum(np.where(pv, dme ** 2 * _fisher(law, me), 0.0), axis=1)
    return lam, 1.0 / np.sqrt(np.maximum(info, 1e-300))


def law_correct_branch(I_sum, sig_sum, mu, f, pv, law, lam_hint, n_iter=60):
    """X-LC with the branch chosen by lam_hint (the ML estimate): if lam_hint lies beyond the maximum of L, solve on the
    decreasing branch. Returns (lam, sigma, beyond_max, dL)."""
    lam, sig, sat, dL = law_correct(I_sum, sig_sum, mu, f, pv, law, n_iter)
    n = len(I_sum); mu_box = np.sum(np.where(pv, mu, 0), 1)
    grid = np.geomspace(1e-2, 1e6, 300)[None, :] * (mu_box[:, None] + 1.0)
    Lg = np.stack([L_of(grid[:, j], mu, f, pv, law) for j in range(grid.shape[1])], 1)
    jpk = np.argmax(Lg, axis=1); lpk = grid[np.arange(n), jpk]; Lpk = Lg[np.arange(n), jpk]
    beyond = np.isfinite(lam_hint) & (lam_hint > lpk) & (jpk < grid.shape[1] - 1)
    if beyond.any():
        idx = np.flatnonzero(beyond)
        a = lpk[idx].copy(); b = grid[idx, -1].copy()
        tgt = np.minimum(I_sum[idx], Lpk[idx])
        for _ in range(n_iter):
            m = np.sqrt(a * b)
            Lm = L_of(m, mu[idx], f[idx], pv[idx], law)
            right = Lm > tgt                   # decreasing branch: L > target -> root is to the right
            a = np.where(right, m, a); b = np.where(right, b, m)
        lb = np.sqrt(a * b)
        hh = 1e-3 * lb
        d2 = (L_of(lb + hh, mu[idx], f[idx], pv[idx], law) - L_of(lb - hh, mu[idx], f[idx], pv[idx], law)) / (2 * hh)
        lam[idx] = lb; dL[idx] = d2; sig[idx] = sig_sum[idx] / np.maximum(np.abs(d2), 1e-12)
    return lam, sig, beyond, dL


# ============================================================================= v1 (design note v1 Sec. 3, fix 3)
def dmu_eff(mu, f, lam, tau):
    """d mu_eff / d lambda for lambda_p = lam f (analytic; the linear continuation below zero)."""
    lp = lam[..., None] * f if np.ndim(lam) < np.ndim(f) else lam * f
    if tau:
        lpp = np.maximum(lp, 0)
        pos = f * (np.exp(-(mu + lpp) * tau) * (1.0 - lpp * tau) - mu * tau * np.exp(-lpp * tau))
    else:
        pos = f * np.ones_like(lp)
    return np.where(lp >= 0, pos, f * np.ones_like(lp))


def _info_grid(mu, f, pv, law, g):
    """Expected Fisher information about the box intensity at each grid value g (n_spots, n_grid)."""
    me = mu_eff(mu[:, None, :], g[:, :, None] * f[:, None, :], law.tau)
    d = dmu_eff(mu[:, None, :], f[:, None, :], g[:, :, None] * 0 + g[:, :, None], law.tau) if False else _dme_grid(mu, f, g, law.tau)
    return np.sum(np.where(pv[:, None, :], d ** 2 * _fisher(law, me), 0.0), axis=2)


def _dme_grid(mu, f, g, tau):
    lp = g[:, :, None] * f[:, None, :]
    m = mu[:, None, :]; ff = f[:, None, :]
    if tau:
        lpp = np.maximum(lp, 0)
        pos = ff * (np.exp(-(m + lpp) * tau) * (1.0 - lpp * tau) - m * tau * np.exp(-lpp * tau))
    else:
        pos = ff * np.ones_like(lp)
    return np.where(lp >= 0, pos, ff * np.ones_like(lp))


def ml_fit_firth(K, mu, f, pv, law, lam0, sig0, I_sum=None, n_grid=81):
    """Firth-penalised profile likelihood: argmax l(lambda) + 1/2 log I(lambda) (Jeffreys prior; removes the O(1/I) bias
    of the ML). Same grid search as ml_fit. Returns (lam, sigma_stat) with sigma_stat from the expected information."""
    n = len(lam0); ar = np.arange(n)
    mu_box = np.sum(np.where(pv, mu, 0), 1)
    lo = -np.minimum(np.min(np.where(pv & (f > 0), mu / np.maximum(f, 1e-12), np.inf), axis=1), 10 * mu_box + 10) * 0.999
    big = np.maximum.reduce([50 * (mu_box + 1.0), 30 * np.abs(np.nan_to_num(lam0)), 30 * np.abs(np.nan_to_num(I_sum if I_sum is not None else lam0))])
    g0 = np.concatenate([lo[:, None] * np.linspace(1, 0, 12)[None, :-1],
                         np.geomspace(1e-3, 1.0, 140)[None, :] * big[:, None]], axis=1)

    def obj(g):
        return _ll(K, mu, f, pv, law, g) + 0.5 * np.log(np.maximum(_info_grid(mu, f, pv, law, g), 1e-300))

    ll0 = obj(g0)
    i0 = np.argmax(ll0, axis=1)
    c = g0[ar, i0]
    s0 = np.maximum(np.abs(g0[ar, np.clip(i0 + 1, 0, g0.shape[1] - 1)] - g0[ar, np.clip(i0 - 1, 0, g0.shape[1] - 1)]), 1e-3 * (mu_box + 1))
    lam = c
    for width in (1.0, 0.1):
        g = c[:, None] + s0[:, None] * width * np.linspace(-1, 1, n_grid)[None, :]
        g = np.maximum(g, lo[:, None])
        ll = obj(g)
        i = np.argmax(ll, axis=1)
        ii = np.clip(i, 1, n_grid - 2)
        y0 = ll[ar, ii - 1]; y1 = ll[ar, ii]; y2 = ll[ar, ii + 1]
        den = y0 - 2 * y1 + y2
        step = g[ar, ii + 1] - g[ar, ii]
        off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
        lam = g[ar, ii] + np.clip(off, -1, 1) * step
        lam = np.where((i == 0) | (i == n_grid - 1), g[ar, i], lam)
        c = lam
    info = _info_grid(mu, f, pv, law, lam[:, None])[:, 0]
    return lam, 1.0 / np.sqrt(np.maximum(info, 1e-300))


def lam_min(mu, f, pv, law, cr=0.45, n_grid=90):
    """Information limit (design note v1 Sec. 2.3): the smallest box intensity lambda (electrons) at which the Cramer-Rao
    relative SD 1 / (lambda sqrt(I(lambda))) of the box intensity is <= cr, from the expected Fisher information under
    the declared law with background mu and profile f (no data of the spot).  inf if never reached."""
    mu_box = np.sum(np.where(pv, mu, 0), 1)
    g = np.geomspace(1e-3, 1e5, n_grid)[None, :] * (mu_box[:, None] + 1.0)
    info = _info_grid(mu, f, pv, law, g)
    rel = 1.0 / (g * np.sqrt(np.maximum(info, 1e-300)))
    ok = rel <= cr
    first = np.where(ok.any(1), np.argmax(ok, 1), -1)
    out = np.full(len(mu_box), np.inf)
    for i in np.flatnonzero(first >= 0):
        j = first[i]
        if j == 0:
            out[i] = g[i, 0]; continue
        x0, x1 = np.log(g[i, j - 1]), np.log(g[i, j]); y0, y1 = np.log(rel[i, j - 1]), np.log(rel[i, j])
        t = (np.log(cr) - y0) / (y1 - y0) if y1 != y0 else 1.0
        out[i] = float(np.exp(x0 + np.clip(t, 0, 1) * (x1 - x0)))
    return out


def pooled_ml(K_list, mu_list, f_list, law):
    """One joint stored-count likelihood for n EQUIVALENT observations of the same reflection (design note v1 Sec. 1.2:
    pooling before inverting shrinks the information-limit bias ~1/n).  K_list / mu_list / f_list: per observation the
    box stored values (offset removed), background electrons per pixel and the box-normalised profile.  Returns
    (lambda, sigma_stat): the common BOX intensity in electrons."""
    K = np.concatenate([np.ravel(k) for k in K_list])[None, :].astype(float)
    mu = np.concatenate([np.ravel(m) for m in mu_list])[None, :].astype(float)
    f = np.concatenate([np.ravel(x) for x in f_list])[None, :].astype(float)
    pv = np.ones_like(K, bool)
    n = len(K_list)
    I = law.epc * (K.sum(1) - law.E(mu).sum(1)) / n
    s = law.epc * np.sqrt(law.g(law.E(mu)).sum(1)) / n
    l0, s0, _a, _b = law_correct(I, s, mu, f / n, pv, law)
    lam, sig = ml_fit(K, mu, f, pv, law, l0, s0, I_sum=I)
    return float(lam[0]), float(sig[0])
