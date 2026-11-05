# MOVED from pod/background/src/podbg/bgsub/law.py at background commit f1e93f5 (file last changed in d8ca202; sha256 9f204939c4a524972ff237012bb57992acd595387649b1c63693f0ec44d3f4c5).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""Count laws for bgsub: variance function g(b), its slope, electrons per count and one-sided tail thresholds.

b is always the expected STORED value of a pixel (what the frame holds).  Nothing in the estimator assumes Poisson for
unbiasedness; the law enters through weights (efficiency), outlier thresholds and variances (design note Sec. 1.1, 1.3.2).

Laws:
- ``CountLaw.apollo(binning, tie_up, dead_time_tau)``: DE Apollo stored-count law (round-half-to-even of the 3-raw-frame
  sum, m = binning^2 / 16 e per count, tie rule), tables from ``bgsub.vendor.binning.response_table`` (vendored podbg.binning) / ``recorded_pmf``.
  Bin 4 is exactly Poisson (m = 1).
- ``CountLaw.poisson(gain)``: a counting / integrating detector with ``gain`` stored units per electron
  (g(b) = gain b + read_var).
- ``CountLaw.from_function(g, epc)``: any variance function (thresholds by a quasi-Poisson tail).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

ALPHA_PX = 1e-6
ALPHA_WIN = 1e-6


@dataclass
class CountLaw:
    kind: str
    epc: float                                  # electrons per stored count
    b_grid: np.ndarray                          # stored-mean grid (increasing)
    g_grid: np.ndarray                          # variance at b_grid
    binning: int | None = None
    tie_up: float | None = None
    gain: float = 1.0
    read_var: float = 0.0
    tau: float = 0.0                            # DECLARED paralyzable dead time per e per output px per stored frame (spot law)
    offset: float = 0.0                         # DECLARED stored value of zero electrons (e.g. 1 for MRC -> TIFF + 1); never inferred
    _kstar: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def apollo(cls, binning: int = 8, tie_up: float | None = 0.35, dead_time_tau: float = 0.0) -> "CountLaw":
        from .vendor import binning as B
        m = B.electrons_per_count(binning)
        if m == 1:
            law = cls.poisson(1.0, _kind="apollo", binning=binning, tie_up=tie_up)
            law.tau = float(dead_time_tau)
            return law
        # g(b) does not depend on dead time (dead time maps mu -> mu e^{-mu tau} before the quantiser); the declared tau
        # is used only for the spot law (integrate X-LC / X-ML)
        mu, e, v = B.response_table(binning, tie_up, 0.0, 0.0, mu_min=1e-5, mu_max=60.0 * m * 64, n=1200)
        ok = np.concatenate([[True], np.diff(e) > 0])
        law = cls("apollo", float(m), e[ok], v[ok], binning=binning, tie_up=tie_up, tau=float(dead_time_tau))
        law._mu = mu[ok]
        return law

    @classmethod
    def poisson(cls, gain: float = 1.0, read_var: float = 0.0, _kind: str = "poisson", binning=None, tie_up=None) -> "CountLaw":
        b = np.geomspace(1e-7, 1e7, 400)
        return cls(_kind, 1.0 / gain, b, gain * b + read_var, binning=binning, tie_up=tie_up, gain=gain, read_var=read_var)

    @classmethod
    def from_function(cls, g, epc: float = 1.0) -> "CountLaw":
        b = np.geomspace(1e-7, 1e7, 400)
        return cls("generic", float(epc), b, np.asarray(g(b), float))

    def with_offset(self, offset: float) -> "CountLaw":
        """The same law with a declared zero level (the estimator works on frame - offset)."""
        import copy
        o = copy.copy(self); o.offset = float(offset); o._kstar = self._kstar
        return o

    # ------------------------------------------------------------------ electrons <-> stored mean (no dead time)
    def _poissonlike(self):
        return self.kind != "apollo" or self.epc == 1.0

    def E(self, mu):
        """Stored mean for electron mean mu (after dead time) per pixel."""
        mu = np.asarray(mu, float)
        if self._poissonlike():
            return mu / self.epc
        out = np.interp(np.log(np.maximum(mu, 1e-300)), np.log(self._mu), self.b_grid)
        return np.where(mu <= self._mu[0], np.maximum(mu, 0) / self._mu[0] * self.b_grid[0], out)

    def E_inv(self, b):
        b = np.asarray(b, float)
        if self._poissonlike():
            return b * self.epc
        out = np.exp(np.interp(b, self.b_grid, np.log(self._mu)))
        return np.where(b <= self.b_grid[0], np.maximum(b, 0) / self.b_grid[0] * self._mu[0], out)

    # ------------------------------------------------------------------ log PMF and Fisher information tables
    def pmf_table(self):
        """(log mu grid, logP[i, k], Fisher information F(mu_i) wrt mu) for the stored value k given electron mean mu."""
        if getattr(self, "_pmf", None) is not None:
            return self._pmf
        mu = np.geomspace(1e-4, 4000.0 * max(self.epc, 1.0) / 4.0, 500)
        if self._poissonlike():
            kmax = int(mu[-1] / self.epc + 12 * np.sqrt(mu[-1] / self.epc) + 30)
            k = np.arange(kmax + 1)
            lp = stats.poisson.logpmf(k[None, :], mu[:, None] / self.epc * self.gain / self.gain)
        else:
            from .vendor import binning as B
            kmax = int(mu[-1] / self.epc + 12 * np.sqrt(mu[-1]) / self.epc + 30)
            P = np.zeros((len(mu), kmax + 1))
            for i, m_ in enumerate(mu):
                p = B.recorded_pmf(float(m_), self.binning, self.tie_up)
                n = min(len(p), kmax + 1); P[i, :n] = p[:n]
            with np.errstate(divide="ignore"):
                lp = np.log(P)
        lp = np.maximum(lp, -700.0)
        P = np.exp(lp)
        dP = np.gradient(P, mu, axis=0)
        F = np.sum(np.where(P > 1e-300, dP ** 2 / np.maximum(P, 1e-300), 0.0), axis=1)
        self._pmf = (np.log(mu), lp, F)
        return self._pmf

    # ------------------------------------------------------------------ variance function
    def g(self, b):
        b = np.asarray(b, float)
        if self.kind in ("poisson",) or (self.kind == "apollo" and self.epc == 1.0):
            return self.gain * np.maximum(b, 0.0) + self.read_var
        return self._g_extrap(b)

    def _g_extrap(self, b):
        bg, gg = self.b_grid, self.g_grid
        out = np.interp(b, bg, gg)
        lo = b < bg[0]
        if np.any(lo):                                          # g ~ b below the table (Bernoulli limit)
            out = np.where(lo, np.maximum(b, 0.0) * gg[0] / bg[0], out)
        hi = b > bg[-1]
        if np.any(hi):                                          # linear extrapolation with the last slope
            s = (gg[-1] - gg[-2]) / (bg[-1] - bg[-2])
            out = np.where(hi, gg[-1] + s * (b - bg[-1]), out)
        return out

    def dg(self, b):
        """dg/db (used by the first-order unbiased box variance)."""
        b = np.asarray(b, float)
        if self.kind == "poisson" or (self.kind == "apollo" and self.epc == 1.0):
            return np.full(b.shape, self.gain)
        lb = np.log(np.maximum(self.b_grid, 1e-300))
        s = np.gradient(self.g_grid, self.b_grid)
        return np.interp(np.log(np.maximum(b, 1e-300)), lb, s, left=s[0], right=s[-1])

    # ------------------------------------------------------------------ thresholds
    def _table(self, alpha: float, window: int):
        key = (alpha, window)
        if key in self._kstar:
            return self._kstar[key]
        bq = np.geomspace(1e-5, 1e5, 701)
        if window == 1 and self.kind == "apollo" and self.epc != 1.0:
            from .vendor import binning as B
            mu = np.interp(bq, self.b_grid, self._mu)
            ks = np.empty(len(bq))
            for i, m_ in enumerate(mu):
                if bq[i] > self.b_grid[-1]:
                    ks[i] = np.inf; continue
                p = B.recorded_pmf(float(m_), self.binning, self.tie_up)
                tail = np.cumsum(p[::-1])[::-1]                 # P(K >= k)
                idx = np.flatnonzero(tail <= alpha)
                ks[i] = idx[0] if len(idx) else len(p)
            # above the table: Normal limit
            big = ~np.isfinite(ks)
            if big.any():
                ks[big] = np.ceil(bq[big] + stats.norm.isf(alpha) * np.sqrt(self.g(bq[big])) + 1)
        else:
            # quasi-Poisson tail of the window sum S (expected window * b, variance window * g):
            # Y = S b / g ~ Poisson(window b^2 / g); threshold on S = (g / b) (isf + 1)
            gq = np.maximum(self.g(bq), 1e-300)
            lam = window * bq ** 2 / gq
            ks = (gq / bq) * (stats.poisson.isf(alpha, lam) + 1.0)
            if window == 1 and (self.kind == "poisson" or self.epc == 1.0) and self.read_var == 0:
                ks = self.gain * (stats.poisson.isf(alpha, bq / self.gain) + 1.0)
        self._kstar[key] = (np.log(bq), ks)
        return self._kstar[key]

    def kstar(self, b, alpha: float = ALPHA_PX, window: int = 1):
        """Smallest stored value (window = 1) or window sum that is significant at one-sided level alpha given the
        expected per-pixel stored mean b.  Evaluated at the next grid point ABOVE b (conservative: fewer flags)."""
        lbq, ks = self._table(alpha, window)
        lb = np.log(np.maximum(np.asarray(b, float), 1e-5))
        j = np.clip(np.searchsorted(lbq, lb, side="left"), 0, len(lbq) - 1)
        return ks[j]

    def tail_mass(self, b: float, kstar: float) -> tuple[float, float]:
        """(P(K >= k*), E[K 1(K >= k*)]) for one pixel at stored mean b: the truncation terms of Sec. 1.5."""
        if self.kind == "apollo" and self.epc != 1.0:
            from .vendor import binning as B
            mu = float(np.interp(b, self.b_grid, self._mu))
            p = B.recorded_pmf(mu, self.binning, self.tie_up)
        else:
            lam = b / self.gain
            k = np.arange(0, int(lam + 40 * np.sqrt(lam + 1) + 40))
            p = stats.poisson.pmf(k, lam)
        k = np.arange(len(p)) * (self.gain if self.kind == "poisson" else 1.0)
        sel = k >= kstar
        return float(p[sel].sum()), float((k * p)[sel].sum())
