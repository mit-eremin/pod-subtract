"""The bgsub v1 synthetic design (used unchanged by v2 and v2.1), on frames rendered from the simulator-v1 bank.

Ported from pod/background scripts/bgsub_v1_synth.py (constants, cells, make_frame) at background commit f1e93f5
(file last changed in 1f1796f; sha256 1c1ab45df4332220a7ce1adb5681ad1020d4de349f349b0a53a97d89270fcec6). The only change:
`podbg.sim.simulate_frames(..., version="v1")` is replaced by a bank entry (bgsub.validation.simbank) holding the same
radial table and geometry, so make_frame returns the SAME arrays as the original (the random stream of make_frame is
its own default_rng(seed); the simulator's internal stream only produced the radial table, which is frozen).
KEEP IN STEP with the source while background still runs this design; afterwards this file is bgsub's.
"""
from __future__ import annotations

import numpy as np

from . import sim_detectors as DT
from . import simbank as SB

TIE = 0.403                                   # the simulator's default film tie p (gain on)
LACEY = {"grid": "lacey_utc", "binnings": [4, 8], "doses": [1.0, 3.0, 10.0]}
ROBUST = {"grids": ["quantifoil_r22", "formvar_carbon_tedpella"], "binnings": [8], "doses": [1.0, 3.0]}
STRUCTS = ["plain", "structured"]
BANDS8 = [(55, 110), (110, 230), (230, 450), (450, 724)]
SEED0 = {"dev": 0, "test": 30000}


def cells(set_name, reps=24, robust_reps=8):
    out = []
    for b in LACEY["binnings"]:
        for d in LACEY["doses"]:
            for st in STRUCTS:
                for rep in range(reps):
                    out.append(("lacey_utc", b, d, st, rep))
    for g in ROBUST["grids"]:
        for b in ROBUST["binnings"]:
            for d in ROBUST["doses"]:
                for st in STRUCTS:
                    for rep in range(robust_reps):
                        out.append((g, b, d, st, rep))
    return [(i + SEED0[set_name],) + c for i, c in enumerate(out)]


def cell_key(grid, b, f, struct, rep) -> str:
    return f"{grid}_b{b}_d{f}_{struct}_r{rep}"


def make_frame(seed, grid, b, f, struct, set_name="dev", entry: SB.BankEntry | None = None, check=True):
    """As bgsub_v1_synth.make_frame: (rng, K, N, mu_c, mask, centre, tau_b, k, info). `entry` defaults to the bank file."""
    from scipy import ndimage as ndi
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    sim_seed = int(rng.integers(2 ** 31))     # consumed exactly as the original (it seeded podbg.sim)
    e = entry if entry is not None else SB.load(SB.entry_path(f"v1synth_{set_name}", cell_key(grid, b, f, struct, _rep_of(seed, set_name))))
    if int(e.meta.get("sim_seed", sim_seed)) != sim_seed:
        raise ValueError(f"bank entry {e.key} was made with sim seed {e.meta.get('sim_seed')}, this cell needs {sim_seed}")
    want_ell = 1.008 if struct == "structured" else 1.0
    if abs(e.ellipticity - want_ell) > 1e-12:
        raise ValueError(f"bank entry {e.key} has ellipticity {e.ellipticity}, the cell needs {want_ell}")
    mu = SB.expected(e).astype(np.float64)
    if check and not SB.verify(e):
        raise ValueError(f"bank entry {e.key} does not reproduce podbg's expected frame")
    c = e.centre_yx
    stop = SB.stop_mask(e)
    n = mu.shape[0]; sc = n / 1024.0
    mask = ndi.binary_dilation(stop, iterations=1)
    mask[:2, :] = mask[-2:, :] = True; mask[:, :2] = mask[:, -2:] = True
    info = {"phib": np.nan}
    if struct == "structured":
        yy, xx = np.indices(mu.shape)
        r = np.hypot(yy - c[0], xx - c[1]); phi = np.arctan2(yy - c[0], xx - c[1])
        k = e.k_A_per_px_centre
        r_tab, mu_tab = e.r_tab, e.mu_tab
        sm = gaussian_filter1d(mu_tab, 60.0 * 8 / b / (r_tab[1] - r_tab[0]))
        phb = rng.uniform(-np.pi, np.pi); info["phib"] = phb
        ramp = np.clip((k * r - 0.3) / 0.3, 0, 1)
        mu = mu + 0.15 * np.interp(r, r_tab, sm) * ramp * np.maximum(0, np.cos(phi - phb))
        disc = np.hypot(yy - e.centre_nominal_yx[0], xx - e.centre_nominal_yx[1]) <= 52.0 * 8 / b
        arm = stop & ~disc
        d, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
        on_arm = arm[iy, ix] | ndi.binary_dilation(arm, iterations=2)[iy, ix]
        T = np.where(on_arm & ~mask, 1 - 0.5 * np.exp(-(d - 1) / sc), 1.0)
        mu = mu * T
    mu[stop] = 0.0
    spec = DT.apollo(b, True, TIE)
    tb = spec.tau_binned
    mu_c = mu * np.exp(-mu * tb)
    N = rng.poisson(mu_c)
    K = DT.quantise_fast(N, spec.m, True, TIE, rng).astype(np.int64)
    return rng, K, N, mu_c, mask, c, tb, e.k_A_per_px_centre, info


def _rep_of(seed, set_name):
    for s, g, b, f, st, rep in cells(set_name):
        if s == seed:
            return rep
    raise KeyError(seed)
