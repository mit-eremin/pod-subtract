"""Render frozen simulator-v1 frames (podbg.sim, version="v1") without the background physics model.

A bank entry (written once by scripts/make_simbank_v1.py from pod/background) holds the per-frame radial table mu(r)
of expected electrons per pixel on r = 0, 0.25, 0.5, ... px, the frame geometry (shape, centre, ellipticity, beam-stop
disc + arm) and the sha256 of podbg's float32 expected frame. `expected(entry)` repeats podbg.sim.core.simulate_frames'
rendering (radius map -> np.interp -> float32 -> stop pixels 0) with the vendored sim_geometry / sim_detectors and so
reproduces podbg's expected_e bit for bit; `verify(entry)` checks it. Noisy stored frames come from the vendored count
law (`stored`), equal in law to podbg's, not in random stream.

Validity: everything simulator v1 is (podbg report 2026-11-04_simulator_v1.md): Apollo 200 kV, lacey preset calibrated,
formvar / Quantifoil presets partly assumed. It is a test bench for the background estimate, not a physics claim.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sim_detectors as DT
from . import sim_geometry as G
from .. import config


@dataclass
class BankEntry:
    key: str
    mu_tab: np.ndarray
    r_tab: np.ndarray
    shape: tuple
    centre_yx: tuple
    centre_nominal_yx: tuple
    ellipticity: float
    binning: int
    stop_r_px: float
    stop_arm_half_px: float
    k_A_per_px_centre: float
    expected_sha256: str
    stop_sha256: str
    meta: dict


def _from_arrays(key: str, z: dict) -> BankEntry:
    n_r = int(z["n_r"]); step = float(z["r_step"])
    r_tab = np.arange(0.0, n_r * step, step)[:n_r]
    if hashlib.sha256(np.ascontiguousarray(r_tab).tobytes()).hexdigest() != str(z["r_tab_sha256"]):
        raise ValueError(f"{key}: radial grid does not reproduce the simulator's r_tab")
    return BankEntry(key, np.asarray(z["mu_tab"], np.float64), r_tab, tuple(int(v) for v in z["shape"]),
                     tuple(float(v) for v in z["centre_yx"]), tuple(float(v) for v in z["centre_nominal_yx"]),
                     float(z["ellipticity"]), int(z["binning"]), float(z["stop_r_px"]), float(z["stop_arm_half_px"]),
                     float(z["k_A_per_px_centre"]), str(z["expected_sha256"]), str(z["stop_sha256"]),
                     {k: (v.item() if np.ndim(v) == 0 else v) for k, v in z.items()
                      if k in ("grid", "sim_seed", "dose_e", "frame_index", "truth_json")})


def load(path) -> BankEntry:
    """One entry file (<bank>/<design>/<key>.npz)."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        return _from_arrays(p.stem, {k: z[k] for k in z.files})


def load_bundle(path) -> dict[str, BankEntry]:
    """A bundle file with several entries stored as '<key>__<field>' (e.g. tests/data/simbank_v1_sample.npz)."""
    with np.load(path, allow_pickle=False) as z:
        keys = sorted({k.split("__")[0] for k in z.files if "__" in k})
        return {key: _from_arrays(key, {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(key + "__")}) for key in keys}


def entry_path(design: str, key: str) -> Path:
    """design: 'v1synth_dev', 'v1synth_test' or 'speed'."""
    return config.SIMBANK_V1 / design / f"{key}.npz"


def stop_mask(e: BankEntry) -> np.ndarray:
    return DT.beam_stop_mask(e.shape, e.centre_nominal_yx, e.stop_r_px, e.stop_arm_half_px)


def expected(e: BankEntry, ellipticity: float | None = None) -> np.ndarray:
    """Expected electrons per pixel (float32), stop pixels 0: podbg's sim.expected_e[i] for this entry."""
    r = G.radius_map(e.shape, e.centre_yx, e.ellipticity if ellipticity is None else ellipticity)
    mu = np.interp(r, e.r_tab, e.mu_tab).astype(np.float32)
    stop = stop_mask(e)
    if stop.any():
        mu[stop] = 0.0
    return mu


def verify(e: BankEntry) -> bool:
    """True when the rendered expected frame and the stop equal podbg's bit for bit (sha256)."""
    ok_mu = hashlib.sha256(np.ascontiguousarray(expected(e)).tobytes()).hexdigest() == e.expected_sha256
    ok_stop = hashlib.sha256(np.ascontiguousarray(stop_mask(e)).tobytes()).hexdigest() == e.stop_sha256
    return ok_mu and ok_stop


def stored(mu: np.ndarray, binning: int, rng, tie=DT.FILM_P, gain_flag: bool = True) -> np.ndarray:
    """A noisy Apollo stored frame for expected electrons mu (dead time, Poisson, rounding with the tie rule)."""
    return DT.record(mu, DT.apollo(binning, gain_flag, tie, shape=mu.shape), rng, noise=True)
