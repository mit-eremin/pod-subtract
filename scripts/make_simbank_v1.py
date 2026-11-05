"""BRIDGE SCRIPT (the only code in this repo that imports podbg): freeze podbg's simulator-v1 radial tables into the
bgsub sim bank, so that bgsub renders the SAME expected frames without the background physics model.

Why: podbg.sim pulls ~9,000 lines of the background's physics model (podbg.model) and five theory tables from
~/pod_data/background/theory/ at run time. For bgsub validation only the per-frame radial table mu(r) (expected
electrons per pixel, r on a 0.25-px grid) and the frame geometry matter; the rendering (radius map, beam stop, count
law) is vendored in bgsub.validation (sim_geometry, sim_detectors) and reproduces podbg's expected_e BIT FOR BIT
(the sha256 of podbg's float32 expected frame is stored with each entry and checked by bgsub.validation.simbank).

Run with the BACKGROUND venv, read-only use of podbg (needs ../background with its .venv):

    ../background/.venv/bin/python scripts/make_simbank_v1.py --design v1synth --set dev     # 352 entries
    ../background/.venv/bin/python scripts/make_simbank_v1.py --design v1synth --set test    # 352 entries
    ../background/.venv/bin/python scripts/make_simbank_v1.py --design speed                 # lacey seed 11, 4 frames x bins 16/8/4
    ../background/.venv/bin/python scripts/make_simbank_v1.py --design sample --out tests/data/simbank_v1_sample.npz

Output: ~/pod_data/bgsub/simbank_v1/<design>[_<set>]/<key>.npz + index.csv (+ provenance sidecar written as JSON).
Entry keys follow the bgsub v1/v2/v2.1 synthetic checkpoints: <grid>_b<b>_d<f>_<struct>_r<rep>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from podbg.sim import simulate_frames, grids_v1 as GR1, detectors as DT

ROOT = Path(__file__).resolve().parents[1]
BANK = Path(os.environ.get("BGSUB_DATA", Path.home() / "pod_data" / "bgsub")) / "simbank_v1"
BG_REPO = ROOT.parent / "background"

# the bgsub v1 synthetic design (pod/background scripts/bgsub_v1_synth.py at f1e93f5; reused unchanged by v2 and v2.1)
LACEY = {"grid": "lacey_utc", "binnings": [4, 8], "doses": [1.0, 3.0, 10.0]}
ROBUST = {"grids": ["quantifoil_r22", "formvar_carbon_tedpella"], "binnings": [8], "doses": [1.0, 3.0]}
STRUCTS = ["plain", "structured"]
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


def entry(grid, b, dose, sim_seed, ellipticity, n_frames=1):
    sim = simulate_frames(grid, dose=dose, n_frames=n_frames, binning=b, seed=sim_seed, beam_stop=True, return_expected=True,
                          return_profiles=True, version="v1", ellipticity=ellipticity)
    spec = DT.apollo(b)
    out = []
    for i in range(n_frames):
        r_tab, mu_tab = sim.profiles[i]
        assert r_tab[0] == 0.0 and np.allclose(np.diff(r_tab), 0.25)
        t = sim.truth[i]
        out.append({
            "mu_tab": np.asarray(mu_tab, np.float64), "r_step": np.array(0.25), "n_r": np.array(len(r_tab)),
            "r_tab_sha256": np.array(hashlib.sha256(np.ascontiguousarray(r_tab).tobytes()).hexdigest()),
            "shape": np.array(sim.mask.shape), "centre_yx": np.array(t["centre_yx"]), "centre_nominal_yx": np.array(sim.meta["centre_nominal_yx"]),
            "ellipticity": np.array(ellipticity), "binning": np.array(b), "stop_r_px": np.array(spec.stop_r_px),
            "stop_arm_half_px": np.array(spec.stop_arm_half_px), "k_A_per_px_centre": np.array(sim.meta["k_A_per_px_centre"]),
            "dose_e": np.array(t["dose_e"]), "grid": np.array(grid), "sim_seed": np.array(sim_seed), "frame_index": np.array(i),
            "expected_sha256": np.array(hashlib.sha256(np.ascontiguousarray(sim.expected_e[i]).tobytes()).hexdigest()),
            "stop_sha256": np.array(hashlib.sha256(np.ascontiguousarray(sim.mask).tobytes()).hexdigest()),
            "truth_json": np.array(json.dumps(t, default=float)),
        })
    return out


def stamp():
    rev = subprocess.run(["git", "-C", str(BG_REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(BG_REPO), "status", "--porcelain", "src/podbg/sim", "src/podbg/model"],
                           capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["v1synth", "speed", "sample"], required=True)
    ap.add_argument("--set", default="dev"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rev = stamp()
    t0 = time.time()
    if a.design == "sample":           # two small entries for the unit test (bin 16 plain, bin 8 structured)
        ents = {}
        for key, (grid, b, f, ell, s) in {"lacey_b16": ("lacey_utc", 16, 1.0, 1.0, 7), "lacey_b8_struct": ("lacey_utc", 8, 3.0, 1.008, 8)}.items():
            e = entry(grid, b, f * GR1.DOSE0_V1, s, ell)[0]
            ents.update({f"{key}__{k}": v for k, v in e.items()})
        ents["background_commit"] = np.array(rev)
        np.savez_compressed(a.out, **ents)
        print("wrote", a.out, f"{time.time() - t0:.1f} s"); return
    od = BANK / (f"v1synth_{a.set}" if a.design == "v1synth" else "speed")
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.design == "v1synth":
        for seed, grid, b, f, struct, rep in cells(a.set):
            rng = np.random.default_rng(seed)
            sim_seed = int(rng.integers(2 ** 31))          # exactly as bgsub_v1_synth.make_frame
            key = f"{grid}_b{b}_d{f}_{struct}_r{rep}"
            e = entry(grid, b, f * GR1.DOSE0_V1, sim_seed, 1.008 if struct == "structured" else 1.0)[0]
            np.savez_compressed(od / f"{key}.npz", **e)
            rows.append({"key": key, "seed": seed, "sim_seed": sim_seed, "grid": grid, "binning": b, "dose_factor": f, "struct": struct,
                         "rep": rep, "expected_sha256": str(e["expected_sha256"])})
    else:
        for b in (16, 8, 4):
            for i, e in enumerate(entry("lacey_utc", b, None, 11, 1.0, n_frames=4)):
                key = f"lacey_utc_b{b}_seed11_f{i}"
                np.savez_compressed(od / f"{key}.npz", **e)
                rows.append({"key": key, "seed": 11, "sim_seed": 11, "grid": "lacey_utc", "binning": b, "dose_factor": 1.0, "struct": "plain",
                             "rep": i, "expected_sha256": str(e["expected_sha256"])})
    with open(od / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    side = {"target": str(od), "produced": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "background_commit": rev, "sim_version": "v1",
            "design": a.design, "set": a.set, "n_entries": len(rows), "argv": sys.argv, "python": sys.version.split()[0],
            "note": "frozen podbg.sim v1 radial tables + geometry; bgsub.validation.simbank renders them (sha256-checked)"}
    (od / "index.csv.provenance.json").write_text(json.dumps(side, indent=2))
    print(f"wrote {len(rows)} entries to {od} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
