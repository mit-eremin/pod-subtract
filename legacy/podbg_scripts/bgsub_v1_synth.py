"""bgsub v1 synthetic validation (design note 2026-11-04_bgsub_algorithm_v1.md Sec. 2.5): simulator v1 frames
(read-only, version="v1"), plain / structured, peaks injected at the electron level with known host electrons,
v0 and v1 algorithms on the same frames.

Usage: .venv/bin/python scripts/bgsub_v1_synth.py --set dev|test [--reps N] [--robust-reps N] [--workers W]
Outputs: ~/pod_data/background/bgsub_v1/synth_<set>_{peaks.parquet,bands.csv,frames.csv} (+ sidecars); checkpoints
per frame under checkpoints/synth_<set>/.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bgsub_v0_common as CM  # noqa: E402

from podbg import paths  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
OUT = paths.SUB / "bgsub_v1"; CK = OUT / "checkpoints"
TIE = 0.403                                   # the simulator's default film tie p (gain on)
LACEY = {"grid": "lacey_utc", "binnings": [4, 8], "doses": [1.0, 3.0, 10.0]}
ROBUST = {"grids": ["quantifoil_r22", "formvar_carbon_tedpella"], "binnings": [8], "doses": [1.0, 3.0]}
STRUCTS = ["plain", "structured"]
BANDS8 = [(55, 110), (110, 230), (230, 450), (450, 724)]
V0_METHODS = ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST")
V1_METHODS = ("G", "GL", "AUTO")
SEED0 = {"dev": 0, "test": 30000}
PREREG = {"prereg_commit": "bc37be9", "sim_version": "v1", "tie": TIE, "lacey": LACEY, "robust": ROBUST,
          "structured": "ellipticity 1.008 + v0 m=1 pedestal 0.15 + arm shadow 1 - 0.5 exp(-(d-1)/(N/1024))",
          "mask": "sim stop dilated 1 px + 2-px border", "centre_error_px": 0.5}


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "logs" / "synth_v1.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def cells(set_name, reps, robust_reps):
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


def make_frame(seed, grid, b, f, struct):
    from scipy import ndimage as ndi
    from scipy.ndimage import gaussian_filter1d
    from podbg.sim import simulate_frames, grids_v1 as GR1, detectors as DT
    rng = np.random.default_rng(seed)
    sim = simulate_frames(grid, dose=f * GR1.DOSE0_V1, n_frames=1, binning=b, seed=int(rng.integers(2 ** 31)), beam_stop=True,
                          return_expected=True, return_profiles=True, version="v1", ellipticity=1.008 if struct == "structured" else 1.0)
    mu = sim.expected_e[0].astype(np.float64)
    c = sim.truth[0]["centre_yx"]
    stop = sim.mask
    n = mu.shape[0]; sc = n / 1024.0
    mask = ndi.binary_dilation(stop, iterations=1)
    mask[:2, :] = mask[-2:, :] = True; mask[:, :2] = mask[:, -2:] = True
    info = {"phib": np.nan}
    if struct == "structured":
        yy, xx = np.indices(mu.shape)
        r = np.hypot(yy - c[0], xx - c[1]); phi = np.arctan2(yy - c[0], xx - c[1])
        k = sim.meta["k_A_per_px_centre"]
        # v0 bar pedestal: 0.15 x ring-free profile x ramp in s (0.3 -> 0.6) x max(0, cos(phi - phi_b))
        r_tab, mu_tab = sim.profiles[0]
        sm = gaussian_filter1d(mu_tab, 60.0 * 8 / b / (r_tab[1] - r_tab[0]))
        phb = rng.uniform(-np.pi, np.pi); info["phib"] = phb
        ramp = np.clip((k * r - 0.3) / 0.3, 0, 1)
        mu = mu + 0.15 * np.interp(r, r_tab, sm) * ramp * np.maximum(0, np.cos(phi - phb))
        # arm shadow (DEV measurement, design note v1 Sec. 1.1 b): pixels whose nearest stop pixel lies on the arm
        disc = np.hypot(yy - sim.meta["centre_nominal_yx"][0], xx - sim.meta["centre_nominal_yx"][1]) <= 52.0 * 8 / b
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
    return rng, K, N, mu_c, mask, c, tb, sim.meta["k_A_per_px_centre"], info


class _TruthRes:
    def __init__(self, res, b):
        self.plan = res.plan; self.law = res.law; self.bg = b.astype(np.float32); self.flags = np.zeros(b.shape, bool)
        self.dispersion = 1.0; self._res = res

    def box_sums(self, *a, **k):
        return self._res.box_sums(*a, **k)


def evaluate(Kp, K0, mask, ce, law, peaks, box, b_true, ver, methods):
    spots = np.array([[q["cy"], q["cx"]] for q in peaks]).reshape(-1, 2)
    ex = BG.disc_mask(Kp.shape, spots, box["r_excl"])
    t0 = time.perf_counter()
    plan = BG.make_plan(Kp.shape, mask, BG.Geometry(tuple(ce)), version=ver)
    tp = time.perf_counter() - t0
    res = BG.estimate_background(Kp, mask, BG.Geometry(tuple(ce)), law, exclude=ex, plan=plan)
    out = BG.integrate_spots(Kp, res, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"],
                             sigmas=np.array([q["sigma"] for q in peaks]), methods=methods, oracle=K0 if ver == "v0" else None)
    info = {f"{ver}_t_fit_s": res.timings["total_s"], f"{ver}_t_plan_s": tp, f"{ver}_disp": res.dispersion}
    if "AUTO" in out:
        info[f"{ver}_auto_frac"] = float(np.mean(out["AUTO"]["auto_use_GL"]))
    return out, info, res


def run_one(args):
    seed, grid, b, f, struct, rep, set_name = args
    ck = CK / f"synth_{set_name}" / f"{grid}_b{b}_d{f}_{struct}_r{rep}.pkl"
    if ck.exists():
        return str(ck)
    t0 = time.time()
    sig8 = np.clip(pd.read_csv(paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv").sigma_px.to_numpy(float), 0.5, 2.0)
    rng, K0, N, mu_c, mask, c, tb, k, finfo = make_frame(seed, grid, b, f, struct)
    law = BG.CountLaw.apollo(b, TIE, dead_time_tau=tb)
    b_true = law.E(mu_c)
    box = CM.geometry_for(b)
    valid = ~mask
    peaks = CM.place_peaks(rng, valid, c, k, box, sig8, b)
    Kp = CM.inject_all(rng, K0, peaks, mu_c, b, TIE, tb, box, law.epc, law.E, N_known=N)
    ce = (c[0] + rng.uniform(-0.5, 0.5), c[1] + rng.uniform(-0.5, 0.5))
    o0, i0, r0 = evaluate(Kp, K0, mask, ce, law, peaks, box, b_true, "v0", V0_METHODS)
    o1, i1, r1 = evaluate(Kp, K0, mask, ce, law, peaks, box, b_true, "v1", V1_METHODS)
    outs = {**{m: d for m, d in o0.items()}, **{"v1:" + m: d for m, d in o1.items()}}
    # derivation check as v0: FLAT on the noise-free truth
    rows = []
    for kk, q in enumerate(peaks):
        btb = float(b_true[q["sl"]][q["bx"]].sum())
        base = {key: q[key] for key in ("dbin", "pb", "s", "sigma", "lam_box", "lam_tot", "box_frac", "B_box_e", "lam_eff")}
        base["b_true_box"] = btb; base["K0_box"] = float(o0["O"]["B_box"][kk])
        base["g_true_box"] = float(law.g(b_true[q["sl"]][q["bx"]]).sum())
        for m, d in outs.items():
            r_ = {**base, "method": m, "I": d["I"][kk], "sigma_I": d["sigma"][kk], "B_box": d["B_box"][kk], "var_B": d["var_B"][kk],
                  "n_box": d["n_box"][kk], "n_ann": d["n_ann"][kk]}
            for extra in ("lam_min5", "auto_use_GL", "auto_D", "sigma_v0form"):
                if extra in d:
                    r_[extra] = d[extra][kk]
            rows.append(r_)
    # spot-free twins: v1 G with masking on / off (R2)
    plan1 = r1.plan
    t_on = BG.estimate_background(K0, mask, BG.Geometry(ce), law, plan=plan1)
    t_off = BG.estimate_background(K0, mask, BG.Geometry(ce), law, plan=plan1, flagging=False)
    rr8 = plan1.rmap * (b / 8.0)
    bands = []
    for lo, hi in BANDS8:
        s = valid & (rr8 >= lo) & (rr8 < hi) & np.isfinite(t_on.bg)
        bands.append({"band": f"{lo}-{hi}", "b_mean": float(b_true[s].mean()), "n_px": int(s.sum()),
                      "d_on": float((t_on.bg[s] - b_true[s]).mean()), "d_off": float((t_off.bg[s] - b_true[s]).mean()),
                      "d_raw": float((K0[s] - b_true[s]).mean())})
    meta = {"grid": grid, "binning": b, "dose": f, "struct": struct, "rep": rep, "seed": seed, **finfo, **i0, **i1, "n_peaks": len(peaks),
            "frame_mean": float(K0[valid].mean())}
    for r_ in rows:
        r_.update(meta)
    for bd in bands:
        bd.update(meta)
    pickle.dump({"rows": rows, "bands": bands, "meta": meta}, open(ck, "wb"))
    return f"{grid} b{b} d{f} {struct} r{rep}: {len(peaks)} peaks, v1 fit {i1['v1_t_fit_s'] * 1e3:.0f} ms, auto {i1.get('v1_auto_frac', np.nan):.2f}/{i0.get('v0_auto_frac', np.nan):.2f}, {time.time() - t0:.1f} s"


def gather(set_name):
    ck = CK / f"synth_{set_name}"
    R, Bd, Fm = [], [], []
    for f in sorted(ck.glob("*.pkl")):
        d = pickle.load(open(f, "rb")); R += d["rows"]; Bd += d["bands"]; Fm.append(d["meta"])
    R = pd.DataFrame(R); Bd = pd.DataFrame(Bd); Fm = pd.DataFrame(Fm)
    for df, name in ((R, f"synth_{set_name}_peaks.parquet"), (Bd, f"synth_{set_name}_bands.csv"), (Fm, f"synth_{set_name}_frames.csv")):
        p = OUT / name
        (df.to_parquet(p) if name.endswith("parquet") else df.to_csv(p, index=False))
        write_sidecar(p, inputs=[paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv"], params={**PREREG, "set": set_name},
                      note="bgsub v1 synthetic validation (design note 2026-11-04_bgsub_algorithm_v1.md)")
    log(f"gathered {len(R)} rows, {len(Fm)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev"); ap.add_argument("--reps", type=int, default=24); ap.add_argument("--robust-reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True); (CK / f"synth_{a.set}").mkdir(parents=True, exist_ok=True)
    jobs = [c + (a.set,) for c in cells(a.set, a.reps, a.robust_reps)]
    log(f"synth {a.set}: {len(jobs)} frames, {a.workers} workers")
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(run_one, jobs):
                log(msg)
    else:
        for j in jobs:
            log(run_one(j))
    gather(a.set)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
