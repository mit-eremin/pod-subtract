"""bgsub v2.1 synthetic validation (design note 2026-11-05_bgsub_algorithm_v2_1.md Sec. 3): the v1 design EXACTLY
(same cells, seeds, frames: simulator pinned to version="v1", frames made by bgsub_v1_synth.make_frame), with the v2 and
v2.1 algorithms on the same frames (the v2 rows are checked against the published v2 rows).

Usage: .venv/bin/python scripts/bgsub_v2_1_synth.py --set dev|test [--reps N] [--robust-reps N] [--workers W]
Outputs: ~/pod_data/background/bgsub_v2_1/synth_<set>_{peaks.parquet,bands.csv,frames.csv} (+ sidecars); checkpoints per
frame under checkpoints/synth_<set>/.
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
import bgsub_v1_synth as S1  # noqa: E402

from podbg import paths  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
OUT = paths.SUB / "bgsub_v2_1"; CK = OUT / "checkpoints"
BANDS8 = S1.BANDS8
V2_METHODS = ("G", "GL", "AUTO", "FLAT")
V21_METHODS = ("G", "GL", "AUTO")
PREREG = {"prereg_commit": "see report Sec. 2-3", "sim_version": "v1", "design": "bgsub v1 synthetic design (S1.cells, S1.make_frame)",
          "tie": S1.TIE}


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "logs" / "synth_v2_1.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def evaluate(Kp, K0, mask, ce, law, peaks, box, ver, methods):
    spots = np.array([[q["cy"], q["cx"]] for q in peaks]).reshape(-1, 2)
    ex = BG.disc_mask(Kp.shape, spots, box["r_excl"])
    t0 = time.perf_counter()
    plan = BG.make_plan(Kp.shape, mask, BG.Geometry(tuple(ce)), version=ver)
    tp = time.perf_counter() - t0
    res = BG.estimate_background(Kp, mask, BG.Geometry(tuple(ce)), law, exclude=ex, plan=plan)
    out = BG.integrate_spots(Kp, res, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"],
                             sigmas=np.array([q["sigma"] for q in peaks]), methods=methods, oracle=K0 if ver == "v2" else None,
                             lam_eval=np.array([q["lam_box"] for q in peaks]))
    info = {f"{ver}_t_fit_s": res.timings["total_s"], f"{ver}_t_plan_s": tp, f"{ver}_disp": res.dispersion}
    if "AUTO" in out:
        info[f"{ver}_auto_frac"] = float(np.mean(out["AUTO"]["auto_use_GL"] > 0))
        if "auto_w" in out["AUTO"]:
            info[f"{ver}_auto_w_mean"] = float(np.mean(out["AUTO"]["auto_w"]))
    if res.centre_info is not None:
        info[f"{ver}_recentred"] = bool(res.centre_info["recentred"])
    if res.ring is not None:
        info[f"{ver}_ring_wmax"] = float(res.ring.w.max()); info[f"{ver}_ring_n"] = int((res.ring.w > 0.5).sum())
        if hasattr(res.ring, "w2"):
            info[f"{ver}_ring2_n"] = int((res.ring.w2 > 0.5).sum())
    return out, info, res


def _job(args):
    try:
        return run_one(args)
    except Exception as e:  # noqa: BLE001
        return f"{args} FAILED: {e!r}"


def run_one(args):
    seed, grid, b, f, struct, rep, set_name = args
    ck = CK / f"synth_{set_name}" / f"{grid}_b{b}_d{f}_{struct}_r{rep}.pkl"
    if ck.exists():
        return str(ck)
    t0 = time.time()
    sig8 = np.clip(pd.read_csv(paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv").sigma_px.to_numpy(float), 0.5, 2.0)
    rng, K0, N, mu_c, mask, c, tb, k, finfo = S1.make_frame(seed, grid, b, f, struct)
    law = BG.CountLaw.apollo(b, S1.TIE, dead_time_tau=tb)
    b_true = law.E(mu_c)
    box = CM.geometry_for(b)
    valid = ~mask
    peaks = CM.place_peaks(rng, valid, c, k, box, sig8, b)
    Kp = CM.inject_all(rng, K0, peaks, mu_c, b, S1.TIE, tb, box, law.epc, law.E, N_known=N)
    ce = (c[0] + rng.uniform(-0.5, 0.5), c[1] + rng.uniform(-0.5, 0.5))
    o1, i1, r1 = evaluate(Kp, K0, mask, ce, law, peaks, box, "v2", V2_METHODS)
    o2, i2, r2 = evaluate(Kp, K0, mask, ce, law, peaks, box, "v2.1", V21_METHODS)
    outs = {**{"v2:" + m: d for m, d in o1.items()}, **{"v2.1:" + m: d for m, d in o2.items()}}
    rows = []
    for kk, q in enumerate(peaks):
        base = {key: q[key] for key in ("dbin", "pb", "s", "sigma", "lam_box", "lam_tot", "box_frac", "B_box_e", "lam_eff")}
        base["b_true_box"] = float(b_true[q["sl"]][q["bx"]].sum()); base["K0_box"] = float(o1["O"]["B_box"][kk])
        base["g_true_box"] = float(law.g(b_true[q["sl"]][q["bx"]]).sum())
        for m, d in outs.items():
            r_ = {**base, "method": m, "I": d["I"][kk], "sigma_I": d["sigma"][kk], "B_box": d["B_box"][kk], "var_B": d["var_B"][kk],
                  "n_box": d["n_box"][kk], "n_ann": d["n_ann"][kk]}
            for extra in ("lam_min5", "auto_use_GL", "auto_w", "auto_u", "auto_D", "sigma_v0form", "sigma_at", "ci_lo", "ci_hi",
                          "ci_lam_lo", "ci_lam_hi", "ci_open"):
                if extra in d:
                    r_[extra] = d[extra][kk]
            rows.append(r_)
    # spot-free twins (R2): v2 and v2.1 G with masking on / off; paired estimator-minus-frame statistic
    rr8 = r1.plan.rmap * (b / 8.0)
    bands = []
    for ver, rk in (("v2", r1), ("v2.1", r2)):
        pl = rk.plan
        t_on = BG.estimate_background(K0, mask, BG.Geometry(ce), law, plan=pl)
        t_off = BG.estimate_background(K0, mask, BG.Geometry(ce), law, plan=pl, flagging=False)
        for lo, hi in BANDS8:
            s = valid & (rr8 >= lo) & (rr8 < hi) & np.isfinite(t_on.bg)
            bands.append({"ver": ver, "band": f"{lo}-{hi}", "b_mean": float(b_true[s].mean()), "n_px": int(s.sum()),
                          "d_on": float((t_on.bg[s] - b_true[s]).mean()), "d_off": float((t_off.bg[s] - b_true[s]).mean()),
                          "d_raw": float((K0[s] - b_true[s]).mean()), "d_pair": float((t_on.bg[s] - K0[s]).mean())})
    meta = {"grid": grid, "binning": b, "dose": f, "struct": struct, "rep": rep, "seed": seed, **finfo, **i1, **i2, "n_peaks": len(peaks),
            "frame_mean": float(K0[valid].mean())}
    for r_ in rows:
        r_.update(meta)
    for bd in bands:
        bd.update(meta)
    pickle.dump({"rows": rows, "bands": bands, "meta": meta}, open(ck, "wb"))
    return (f"{grid} b{b} d{f} {struct} r{rep}: {len(peaks)} peaks, v2.1 fit {i2['v2.1_t_fit_s'] * 1e3:.0f} ms, w {i2.get('v2.1_auto_w_mean', np.nan):.2f}, "
            f"ring {i2.get('v2.1_ring_n', 0)}/{i2.get('v2.1_ring2_n', 0)}, {time.time() - t0:.1f} s")


def gather(set_name, n_planned):
    ck = CK / f"synth_{set_name}"
    R, Bd, Fm = [], [], []
    for f in sorted(ck.glob("*.pkl")):
        d = pickle.load(open(f, "rb")); R += d["rows"]; Bd += d["bands"]; Fm.append(d["meta"])
    R = pd.DataFrame(R); Bd = pd.DataFrame(Bd); Fm = pd.DataFrame(Fm)
    log(f"completeness: {len(Fm)} / {n_planned} planned frames have checkpoints")
    for df, name in ((R, f"synth_{set_name}_peaks.parquet"), (Bd, f"synth_{set_name}_bands.csv"), (Fm, f"synth_{set_name}_frames.csv")):
        p = OUT / name
        (df.to_parquet(p) if name.endswith("parquet") else df.to_csv(p, index=False))
        write_sidecar(p, inputs=[paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv"], params={**PREREG, "set": set_name, "n_planned": n_planned},
                      note="bgsub v2.1 synthetic validation (design note 2026-11-05_bgsub_algorithm_v2_1.md)")
    log(f"gathered {len(R)} rows, {len(Fm)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev"); ap.add_argument("--reps", type=int, default=24); ap.add_argument("--robust-reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True); (CK / f"synth_{a.set}").mkdir(parents=True, exist_ok=True)
    jobs = [c + (a.set,) for c in S1.cells(a.set, a.reps, a.robust_reps)]
    log(f"synth {a.set}: {len(jobs)} frames, {a.workers} workers")
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_job, jobs):
                log(msg)
    else:
        for j in jobs:
            log(_job(j))
    gather(a.set, len(jobs))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
