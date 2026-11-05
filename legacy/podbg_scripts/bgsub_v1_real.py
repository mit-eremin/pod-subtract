"""bgsub v1 real-frame validation (design note v1 Sec. 2.5): the v0 harness with the SAME sessions, hosts, replicates and
seeds as bgsub v0 (so the injected frames are identical), evaluated with the v0 and the v1 algorithms.

Usage: .venv/bin/python scripts/bgsub_v1_real.py --set dev|test [--workers W]
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bgsub_v0_common as CM  # noqa: E402
import bgsub_v0_real as RV  # noqa: E402

from podbg import paths, detector as D  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = paths.SUB / "bgsub_v1"; CK = OUT / "checkpoints"
V0_METHODS = ("G", "GL", "AUTO", "FLAT")
V1_METHODS = ("G", "GL", "AUTO")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "logs" / "real_v1.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def run_host(rng, K0, bad, centre, law, binning, tie, tau_b, sig8, meta):
    """Exactly v0's run_host up to the injection (same rng sequence), then v0 and v1 evaluation."""
    box = CM.geometry_for(binning)
    valid = ~bad
    mu_map, m_map = RV.host_mu_map(K0, valid, centre, law)
    peaks = CM.place_peaks(rng, valid, centre, CM.K8 * binning / 8.0, box, sig8, binning)
    Kp = CM.inject_all(rng, K0, peaks, mu_map, binning, tie, tau_b, box, law.epc, law.E)
    spots = np.array([[q["cy"], q["cx"]] for q in peaks]).reshape(-1, 2)
    ex = BG.disc_mask(Kp.shape, spots, box["r_excl"])
    outs = {}; info = {}
    for ver, methods in (("v0", V0_METHODS), ("v1", V1_METHODS)):
        t = time.perf_counter()
        plan = BG.make_plan(Kp.shape, bad, BG.Geometry(centre), version=ver)
        info[f"{ver}_t_plan_s"] = time.perf_counter() - t
        res = BG.estimate_background(Kp, bad, BG.Geometry(centre), law, exclude=ex, plan=plan)
        o = BG.integrate_spots(Kp, res, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"],
                               sigmas=np.array([q["sigma"] for q in peaks]), methods=methods, oracle=K0 if ver == "v0" else None)
        info[f"{ver}_t_fit_s"] = res.timings["total_s"]; info["dispersion" if ver == "v0" else "v1_dispersion"] = res.dispersion
        for m, d in o.items():
            outs[m if ver == "v0" else "v1:" + m] = d
    rows = []
    for k, q in enumerate(peaks):
        base = {kk: q[kk] for kk in ("dbin", "pb", "s", "sigma", "lam_box", "lam_tot", "box_frac", "B_box_e", "lam_eff")}
        base["K0_box"] = float(outs["O"]["B_box"][k])
        base["mh_box"] = float(m_map[q["sl"]][q["bx"]].sum()); base["g_box"] = float(law.g(m_map[q["sl"]][q["bx"]]).sum())
        for m, d in outs.items():
            r_ = {**base, "method": m, "I": d["I"][k], "sigma_I": d["sigma"][k], "B_box": d["B_box"][k], "var_B": d["var_B"][k],
                  "n_box": d["n_box"][k], "n_ann": d["n_ann"][k], "auto_use_GL": d["auto_use_GL"][k] if "auto_use_GL" in d else np.nan}
            for extra in ("lam_min5", "auto_D", "sigma_v0form"):
                if extra in d:
                    r_[extra] = d[extra][k]
            rows.append(r_)
    for r_ in rows:
        r_.update(meta); r_.update({f"info_{kk}": v for kk, v in info.items()})
    return rows


def run_fly(args):
    sid, set_name = args
    f = CK / f"real_{set_name}" / f"{sid}.pkl"
    if f.exists():
        return f"{sid} cached"
    t0 = time.time()
    sig8 = np.clip(pd.read_csv(paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv").sigma_px.to_numpy(float), 0.5, 2.0)
    z = np.load(RV.HOSTS / f"{sid}_hosts_v0.npz")
    tab = pd.read_parquet(paths.SUB / "features" / f"{sid}_bin8_features_v1.parquet", columns=["row", "frame", "cy", "cx"]).set_index(["row", "frame"])
    flag = str(np.load(paths.SUB / "atlas" / "v1b" / f"{sid}_build_v1b.npz")["gain_flag"])
    tie = 0.35 if flag.lower().startswith("enable") else None
    tau_b = D.tau_for_binning(D.TAU_PHYS_V1, 8)
    law = BG.CountLaw.apollo(8, tie, dead_time_tau=tau_b)
    bad = RV.fly_mask(sid)
    rows = []
    for h in range(len(z["pos"])):
        if not z["addr_ok"][h]:
            continue
        row, frame, cls = int(z["row"][h]), int(z["frame"][h]), str(z["cls"][h])
        tr = tab.loc[(row, frame)]; centre = (float(tr.cy), float(tr.cx))
        K0 = z["frames"][h].astype(np.int64)
        for rep in range(RV.REPS):
            rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 31) + 7_000_003 * int(z["pos"][h]) + 101 * rep + 55)
            rows += run_host(rng, K0, bad, centre, law, 8, tie, tau_b, sig8,
                             {"sid": sid, "regime": cls, "row": row, "frame": frame, "rep": rep, "binning": 8, "gain_flag": flag})
    pickle.dump(rows, open(f, "wb"))
    return f"{set_name} fly {sid}: {len(rows)} rows, {time.time() - t0:.0f} s"


def run_classic(args):
    sid, set_name = args
    f = CK / f"real_{set_name}" / f"classic_{sid}.pkl"
    if f.exists():
        return f"classic {sid} cached"
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("c0", ROOT / "scripts" / "classic_v0.py")
    c0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c0)
    prof = pickle.load(open(paths.SUB / "s_level_bin4_v0" / "classic_profiles_v0.pkl", "rb"))
    if sid not in prof or prof[sid].get("profile") is None:
        return f"classic {sid} no profile"
    sig8 = np.clip(pd.read_csv(paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv").sigma_px.to_numpy(float), 0.5, 2.0)
    files = sorted((RV.FRD / sid).glob("*.npz"))
    frames = [np.load(p)["frame"] for p in files]
    frames = [a for a in frames if np.issubdtype(a.dtype, np.integer)]
    if len(frames) < 8:
        return f"classic {sid} < 8 frames"
    S0 = np.sum([a.astype(float) for a in frames], axis=0)
    bad0, stop, sinfo = c0.bad_mask_for(S0)
    from scipy import ndimage as ndi
    bad = ndi.binary_dilation(bad0, iterations=1); bad[:2, :] = bad[-2:, :] = True; bad[:, :2] = bad[:, -2:] = True
    centre = tuple(float(v) for v in prof[sid]["profile"].meta["centre"])
    law = BG.CountLaw.apollo(4, None, dead_time_tau=0.0)
    hosts = np.unique(np.round(np.linspace(0, len(frames) - 1, min(6, len(frames)))).astype(int))
    rows = []
    for h in hosts:
        K0 = frames[h].astype(np.int64)
        for rep in range(RV.REPS):
            rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 31) + 9_000_011 * int(h) + 103 * rep + 77)
            rows += run_host(rng, K0, bad, centre, law, 4, None, 0.0, sig8,
                             {"sid": sid, "regime": "classic", "row": -1, "frame": int(h), "rep": rep, "binning": 4, "gain_flag": "raw"})
    pickle.dump(rows, open(f, "wb"))
    return f"{set_name} classic {sid}: {len(rows)} rows, {time.time() - t0:.0f} s"


def _job(a):
    kind, sid, set_name = a
    try:
        return (run_fly if kind == "fly" else run_classic)((sid, set_name))
    except Exception as e:  # noqa: BLE001
        return f"{kind} {sid} FAILED: {e!r}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True); (CK / f"real_{a.set}").mkdir(parents=True, exist_ok=True)
    jobs = [("fly", s, a.set) for s in RV.fly_sessions(a.set)] + [("classic", s, a.set) for s in RV.classic_sessions(a.set)]
    log(f"real {a.set}: {len(jobs)} sessions")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_job, jobs):
            log(msg)
    rows = []
    for f in sorted((CK / f"real_{a.set}").glob("*.pkl")):
        rows += pickle.load(open(f, "rb"))
    R = pd.DataFrame(rows)
    p = OUT / f"real_{a.set}_peaks.parquet"
    R.to_parquet(p)
    write_sidecar(p, inputs=[RV.HOSTS, RV.FRD, paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv"],
                  params={"set": a.set, "prereg_commit": "bc37be9", "harness": "bgsub v0 real (same seeds, hosts, sessions)",
                          "v0_methods": V0_METHODS, "v1_methods": V1_METHODS},
                  note="bgsub v1 real-frame injected-peak validation (v0 and v1 on identical frames)")
    log(f"gathered {len(R)} rows")


if __name__ == "__main__":
    main()
