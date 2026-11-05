"""bgsub v2 real-frame validation (design note v2 Sec. 3): the v0/v1 harness with the SAME sessions, hosts, replicates and
seeds (so the injected frames are identical to v0's and v1's), evaluated with v1 and v2 on each frame, plus the
pre-registered, method-independent HOST-OVERLAP flag (a sample Bragg spot of the host frame inside the box).

Usage: .venv/bin/python scripts/bgsub_v2_real.py --set dev|test|confirm [--workers W]
  confirm = the 9 classic sessions with 4-7 cached raw frames that no earlier run used (classic confirmation set).
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
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bgsub_v0_common as CM  # noqa: E402
import bgsub_v0_real as RV  # noqa: E402

from podbg import paths, detector as D  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = paths.SUB / "bgsub_v2"; CK = OUT / "checkpoints"
V1_METHODS = ("G", "GL", "AUTO", "FLAT")
V2_METHODS = ("G", "GL", "AUTO")
OVL_Z = 3.0; OVL_PEAK = 6.0


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "logs" / "real_v2.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def overlap_flags(K0, valid, m_map, law, peaks, box, binning):
    """Pre-registered host-overlap flag (design note v2 Sec. 3.3): on the spot-free host, z = (K0 - shell mean) /
    sqrt(g(shell mean)); clusters (8-connected) of z > 3 with >= A_min pixels (4 at bin 8, 8 at bin 4) and peak z >= 6
    are sample diffraction; a spot overlaps if such a cluster has a pixel within r_box + 2 (x 8 / binning) px."""
    gm = np.maximum(law.g(np.maximum(m_map, 0.0)), float(law.g(np.array([1e-3]))[0]))
    z = np.where(valid, (K0 - m_map) / np.sqrt(gm), 0.0)
    hot = valid & (z > OVL_Z)
    lab, nl = ndi.label(hot, structure=np.ones((3, 3)))
    a_min = 4 if binning >= 8 else 8
    flagged = np.zeros(K0.shape, bool)
    if nl:
        area = np.bincount(lab.ravel())
        peak = ndi.maximum(z, lab, index=np.arange(nl + 1))
        good = (area >= a_min) & (np.asarray(peak) >= OVL_PEAK); good[0] = False
        flagged = good[lab]
    reach = box["r_box"] + 2.0 * 8.0 / binning
    out = []
    for q in peaks:
        iy, ix = int(round(q["cy"])), int(round(q["cx"])); R = int(np.ceil(reach)) + 1
        y0, y1 = max(iy - R, 0), min(iy + R + 1, K0.shape[0]); x0, x1 = max(ix - R, 0), min(ix + R + 1, K0.shape[1])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append(bool((flagged[y0:y1, x0:x1] & (np.hypot(yy - q["cy"], xx - q["cx"]) <= reach)).any()))
    return np.array(out)


def run_host(rng, K0, bad, centre, law, binning, tie, tau_b, sig8, meta):
    """Exactly v0's run_host up to the injection (same rng sequence), then v1 and v2 evaluation."""
    box = CM.geometry_for(binning)
    valid = ~bad
    mu_map, m_map = RV.host_mu_map(K0, valid, centre, law)
    peaks = CM.place_peaks(rng, valid, centre, CM.K8 * binning / 8.0, box, sig8, binning)
    Kp = CM.inject_all(rng, K0, peaks, mu_map, binning, tie, tau_b, box, law.epc, law.E)
    ovl = overlap_flags(K0.astype(float), valid, m_map, law, peaks, box, binning)
    spots = np.array([[q["cy"], q["cx"]] for q in peaks]).reshape(-1, 2)
    ex = BG.disc_mask(Kp.shape, spots, box["r_excl"])
    lam = np.array([q["lam_box"] for q in peaks])
    outs = {}; info = {}
    for ver, methods in (("v1", V1_METHODS), ("v2", V2_METHODS)):
        t = time.perf_counter()
        plan = BG.make_plan(Kp.shape, bad, BG.Geometry(centre), version=ver)
        info[f"{ver}_t_plan_s"] = time.perf_counter() - t
        res = BG.estimate_background(Kp, bad, BG.Geometry(centre), law, exclude=ex, plan=plan)
        o = BG.integrate_spots(Kp, res, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"],
                               sigmas=np.array([q["sigma"] for q in peaks]), methods=methods, oracle=K0 if ver == "v1" else None,
                               lam_eval=lam)
        info[f"{ver}_t_fit_s"] = res.timings["total_s"]; info[f"{ver}_dispersion"] = res.dispersion
        if res.centre_info is not None:
            info["v2_recentred"] = bool(res.centre_info["recentred"]); info["v2_shift_in"] = float(np.hypot(*res.centre_info["shift_in"]))
        if res.ring is not None:
            info["v2_ring_n"] = int((res.ring.w > 0.5).sum()); info["v2_ring_absmax"] = float(np.abs(res.ring.c).max())
        for m, d in o.items():
            outs[f"{ver}:{m}"] = d
    rows = []
    for k, q in enumerate(peaks):
        base = {kk: q[kk] for kk in ("dbin", "pb", "s", "sigma", "lam_box", "lam_tot", "box_frac", "B_box_e", "lam_eff")}
        base["K0_box"] = float(outs["v1:O"]["B_box"][k]); base["overlap"] = bool(ovl[k])
        base["mh_box"] = float(m_map[q["sl"]][q["bx"]].sum()); base["g_box"] = float(law.g(m_map[q["sl"]][q["bx"]]).sum())
        for m, d in outs.items():
            r_ = {**base, "method": m, "I": d["I"][k], "sigma_I": d["sigma"][k], "B_box": d["B_box"][k], "var_B": d["var_B"][k],
                  "n_box": d["n_box"][k], "n_ann": d["n_ann"][k]}
            for extra in ("lam_min5", "auto_use_GL", "auto_w", "auto_D", "sigma_v0form", "sigma_at", "ci_lo", "ci_hi"):
                if extra in d:
                    r_[extra] = d[extra][k]
            rows.append(r_)
    for r_ in rows:
        r_.update(meta); r_.update({f"info_{kk}": v for kk, v in info.items()})
    return rows


def run_fly(sid, set_name):
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


def confirm_sessions():
    """Classic sessions with 4-7 cached frames (never used by v0/v1: classic_sessions() needs >= 8)."""
    return sorted(d.name for d in RV.FRD.iterdir() if 4 <= len(list(d.glob("*.npz"))) < 8)


def run_classic(sid, set_name):
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
    n_min = 4 if set_name == "confirm" else 8
    if len(frames) < n_min:
        return f"classic {sid} < {n_min} integer frames"
    S0 = np.sum([a.astype(float) for a in frames], axis=0)
    bad0, stop, sinfo = c0.bad_mask_for(S0)
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
        return (run_fly if kind == "fly" else run_classic)(sid, set_name)
    except Exception as e:  # noqa: BLE001
        return f"{kind} {sid} FAILED: {e!r}"


def jobs_for(set_name):
    if set_name == "confirm":
        return [("classic", s, set_name) for s in confirm_sessions()]
    return [("fly", s, set_name) for s in RV.fly_sessions(set_name)] + [("classic", s, set_name) for s in RV.classic_sessions(set_name)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True); (CK / f"real_{a.set}").mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.set)
    log(f"real {a.set}: {len(jobs)} sessions planned")
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_job, jobs):
                log(msg)
    else:
        for j in jobs:
            log(_job(j))
    rows = []; done = []
    for kind, sid, _s in jobs:
        f = CK / f"real_{a.set}" / (f"{sid}.pkl" if kind == "fly" else f"classic_{sid}.pkl")
        if f.exists():
            rows += pickle.load(open(f, "rb")); done.append(sid)
    log(f"completeness: {len(done)} / {len(jobs)} planned sessions have checkpoints; missing {sorted(set(s for _k, s, _ in jobs) - set(done))}")
    R = pd.DataFrame(rows)
    p = OUT / f"real_{a.set}_peaks.parquet"
    R.to_parquet(p)
    write_sidecar(p, inputs=[RV.HOSTS, RV.FRD, paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv"],
                  params={"set": a.set, "harness": "bgsub v0 real (same seeds, hosts, sessions)", "v1_methods": V1_METHODS,
                          "v2_methods": V2_METHODS, "overlap": {"z": OVL_Z, "peak": OVL_PEAK}, "n_planned": len(jobs), "n_done": len(done)},
                  note="bgsub v2 real-frame injected-peak validation (v1 and v2 on identical frames)")
    log(f"gathered {len(R)} rows")


if __name__ == "__main__":
    main()
