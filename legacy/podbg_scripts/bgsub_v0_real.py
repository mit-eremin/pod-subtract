"""bgsub v0 stage 4: injected-peak validation on REAL archive frames (design note Sec. 1.8, amendments 1-4).

Host frames are peak recovery v0's (read-only): fly-scan F / BA / CN hosts of the 20 held-out sessions (bin 8) and the
cached classic raw-count bin-4 No frames of s level bin4 v0. Injection reuses `podbg.peaks` (latent posterior, dead
time, stored-count law). Truth: the oracle K0 (pre-injection frame) and lambda_box / lambda_eff.

Usage: .venv/bin/python scripts/bgsub_v0_real.py --set dev|test
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

from podbg import paths, detector as D  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg import peaks as PK  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = paths.SUB / "bgsub_v0"; CK = OUT / "checkpoints"
HOSTS = paths.SUB / "peak_recovery_v0" / "hosts"
FRD = paths.SUB / "s_level_bin4_v0" / "frames"
DEV_FLY = ["s0047f887", "s4fead7c7", "sc19f33b9"]
N_DEV_CLASSIC = 4
REPS = 2
K8 = CM.K8


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "logs" / "real_v0.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def seam_bad(shape):
    from podbg import atlas as A
    m = np.zeros(shape, bool)
    for c in A.SEAM_LINES_BIN8["cols"]:
        m[:, max(c - 1, 0):c + 2] = True
    for r in A.SEAM_LINES_BIN8["rows"]:
        m[max(r - 1, 0):r + 2, :] = True
    return m


def fly_mask(sid):
    from scipy import ndimage as ndi
    z = np.load(paths.SUB / "features" / f"{sid}_bin8_shells_v1.npz")
    bad = ndi.binary_dilation(z["mask"].astype(bool), iterations=1) | seam_bad(z["mask"].shape)
    bad[:2, :] = bad[-2:, :] = True; bad[:, :2] = bad[:, -2:] = True
    return bad


def host_mu_map(K0, valid, centre, law):
    """The host's own masked shell means (stored), inverted to electrons per pixel, as a map (peak recovery v0)."""
    nsh = int(np.hypot(*K0.shape)) + 2
    mh, n = PK.shell_means(K0, valid, centre, nsh)
    yy, xx = np.indices(K0.shape)
    r = np.hypot(yy - centre[0], xx - centre[1])
    m_map = np.clip(PK.profile_at(mh, r), 0, None)
    return law.E_inv(m_map), m_map


def run_host(rng, K0, bad, centre, law, binning, tie, tau_b, sig8, meta):
    box = CM.geometry_for(binning)
    valid = ~bad
    mu_map, m_map = host_mu_map(K0, valid, centre, law)
    peaks = CM.place_peaks(rng, valid, centre, K8 * binning / 8.0, box, sig8, binning)
    Kp = CM.inject_all(rng, K0, peaks, mu_map, binning, tie, tau_b, box, law.epc, law.E)
    rows, info, res = CM.evaluate_frame(Kp, K0, bad, centre, law, peaks, box, b_true=None)
    for k, q in enumerate(peaks):
        mh_box = float(m_map[q["sl"]][q["bx"]].sum())
        g_box = float(law.g(m_map[q["sl"]][q["bx"]]).sum())
        for r_ in rows:
            if r_["pb"] == q["pb"] and r_["dbin"] == q["dbin"]:
                r_["mh_box"] = mh_box; r_["g_box"] = g_box
    for r_ in rows:
        r_.update(meta); r_.update({f"info_{k}": v for k, v in info.items()})
    return rows, info


def fly_sessions(set_name):
    sids = sorted(p.name.split("_")[0] for p in HOSTS.glob("*_hosts_v0.npz"))
    return [s for s in sids if (s in DEV_FLY) == (set_name == "dev")]


def classic_sessions(set_name):
    ok = sorted(d.name for d in FRD.iterdir() if len(list(d.glob("*.npz"))) >= 8)
    return ok[:N_DEV_CLASSIC] if set_name == "dev" else ok[N_DEV_CLASSIC:]


def run_fly(sid, set_name, sig8):
    f = CK / f"real_{set_name}" / f"{sid}.pkl"
    if f.exists():
        return
    t0 = time.time()
    z = np.load(HOSTS / f"{sid}_hosts_v0.npz")
    tab = pd.read_parquet(paths.SUB / "features" / f"{sid}_bin8_features_v1.parquet", columns=["row", "frame", "cy", "cx"]).set_index(["row", "frame"])
    flag = str(np.load(paths.SUB / "atlas" / "v1b" / f"{sid}_build_v1b.npz")["gain_flag"])
    tie = 0.35 if flag.lower().startswith("enable") else None
    tau_b = D.tau_for_binning(D.TAU_PHYS_V1, 8)
    law = BG.CountLaw.apollo(8, tie, dead_time_tau=tau_b)
    bad = fly_mask(sid)
    rows = []
    for h in range(len(z["pos"])):
        if not z["addr_ok"][h]:
            continue
        row, frame, cls = int(z["row"][h]), int(z["frame"][h]), str(z["cls"][h])
        tr = tab.loc[(row, frame)]; centre = (float(tr.cy), float(tr.cx))
        K0 = z["frames"][h].astype(np.int64)
        for rep in range(REPS):
            rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 31) + 7_000_003 * int(z["pos"][h]) + 101 * rep + 55)
            r_, info = run_host(rng, K0, bad, centre, law, 8, tie, tau_b, sig8,
                                {"sid": sid, "regime": cls, "row": row, "frame": frame, "rep": rep, "binning": 8, "gain_flag": flag})
            rows += r_
    pickle.dump(rows, open(f, "wb"))
    log(f"{set_name} fly {sid}: {len(rows)} rows, {time.time() - t0:.0f} s")


def run_classic(sid, set_name, sig8, prof, c0):
    f = CK / f"real_{set_name}" / f"classic_{sid}.pkl"
    if f.exists() or sid not in prof or prof[sid].get("profile") is None:
        return
    t0 = time.time()
    files = sorted((FRD / sid).glob("*.npz"))
    frames = [np.load(p)["frame"] for p in files]
    frames = [a for a in frames if np.issubdtype(a.dtype, np.integer)]
    if len(frames) < 8:
        return
    S0 = np.sum([a.astype(float) for a in frames], axis=0)
    bad0, stop, sinfo = c0.bad_mask_for(S0)
    from scipy import ndimage as ndi
    bad = ndi.binary_dilation(bad0, iterations=1); bad[:2, :] = bad[-2:, :] = True; bad[:, :2] = bad[:, -2:] = True
    centre = tuple(float(v) for v in prof[sid]["profile"].meta["centre"])
    law = BG.CountLaw.apollo(4, None, dead_time_tau=0.0)          # raw counts, 1 count = 1 e; no dead time (as peak recovery)
    hosts = np.unique(np.round(np.linspace(0, len(frames) - 1, min(6, len(frames)))).astype(int))
    rows = []
    for h in hosts:
        K0 = frames[h].astype(np.int64)
        for rep in range(REPS):
            rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 31) + 9_000_011 * int(h) + 103 * rep + 77)
            r_, info = run_host(rng, K0, bad, centre, law, 4, None, 0.0, sig8,
                                {"sid": sid, "regime": "classic", "row": -1, "frame": int(h), "rep": rep, "binning": 4, "gain_flag": "raw"})
            rows += r_
    pickle.dump(rows, open(f, "wb"))
    log(f"{set_name} classic {sid}: {len(rows)} rows, {time.time() - t0:.0f} s")


def main(set_name):
    (OUT / "logs").mkdir(parents=True, exist_ok=True); (CK / f"real_{set_name}").mkdir(parents=True, exist_ok=True)
    sig8 = np.clip(pd.read_csv(paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv").sigma_px.to_numpy(float), 0.5, 2.0)
    for sid in fly_sessions(set_name):
        run_fly(sid, set_name, sig8)
    spec = importlib.util.spec_from_file_location("c0", ROOT / "scripts" / "classic_v0.py")
    c0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c0)
    pp = paths.SUB / "s_level_bin4_v0" / "classic_profiles_v0.pkl"
    prof = pickle.load(open(pp, "rb"))
    for sid in classic_sessions(set_name):
        try:
            run_classic(sid, set_name, sig8, prof, c0)
        except Exception as e:  # noqa: BLE001
            log(f"classic {sid} FAILED: {e!r}")
    rows = []
    for f in sorted((CK / f"real_{set_name}").glob("*.pkl")):
        rows += pickle.load(open(f, "rb"))
    R = pd.DataFrame(rows)
    p = OUT / f"real_{set_name}_peaks_v0.parquet"
    R.to_parquet(p)
    write_sidecar(p, inputs=[HOSTS, FRD, paths.SUB / "peak_recovery_v0" / "spot_shape_v0.csv", pp],
                  params={"set": set_name, "dev_fly": DEV_FLY, "n_dev_classic": N_DEV_CLASSIC, "reps": REPS,
                          "box": {b: CM.geometry_for(b) for b in (4, 8)}, "dbins": CM.DBINS, "ladder": CM.LADDER,
                          "law": "fly-scan: CountLaw.apollo(8, tie of the session gain flag, tau_b TAU_PHYS_V1); classic: Poisson, no dead time"},
                  note="bgsub v0 real-frame injected-peak validation")
    log(f"gathered {len(R)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev")
    main(ap.parse_args().set)
