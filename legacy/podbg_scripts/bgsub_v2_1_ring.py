"""bgsub v2.1 fix 2 (A267): the sharp-ring zone on spot-free classic hosts, v2 vs v2.1 global background (same hosts as
the v0-v2.1 harness, no injection).  Per host: pixel dispersion of K - G against the count law in the 4.6-A ring zone
(|s - 0.2174| <= 3 px x k4) and in the rest of D0 (s 0.13-0.25), mean residual and the range of 16 sector means in the
ring zone (stored counts per pixel), ellipticity used by v2.1, re-flag rounds.

Usage: .venv/bin/python scripts/bgsub_v2_1_ring.py --set dev|test|confirm
Output: ~/pod_data/background/bgsub_v2_1/ring_<set>.csv (+ sidecar)
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bgsub_v0_real as RV  # noqa: E402
import bgsub_v2_1_real as R21  # noqa: E402
from podbg import paths  # noqa: E402
from podbg import bgsub as BG  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = paths.SUB / "bgsub_v2_1"
K4 = 0.0022811 / 2
S_RING = 1.0 / 4.60
HALF_PX = 3.0


def session(sid, c0, prof, set_name):
    if sid not in prof or prof[sid].get("profile") is None:
        return []
    files = sorted((RV.FRD / sid).glob("*.npz"))
    frames = [a for a in (np.load(p)["frame"] for p in files) if np.issubdtype(a.dtype, np.integer)]
    if len(frames) < (4 if set_name == "confirm" else 8):
        return []
    S0 = np.sum([a.astype(float) for a in frames], axis=0)
    bad0, stop, sinfo = c0.bad_mask_for(S0)
    bad = ndi.binary_dilation(bad0, iterations=1); bad[:2, :] = bad[-2:, :] = True; bad[:, :2] = bad[:, -2:] = True
    centre = tuple(float(v) for v in prof[sid]["profile"].meta["centre"])
    law = BG.CountLaw.apollo(4, None)
    hosts = np.unique(np.round(np.linspace(0, len(frames) - 1, min(6, len(frames)))).astype(int))
    yy, xx = np.indices(S0.shape)
    rows = []
    for h in hosts:
        K = frames[h].astype(float)
        for ver in ("v2", "v2.1"):
            res = BG.estimate_background(K, bad, BG.Geometry(centre), law, version=ver)
            cy, cx = res.plan.geometry.centre
            r = np.hypot(yy - cy, xx - cx); s = r * K4; phi = np.arctan2(yy - cy, xx - cx)
            ok = ~bad & np.isfinite(res.bg) & ~res.flags
            okall = ~bad & np.isfinite(res.bg)
            ring = okall & (np.abs(s - S_RING) <= HALF_PX * K4)
            d0 = ok & (s >= 0.13) & (s < 0.25) & ~ring
            R = K - res.bg
            g = np.maximum(law.g(np.maximum(res.bg, 0)), 1e-3)
            sec = np.floor((phi[ring] + np.pi) / (2 * np.pi) * 16).astype(int)
            sm = np.bincount(sec, R[ring], 16) / np.maximum(np.bincount(sec, minlength=16), 1)
            ci = res.centre_info or {}
            rows.append({"set": set_name, "sid": sid, "host": int(h), "ver": ver, "b_ring": float(res.bg[ring].mean()),
                         "disp_ring": float(np.mean(R[ring] ** 2 / g[ring])), "mean_ring": float(R[ring].mean()),
                         "sector_range": float(sm.max() - sm.min()), "disp_d0": float(np.mean(R[d0] ** 2 / g[d0])),
                         "frac_flag_ring": float(res.flags[ring].mean()), "recentred": bool(ci.get("recentred", False)),
                         "ring_reflag": int(ci.get("ring_reflag", 0)), "ell": str(ci.get("ellipticity", "")),
                         "ring_cmax": float(np.abs(res.ring.c).max()) if res.ring is not None else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("c0", ROOT / "scripts" / "classic_v0.py")
    c0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c0)
    prof = pickle.load(open(paths.SUB / "s_level_bin4_v0" / "classic_profiles_v0.pkl", "rb"))
    sids = R21.confirm_sessions() if a.set == "confirm" else RV.classic_sessions(a.set)
    rows = []
    for sid in sids:
        rows += session(sid, c0, prof, a.set)
        print(sid, "done", flush=True)
    df = pd.DataFrame(rows)
    p = OUT / f"ring_{a.set}.csv"; df.to_csv(p, index=False)
    write_sidecar(p, inputs=[RV.FRD], params={"s_ring": S_RING, "half_px": HALF_PX, "k4": K4, "set": a.set, "n_sessions": int(df.sid.nunique())},
                  note="bgsub v2.1 fix 2: sharp-ring zone, v2 vs v2.1 G on spot-free classic hosts")
    print(df.groupby(["sid", "ver"])[["disp_ring", "mean_ring", "sector_range", "disp_d0", "frac_flag_ring"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
