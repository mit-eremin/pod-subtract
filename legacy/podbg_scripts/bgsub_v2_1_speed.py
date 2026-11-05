"""bgsub v2.1 speed (R5, design note v2.1 Sec. 2.4): G fit + map per frame (plan cached; ring refinement on,
re-centring not triggered), numba and numpy, on simulator v1 Apollo frames (lacey, beam stop) at 512^2 (bin 16),
1024^2 (bin 8), 2048^2 (bin 4); v1, v2 and v2.1 in the same run; integration of 35 spots at bin 8 (G, GL, AUTO, -ML,
intervals).  Each attempt waits (<= 10 min) for the 1-min load average to fall below 4; up to 3 attempts, the
lowest-load attempt is the R5 measurement (all attempts are written)."""
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

from podbg import bgsub as BG, paths
from podbg.sim import simulate_frames
from podbg.provenance import write_sidecar

LOAD_MAX = 4.0


def attempt():
    out = []
    for b in (16, 8, 4):
        r = simulate_frames("lacey_utc", n_frames=4, seed=11, binning=b, beam_stop=True, version="v1")
        m = ndi.binary_dilation(r.mask, iterations=1); m[:2] = m[-2:] = True; m[:, :2] = m[:, -2:] = True
        c = r.truth[0]["centre_yx"]
        law = BG.CountLaw.apollo(b, 0.403)
        for ver in ("v1", "v2", "v2.1"):
            t = time.perf_counter(); plan = BG.make_plan(m.shape, m, BG.Geometry(c), version=ver); tp = time.perf_counter() - t
            for nb in (True, False):
                ts = []; rec = []
                for i in range(16 if nb else 4):
                    t = time.perf_counter(); res = BG.estimate_background(r.frames[i % 4], m, BG.Geometry(c), law, plan=plan, use_numba=nb)
                    ts.append(time.perf_counter() - t)
                    if res.centre_info is not None:
                        rec.append(res.centre_info["recentred"])
                out.append({"N": m.shape[0], "binning": b, "version": ver, "numba": nb, "median_ms": round(1e3 * float(np.median(ts[1:])), 1),
                            "plan_ms": round(1e3 * tp, 1), "recentred_any": bool(any(rec)), "load1": os.getloadavg()[0]})
                print(out[-1], flush=True)
            if b == 8:
                rng = np.random.default_rng(0)
                yy = c[0] + rng.uniform(-400, 400, 60); xx = c[1] + rng.uniform(-400, 400, 60)
                ok = (np.hypot(yy - c[0], xx - c[1]) > 90) & (yy > 15) & (yy < 1008) & (xx > 15) & (xx < 1008)
                spots = np.column_stack([yy[ok], xx[ok]])[:35]
                ex = BG.disc_mask(m.shape, spots, 9.0)
                res = BG.estimate_background(r.frames[0], m, BG.Geometry(c), law, exclude=ex, plan=plan)
                ts = []
                for i in range(5):
                    t = time.perf_counter()
                    BG.integrate_spots(r.frames[0], res, spots, 3.0, 5.0, 9.0, 9.0, sigmas=np.full(len(spots), 1.2), methods=("G", "GL", "AUTO"))
                    ts.append(time.perf_counter() - t)
                out.append({"N": 1024, "binning": 8, "version": ver, "what": "integrate_spots 35 spots (G, GL, AUTO, ML, intervals)",
                            "median_ms": round(1e3 * float(np.median(ts[1:])), 1), "load1": os.getloadavg()[0]})
                print(out[-1], flush=True)
    return out


def main():
    attempts = []
    for k in range(3):
        t0 = time.time()
        while os.getloadavg()[0] >= LOAD_MAX and time.time() - t0 < 600:
            time.sleep(30)
        l0 = os.getloadavg()[0]
        rows = attempt()
        attempts.append({"attempt": k, "load_start": l0, "load_end": os.getloadavg()[0], "rows": rows})
        if l0 < LOAD_MAX:
            break
    best = min(attempts, key=lambda a: max(a["load_start"], a["load_end"]))
    p = paths.SUB / "bgsub_v2_1" / "speed_v2_1.json"
    json.dump({"best_attempt": best["attempt"], "attempts": attempts}, open(p, "w"), indent=1)
    write_sidecar(p, inputs=[], params={"load_max": LOAD_MAX, "sim": "v1 lacey, beam stop", "n_attempts": len(attempts)},
                  note="bgsub v2.1 speed (R5)")
    r5 = [r for r in best["rows"] if r.get("N") == 1024 and r.get("numba") and r["version"] == "v2.1" and "what" not in r]
    print("R5 v2.1 1k numba median ms:", r5[0]["median_ms"], "load", best["load_start"], best["load_end"])


if __name__ == "__main__":
    main()
