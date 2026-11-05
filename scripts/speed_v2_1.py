"""R5 speed of the background estimate (ported from pod/background scripts/bgsub_v2_1_speed.py at f1e93f5): G fit + map
per frame (plan cached; ring refinement on, re-centring not triggered), numba and numpy, on simulator-v1 Apollo lacey
frames with the beam stop at 512^2 (bin 16), 1024^2 (bin 8), 2048^2 (bin 4); v1, v2, v2.1 in one run. Integration
timing is kept as information only (not a product criterion since the 2026-11-05 scope note).

Difference from the original: the four frames per binning are rendered from the bank's expected frames (bit-identical
to podbg's) with this package's vendored count law and seed 11, so the noise realisation differs from the published
run (equal in law). Each attempt waits (<= 10 min) for the 1-min load average to fall below 4; up to 3 attempts; the
lowest-load attempt is the R5 measurement. Record the load with the number: R5 failed at 54.2 ms under load (A278).

Usage: .venv/bin/python scripts/speed_v2_1.py [--out speed_v2_1]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

import bgsub as BG
from bgsub import config
from bgsub.validation import simbank as SB
from bgsub.vendor.provenance import write_sidecar

LOAD_MAX = 4.0


def frames_for(b):
    rng = np.random.default_rng(11)
    ents = [SB.load(SB.entry_path("speed", f"lacey_utc_b{b}_seed11_f{i}")) for i in range(4)]
    return ents, [SB.stored(SB.expected(e), b, rng) for e in ents]


def attempt():
    out = []
    for b in (16, 8, 4):
        ents, frames = frames_for(b)
        m = ndi.binary_dilation(SB.stop_mask(ents[0]), iterations=1); m[:2] = m[-2:] = True; m[:, :2] = m[:, -2:] = True
        c = ents[0].centre_yx
        law = BG.CountLaw.apollo(b, 0.403)
        for ver in ("v1", "v2", "v2.1"):
            t = time.perf_counter(); plan = BG.make_plan(m.shape, m, BG.Geometry(c), version=ver); tp = time.perf_counter() - t
            for nb in (True, False):
                ts = []; rec = []
                for i in range(16 if nb else 4):
                    t = time.perf_counter(); res = BG.estimate_background(frames[i % 4], m, BG.Geometry(c), law, plan=plan, use_numba=nb)
                    ts.append(time.perf_counter() - t)
                    if res.centre_info is not None:
                        rec.append(res.centre_info["recentred"])
                out.append({"N": m.shape[0], "binning": b, "version": ver, "numba": nb, "median_ms": round(1e3 * float(np.median(ts[1:])), 1),
                            "plan_ms": round(1e3 * tp, 1), "recentred_any": bool(any(rec)), "load1": os.getloadavg()[0]})
                print(out[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="speed_v2_1"); a = ap.parse_args()
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
    best = min(attempts, key=lambda x: max(x["load_start"], x["load_end"]))
    p = config.out_dir(a.out) / "speed.json"
    json.dump({"best_attempt": best["attempt"], "attempts": attempts}, open(p, "w"), indent=1)
    write_sidecar(p, inputs=[config.SIMBANK_V1 / "speed"], params={"load_max": LOAD_MAX, "sim": "v1 lacey bank, beam stop", "n_attempts": len(attempts)},
                  note="bgsub speed (R5)")
    r5 = [r for r in best["rows"] if r.get("N") == 1024 and r.get("numba") and r["version"] == "v2.1"]
    print("R5 v2.1 1k numba median ms:", r5[0]["median_ms"], "load", best["load_start"], best["load_end"])


if __name__ == "__main__":
    main()
