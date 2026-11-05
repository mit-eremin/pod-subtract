# VENDORED COPY. Source: pod/background/scripts/bgsub_v0_common.py at background commit f1e93f5 (file last changed in 0cd35b9);
# sha256 of the source file: 1489094426b65af2ef38bf23d3869990d1d4809f1c2308d2ffe7cdc654c781d4. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: `from podbg import peaks/bgsub` -> `from . import peaks as PK` / `import bgsub as BG`.
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""bgsub v0 validation helpers shared by the synthetic and the real-frame runs (design note Sec. 1.8).

Placement and injection follow peak recovery v0 Sec. 1.2 (reusing `podbg.peaks` read-only: spot_pixels, patch_radius,
box_and_annulus, inject); the integration is podbg.bgsub's own.
"""
from __future__ import annotations

import numpy as np

from . import peaks as PK
import bgsub as BG

K8 = 0.0022811
DBINS = {"D0": (0.13, 0.25), "D4": (0.25, 0.50), "D3": (0.50, 0.80), "D2": (0.80, 1.00), "D1": (1.00, 1.163)}
LADDER = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
METHODS = ("G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST")


def geometry_for(binning: int) -> dict:
    f = 8.0 / binning
    if binning == 16:
        return {"r_box": 1.5, "r_in": 2.5, "r_out": 4.5, "r_excl": 4.5, "half": 6, "spacing": 12.0}
    return {"r_box": 3.0 * f, "r_in": 5.0 * f, "r_out": 9.0 * f, "r_excl": 9.0 * f, "half": int(round(10 * f)), "spacing": 24.0 * f}


def place_peaks(rng, valid, centre, k_per_px, box, sigmas_bin8, binning, dbins=DBINS, ladder=LADDER):
    ny, nx = valid.shape; half = box["half"]; out = []
    for db, (lo, hi) in dbins.items():
        for pb in ladder:
            for _ in range(200):
                s = rng.uniform(lo, hi); r = s / k_per_px - 0.5; ph = rng.uniform(0, 2 * np.pi)
                cy = centre[0] + r * np.sin(ph); cx = centre[1] + r * np.cos(ph)
                iy, ix = int(round(cy)), int(round(cx))
                if iy - half < 0 or ix - half < 0 or iy + half >= ny or ix + half >= nx:
                    continue
                if any(np.hypot(cy - q["cy"], cx - q["cx"]) < box["spacing"] for q in out):
                    continue
                rp = PK.patch_radius(iy - half, ix - half, 2 * half + 1, cy, cx)
                bx, an = PK.box_and_annulus(rp, box["r_box"], box["r_in"], box["r_out"])
                vp = valid[iy - half:iy + half + 1, ix - half:ix + half + 1]
                if not vp[bx | an].all():
                    continue
                out.append({"cy": cy, "cx": cx, "s": s, "dbin": db, "pb": pb, "sigma": float(rng.choice(sigmas_bin8)) * 8.0 / binning})
                break
    return out


def inject_all(rng, K0, peaks, mu_h_map, binning, tie, tau_b, box, epc, E0, N_known=None):
    """Inject every peak (peaks.inject); mu_h_map = host electron mean per pixel (post dead time).
    E0(mu) = stored mean without dead time (for lambda_eff, as peak recovery v0)."""
    Kp = K0.astype(np.int64).copy(); half = box["half"]
    for q in peaks:
        y0, x0, unit = PK.spot_pixels(q["cy"], q["cx"], q["sigma"], 1.0, half)
        sl = (slice(y0, y0 + 2 * half + 1), slice(x0, x0 + 2 * half + 1))
        rp = PK.patch_radius(y0, x0, 2 * half + 1, q["cy"], q["cx"])
        bx, _ = PK.box_and_annulus(rp, box["r_box"], box["r_in"], box["r_out"])
        mu_h = np.clip(mu_h_map[sl], 0, None)
        B_box_e = float(mu_h[bx].sum())
        frac = float(unit[bx].sum()); lam = unit * (q["pb"] * B_box_e / max(frac, 1e-12))
        Nk = None if N_known is None else N_known[sl]
        Kp[sl] = PK.inject(K0[sl], mu_h, lam, binning, tie, tau_b, rng, N_known=Nk)
        lc = lam * np.exp(-(mu_h + lam) * tau_b); mh = mu_h * np.exp(-lam * tau_b)
        q.update({"lam_box": float(lam[bx].sum()), "lam_tot": float(lam.sum()), "box_frac": frac, "B_box_e": B_box_e,
                  "lam_eff": float(epc * (E0(mh + lc) - E0(mu_h))[bx].sum()), "sl": sl, "bx": bx})
    return Kp


def evaluate_frame(Kp, K0, mask, centre_est, law, peaks, box, b_true=None, plan=None, blind=True, pf=True):
    """All methods on one injected frame. Returns (rows, info)."""
    spots = np.array([[q["cy"], q["cx"]] for q in peaks]).reshape(-1, 2)
    ex = BG.disc_mask(Kp.shape, spots, box["r_excl"])
    geo = BG.Geometry(tuple(centre_est))
    res = BG.estimate_background(Kp, mask, geo, law, exclude=ex, plan=plan)
    out = BG.integrate_spots(Kp, res, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"],
                             sigmas=np.array([q["sigma"] for q in peaks]) if pf else None, methods=METHODS, oracle=K0)
    info = {"t_total_s": res.timings["total_s"], "dispersion": res.dispersion, "n_flag": int(res.flags.sum())}
    if "AUTO" in out and "auto_p" in out["AUTO"]:
        info["auto_p"] = float(out["AUTO"]["auto_p"][0]); info["auto_use_GL"] = bool(out["AUTO"]["auto_use_GL"][0])
    if blind:
        rb = BG.estimate_background(Kp, mask, geo, law, plan=res.plan)
        ob = BG.integrate_spots(Kp, rb, spots, box["r_box"], box["r_in"], box["r_out"], box["r_excl"], methods=("G",))
        out["G_blind"] = ob["G"]
        info["t_blind_s"] = rb.timings["total_s"]
    rows = []
    for k, q in enumerate(peaks):
        bt = float(b_true[q["sl"]][q["bx"]].sum()) if b_true is not None else np.nan
        base = {kk: q[kk] for kk in ("dbin", "pb", "s", "sigma", "lam_box", "lam_tot", "box_frac", "B_box_e", "lam_eff")}
        base["b_true_box"] = bt; base["K0_box"] = float(out["O"]["B_box"][k])
        for m, d in out.items():
            rows.append({**base, "method": m, "I": d["I"][k], "sigma_I": d["sigma"][k], "sigma_plugin": d["sigma_plugin"][k],
                         "B_box": d["B_box"][k], "var_B": d["var_B"][k], "n_box": d["n_box"][k], "n_ann": d["n_ann"][k]})
    return rows, info, res
