"""bgsub v2.1 tables and pre-registered rules (design note v2.1 Sec. 2.4) for the synthetic and real runs; v2 and v2.1
paired on identical frames; the v2 rows of this run are compared with the published v2 rows (identity check).

Usage: .venv/bin/python scripts/bgsub_v2_1_tables.py --kind synth|real --set dev|test|confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bgsub_v0_tables as T0  # noqa: E402
from podbg import paths  # noqa: E402
from podbg.provenance import write_sidecar  # noqa: E402

warnings.simplefilter("ignore")
OUT = paths.SUB / "bgsub_v2_1"
OUT_V2 = paths.SUB / "bgsub_v2"
COV_LO, COV_HI = 0.63, 0.73     # R3 v2.1: coverage of the 68.3 % interval
Z_R1 = 3.2          # R1: two-sided Bonferroni over 60 cells (as v1)
Z_R2 = 3.1          # R2 v2: two-sided Bonferroni over 24 cells at family 0.05
LAWM = lambda m: m.endswith("-ML") or m.endswith("-LC")  # noqa: E731
METH = {"v2": ("v2:G", "v2:AUTO"), "v2.1": ("v2.1:G", "v2.1:AUTO")}


def rsd(x):
    return T0.rsd(x)


def pull_cols(g, m):
    """Pulls and coverage.  v2.1 rule coverage (R3): the confidence set for the box intensity contains the truth,
    lam_box in [ci_lam_lo, ci_lam_hi] (summation) or [ci_lo, ci_hi] (ML); the E-image interval of the summation against
    lam_eff (the v2 statistic) is reported as cov68_E."""
    tr = g.lam_box if LAWM(m) else g.lam_eff
    wald = (g.I - tr) / g.sigma_I
    at = (g.I - tr) / g.sigma_at if "sigma_at" in g and g.sigma_at.notna().any() else pd.Series(np.nan, index=g.index)
    covE = ((g.ci_lo <= tr) & (tr <= g.ci_hi)).astype(float).where(g.ci_lo.notna() | g.I.notna()) if "ci_lo" in g else pd.Series(np.nan, index=g.index)
    if "ci_lo" in g and g.ci_lo.isna().all():
        covE = pd.Series(np.nan, index=g.index)
    if not LAWM(m) and "ci_lam_lo" in g and g.ci_lam_lo.notna().any():
        cov = ((g.ci_lam_lo <= g.lam_box) & (g.lam_box <= g.ci_lam_hi)).astype(float)
    else:
        cov = covE
    # degenerate injections (lambda_box = 0 on a zero background: 2 of 48,930 real spots) give 0 / 0: dropped
    return wald.replace([np.inf, -np.inf], np.nan), at.replace([np.inf, -np.inf], np.nan), cov, covE


def v2_identity(R, name, key):
    """The v2 rows of this run against the published v2 rows (same frames => identical numbers)."""
    p = OUT_V2 / name
    if not p.exists():
        return {"published": False}
    P2 = pd.read_parquet(p)
    out = {"published": True}
    for m in ("v2:G", "v2:AUTO", "v2:GL", "v2:G-ML", "v2:AUTO-ML", "v2:FLAT" if "v2:FLAT" in set(R.method) else "v2:G"):
        a = R[R.method == m]; b = P2[P2.method == m]
        if not len(b):
            continue
        cols = key + ["dbin", "pb"] + (["rep", "row", "frame"] if "frame" in a else [])
        j = a.merge(b, on=cols + (["sid"] if "sid" in a else []), suffixes=("", "_pub"))
        out[m] = {"n_this": len(a), "n_pub": len(b), "n_matched": len(j), "max_abs_dB": float((j.B_box - j.B_box_pub).abs().max()),
                  "max_abs_dI": float((j.I - j.I_pub).abs().max())}
    return out


def synth(set_name):
    rng = np.random.default_rng(0)
    R = pd.read_parquet(OUT / f"synth_{set_name}_peaks.parquet")
    R["cluster"] = R.seed
    R["e"] = R.B_box - R.b_true_box
    rows = []
    for k, g in R.groupby(["method", "grid", "binning", "dose", "struct", "dbin"]):
        b, se = T0.ratio_of_sums(g.e, g.b_true_box, g.cluster, rng)
        noise = np.nanmean(g.g_true_box)
        rows.append({"method": k[0], "grid": k[1], "binning": k[2], "dose": k[3], "struct": k[4], "dbin": k[5], "n": len(g),
                     "beta": b, "beta_se": se, "rmse_bg": float(np.sqrt(np.nanmean(g.e ** 2) / noise)) if noise > 0 else np.nan,
                     "w_mean": float(np.nanmean(g.auto_w)) if "auto_w" in g and g.auto_w.notna().any() else np.nan})
    C = pd.DataFrame(rows)
    rules = {}
    for m in ("v2.1:G", "v2.1:AUTO", "v2:G", "v2:AUTO"):
        c = C[(C.method == m) & (C.grid == "lacey_utc")]
        ok = c.beta.abs() <= np.maximum(0.005, Z_R1 * c.beta_se)
        rules[f"R1_{m}"] = {"pass": bool(ok.all() and len(c) == 60), "n_cells": int(len(c)), "n_fail": int((~ok).sum()),
                            "n_fail_at_2SE": int((~(c.beta.abs() <= np.maximum(0.005, 2 * c.beta_se))).sum()),
                            "max_abs_beta": float(c.beta.abs().max()), "median_se": float(c.beta_se.median()),
                            "fails": c[~ok][["binning", "dose", "struct", "dbin", "beta", "beta_se"]].round(5).to_dict("records")}
    # R2 v2: paired estimator-minus-frame bias (expectation = bias against E[K]); v1 form reported
    Bd = pd.read_csv(OUT / f"synth_{set_name}_bands.csv")
    Bd = Bd[Bd.grid == "lacey_utc"]
    tb = []
    for k, gg in Bd.groupby(["ver", "binning", "dose", "band"]):
        pair = gg.d_pair / gg.b_mean; on = gg.d_on / gg.b_mean; raw = gg.d_raw / gg.b_mean; om = (gg.d_on - gg.d_off) / gg.b_mean
        se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan  # noqa: E731
        tb.append({"ver": k[0], "binning": k[1], "dose": k[2], "band": k[3], "n": len(gg), "b_mean": gg.b_mean.mean(),
                   "bias_pair": pair.mean(), "bias_pair_se": se(pair), "bias_v1form": on.mean(), "bias_v1form_se": se(on),
                   "raw": raw.mean(), "raw_se": se(raw), "on_minus_off": om.mean()})
    TB = pd.DataFrame(tb)
    for ver in ("v2", "v2.1"):
        t = TB[TB.ver == ver]
        ok = (t.bias_pair.abs() <= np.maximum(0.003, Z_R2 * t.bias_pair_se)) & (t.on_minus_off.abs() <= 0.001)
        okv1 = (t.bias_v1form.abs() <= np.maximum(0.003, 2 * t.bias_v1form_se)) & (t.on_minus_off.abs() <= 0.001)
        rules[f"R2_{ver}"] = {"pass": bool(ok.all() and len(t) == 24), "n_cells": int(len(t)), "n_fail": int((~ok).sum()),
                              "fails": t[~ok].round(5).to_dict("records"), "v1_form_n_fail": int((~okv1).sum()),
                              "max_abs_pair": float(t.bias_pair.abs().max())}
    # R3 v2: pulls with sigma at the TRUE lambda (= the inverted-interval statistic), informative spots
    lac = R[R.grid == "lacey_utc"]
    lm = lac[lac.method == "v2.1:AUTO-ML"][["seed", "dbin", "pb", "lam_min5"]].rename(columns={"lam_min5": "lm5"})
    L = lac.merge(lm, on=["seed", "dbin", "pb"], how="left")
    L["pbg"] = np.where(L.pb <= 1.0, "low", np.where(L.pb >= 10, "high", "mid"))
    L["informative"] = L.lam_box >= L.lm5
    r3 = []
    for m in ("v2.1:AUTO", "v2.1:AUTO-ML", "v2.1:G", "v2.1:G-ML", "v2:AUTO", "v2:AUTO-ML", "v2:G", "v2:G-ML"):
        for sel_name, sel in (("informative", L.informative), ("all", L.informative | ~L.informative)):
            for k, g in L[(L.method == m) & sel].groupby(["binning", "pbg"]):
                wald, at, cov, covE = pull_cols(g, m)
                r3.append({"method": m, "subset": sel_name, "binning": k[0], "pbg": k[1], "n": len(g),
                           "wald_mean": float(np.nanmean(wald)), "wald_rsd": rsd(wald), "at_mean": float(np.nanmean(at)) if at.notna().any() else np.nan,
                           "at_rsd": rsd(at) if at.notna().any() else np.nan, "cov68": float(np.nanmean(cov)) if cov.notna().any() else np.nan,
                           "cov68_E": float(np.nanmean(covE)) if covE.notna().any() else np.nan})
    P = pd.DataFrame(r3)
    for m in ("v2.1:AUTO", "v2.1:AUTO-ML", "v2:AUTO", "v2:AUTO-ML"):
        t = P[(P.method == m) & (P.subset == "informative") & P.pbg.isin(["low", "high"])]
        okm = t.at_rsd.between(0.9, 1.1) & (t.at_mean.abs() <= 0.1)
        okc = t.cov68.between(COV_LO, COV_HI)
        rules[f"R3_{m}"] = {"pass": bool(okm.all() and okc.all() and len(t) == 4), "pass_pulls": bool(okm.all() and len(t) == 4),
                            "pass_coverage": bool(okc.all() and len(t) == 4), "cells": t.round(3).to_dict("records")}
    # R6 (reported)
    rr = []
    for k, g in L[L.method.isin(["v2:G-ML", "v2.1:G-ML"])].groupby(["method", "binning", "dose", "dbin", "pb"]):
        gi = g[g.informative]
        rr.append({"method": k[0], "binning": k[1], "dose": k[2], "dbin": k[3], "pb": k[4], "n": len(g), "ratio": float(g.I.sum() / g.lam_box.sum()),
                   "ratio_se": T0.cluster_se(g.I / g.lam_box, g.cluster, rng), "n_inf": len(gi),
                   "ratio_inf": float(gi.I.sum() / gi.lam_box.sum()) if len(gi) else np.nan})
    R6 = pd.DataFrame(rr)
    a = R6[(R6.method == "v2.1:G-ML") & (R6.binning == 8) & (R6.dbin == "D2") & (R6.pb == 1.0)]
    rules["R6a"] = {"pass": bool((a.ratio.sub(1).abs() <= 0.05).all()), "cells": a.round(3).to_dict("records")}
    bb = R6[(R6.method == "v2.1:G-ML") & (R6.pb >= 1) & R6.dbin.isin(["D1", "D2", "D3", "D4"]) & (R6.n_inf >= 20)]
    rules["R6b"] = {"pass": bool((bb.ratio_inf.sub(1).abs() <= 0.05).all()), "n_cells": int(len(bb)),
                    "n_fail": int((bb.ratio_inf.sub(1).abs() > 0.05).sum())}
    F = pd.read_csv(OUT / f"synth_{set_name}_frames.csv")
    rules["n_frames"] = int(len(F))
    rules["timing_ms_fit"] = {v: F.groupby("binning")[f"{v}_t_fit_s"].median().mul(1e3).round(1).to_dict() for v in ("v2", "v2.1")}
    rules["auto_w"] = F.groupby(["grid", "struct"])[["v2_auto_w_mean", "v2.1_auto_w_mean"]].mean().round(3).reset_index().to_dict("records")
    rules["recentred_frac"] = {v: float(F[f"{v}_recentred"].mean()) for v in ("v2", "v2.1")}
    rules["v2_identity"] = v2_identity(R, f"synth_{set_name}_peaks.parquet", ["seed"])
    for df, name in ((C, "cells"), (TB, "trunc"), (P, "pulls"), (R6, "ml")):
        p = OUT / f"synth_{set_name}_{name}.csv"; df.to_csv(p, index=False)
        write_sidecar(p, inputs=[OUT / f"synth_{set_name}_peaks.parquet"], params={"set": set_name}, note="bgsub v2.1 synthetic tables")
    p = OUT / f"synth_{set_name}_rules.json"
    json.dump(rules, open(p, "w"), indent=1, default=str)
    write_sidecar(p, inputs=[OUT / f"synth_{set_name}_peaks.parquet"], params={"set": set_name, "z_r1": Z_R1, "z_r2": Z_R2}, note="bgsub v2.1 synthetic rules")
    return rules


def real_cells(R, rng, sel_name):
    rows = []
    for k, g in R.groupby(["method", "regime", "dbin"]):
        b, se = T0.ratio_of_sums(g.e, g.mh_box, g.cluster, rng)
        ex = np.nanmean(g.e ** 2) - np.nanmean(g.g_box) * g.info_v1_dispersion.mean()
        rows.append({"subset": sel_name, "method": k[0], "regime": k[1], "dbin": k[2], "n": len(g), "beta": b, "beta_se": se,
                     "rmse_bg": float(np.sqrt(max(ex, 0) / np.nanmean(g.g_box))) if not k[0].endswith(":O") else 0.0,
                     "w_mean": float(np.nanmean(g.auto_w)) if "auto_w" in g and g.auto_w.notna().any() else np.nan})
    return pd.DataFrame(rows)


def r4(C, gm, flat="v2:FLAT"):
    G = C[C.method == gm].set_index(["regime", "dbin"]); Fl = C[C.method == flat].set_index(["regime", "dbin"])
    okb = G.beta.abs() <= np.maximum(0.01, 2 * G.beta_se); okr = G.rmse_bg < Fl.rmse_bg; okc = G.beta.abs() <= Fl.beta.abs() + 2 * G.beta_se
    out = {}
    for reg in sorted(C.regime.unique()):
        s = G.index.get_level_values(0) == reg
        out[reg] = {"pass": bool(okb[s].mean() >= 0.9 and okr[s].all() and okc[s].all()), "frac_beta_ok": float(okb[s].mean()),
                    "rmse_better_all": bool(okr[s].all()), "beta_vs_flat_all": bool(okc[s].all()),
                    "rmse": {k[1]: v for k, v in G.rmse_bg[s].round(3).to_dict().items()},
                    "FLAT_rmse": {k[1]: v for k, v in Fl.rmse_bg[s].round(3).to_dict().items()},
                    "beta": {k[1]: v for k, v in G.beta[s].round(4).to_dict().items()},
                    "beta_se": {k[1]: v for k, v in G.beta_se[s].round(4).to_dict().items()}}
    return out


def real(set_name):
    rng = np.random.default_rng(0)
    R = pd.read_parquet(OUT / f"real_{set_name}_peaks.parquet")
    R["cluster"] = R.sid + ":" + R.frame.astype(str)
    R["e"] = R.B_box - R.K0_box
    if "info_v1_dispersion" not in R:
        R["info_v1_dispersion"] = R["info_v2_dispersion"]          # the noise floor of rmse_bg uses the v2 dispersion
    R["sb"] = np.where(R.s < 0.19, "D0a", "D0b")
    C = pd.concat([real_cells(R[~R.overlap], rng, "no_overlap"), real_cells(R, rng, "all")])
    rules = {"overlap_frac": R[R.method == "v2:G"].groupby("regime").overlap.mean().round(4).to_dict(),
             "n_spots_per_method": R[R.method == "v2:G"].groupby("regime").size().to_dict(),
             "n_sessions": int(R.sid.nunique())}
    for sub in ("no_overlap", "all"):
        Cs = C[C.subset == sub]
        for gm in ("v2.1:AUTO", "v2.1:G", "v2:AUTO", "v2:G"):
            for reg, v in r4(Cs, gm).items():
                rules[f"R4_{sub}_{gm}_{reg}"] = v
    rows = []
    for k, g in R[(R.dbin == "D0")].groupby(["overlap", "method", "regime", "sb"]):
        b, se = T0.ratio_of_sums(g.e, g.mh_box, g.cluster, rng)
        ex = np.nanmean(g.e ** 2) - np.nanmean(g.g_box) * g.info_v1_dispersion.mean()
        rows.append({"overlap": k[0], "method": k[1], "regime": k[2], "sbin": k[3], "n": len(g), "beta": b, "beta_se": se,
                     "rmse_bg": float(np.sqrt(max(ex, 0) / np.nanmean(g.g_box)))})
    D0 = pd.DataFrame(rows)
    # per session D0 (diagnostic)
    rows = []
    for k, g in R[(R.dbin == "D0") & ~R.overlap].groupby(["sid", "regime", "method"]):
        ex = np.nanmean(g.e ** 2) - np.nanmean(g.g_box) * g.info_v1_dispersion.mean()
        rows.append({"sid": k[0], "regime": k[1], "method": k[2], "n": len(g), "rmse_bg": float(np.sqrt(max(ex, 0) / np.nanmean(g.g_box)))})
    DS = pd.DataFrame(rows)
    R["pbg"] = np.where(R.pb <= 1.0, "low", np.where(R.pb >= 10, "high", "mid"))
    lm = R[R.method == "v2.1:AUTO-ML"][["cluster", "rep", "dbin", "pb", "lam_min5"]].rename(columns={"lam_min5": "lm5"})
    R = R.merge(lm, on=["cluster", "rep", "dbin", "pb"], how="left")
    R["informative"] = R.lam_box >= R.lm5
    pr = []
    for sel_name, sel in (("informative", R.informative), ("all", R.informative | ~R.informative)):
        for k, g in R[sel & ~R.method.str.endswith(":O")].groupby(["method", "regime", "pbg"]):
            wald, at, cov, covE = pull_cols(g, k[0])
            pr.append({"subset": sel_name, "method": k[0], "regime": k[1], "pbg": k[2], "n": len(g),
                       "wald_mean": float(np.nanmean(wald)), "wald_median": float(np.nanmedian(wald)), "wald_rsd": rsd(wald),
                       "at_mean": float(np.nanmean(at)) if at.notna().any() else np.nan,
                       "at_rsd": rsd(at) if at.notna().any() else np.nan, "cov68": float(np.nanmean(cov)) if cov.notna().any() else np.nan,
                       "cov68_E": float(np.nanmean(covE)) if covE.notna().any() else np.nan})
    PR = pd.DataFrame(pr)
    rows = []
    for k, g in R.groupby(["method", "regime", "dbin", "pb"]):
        tr = g.lam_box if LAWM(k[0]) else g.lam_eff
        rows.append({"method": k[0], "regime": k[1], "dbin": k[2], "pb": k[3], "n": len(g), "ratio_sum": float(g.I.sum() / tr.sum()),
                     "ratio_se": T0.cluster_se(g.I / tr, g.cluster, rng)})
    CP = pd.DataFrame(rows)
    rules["timing_ms"] = {v: float(R[f"info_{v}_t_fit_s"].median() * 1e3) for v in ("v2", "v2.1")}
    rules["recentred_frac"] = {v: R[R.method == "v2:G"].groupby("regime")[f"info_{v}_recentred"].mean().round(4).to_dict() for v in ("v2", "v2.1")}
    rules["w_mean"] = {v: R[R.method == f"{v}:AUTO"].groupby("regime").auto_w.mean().round(3).to_dict() for v in ("v2", "v2.1")}
    rules["v2_identity"] = v2_identity(R, f"real_{set_name}_peaks.parquet", ["sid"])
    for df, name in ((C, "cells"), (D0, "d0"), (DS, "d0_by_session"), (PR, "pulls"), (CP, "cells_pb")):
        p = OUT / f"real_{set_name}_{name}.csv"; df.to_csv(p, index=False)
        write_sidecar(p, inputs=[OUT / f"real_{set_name}_peaks.parquet"], params={"set": set_name}, note="bgsub v2.1 real tables")
    p = OUT / f"real_{set_name}_rules.json"
    json.dump(rules, open(p, "w"), indent=1, default=str)
    write_sidecar(p, inputs=[OUT / f"real_{set_name}_peaks.parquet"], params={"set": set_name}, note="bgsub v2.1 real rules")
    return rules


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", default="synth"); ap.add_argument("--set", default="dev")
    a = ap.parse_args()
    r = synth(a.set) if a.kind == "synth" else real(a.set)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("fails", "cells", "beta", "beta_se")} if isinstance(v, dict) else v)
                      for k, v in r.items()}, indent=1, default=str)[:9000])
