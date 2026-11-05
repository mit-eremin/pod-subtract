"""bgsub v0: tables and decision rules (design note Sec. 1.9, amendments 3-4) for the synthetic and real runs.

Usage: .venv/bin/python scripts/bgsub_v0_tables.py --kind synth|real --set dev|test
"""
from __future__ import annotations

import argparse
import warnings
warnings.simplefilter("ignore", RuntimeWarning)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from podbg import paths
from podbg.provenance import write_sidecar

OUT = paths.SUB / "bgsub_v0"
BG_METHODS = ["O", "G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G_blind"]
SUM_METHODS = ["O", "G", "GL", "AUTO", "FLAT", "PLANE", "ROBUST", "G_blind", "G_PF"]
LAW_METHODS = ["O-LC", "G-LC", "GL-LC", "AUTO-LC", "FLAT-LC", "PLANE-LC", "ROBUST-LC", "G-ML", "AUTO-ML"]
N_BOOT = 400


def cluster_se(x, cl, rng):
    x = np.asarray(x, float); cl = np.asarray(cl)
    ok = np.isfinite(x); x = x[ok]; cl = cl[ok]
    if len(x) < 3:
        return np.nan
    u, inv = np.unique(cl, return_inverse=True)
    if len(u) < 3:
        return float(np.std(x) / np.sqrt(len(x)))
    s = np.bincount(inv, weights=x, minlength=len(u)); n = np.bincount(inv, minlength=len(u))
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    means = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.std(means))


def ratio_of_sums(num, den, cl, rng):
    num = np.asarray(num, float); den = np.asarray(den, float); cl = np.asarray(cl)
    ok = np.isfinite(num) & np.isfinite(den); num, den, cl = num[ok], den[ok], cl[ok]
    if len(num) == 0 or den.sum() <= 0:
        return np.nan, np.nan
    est = float(num.sum() / den.sum())
    u, inv = np.unique(cl, return_inverse=True)
    if len(u) < 3:
        return est, np.nan
    sn = np.bincount(inv, weights=num, minlength=len(u)); sd = np.bincount(inv, weights=den, minlength=len(u))
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    return est, float(np.std(sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-300)))


def rsd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 2 else np.nan


def summarise(R, keys, truth_bg, rng):
    """Per method x keys cell: background bias beta (mean, SE), rmse_bg, ratios, pulls."""
    rows = []
    R = R.copy()
    R["beta"] = (R.B_box - R[truth_bg]) / R[truth_bg]
    R["ratio_eff"] = R.I / R.lam_eff
    R["ratio_box"] = R.I / R.lam_box
    R["pull_eff"] = (R.I - R.lam_eff) / R.sigma_I
    R["pull_box"] = (R.I - R.lam_box) / R.sigma_I
    R["pull_plugin"] = (R.I - R.lam_eff) / R.sigma_plugin
    for k, g in R.groupby(["method"] + keys, sort=True):
        m = k[0]
        law = m in LAW_METHODS
        d = {"method": m, **dict(zip(keys, k[1:])), "n": len(g)}
        if m in BG_METHODS or m.endswith("-LC") or m.endswith("-ML"):
            # Amendment 4: the cell's mean beta is the ratio of sums sum(B - b) / sum(b) (a mean of per-peak ratios is
            # dominated by boxes with b ~ 0 at bin 16); SE by cluster bootstrap of the same ratio
            d["beta_mean"], d["beta_se"] = ratio_of_sums(g.B_box - g[truth_bg], g[truth_bg], g.cluster, rng)
            if "beta_flat_truth" in g:
                d["beta_flat_truth"] = float(np.nanmean(g.beta_flat_truth)); d["beta_flat_pred"] = float(np.nanmean(g.beta_flat_pred))
            d["beta_median"] = float(np.nanmedian(g.beta))
        oracle_noise = np.sqrt(np.nanmean(g.var_box_true)) if "var_box_true" in g else np.nan
        d["rmse_bg"] = float(np.sqrt(np.nanmean((g.B_box - g[truth_bg]) ** 2))) / oracle_noise if np.isfinite(oracle_noise) and oracle_noise > 0 else np.nan
        rt = "ratio_box" if law else "ratio_eff"
        pl = "pull_box" if law else "pull_eff"
        d["ratio_mean"] = float(np.nanmean(g[rt])); d["ratio_se"] = cluster_se(g[rt], g.cluster, rng)
        d["ratio_median"] = float(np.nanmedian(g[rt]))
        d["ratio_box_mean"] = float(np.nanmean(g.ratio_box))
        d["pull_mean"] = float(np.nanmean(g[pl])); d["pull_sd"] = float(np.nanstd(g[pl])); d["pull_rsd"] = rsd(g[pl])
        d["pull_plugin_rsd"] = rsd(g.pull_plugin) if m == "FLAT" else np.nan
        d["frac_nonpos"] = float(np.mean(g.I <= 0))
        rows.append(d)
    return pd.DataFrame(rows), R


def synth(set_name):
    rng = np.random.default_rng(0)
    R = pd.read_parquet(OUT / f"synth_{set_name}_peaks_v0.parquet")
    R["cluster"] = R.seed
    # oracle noise of the box background: sum_box g(b) in stored counts^2 (from O's rows: var of K0_box around b_true)
    o = R[R.method == "O"][["seed", "dbin", "pb", "K0_box", "b_true_box"]]
    R["var_box_true"] = np.nan
    vb = R[R.method == "O"].groupby(["binning", "dose", "dbin"]).apply(lambda g: np.mean((g.K0_box - g.b_true_box) ** 2)).rename("vbt")
    R = R.merge(vb.reset_index(), on=["binning", "dose", "dbin"], how="left"); R["var_box_true"] = R.vbt
    T, Rf = summarise(R, ["binning", "dose", "struct", "dbin"], "b_true_box", rng)
    Tp, _ = summarise(R, ["binning", "dose", "struct", "dbin", "pb"], "b_true_box", rng)
    Rf["pbg"] = np.where(Rf.pb <= 1.0, "low", np.where(Rf.pb >= 10, "high", "mid"))
    Tpull, _ = summarise(R.assign(pbg=Rf.pbg), ["binning", "pbg"], "b_true_box", rng)
    # rules
    rules = {}
    g = T[T.method == "G"]
    ok1 = (g.beta_mean.abs() <= np.maximum(0.005, 2 * g.beta_se))
    rules["R1"] = {"pass": bool(ok1.all()), "n_cells": int(len(g)), "n_fail": int((~ok1).sum()),
                   "fails": g[~ok1][["binning", "dose", "struct", "dbin", "beta_mean", "beta_se"]].round(5).to_dict("records")}
    # R1 derivation check for FLAT at D0
    Bd = pd.read_csv(OUT / f"synth_{set_name}_bands_v0.csv")
    Bd["on"] = Bd.d_on / Bd.b_mean; Bd["off"] = Bd.d_off / Bd.b_mean; Bd["raw"] = Bd.d_raw / Bd.b_mean
    Bd["est_minus_raw"] = Bd.on - Bd.raw
    tb = []
    for k, gg in Bd.groupby(["binning", "dose", "band"]):
        tb.append({"binning": k[0], "dose": k[1], "band": k[2], "b_mean": gg.b_mean.mean(), "bias_on": gg.on.mean(),
                   "bias_on_se": gg.on.std(ddof=1) / np.sqrt(len(gg)) if len(gg) > 1 else np.nan,
                   "on_minus_off": (gg.on - gg.off).mean(), "frame_noise": gg.raw.mean()})
    TB = pd.DataFrame(tb)
    ok2 = (TB.bias_on.abs() <= np.maximum(0.003, 2 * TB.bias_on_se.fillna(np.inf))) & (TB.on_minus_off.abs() <= 0.001)
    rules["R2"] = {"pass": bool(ok2.all()), "n_fail": int((~ok2).sum()), "fails": TB[~ok2].round(5).to_dict("records")}
    r3 = {}
    for m in ["G", "G-LC", "G-ML", "AUTO-LC"]:
        tt = Tpull[(Tpull.method == m) & Tpull.pbg.isin(["low", "high"])]
        okm = tt.pull_rsd.between(0.9, 1.1) & (tt.pull_mean.abs() <= 0.1)
        r3[m] = {"pass": bool(okm.all()), "cells": tt[["binning", "pbg", "pull_mean", "pull_sd", "pull_rsd"]].round(3).to_dict("records")}
    rules["R3"] = r3
    lc = Tp[(Tp.method == "G-LC") & (Tp.pb >= 1) & Tp.binning.isin([4, 8]) & Tp.dbin.isin(["D2", "D3", "D4"])]
    lcg = lc.groupby(["binning", "dose", "dbin", "pb"]).ratio_mean.mean()
    rules["R6"] = {"pass": bool((lcg.sub(1).abs() <= 0.05).all()), "n_fail": int((lcg.sub(1).abs() > 0.05).sum()),
                   "worst": {str(k): v for k, v in lcg.sub(1).abs().sort_values().tail(5).round(4).to_dict().items()}}
    F = pd.read_csv(OUT / f"synth_{set_name}_frames_v0.csv")
    rules["timing_ms"] = F.groupby("binning").t_total_s.median().mul(1e3).round(1).to_dict()
    for df, name in ((T, f"synth_{set_name}_cells_v0.csv"), (Tp, f"synth_{set_name}_cells_pb_v0.csv"), (Tpull, f"synth_{set_name}_pulls_v0.csv"), (TB, f"synth_{set_name}_trunc_v0.csv")):
        df.to_csv(OUT / name, index=False)
        write_sidecar(OUT / name, inputs=[OUT / f"synth_{set_name}_peaks_v0.parquet"], params={"set": set_name}, note="bgsub v0 synthetic tables")
    json.dump(rules, open(OUT / f"synth_{set_name}_rules_v0.json", "w"), indent=1, default=str)
    write_sidecar(OUT / f"synth_{set_name}_rules_v0.json", inputs=[OUT / f"synth_{set_name}_peaks_v0.parquet"], params={"set": set_name}, note="bgsub v0 rules")
    return T, Tp, Tpull, TB, rules


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", default="synth"); ap.add_argument("--set", default="dev")
    a = ap.parse_args()
    if a.kind == "synth":
        T, Tp, Tpull, TB, rules = synth(a.set)
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
        print(json.dumps({k: (v if k != "R1" else {kk: vv for kk, vv in v.items() if kk != "fails"}) for k, v in rules.items()}, indent=1, default=str)[:4000])


def real(set_name):
    rng = np.random.default_rng(0)
    R = pd.read_parquet(OUT / f"real_{set_name}_peaks_v0.parquet")
    R["cluster"] = R.sid + ":" + R.frame.astype(str)
    R["e"] = R.B_box - R.K0_box
    rows = []
    for k, g in R.groupby(["method", "regime", "dbin"]):
        m = k[0]; law = m in LAW_METHODS
        b, se = ratio_of_sums(g.e, g.mh_box, g.cluster, rng)
        disp = g.info_dispersion.mean()
        ex = np.nanmean(g.e ** 2) - np.nanmean(g.g_box) * disp
        rt = (g.I / g.lam_box) if law else (g.I / g.lam_eff)
        pl = ((g.I - g.lam_box) if law else (g.I - g.lam_eff)) / g.sigma_I
        rows.append({"method": m, "regime": k[1], "dbin": k[2], "n": len(g), "beta": b, "beta_se": se,
                     "rmse_bg": float(np.sqrt(max(ex, 0) / np.nanmean(g.g_box))) if m not in ("O", "O-LC") else 0.0,
                     "ratio_mean": float(np.nanmean(rt)), "ratio_median": float(np.nanmedian(rt)),
                     "ratio_box_mean": float(np.nanmean(g.I / g.lam_box)), "pull_mean": float(np.nanmean(pl)), "pull_rsd": rsd(pl),
                     "pull_sd": float(np.nanstd(pl))})
    T = pd.DataFrame(rows)
    # per P/B
    rows = []
    for k, g in R.groupby(["method", "regime", "dbin", "pb"]):
        m = k[0]; law = m in LAW_METHODS
        rt = (g.I / g.lam_box) if law else (g.I / g.lam_eff)
        rows.append({"method": m, "regime": k[1], "dbin": k[2], "pb": k[3], "n": len(g), "ratio_mean": float(np.nanmean(rt)),
                     "ratio_se": cluster_se(rt, g.cluster, rng), "ratio_median": float(np.nanmedian(rt)),
                     "ratio_box_mean": float(np.nanmean(g.I / g.lam_box)), "frac_nonpos": float(np.mean(g.I <= 0)),
                     "frac_lt3sig": float(np.mean(g.I / g.sigma_I < 3))})
    Tp = pd.DataFrame(rows)
    # pulls by P/B group
    R["pbg"] = np.where(R.pb <= 1.0, "low", np.where(R.pb >= 10, "high", "mid"))
    rows = []
    for k, g in R.groupby(["method", "regime", "pbg"]):
        m = k[0]; law = m in LAW_METHODS
        pl = ((g.I - g.lam_box) if law else (g.I - g.lam_eff)) / g.sigma_I
        rows.append({"method": m, "regime": k[1], "pbg": k[2], "pull_mean": float(np.nanmean(pl)), "pull_rsd": rsd(pl), "pull_sd": float(np.nanstd(pl))})
    Tpull = pd.DataFrame(rows)
    rules = {}
    G = T[T.method == "G"].set_index(["regime", "dbin"]); Fl = T[T.method == "FLAT"].set_index(["regime", "dbin"])
    okb = G.beta.abs() <= np.maximum(0.01, 2 * G.beta_se)
    okr = G.rmse_bg < Fl.rmse_bg
    okc = G.beta.abs() <= Fl.beta.abs() + 2 * G.beta_se
    for reg in sorted(T.regime.unique()):
        s = G.index.get_level_values(0) == reg
        rules[f"R4_{reg}"] = {"pass": bool(okb[s].mean() >= 0.9 and okr[s].all() and okc[s].all()),
                              "frac_beta_ok": float(okb[s].mean()), "rmse_better_all": bool(okr[s].all()), "beta_vs_flat_all": bool(okc[s].all()),
                              "G_beta": {k[1]: v for k, v in G.beta[s].round(4).to_dict().items()},
                              "G_beta_se": {k[1]: v for k, v in G.beta_se[s].round(4).to_dict().items()},
                              "FLAT_beta": {k[1]: v for k, v in Fl.beta[s].round(4).to_dict().items()},
                              "G_rmse": {k[1]: v for k, v in G.rmse_bg[s].round(3).to_dict().items()},
                              "FLAT_rmse": {k[1]: v for k, v in Fl.rmse_bg[s].round(3).to_dict().items()}}
    for m in ["G", "G-ML", "AUTO-ML", "G-LC"]:
        tt = Tpull[(Tpull.method == m) & Tpull.pbg.isin(["low", "high"])]
        ok = tt.pull_rsd.between(0.85, 1.15) & (tt.pull_mean.abs() <= 0.25)
        rules[f"R3_{m}"] = {"pass": bool(ok.all()), "cells": tt.round(3).to_dict("records")}
    # D1 (dev only): GL replaces G in a regime?
    d1 = {}
    for reg, g in R[R.method.isin(["G", "GL"])].groupby("regime"):
        out = {}
        for m in ("G", "GL"):
            gg = g[g.method == m]; out[m] = ratio_of_sums(gg.e, gg.mh_box, gg.cluster, rng)
        d1[reg] = {"G": out["G"], "GL": out["GL"], "GL_replaces_G": bool(abs(out["G"][0]) > max(0.01, 2 * out["G"][1]) and abs(out["GL"][0]) <= max(0.01, 2 * out["GL"][1]))}
    rules["D1"] = d1
    rules["timing_ms"] = R.groupby("binning").info_t_total_s.median().mul(1e3).round(1).to_dict()
    rules["auto_GL_frac"] = R.groupby("regime").info_auto_use_GL.mean().round(3).to_dict()
    for df, name in ((T, f"real_{set_name}_cells_v0.csv"), (Tp, f"real_{set_name}_cells_pb_v0.csv"), (Tpull, f"real_{set_name}_pulls_v0.csv")):
        df.to_csv(OUT / name, index=False)
        write_sidecar(OUT / name, inputs=[OUT / f"real_{set_name}_peaks_v0.parquet"], params={"set": set_name}, note="bgsub v0 real tables")
    json.dump(rules, open(OUT / f"real_{set_name}_rules_v0.json", "w"), indent=1, default=str)
    write_sidecar(OUT / f"real_{set_name}_rules_v0.json", inputs=[OUT / f"real_{set_name}_peaks_v0.parquet"], params={"set": set_name}, note="bgsub v0 real rules")
    return T, Tp, Tpull, rules


if __name__ == "__main__" and "--kind" in sys.argv and sys.argv[sys.argv.index("--kind") + 1] == "real":
    s = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "dev"
    T, Tp, Tpull, rules = real(s)
    print(json.dumps({k: v for k, v in rules.items() if not k.startswith("R3")}, indent=1, default=str)[:6000])
    for k, v in rules.items():
        if k.startswith("R3"):
            print(k, v["pass"], [(c["regime"], c["pbg"], c["pull_mean"], c["pull_rsd"]) for c in v["cells"]])
