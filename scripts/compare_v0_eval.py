"""compare v0, evaluation: one common per-peak table measured on the ORIGINAL frames, cross-method matching, radial
residuals of each background on spot-free pixels, and the figures.  Inputs are the outputs of compare_v0.py
(select, bgsub, pyreyes, xds).  Run:  .venv/bin/python scripts/compare_v0_eval.py [--versions v2.1 v1]

Per peak (any method), in a disc of radius RB (bin 8: 3 px, bin 4: 5 px) about the method's centroid:
  S_raw        sum of the stored counts of the original frame (offset included as stored)
  B_<m>        background under the disc by estimator m: bgsub v2.1 / v1 map; xds = BKGINIT/1000 (bin 8: the table of
               that frame's own single-frame INIT; bin 4: the sweep table rescaled per frame by RATIO = sum(frame)/
               sum(BKGINIT) over pixels outside every detected spot, COLSPOT's rule); flat = median of the annulus
               [RB+2, RB+6] x n_disc (the XDS/DIALS-style local flat background)
  I_<m>        S_raw - B_<m>;  frac_<m> = B_<m> / S_raw
  pyREyes has no background: dog_sum = sum of its DoG image in the disc (a band-pass response, not a subtraction).
bgsub "peaks" = 8-connected components of its count-law outlier flags (area >= the same minimum pixels as pyREyes:
3 at bin 8, 5 at bin 4; spots AND hot pixels / zingers, which the area cut removes), centroid weighted by (K - B)+.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bgsub.config import out_dir
from bgsub.vendor.provenance import write_sidecar
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_v0 import OUT, FR, MK, read_cbf_byte_offset, law_for

FIG = Path(__file__).resolve().parents[1] / "notes" / "figures" / "compare_v0"
RB = {8: 3, 4: 5}; MINPX = {8: 3, 4: 5}; MATCH = {8: 3.0, 4: 4.0}
METHODS = ["bgsub", "pyreyes", "xds"]; EXTRA = ["bgsub_flags"]
COL = {"bgsub": "#d95f02", "pyreyes": "#1b9e77", "xds": "#7570b3", "xds_ref": "#e7298a", "bgsub_flags": "#fdbf6f"}
Z_DET = 5.0   # bgsub detector: 3x3 box excess over the background >= Z_DET sigma of the count-law variance


def disc_offsets(rb):
    yy, xx = np.mgrid[-rb:rb + 1, -rb:rb + 1]; d = np.hypot(yy, xx) <= rb + 0.01
    return yy[d], xx[d]


def ann_offsets(r0, r1):
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]; r = np.hypot(yy, xx); d = (r >= r0) & (r <= r1)
    return yy[d], xx[d]


def sums_at(img, rows, cols, oy, ox, fn=np.sum):
    H, W = img.shape; out = np.full(len(rows), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        y = np.clip(int(round(r)) + oy, 0, H - 1); x = np.clip(int(round(c)) + ox, 0, W - 1)
        v = img[y, x]; out[i] = fn(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
    return out


def bgsub_peaks(frame, bg, flags, mask, minpx):
    good = flags & ~mask & np.isfinite(bg)
    lab, n = ndi.label(good, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=["row", "col", "area"])
    w = np.clip(frame - np.nan_to_num(bg), 0, None) + 1e-6
    idx = np.arange(1, n + 1); area = ndi.sum(good, lab, idx)
    cm = ndi.center_of_mass(w, lab, idx)
    df = pd.DataFrame({"row": [c[0] for c in cm], "col": [c[1] for c in cm], "area": area})
    return df[df.area >= minpx].reset_index(drop=True)


def bgsub_detect(frame, bg, var, mask, minpx, flags, z=Z_DET, min_excess=3.0):
    """bgsub's own count-law outlier flags (exact one-sided tails per pixel and per 3 x 3 window at alpha 1e-6; the
    exclusion mask of the fit), as 8-connected components with >= minpx pixels, KEPT as a peak when the component's net
    excess sum(K - B) is >= z sigma of the summed count-law variance at B and >= min_excess counts (a Gaussian test on
    the component sum, which is fair once several counts are summed; single stray counts at ~1e-4 counts/px are not
    5-sigma events under the exact tail and are dropped by min_excess).  Centroid weighted by (K - B)+."""
    good = flags & ~mask & np.isfinite(bg) & np.isfinite(var)
    lab, n = ndi.label(good, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=["row", "col", "area", "excess", "z"])
    idx = np.arange(1, n + 1); area = ndi.sum(good, lab, idx)
    d = np.where(good, frame - bg, 0.0); ex = ndi.sum(d, lab, idx); vs = ndi.sum(np.where(good, var, 0.0), lab, idx)
    zc = ex / np.sqrt(np.maximum(vs, 1e-9))
    cm = ndi.center_of_mass(np.clip(d, 0, None) + 1e-6, lab, idx)
    df = pd.DataFrame({"row": [c[0] for c in cm], "col": [c[1] for c in cm], "area": area, "excess": ex, "z": zc})
    return df[(df.area >= minpx) & (df.z >= z) & (df.excess >= min_excess)].reset_index(drop=True)


def match(a, b, tol):
    """For each row of a, the nearest row of b within tol (index or -1)."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), -1)
    from scipy.spatial import cKDTree
    d, j = cKDTree(b[["row", "col"]].to_numpy()).query(a[["row", "col"]].to_numpy(), distance_upper_bound=tol)
    return np.where(np.isfinite(d), j, -1)


def main(versions):
    FIG.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(OUT / "manifest.csv")
    pk_py = pd.concat([pd.read_csv(p) for p in glob.glob(str(OUT / "peaks_pyreyes*.csv"))]).drop_duplicates(["id", "label"])
    pk_xds = pd.concat([pd.read_csv(p) for p in glob.glob(str(OUT / "peaks_xds*.csv")) if Path(p).stat().st_size > 10]) if glob.glob(str(OUT / "peaks_xds*.csv")) else pd.DataFrame(columns=["id", "row", "col", "I_xds"])
    if "setting" in pk_xds.columns:
        pk_xds = pk_xds[pk_xds.setting == "default"]
    ref = pd.read_csv(OUT / "xds_spots_mov11.csv")
    rows, resid, counts = [], [], []
    for r in man.itertuples():
        frame = np.load(FR / f"{r.id}.npz")["frame"].astype(float); mask = np.load(MK / f"{r.mask}.npz")["mask"]
        rb, minpx, tol = RB[r.binning], MINPX[r.binning], MATCH[r.binning]
        oy, ox = disc_offsets(rb); ay, ax = ann_offsets(rb + 2, rb + 6); nd = len(oy)
        off = float(law_for(r).offset)          # bgsub returns bg WITHOUT the declared zero level (core.py:556); add it back in stored counts
        bgs = {v: np.load(OUT / "bgsub" / f"{r.id}_{v}.npz") for v in versions}
        bgmap = {v: bgs[v]["bg"].astype(float) + off for v in versions}
        bg21 = bgmap[versions[0]]; flags = bgs[versions[0]]["flags"]
        law = law_for(r); var21 = law.g(np.clip(bg21 - off, 0, None))   # count-law variance of the stored value at the background
        dog = np.load(OUT / "pyreyes" / f"{r.id}.npz")["dog"].astype(float)
        # XDS background table
        xd = OUT / "xds" / (f"pseudo_{r.sid}_default" if r.binning == 8 else "mov11_sweep_default")   # the tables behind the XDS spots
        bkx = None
        if (xd / "BKGINIT.cbf").exists():
            t = read_cbf_byte_offset(xd / "BKGINIT.cbf").astype(float); bkx = np.where(t >= 0, t / 1000.0, np.nan)
        # peaks per method
        P = {"bgsub": bgsub_detect(frame, bg21, var21, mask, minpx, flags), "bgsub_flags": bgsub_peaks(frame, bg21, flags, mask, minpx),
             "pyreyes": pk_py[pk_py.id == r.id][["row", "col", "area"]].reset_index(drop=True),
             "xds": pk_xds[pk_xds.id == r.id][["row", "col", "I_xds"]].reset_index(drop=True) if len(pk_xds) else pd.DataFrame(columns=["row", "col", "I_xds"])}
        # XDS per-frame RATIO (bin 4): outside every detected spot
        ratio = 1.0
        if bkx is not None:                       # COLSPOT's per-frame scale: RATIO = sum(frame) / sum(BKGINIT) over background pixels
            sp = np.zeros(frame.shape, bool)
            for m in METHODS:
                for rr, cc in zip(P[m].row, P[m].col):
                    y = np.clip(int(round(rr)) + oy, 0, frame.shape[0] - 1); x = np.clip(int(round(cc)) + ox, 0, frame.shape[1] - 1); sp[y, x] = True
            ok = ~sp & ~mask & np.isfinite(bkx) & (bkx > 0); ratio = float(frame[ok].sum() / bkx[ok].sum())
            bkx = bkx * ratio
        for m in METHODS + EXTRA:
            df = P[m].copy(); df.insert(0, "method", m); df.insert(0, "id", r.id)
            df["binning"] = r.binning; df["quality"] = r.quality
            df["r_px"] = np.hypot(df.row - r.cy, df.col - r.cx)
            df["S_raw"] = sums_at(frame, df.row, df.col, oy, ox)
            df["max_px"] = sums_at(frame, df.row, df.col, oy, ox, np.max)
            for v in versions:
                df[f"B_bgsub_{v}"] = sums_at(bgmap[v], df.row, df.col, oy, ox)
            df["B_flat"] = sums_at(frame, df.row, df.col, ay, ax, np.mean) * nd          # annulus MEAN (XDS / DIALS-style background pixels)
            df["B_flat_median"] = sums_at(frame, df.row, df.col, ay, ax, np.median) * nd  # the median is 0 at sparse counts
            df["B_xds"] = sums_at(bkx, df.row, df.col, oy, ox) if bkx is not None else np.nan
            df["dog_sum"] = sums_at(dog, df.row, df.col, oy, ox)
            df["on_mask"] = sums_at(mask.astype(float), df.row, df.col, oy, ox) > 0
            for b in [f"B_bgsub_{v}" for v in versions] + ["B_flat", "B_flat_median", "B_xds"]:
                df["I" + b[1:]] = df.S_raw - df[b]; df["frac" + b[1:]] = df[b] / df.S_raw
            for m2 in METHODS:
                if m2 != m:
                    df[f"in_{m2}"] = match(df, P[m2], tol) >= 0
            if r.binning == 4:
                rf = ref[(ref.frame - r.frame).abs() <= 1]; df["in_xds_ref"] = match(df, rf, tol) >= 0
            rows.append(df)
            counts.append({"id": r.id, "binning": r.binning, "quality": r.quality, "method": m, "n": len(df), "n_on_mask": int(df.on_mask.sum())})
        if r.binning == 4:
            rf = ref[ref.frame == r.frame]; counts.append({"id": r.id, "binning": 4, "quality": r.quality, "method": "xds_ref", "n": len(rf), "n_on_mask": 0})
        # radial residuals of each background on spot-free pixels (outside every method's discs and the flags)
        sp = flags.copy()
        for m in METHODS:
            for rr, cc in zip(P[m].row, P[m].col):
                y = np.clip(int(round(rr)) + oy, 0, frame.shape[0] - 1); x = np.clip(int(round(cc)) + ox, 0, frame.shape[1] - 1); sp[y, x] = True
        yy, xx = np.indices(frame.shape); rr_ = np.hypot(yy - r.cy, xx - r.cx); band = (rr_ // 32).astype(int)
        ok = ~sp & ~mask & np.isfinite(bg21)
        for name, B in [(f"bgsub_{versions[0]}", bg21), ("xds", bkx), ("flat_annulus", None)]:
            if B is None and name == "flat_annulus":
                okk = (~sp & ~mask).astype(float)                      # local MEAN of spot-free pixels, same footprint as the annulus
                B = ndi.uniform_filter(np.where(okk > 0, frame, 0.0), 2 * (rb + 6) + 1) / np.maximum(ndi.uniform_filter(okk, 2 * (rb + 6) + 1), 1e-6)
            if B is None:
                continue
            okb = ok & np.isfinite(B)
            for b_ in np.unique(band[okb]):
                s = okb & (band == b_)
                if s.sum() < 500:
                    continue
                mK = float(frame[s].mean()); mR = float((frame - B)[s].mean()); mK0 = mK - off      # counts above the declared zero level
                resid.append({"id": r.id, "binning": r.binning, "quality": r.quality, "bg": name, "r_lo": int(b_ * 32), "n": int(s.sum()),
                              "mean_K": mK, "mean_B": float(B[s].mean()), "mean_resid": mR,
                              "rel_resid": (mR / mK0) if mK0 > 0.02 else np.nan})       # relative to the counts above zero; undefined where they vanish
        print(r.id, {m: len(P[m]) for m in METHODS}, f"xds ratio {ratio:.3f}", flush=True)
    peaks = pd.concat(rows, ignore_index=True); peaks.to_csv(OUT / "peaks_all.csv", index=False)
    cnt = pd.DataFrame(counts); cnt.to_csv(OUT / "peak_counts.csv", index=False)
    res = pd.DataFrame(resid); res.to_csv(OUT / "radial_residuals.csv", index=False)
    for f in ("peaks_all.csv", "peak_counts.csv", "radial_residuals.csv"):
        write_sidecar(OUT / f, inputs=[str(OUT / "manifest.csv")], params={"RB": RB, "MINPX": MINPX, "MATCH": MATCH, "versions": versions})
    figures(man, peaks, cnt, res, versions)


def figures(man, peaks, cnt, res, versions):
    v = versions[0]
    # 1. counts per frame
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, b in zip(axes, (8, 4)):
        c = cnt[cnt.binning == b].pivot_table(index="id", columns="method", values="n").fillna(0); c = c[[m for m in ["bgsub", "bgsub_flags", "pyreyes", "xds", "xds_ref"] if m in c.columns]]
        c.plot.bar(ax=ax, color=[COL[m] for m in c.columns], width=0.8); ax.set_title(f"peaks found per frame, bin {b} ({'1024' if b == 8 else '2048'}^2)"); ax.set_ylabel("peaks")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout(); fig.savefig(FIG / "peak_counts.png", dpi=130); plt.close(fig)
    # 2. intensity distributions (net, bgsub background) per method, and raw
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, b in zip(axes, (8, 4)):
        for m in METHODS:
            d = peaks[(peaks.binning == b) & (peaks.method == m) & ~peaks.on_mask]
            if len(d):
                x = np.sort(np.clip(d[f"I_bgsub_{v}"], 0.5, None)); ax.step(x, np.arange(1, len(x) + 1), color=COL[m], label=f"{m} (n={len(d)})")
        ax.set_xscale("log"); ax.set_xlabel(f"net intensity in the disc, S_raw - B_bgsub_{v} (stored counts)"); ax.set_ylabel("cumulative peaks"); ax.set_title(f"bin {b}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "intensity_cdf.png", dpi=130); plt.close(fig)
    # 3. background fraction vs raw sum, by background estimator (on the union of peaks)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, b in zip(axes, (8, 4)):
        d = peaks[(peaks.binning == b) & ~peaks.on_mask]
        for name, col in [(f"frac_bgsub_{v}", COL["bgsub"]), ("frac_flat", "#666666"), ("frac_xds", COL["xds"])]:
            if name in d and d[name].notna().any():
                ax.scatter(d.S_raw, d[name], s=5, alpha=0.4, color=col, label=name.replace("frac_", "B/S_raw: "))
        ax.set_xscale("log"); ax.set_ylim(-0.1, 1.3); ax.axhline(1, color="k", lw=0.5); ax.set_xlabel("S_raw (disc sum, stored counts)"); ax.set_ylabel("background fraction of the disc"); ax.set_title(f"bin {b}: how much of each peak is background"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "background_fraction.png", dpi=130); plt.close(fig)
    # 4. radial residuals
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, b in zip(axes, (8, 4)):
        d = res[res.binning == b]
        for name, col in [(f"bgsub_{v}", COL["bgsub"]), ("xds", COL["xds"]), ("flat_annulus", "#666666")]:
            g = d[d.bg == name].groupby("r_lo").rel_resid
            if len(g):
                ax.errorbar(g.mean().index + 16, g.mean(), yerr=g.std() / np.sqrt(g.count()), color=col, label=name, lw=1.2, capsize=2)
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("radius (px)"); ax.set_ylabel("mean (K - B) / mean (K - zero), spot-free pixels"); ax.set_title(f"bin {b}: background residual by radius (frames averaged; flat = local mean of the same pixels, circular by construction)"); ax.legend(fontsize=8); ax.set_ylim(-0.3, 0.3)
    fig.tight_layout(); fig.savefig(FIG / "radial_residuals.png", dpi=130); plt.close(fig)
    # 5. overlays: one Good bin-8 frame, one bin-4 frame: original image with every method's peaks
    ex = [man[(man.binning == 8) & (man.quality == "Good")].iloc[0].id, man[man.binning == 4].iloc[0].id]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, fid in zip(axes, ex):
        frame = np.load(FR / f"{fid}.npz")["frame"].astype(float); r = man[man.id == fid].iloc[0]
        ax.imshow(np.log1p(frame), cmap="gray", vmax=np.log1p(np.percentile(frame, 99.95)))
        mk = {"bgsub": ("o", 9), "pyreyes": ("s", 13), "xds": ("^", 17)}
        for m in METHODS:
            d = peaks[(peaks.id == fid) & (peaks.method == m)]
            ax.scatter(d.col, d.row, s=mk[m][1] ** 2, facecolors="none", edgecolors=COL[m], marker=mk[m][0], linewidths=0.8, label=f"{m} ({len(d)})")
        if r.binning == 4:
            rf = pd.read_csv(OUT / "xds_spots_mov11.csv"); rf = rf[(rf.frame - r.frame).abs() <= 0]
            ax.scatter(rf.col, rf.row, s=22 ** 2, facecolors="none", edgecolors=COL["xds_ref"], marker="D", linewidths=0.7, label=f"XDS reference run, this frame ({len(rf)})")
        ax.plot(r.cx, r.cy, "+", color="yellow", ms=10); ax.set_title(f"{fid}: peaks of every method on the original frame (log)"); ax.legend(fontsize=8, loc="lower right"); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(FIG / "overlays.png", dpi=110); plt.close(fig)
    # 6. per-method zoom crops of the same spots (first 8 matched-by-all on each example frame)
    for fid in ex:
        d = peaks[(peaks.id == fid) & (peaks.method == "bgsub")]
        allm = d[[c for c in d.columns if c.startswith("in_") and c != "in_xds_ref"]].all(axis=1) if len(d) else d
        d = d[allm].sort_values("S_raw", ascending=False).head(8)
        if not len(d):
            continue
        frame = np.load(FR / f"{fid}.npz")["frame"].astype(float); bg = np.load(OUT / "bgsub" / f"{fid}_{v}.npz")["bg"].astype(float) + float(law_for(man[man.id == fid].iloc[0]).offset)
        fig, axes = plt.subplots(2, len(d), figsize=(2.2 * len(d), 4.6), squeeze=False)
        for j, (_, p) in enumerate(d.iterrows()):
            y, x = int(round(p["row"])), int(round(p["col"])); h = 12
            sl = (slice(max(y - h, 0), y + h + 1), slice(max(x - h, 0), x + h + 1))
            raw = frame[sl]; sub = frame[sl] - np.nan_to_num(bg[sl]); vmax = np.percentile(raw, 99.5)
            axes[0, j].imshow(raw, cmap="viridis", vmin=0, vmax=vmax); axes[0, j].set_title(f"raw S={p['S_raw']:.0f}  bg={p[f'B_bgsub_{v}']:.0f}", fontsize=8)
            axes[1, j].imshow(sub, cmap="viridis", vmin=0, vmax=vmax); axes[1, j].set_title(f"after: I={p['S_raw'] - p[f'B_bgsub_{v}']:.0f} (flat {p['I_flat']:.0f})", fontsize=8)
            for a in axes[:, j]: a.set_axis_off()
        fig.suptitle(f"{fid}: spots found by all three methods, raw (top) and after bgsub {v} subtraction (bottom), same colour scale; disc sums in stored counts", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / f"zoom_{fid}.png", dpi=120); plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--versions", nargs="+", default=["v2.1", "v1"]); a = ap.parse_args(); main(a.versions)
