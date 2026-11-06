"""compare v1: the three background-subtraction methods on REAL frames from every binning and all three detectors,
with the images.  Datasets (one per detector x binning; frames referenced, private copies under
~/pod_data/bgsub/compare_v1/private/, never in the repo):
  apollo_b4   Arctica-Apollo bin 4, 2048^2, rotation movie mov11 (xds_merge_v0), 120 frames, XDS sweep + archive SPOT.XDS
  apollo_b8   Arctica-Apollo bin 8, 1024^2, fly-scan Good snapshots (compare_v0 b8_s73dc36c8_*), 15-frame pseudo-sweep
  apollo_b16  Arctica-Apollo bin 16, 512^2, fly-scan verify miniscan 2026-08-31 (55 single frames, 0.196 s), pseudo-sweep
  ceta_b2     Arctica-CETA bin 2, 2048^2, rotation movie (grossular mov11, 112 frames, 0.9 deg), XDS sweep + archive SPOT.XDS
  tvips_b2    F30-TVIPS bin 2, 2048^2, rotation movie (mov12, 101 frames, 0.5 deg), XDS sweep + archive SPOT.XDS
Methods per frame: bgsub v2.1 (global map + count-law flags -> detector peaks), pyREyes 3.6.3 (exact re-implementation),
superxds (the validated numpy port, scripts/superxds_py.py; XDS defaults 2 / 3 / 6; its BKGINIT x RATIO map per frame).
Count laws: Apollo bin 4 / 8 / 16 = CountLaw.apollo(binning, tie 0.35, tau_b) (+1 offset for the bin-4 TIFFs); CETA-D and
TVIPS (integrating): CountLaw.poisson(gain g).with_offset(1), g ESTIMATED from the frame as the Pearson dispersion of a
gain-1 fit (declared, not calibrated; XDS INIT's gain is reported alongside).  Centres: Friedel pairs of the archive
SPOT.XDS where a sweep exists, else the compare_v0 value (bin 8) or image symmetry (bin 16).
Outputs: ~/pod_data/bgsub/compare_v1/ (manifest, per-frame npz, peaks_v1.csv, counts_v1.csv), figures
notes/figures/compare_v1/panel_<dataset>.png and crops.   Run: .venv/bin/python scripts/compare_v1.py [--only ds]
"""
from __future__ import annotations
import argparse, glob, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bgsub import CountLaw, Geometry, estimate_background, make_plan
from bgsub.config import BACKGROUND, out_dir
from bgsub.vendor import detector as D
from bgsub.vendor.centre import centre_from_symmetry
from bgsub.vendor.provenance import write_sidecar
import compare_v0 as C0
import compare_v0_eval as EV
import superxds_py as SX

OUT = out_dir("compare_v1"); PRIV = OUT / "private"
FIG = Path(__file__).resolve().parents[1] / "notes" / "figures" / "compare_v1"
XDS_DEF = dict(sigma=2.0, limit=3.0, minpix=6, maxdev=3.0)


def read_mrc(p):
    b = open(p, "rb").read(); nx, ny, nz, mode = np.frombuffer(b[:16], "<i4"); ext = int(np.frombuffer(b[92:96], "<i4")[0])
    dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[int(mode)]
    return np.frombuffer(b[1024 + ext:], dt)[: nx * ny * nz].reshape(nz, ny, nx)[0].astype(np.int64)


def friedel_centre(sp, shape, tol=2.0):
    """Beam centre from Friedel mates in a SPOT.XDS table (compare_v0 convention: col = X - 0.5, row = Y - 0.5)."""
    P = sp[["row", "col"]].to_numpy(); fr = sp.frame.to_numpy()
    def score(c, t):
        M = 2 * np.asarray(c) - P; n = 0
        for i in range(len(P)):
            d = np.hypot(*(P - M[i]).T); ok = (d < t) & (np.abs(fr - fr[i]) <= 2); ok[i] = False; n += ok.any()
        return n
    cy0, cx0 = shape[0] / 2, shape[1] / 2; best = None
    for cy in np.arange(cy0 - 120, cy0 + 121, 6):
        for cx in np.arange(cx0 - 120, cx0 + 121, 6):
            s = score((cy, cx), 4.0)
            if best is None or s > best[0]: best = (s, cy, cx)
    s0, cy0, cx0 = best; fine = []
    for cy in np.arange(cy0 - 6, cy0 + 6.1, 0.5):
        for cx in np.arange(cx0 - 6, cx0 + 6.1, 0.5):
            fine.append((score((cy, cx), tol), cy, cx))
    fine.sort(reverse=True); return (float(fine[0][1]), float(fine[0][2])), int(fine[0][0]), len(P)


def stop_mask(frames, centre, offset, frac=0.25, floor=0.3, r_max=None, dilate=None):
    """Beam stop from the stack: the 15-px local mean of the summed frames above the declared zero level, below
    max(frac x median of that local mean within r_max, floor) per frame, within r_max of the centre; closed; the component
    holding the beam centre (else the nearest); dilated (4 px at <= 1024, 6 px at 2048).  Plus a 2-px border."""
    n = len(frames); summed = np.sum([f.astype(np.float64) for f in frames], axis=0) - n * offset
    yy, xx = np.indices(summed.shape); r = np.hypot(yy - centre[0], xx - centre[1]); r_max = r_max or 0.45 * summed.shape[0]
    local = ndi.uniform_filter(summed, 15) / n
    thr = max(frac * float(np.median(local[r < r_max])), floor)
    dark = (local < thr) & (r < r_max)
    dark = ndi.binary_closing(dark, iterations=3); lab, k = ndi.label(dark)
    if k == 0:
        stop = np.zeros_like(dark)
    else:
        cy, cx = int(round(centre[0])), int(round(centre[1])); lc = lab[cy, cx]
        if lc == 0:
            d2 = ndi.distance_transform_edt(lab == 0, return_indices=True)[1]; lc = lab[d2[0][cy, cx], d2[1][cy, cx]]
        stop = ndi.binary_dilation(lab == lc, iterations=dilate or (6 if summed.shape[0] >= 2048 else 4))
    border = np.zeros_like(stop); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
    return stop, border


def datasets():
    ds = {}
    # Apollo bin 4: mov11 (compare_v0 frames + the 120-frame movie)
    files = sorted(glob.glob(str(C0.BIN4["dir"] / "*_???.tif")))
    ds["apollo_b4"] = dict(detector="Arctica-Apollo", binning=4, load=lambda i, files=files: tifffile.imread(files[i - 1]).astype(np.int64),
                           n_sweep=len(files), pick=[1, 25, 49, 73, 97], law="apollo4", offset=1, min_valid=1, fixed=4,
                           xds_ref=OUT.parent / "compare_v0" / "xds_spots_mov11.csv", centre=(1028.5, 954.5), note="rotation movie, 1 deg per frame, CL 420")
    # Apollo bin 8: the 15 compare_v0 frames of s73dc36c8 (10 Good + 5 Bad), picks = Good 0..4
    man0 = pd.read_csv(OUT.parent / "compare_v0" / "manifest.csv"); r8 = man0[man0.sid == "s73dc36c8"].reset_index(drop=True)
    ds["apollo_b8"] = dict(detector="Arctica-Apollo", binning=8, load=lambda i, r8=r8: np.load(C0.FR / f"{r8.id[i - 1]}.npz")["frame"].astype(np.int64),
                           n_sweep=len(r8), pick=[1, 2, 3, 4, 5], law="apollo8", offset=0, min_valid=0, fixed=8, xds_ref=None,
                           centre=(float(r8.cy[0]), float(r8.cx[0])), mask_from=C0.MK / "bin8_s73dc36c8.npz", note="fly-scan Good snapshots, 49 ms, CL 420 (pseudo-sweep of 15)")
    # Apollo bin 16: the 55 verify frames (sweep order = acquisition order), picks = frames with spots
    f16 = sorted(glob.glob(str(PRIV / "apollo_bin16_verify" / "*" / "*_0_movie.mrc")))
    names = [Path(p).name[:15] for p in f16]; want = ["20260831_120676", "20260831_120684", "20260831_120701", "20260831_120711", "20260831_120722"]
    ds["apollo_b16"] = dict(detector="Arctica-Apollo", binning=16, load=lambda i, f16=f16: read_mrc(f16[i - 1]), n_sweep=len(f16),
                            pick=[names.index(w) + 1 for w in want], law="apollo16", offset=0, min_valid=0, fixed=16, xds_ref=None, centre=None,
                            note="fly-scan verify miniscan 2026-08-31, 0.196 s, CL 420, bin 16 (pseudo-sweep of 55)")
    # CETA-D and TVIPS rotation movies
    for key, sub, det, picks, note in [("ceta_b2", "ceta_mov11", "Arctica-CETA", [1, 28, 56, 84, 112], "rotation movie, 0.9 deg per frame, CL 960, CETA-D 4k binned 2"),
                                        ("tvips_b2", "tvips_mov12", "F30-TVIPS", [1, 25, 50, 75, 101], "rotation movie, 0.5 deg per frame, CL 1320 (300 kV), TVIPS 4k binned 2")]:
        fl = sorted(glob.glob(str(PRIV / sub / "images" / "*.tif")))
        ds[key] = dict(detector=det, binning=2, load=lambda i, fl=fl: tifffile.imread(fl[i - 1]).astype(np.int64), n_sweep=len(fl), pick=picks,
                       law="poisson_est", offset=1, min_valid=1, fixed=None, dilate=12, xds_ref_raw=PRIV / sub / "auto_process" / "SPOT.XDS", xds_init=PRIV / sub / "auto_process" / "INIT.LP",
                       centre=None, note=note)
    return ds


def law_for(d, gain=1.0):
    if d["law"] == "apollo4":
        return CountLaw.apollo(4, None, dead_time_tau=D.tau_for_binning(D.TAU_PHYS_V1, 4)).with_offset(1)
    if d["law"] == "apollo8":
        return CountLaw.apollo(8, 0.35, dead_time_tau=D.tau_for_binning(D.TAU_PHYS_V1, 8))
    if d["law"] == "apollo16":
        return CountLaw.apollo(16, 0.35, dead_time_tau=D.tau_for_binning(D.TAU_PHYS_V1, 16))
    return CountLaw.poisson(gain).with_offset(d["offset"])


def run_dataset(key, d, verbose=True):
    t0 = time.perf_counter()
    sweep = [d["load"](i) for i in range(1, d["n_sweep"] + 1)]
    ny, nx = sweep[0].shape; minpx = {2: 5, 4: 5, 8: 3, 16: 3}[d["binning"]]; rb = {2: 5, 4: 5, 8: 3, 16: 2}[d["binning"]]
    # reference spots and centre
    ref = None
    if d.get("xds_ref") is not None and Path(d["xds_ref"]).exists():
        ref = pd.read_csv(d["xds_ref"])
    elif d.get("xds_ref_raw") is not None and Path(d["xds_ref_raw"]).exists():
        ref = C0.read_spot_xds(d["xds_ref_raw"])
    centre = d["centre"]; cinfo = {"source": "given"}
    if centre is None and ref is not None and len(ref) > 50:
        centre, npair, ntot = friedel_centre(ref, (ny, nx)); cinfo = {"source": "friedel", "paired": npair, "of": ntot}
    if centre is None:
        summed = np.sum([f.astype(float) for f in sweep], axis=0)
        c = centre_from_symmetry(summed, init=(ny / 2, nx / 2), search_px=max(20.0, ny / 16)); centre = tuple(float(v) for v in c.centre); cinfo = {"source": "symmetry", "sigma": (float(c.sigma_y), float(c.sigma_x))}
    # mask
    if d.get("mask_from"):
        mask = np.load(d["mask_from"])["mask"]; stop = np.load(d["mask_from"])["stop"]
    else:
        stop, border = stop_mask(sweep, centre, d["offset"], dilate=d.get("dilate")); mask = stop | border
        if d["fixed"]:
            fp = D.fixed_pattern_mask(d["fixed"]); mask |= fp[:ny, :nx] if fp.shape == (ny, nx) else False
    # gain estimate for integrating detectors: Pearson dispersion of a gain-1 fit on the first picked frame
    gain = 1.0; read_var = 0.0; gain_info = {}
    if d["law"] == "poisson_est":
        # Integrating detector, no calibrated law: (1) a first pass with a deliberately generous Poisson gain (50) so that only
        # bright spots are flagged and the fit is not truncated; (2) per 32-px band the ROBUST (MAD) variance of the residual
        # and the variance of 3 x 3 box sums of the residual (spot-free pixels); (3) weighted LS: var = gain x mean + read_var;
        # (4) the window inflation f = var(3x3 sum) / (9 x pixel var) measures the spatial correlation of the noise (CETA /
        # TVIPS pixels are not independent); the declared law is poisson(gain f, read_var f) so that bgsub's 3 x 3 window test
        # has the right variance (its per-pixel test becomes conservative).  Declared, not calibrated.
        f0 = sweep[d["pick"][0] - 1]
        r0 = estimate_background(f0, mask, Geometry(centre), CountLaw.poisson(50.0).with_offset(d["offset"]), version="v2.1")
        bg0 = r0.bg.astype(float); okp = ~mask & ~r0.flags & np.isfinite(bg0)
        yy_, xx_ = np.indices(f0.shape); rr_ = np.hypot(yy_ - centre[0], xx_ - centre[1]); band = (rr_ // 32).astype(int)
        res0 = np.where(okp, f0.astype(float) - d["offset"] - bg0, 0.0)
        box = ndi.uniform_filter(res0, 3) * 9; nbox = ndi.uniform_filter(okp.astype(float), 3) * 9; okb = okp & (nbox >= 8.5)
        ms, vs, ws, fs = [], [], [], []
        for b_ in np.unique(band[okp]):
            sel = okp & (band == b_)
            if sel.sum() < 2000: continue
            madv = (1.4826 * np.median(np.abs(res0[sel] - np.median(res0[sel])))) ** 2
            selb = okb & (band == b_); boxv = (1.4826 * np.median(np.abs(box[selb] - np.median(box[selb])))) ** 2 if selb.sum() > 2000 else np.nan
            ms.append(float(bg0[sel].mean())); vs.append(float(madv)); ws.append(float(sel.sum())); fs.append(float(boxv / (9 * madv)) if np.isfinite(boxv) and madv > 0 else np.nan)
        A = np.vstack([np.asarray(ms), np.ones(len(ms))]).T; W = np.sqrt(np.asarray(ws))
        (gain, read_var), *_ = np.linalg.lstsq(A * W[:, None], np.asarray(vs) * W, rcond=None)
        gain = float(max(gain, 0.05)); read_var = float(max(read_var, 0.0)); fw = float(np.nanmedian(fs)) if np.isfinite(np.nanmedian(fs)) else 1.0; fw = max(fw, 1.0)
        gain_info = {"gain_est": gain, "read_var_est": read_var, "window_factor": fw, "n_bands": len(ms), "dispersion_gain50": float(r0.dispersion),
                     "method": "pass 1 poisson(50); robust MAD variance per 32-px band vs model mean, weighted LS var = gain mean + read_var; 3x3 window factor from box-sum MAD variance; declared law poisson(gain f, read_var f)"}
        gain, read_var = gain * fw, read_var * fw
        if d.get("xds_init") and Path(d["xds_init"]).exists():
            lp = open(d["xds_init"]).read(); g = re.findall(r"MEAN GAIN VALUE\s+([0-9.]+)", lp); gain_info["xds_init_gain"] = float(g[0]) if g else None
    law = CountLaw.poisson(gain, read_var).with_offset(d["offset"]) if d["law"] == "poisson_est" else law_for(d, gain)
    # superxds port over the whole sweep (INIT on all frames, COLSPOT per frame, JOINCOLSPOT)
    good = SX.goodpix(nx, ny, 3, 3, 0.0, 1.4)
    t1 = time.perf_counter(); tabs = SX.init_tables(sweep, good, d["min_valid"], 3, 3); t_init = time.perf_counter() - t1
    per = {}; t_col = 0.0
    for i, fr in enumerate(sweep, 1):
        t1 = time.perf_counter(); per[i] = SX.colspot_frame(fr, tabs, d["min_valid"], 3, 3, XDS_DEF["sigma"], XDS_DEF["limit"]); t_col += time.perf_counter() - t1
    spots, jinfo = SX.joincolspot(per, nx, ny, XDS_DEF["minpix"], XDS_DEF["maxdev"], bkginit=tabs["bkginit"])
    spots["frame"] = np.floor(spots.z + 0.5).astype(int) + 1 if len(spots) else []   # z was shifted by -0.5: frame = round(z + 0.5)
    rows, frames_out, allpeaks = [], {}, []
    plan = None
    for i in d["pick"]:
        fr = sweep[i - 1]
        t1 = time.perf_counter()
        if plan is None:
            plan = make_plan(fr.shape, mask, Geometry(centre), version="v2.1")
        res = estimate_background(fr, mask, Geometry(centre), law, plan=plan); t_bg = time.perf_counter() - t1
        bg = res.bg.astype(float) + d["offset"]; var = law.g(np.clip(res.bg.astype(float), 0, None))
        pk_bg = EV.bgsub_detect(fr.astype(float), bg, var, mask, minpx, res.flags)
        t1 = time.perf_counter(); dog, binary, lab, pk_py, pinfo = C0.pyreyes_detect(fr); t_py = time.perf_counter() - t1
        px = per[i]; E = np.where(tabs["bkginit"] >= 0, tabs["bkginit"] * px["ratio"], np.nan)   # RATIO = sum(img) / sum(BKGINIT table), so the x1000 of the table is already in it
        pk_sx = spots[spots.frame == i][["x", "y", "intensity", "npix"]].copy() if len(spots) else pd.DataFrame(columns=["x", "y", "intensity", "npix"])
        pk_sx["row"] = pk_sx.y - 0.5; pk_sx["col"] = pk_sx.x - 0.5     # XDS pixel units -> array (compare_v0 convention)
        near = ndi.binary_dilation(mask, iterations=3)                  # peaks on or next to the mask are dropped for every method
        def offmask(p):
            if not len(p): return p
            yy_ = np.clip(np.round(p.row).astype(int), 0, ny - 1); xx_ = np.clip(np.round(p.col).astype(int), 0, nx - 1)
            return p[~near[yy_, xx_]].reset_index(drop=True)
        pk_bg, pk_py, pk_sx = offmask(pk_bg), offmask(pk_py), offmask(pk_sx)
        nref = int((ref.frame == i).sum()) if ref is not None else None
        frames_out[i] = dict(frame=fr, bg=bg, flags=res.flags, dog=dog, E=E, sx_mask=px["mask"], peaks={"bgsub": pk_bg, "pyreyes": pk_py, "superxds": pk_sx},
                             ref=(ref[ref.frame == i] if ref is not None else None))
        for mname, pk in (("bgsub", pk_bg), ("pyreyes", pk_py), ("superxds", pk_sx)):
            if len(pk):
                q = pk[["row", "col"]].copy(); q.insert(0, "method", mname); q.insert(0, "frame", i); q.insert(0, "dataset", key); allpeaks.append(q)
        rows.append({"dataset": key, "frame": i, "n_bgsub": len(pk_bg), "n_bgsub_flags": int(ndi.label(res.flags & ~mask, structure=np.ones((3, 3)))[1]),
                     "n_pyreyes": len(pk_py), "n_superxds": len(pk_sx), "n_xds_archive": nref, "sx_nstrong": px["nstrong"], "sx_unflagged_frac": px["n_unflagged"] / max(px["n_valid"], 1),
                     "sx_ratio": px["ratio"], "bgsub_dispersion": float(res.dispersion), "t_bgsub_s": t_bg, "t_pyreyes_s": t_py, "t_superxds_colspot_s": t_col / d["n_sweep"],
                     "mean_counts": float(fr[~mask].mean()), "gain": gain, "read_var": read_var})
        np.savez_compressed(OUT / f"{key}_f{i:03d}.npz", frame=fr.astype(np.int32), bg=bg.astype(np.float32), E=np.nan_to_num(E, nan=-1).astype(np.float32), dog=dog.astype(np.float32), flags=res.flags, mask=mask)
        if verbose:
            print(f"{key} frame {i}: bgsub {len(pk_bg)} (flags comps {rows[-1]['n_bgsub_flags']}), pyREyes {len(pk_py)}, superxds {len(pk_sx)} (strong px {px['nstrong']}), XDS archive {nref}; phi {res.dispersion:.3f}; t bgsub {t_bg*1e3:.0f} ms, pyREyes {t_py*1e3:.0f} ms", flush=True)
    meta = {"dataset": key, **{k: v for k, v in d.items() if k in ("detector", "binning", "n_sweep", "pick", "law", "offset", "min_valid", "note")}, "centre": centre, "centre_info": cinfo,
            "gain_info": gain_info, "superxds": {"t_init_s": t_init, "t_colspot_per_frame_s": t_col / d["n_sweep"], "n_spots_sweep": int(len(spots)), **jinfo}, "shape": [ny, nx], "t_total_s": time.perf_counter() - t0}
    json.dump(meta, open(OUT / f"{key}_meta.json", "w"), indent=1, default=str)
    (pd.concat(allpeaks) if allpeaks else pd.DataFrame(columns=["dataset", "frame", "method", "row", "col"])).to_csv(OUT / f"peaks_{key}.csv", index=False)
    panel(key, d, frames_out, centre, mask, rb)
    return rows, meta


def panel(key, d, frames_out, centre, mask, rb):
    FIG.mkdir(parents=True, exist_ok=True)
    i = d["pick"][len(d["pick"]) // 2]; fo = frames_out[i]; fr = fo["frame"].astype(float)
    mk = {"bgsub": ("o", "#d95f02"), "pyreyes": ("s", "#1b9e77"), "superxds": ("^", "#7570b3")}
    def show(ax, img, title, vmin=None, vmax=None, log=True):
        im = np.where(mask, np.nan, img)
        if log:
            ax.imshow(np.log1p(np.clip(im, 0, None)), cmap="gray", vmax=np.log1p(vmax if vmax is not None else np.nanpercentile(im, 99.9)))
        else:
            ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9); ax.set_axis_off()
    sub_vmax = np.nanpercentile(np.where(mask, np.nan, fr - fo["bg"]), 99.9)
    for crop in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
        show(axes[0, 0], fr, f"{key}: original frame {i} (log), all methods' peaks")
        for m, (mrk, col) in mk.items():
            p = fo["peaks"][m]; axes[0, 0].scatter(p.col, p.row, s=60, facecolors="none", edgecolors=col, marker=mrk, linewidths=0.7, label=f"{m} ({len(p)})")
        if fo["ref"] is not None and len(fo["ref"]):
            axes[0, 0].scatter(fo["ref"].col, fo["ref"].row, s=110, facecolors="none", edgecolors="#e7298a", marker="D", linewidths=0.6, label=f"XDS archive run ({len(fo['ref'])})")
        axes[0, 0].legend(fontsize=7, loc="lower right"); axes[0, 0].plot(centre[1], centre[0], "+", color="yellow", ms=9)
        bmax = float(np.nanmax(np.where(mask, np.nan, fo["bg"])))
        show(axes[0, 1], fo["bg"], "bgsub v2.1 background (log)", vmax=bmax)
        show(axes[0, 2], fo["E"], "superxds / XDS background = BKGINIT x RATIO (log, same scale)", vmax=bmax)
        show(axes[1, 0], fr - fo["bg"], f"bgsub: frame - background, peaks ({len(fo['peaks']['bgsub'])})", vmin=-sub_vmax * 0.1, vmax=sub_vmax, log=False)
        p = fo["peaks"]["bgsub"]; axes[1, 0].scatter(p.col, p.row, s=60, facecolors="none", edgecolors=mk["bgsub"][1], marker="o", linewidths=0.6)
        show(axes[1, 1], fr - np.nan_to_num(fo["E"], nan=0), f"superxds: frame - BKGINIT x RATIO, spots ({len(fo['peaks']['superxds'])})", vmin=-sub_vmax * 0.1, vmax=sub_vmax, log=False)
        p = fo["peaks"]["superxds"]; axes[1, 1].scatter(p.col, p.row, s=60, facecolors="none", edgecolors=mk["superxds"][1], marker="^", linewidths=0.6)
        dog = fo["dog"]; show(axes[1, 2], dog, f"pyREyes: difference of Gaussians (no background), peaks ({len(fo['peaks']['pyreyes'])})", vmin=np.nanpercentile(dog, 1), vmax=np.nanpercentile(dog, 99.9), log=False)
        p = fo["peaks"]["pyreyes"]; axes[1, 2].scatter(p.col, p.row, s=60, facecolors="none", edgecolors=mk["pyreyes"][1], marker="s", linewidths=0.6)
        if crop:
            h = int(0.22 * fr.shape[0]); cy, cx = int(centre[0]), int(centre[1]); cy = min(max(cy, h), fr.shape[0] - h); cx = min(max(cx, h), fr.shape[1] - h)
            for ax in axes.ravel():
                ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h)
        fig.suptitle(f"{d['detector']} bin {d['binning']} ({fr.shape[0]}^2): {d['note']}{' - central crop' if crop else ''}", fontsize=11)
        fig.tight_layout(); fig.savefig(FIG / f"panel_{key}{'_crop' if crop else ''}.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); a = ap.parse_args()
    ds = datasets(); keys = a.only or list(ds)
    allrows, metas = [], {}
    for k in keys:
        print(f"=== {k}", flush=True)
        rows, meta = run_dataset(k, ds[k]); allrows += rows; metas[k] = meta
        pd.DataFrame(allrows).to_csv(OUT / f"counts_v1{'_' + '_'.join(keys) if a.only else ''}.csv", index=False)
    cnt = pd.DataFrame(allrows); out = OUT / f"counts_v1{'_' + '_'.join(keys) if a.only else ''}.csv"; cnt.to_csv(out, index=False)
    write_sidecar(out, inputs=[str(OUT / f"{k}_meta.json") for k in keys], params={"xds_defaults": XDS_DEF, "datasets": {k: metas[k]["note"] for k in keys}})
    print(cnt.groupby("dataset")[["n_bgsub", "n_pyreyes", "n_superxds", "n_xds_archive", "bgsub_dispersion", "gain", "t_bgsub_s", "t_pyreyes_s", "t_superxds_colspot_s"]].median().round(3).to_string())
