"""compare v0: bgsub against two external references, pyREyes (DoG peak counting) and superxds (XDS COLSPOT
re-implementation, Fortran), on REAL Apollo frames: bin 8 (1024^2, fly-scan Good/Bad frames) and bin 4 (2048^2,
rotation movie mov11 with XDS's own SPOT.XDS as a third reference).  Owner's questions (2026-11-05): how many peaks
each finds, the intensity distribution of the found peaks, peaks mapped back onto the original image, and how
intensities change after the background subtraction; and what mechanism each tool uses.

Stages (run in this order; each writes under ~/pod_data/bgsub/compare_v0/ with provenance sidecars):
  select    frames/<id>.npz + .tif (uint16), manifest.csv, masks/<key>.npz, xds_spots_mov11.csv
  bgsub     bgsub/<id>_<version>.npz (bg float32, flags, used, dispersion, timings)  for version in v1, v2.1
  pyreyes   pyreyes/<id>.npz (dog float32, binary) + peaks_pyreyes.csv (exact re-implementation of
            pyREyes 3.6.3 dif_map.py:419-502, parameters dif_map.py:59-64 and config/microscope_configs.json)
  superxds  (added after the build report)      evaluate / figures (common peak table, overlays, distributions)

Run:  .venv/bin/python scripts/compare_v0.py select|bgsub|pyreyes [--versions v1 v2.1]
"""
from __future__ import annotations
import argparse, json, sys, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from bgsub import CountLaw, Geometry, estimate_background, make_plan
from bgsub.config import BACKGROUND, out_dir
from bgsub.vendor import detector as D
from bgsub.vendor.masks import beamstop_mask_v1
from bgsub.vendor.centre import centre_from_symmetry
from bgsub.vendor.provenance import write_sidecar

OUT = out_dir("compare_v0")
FR, MK = OUT / "frames", OUT / "masks"
BIN8 = {"sids": ["s73dc36c8", "s8bdcf271"], "n_good": 10, "n_bad": 5, "seed": 0,
        "source": "frames/atlas_spots_v0_<sid>.npz (cached REyes Good / Bad fly-scan frames, bin 8, uint8 stored counts)"}
BIN4 = {"movie": "mov11", "frames": list(range(1, 121, 12)),
        "dir": BACKGROUND / "xds_merge_v0" / "private" / "frames" / "mov11",
        "xds_run": BACKGROUND / "xds_merge_v0" / "private" / "runs" / "probe_mov11_V0"}
PYREYES_REF = "pyREyes 3.6.3 (commit c0681ee) src/pyREyes/dif_map.py"


# ----------------------------------------------------------------------------------------------------- helpers
def sidecar(p, **kw):
    write_sidecar(p, inputs=kw.pop("inputs", []), params=kw)


def border_mask(shape, px):
    m = np.zeros(shape, bool); m[:px] = m[-px:] = True; m[:, :px] = m[:, -px:] = True; return m


def read_spot_xds(path):
    """SPOT.XDS: X Y Z INTENSITY [h k l]; X, Y in XDS pixel units (pixel ix spans [ix-1, ix], ix 1-based), so the
    0-based array coordinates are col = X - 0.5, row = Y - 0.5; Z is the continuous frame coordinate, frame = floor(Z) + 1."""
    rows = []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        x, y, z, I = map(float, f[:4]); hkl = tuple(int(v) for v in f[4:7]) if len(f) >= 7 else (0, 0, 0)
        rows.append({"x_xds": x, "y_xds": y, "z_xds": z, "I_xds": I, "h": hkl[0], "k": hkl[1], "l": hkl[2],
                     "row": y - 0.5, "col": x - 0.5, "frame": int(np.floor(z)) + 1})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------- select
def stage_select():
    FR.mkdir(exist_ok=True); MK.mkdir(exist_ok=True)
    man = []
    rng = np.random.default_rng(BIN8["seed"])
    for sid in BIN8["sids"]:
        z = np.load(BACKGROUND / "frames" / f"atlas_spots_v0_{sid}.npz")
        tab = pd.read_parquet(BACKGROUND / "features" / f"{sid}_bin8_features_v1.parquet",
                              columns=["row", "frame", "cy", "cx"]).set_index(["row", "frame"])
        flag = str(np.load(BACKGROUND / "atlas" / "v1b" / f"{sid}_build_v1b.npz")["gain_flag"])
        q = z["quality"]
        pick = (list(rng.choice(np.flatnonzero(q == "Good diffraction"), BIN8["n_good"], replace=False)) +
                list(rng.choice(np.flatnonzero(q == "Bad diffraction"), BIN8["n_bad"], replace=False)))
        # mask: the session's shells mask (stop + arm) dilated by 1 px, the fixed pattern (seams, row 0), a 2-px border
        sh = np.load(BACKGROUND / "features" / f"{sid}_bin8_shells_v1.npz")
        bad = ndi.binary_dilation(sh["mask"].astype(bool), iterations=1) | D.fixed_pattern_mask(8) | border_mask(sh["mask"].shape, 2)
        np.savez_compressed(MK / f"bin8_{sid}.npz", mask=bad, stop=sh["mask"].astype(bool))
        sidecar(MK / f"bin8_{sid}.npz", inputs=[str(BACKGROUND / "features" / f"{sid}_bin8_shells_v1.npz")], rule="shells mask dilated 1 | fixed_pattern_mask(8) | border 2")
        for k, i in enumerate(pick):
            fid = f"b8_{sid}_{k:02d}"
            frame = z["frames"][i].astype(np.uint16)
            row, fno = int(z["row"][i]), int(z["frame"][i]); c = tab.loc[(row, fno)]
            np.savez_compressed(FR / f"{fid}.npz", frame=frame)
            tifffile.imwrite(FR / f"{fid}.tif", frame)
            man.append({"id": fid, "binning": 8, "n": 1024, "sid": sid, "row": row, "frame": fno, "quality": str(q[i]).split()[0],
                        "cy": float(c.cy), "cx": float(c.cx), "gain_flag": flag, "mask": f"bin8_{sid}", "tif": str(FR / f"{fid}.tif"),
                        "source": f"atlas_spots_v0_{sid}.npz[{i}]"})
    # bin 4: mov11, 10 frames spread over the 120; TIFF = MRC + 1 (declared through the law's offset)
    files = {}
    for fno in BIN4["frames"]:
        g = sorted(glob.glob(str(BIN4["dir"] / f"*_{fno:03d}.tif")))
        assert len(g) == 1, (fno, g); files[fno] = g[0]
    frames = {fno: tifffile.imread(p).astype(np.uint16) for fno, p in files.items()}
    summed = np.sum([f.astype(np.float64) for f in frames.values()], axis=0)
    fixed = D.fixed_pattern_mask(4) | border_mask(summed.shape, 2)
    try:
        stop, info = beamstop_mask_v1(summed)
    except ValueError as e:   # no beam stop in the rotation movies: mask the saturated direct beam instead
        sat = np.any([f >= 65000 for f in frames.values()], axis=0) | (summed / len(frames) >= 30000)
        stop = ndi.binary_dilation(sat, iterations=4); info = {"is_disc": False, "rule": f"no stop ({e}); saturated >= 65000 in any frame or mean >= 30000, dilated 4 px", "n_sat": int(sat.sum())}
    txt = open(BIN4["xds_run"] / "XDS.INP").read()
    import re
    orgx = float(re.search(r"ORGX=\s*([0-9.]+)", txt).group(1)); orgy = float(re.search(r"ORGY=\s*([0-9.]+)", txt).group(1))
    init = (orgy - 0.5, orgx - 0.5)
    cres = centre_from_symmetry(summed, mask=(stop | fixed), init=init, search_px=40.0)
    centre = tuple(float(v) for v in cres.centre)
    bad = stop | fixed
    np.savez_compressed(MK / "bin4_mov11.npz", mask=bad, stop=stop)
    sidecar(MK / "bin4_mov11.npz", inputs=list(files.values()), rule="beamstop_mask_v1(sum of the 10 frames) | fixed_pattern_mask(4) | border 2",
            stop_info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()},
            centre_xds_init=init, centre_symmetry=centre, centre_err=getattr(cres, "err", None).__repr__())
    for fno, p in files.items():
        fid = f"b4_mov11_{fno:03d}"
        np.savez_compressed(FR / f"{fid}.npz", frame=frames[fno])
        tifffile.imwrite(FR / f"{fid}.tif", frames[fno])
        man.append({"id": fid, "binning": 4, "n": 2048, "sid": "mov11", "row": 0, "frame": fno, "quality": "movie",
                    "cy": centre[0], "cx": centre[1], "gain_flag": "n/a", "mask": "bin4_mov11", "tif": str(FR / f"{fid}.tif"), "source": p})
    m = pd.DataFrame(man); m.to_csv(OUT / "manifest.csv", index=False)
    sidecar(OUT / "manifest.csv", bin8=BIN8 | {"source": BIN8["source"]}, bin4={k: str(v) for k, v in BIN4.items()},
            centre_bin4={"xds_init_rowcol": init, "symmetry_rowcol": centre})
    sp = read_spot_xds(BIN4["xds_run"] / "SPOT.XDS")
    sp["selected"] = sp.frame.isin(BIN4["frames"])
    sp.to_csv(OUT / "xds_spots_mov11.csv", index=False)
    sidecar(OUT / "xds_spots_mov11.csv", inputs=[str(BIN4["xds_run"] / "SPOT.XDS")], convention="col = X - 0.5, row = Y - 0.5, frame = floor(Z) + 1")
    print(f"manifest {len(m)} frames; bin-4 centre: XDS init {init}, symmetry {centre}; stop is_disc={info.get('is_disc')}; "
          f"XDS spots {len(sp)} total, {int(sp.selected.sum())} on the selected frames")


# ----------------------------------------------------------------------------------------------------- bgsub
def law_for(r):
    if r.binning == 8:
        tie = 0.35 if str(r.gain_flag).lower().startswith("enable") else None
        return CountLaw.apollo(8, tie, dead_time_tau=D.tau_for_binning(D.TAU_PHYS_V1, 8))
    return CountLaw.apollo(4, None, dead_time_tau=D.tau_for_binning(D.TAU_PHYS_V1, 4)).with_offset(1)


def manifest(only):
    man = pd.read_csv(OUT / "manifest.csv")
    return man[man.id.str.startswith(only)] if only else man


def stage_bgsub(versions, only=""):
    man = manifest(only); od = OUT / "bgsub"; od.mkdir(exist_ok=True)
    plans, log = {}, []
    for r in man.itertuples():
        frame = np.load(FR / f"{r.id}.npz")["frame"].astype(np.int64)
        mask = np.load(MK / f"{r.mask}.npz")["mask"]
        law = law_for(r)
        for ver in versions:
            key = (r.mask, round(r.cy, 3), round(r.cx, 3), ver)
            t0 = time.perf_counter()
            if key not in plans:
                plans[key] = make_plan(frame.shape, mask, Geometry((r.cy, r.cx)), version=ver)
            t_plan = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = estimate_background(frame, mask, Geometry((r.cy, r.cx)), law, plan=plans[key])
            t_fit = time.perf_counter() - t0
            p = od / f"{r.id}_{ver}.npz"
            np.savez_compressed(p, bg=res.bg.astype(np.float32), flags=res.flags, used=res.used)
            rec = {"id": r.id, "version": ver, "dispersion": float(res.dispersion), "t_plan_s": t_plan, "t_fit_s": t_fit,
                   "n_flagged": int(res.flags.sum()), "n_used": int(res.used.sum()),
                   "centre_refined": json.dumps(res.centre_info, default=str)[:200] if res.centre_info else ""}
            rec.update({f"t_{k}": float(v) for k, v in (res.timings or {}).items() if np.isscalar(v)})
            log.append(rec); print(f"{r.id} {ver}: fit {t_fit*1e3:.0f} ms (plan {t_plan:.2f} s), phi {res.dispersion:.3f}, flagged {rec['n_flagged']}", flush=True)
    lp = OUT / f"bgsub_log{('_' + only) if only else ''}.csv"; pd.DataFrame(log).to_csv(lp, index=False)
    sidecar(lp, versions=versions, law="bin 8: apollo(8, tie 0.35 if gain Enable, tau_b); bin 4: apollo(4, no tie, tau_b).with_offset(1)")


# ----------------------------------------------------------------------------------------------------- pyREyes
def pyreyes_params(shape):
    """dif_map.py:400-417 with config Arctica-Apollo-SM (2048: light 1, harsh 5, threshold 10 std, min 5 px) and
    SMALL_FRAME_PARAMS (dif_map.py:59-64; 1024: light 1.0, harsh 3, threshold 6 std, min 3 px)."""
    if max(shape) >= 2048:
        return {"light": 1.0, "harsh": 5.0, "thr": 10.0, "minpx": 5}
    return {"light": 1.0, "harsh": 3.0, "thr": 6.0, "minpx": 3}


def pyreyes_detect(frame, exclude_center=25.0):
    """Exact re-implementation of ImageProcessor.create_binary_image (dif_map.py:419-455) and get_centroids
    (471-502) as called by process_mrc_file (560-564: exclude_center=25)."""
    from skimage.measure import label, regionprops_table
    p = pyreyes_params(frame.shape)
    data = np.asarray(frame, dtype=np.float64)
    dog = ndi.gaussian_filter(data, sigma=p["light"]) - ndi.gaussian_filter(data, sigma=p["harsh"])
    thr = np.mean(dog) + p["thr"] * np.std(dog)
    binary = dog > thr
    bx = max(1, int(binary.shape[1] * 0.03)); by = max(1, int(binary.shape[0] * 0.03))
    binary[:, :bx] = False; binary[:, -bx:] = False; binary[:by, :] = False; binary[-by:, :] = False
    lab = label(binary)
    pr = regionprops_table(lab, properties=("centroid", "area", "label"))
    cx0, cy0 = binary.shape[1] // 2, binary.shape[0] // 2
    er = (exclude_center / 100.0) * min(cx0, cy0)
    rows = []
    for r0, c0, a, lb in zip(pr["centroid-0"], pr["centroid-1"], pr["area"], pr["label"]):
        if a < p["minpx"]:
            continue
        if (cx0 - er < c0 < cx0 + er) and (cy0 - er < r0 < cy0 + er):
            continue
        rows.append({"row": float(r0), "col": float(c0), "area": int(a), "label": int(lb)})
    return dog.astype(np.float32), binary, lab, pd.DataFrame(rows), {"threshold": float(thr), **p, "exclude_half_px": er}


def stage_pyreyes(only=""):
    man = manifest(only); od = OUT / "pyreyes"; od.mkdir(exist_ok=True)
    allp, log = [], []
    for r in man.itertuples():
        frame = np.load(FR / f"{r.id}.npz")["frame"]
        t0 = time.perf_counter(); dog, binary, lab, pk, info = pyreyes_detect(frame); dt = time.perf_counter() - t0
        np.savez_compressed(od / f"{r.id}.npz", dog=dog, binary=binary, labels=lab.astype(np.int32))
        pk.insert(0, "id", r.id); allp.append(pk)
        log.append({"id": r.id, "n_peaks": len(pk), "t_s": dt, **info}); print(f"{r.id}: pyREyes {len(pk)} peaks, {dt*1e3:.0f} ms, thr {info['threshold']:.3f}", flush=True)
    suf = ('_' + only) if only else ''
    pd.concat(allp).to_csv(OUT / f"peaks_pyreyes{suf}.csv", index=False); pd.DataFrame(log).to_csv(OUT / f"pyreyes_log{suf}.csv", index=False)
    for f in (f"peaks_pyreyes{suf}.csv", f"pyreyes_log{suf}.csv"):
        sidecar(OUT / f, reference=PYREYES_REF, method="DoG(light) - DoG(harsh); mean + k std threshold; 3 % border; 8-connected components >= min px; centre box excluded (25 % of half-size)")


# ----------------------------------------------------------------------------------------------------- XDS (COLSPOT)
# superxds (github.com/XMB5/superxds) is a line-by-line port of XDS and does not build on macOS/arm64 (GCC >= 16,
# AVX-512 kernels, glibc).  The installed XDS (/usr/local/bin/xds, VERSION Apr 16 2026) is run as the executable
# stand-in; the mechanism is documented from the superxds source in the report.  Settings = the owner's electron
# XDS.INP for mov11 (BACKGROUND_PIXEL 4, SIGNAL_PIXEL 7, MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT 5), both binnings.
XDS_BIN = "/usr/local/bin/xds"
COLSPOT_PARAMS = {"BACKGROUND_PIXEL": 4, "SIGNAL_PIXEL": 7, "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT": 5}


def write_smv(path, frame):
    """ADSC/SMV: 512-byte ASCII header + little-endian uint16, row-major (the frame as stored, no flip)."""
    h = ("{\nHEADER_BYTES=  512;\nDIM=2;\nBYTE_ORDER=little_endian;\nTYPE=unsigned_short;\n"
         f"SIZE1={frame.shape[1]};\nSIZE2={frame.shape[0]};\nPIXEL_SIZE=0.016;\n}}\n")
    b = h.encode().ljust(512, b" ")
    with open(path, "wb") as f:
        f.write(b); f.write(np.ascontiguousarray(frame, dtype="<u2").tobytes())


def xds_inp(template, nx, qx, n_frames, min_valid, orgx, orgy):
    p = " ".join(f"{k}= {v}" for k, v in COLSPOT_PARAMS.items())
    return f"""JOB= XYCORR INIT COLSPOT
NAME_TEMPLATE_OF_DATA_FRAMES= {template}
DATA_RANGE= 1 {n_frames}
BACKGROUND_RANGE= 1 {n_frames}
SPOT_RANGE= 1 {n_frames}
DETECTOR= ADSC MINIMUM_VALID_PIXEL_VALUE= {min_valid} OVERLOAD= 65000
NX= {nx} NY= {nx} QX= {qx:.4f} QY= {qx:.4f}
ORGX= {orgx:.1f} ORGY= {orgy:.1f}
DETECTOR_DISTANCE= 560.0
OSCILLATION_RANGE= 1.0
STARTING_ANGLE= 0.0
X-RAY_WAVELENGTH= 0.0251
ROTATION_AXIS= -1 0 0
INCIDENT_BEAM_DIRECTION= 0 0 1
DIRECTION_OF_DETECTOR_X-AXIS= 1 0 0
DIRECTION_OF_DETECTOR_Y-AXIS= 0 1 0
TRUSTED_REGION= 0.0 1.4
{p}
MAXIMUM_NUMBER_OF_JOBS= 1
MAXIMUM_NUMBER_OF_PROCESSORS= 4
"""


def read_cbf_byte_offset(path):
    """Minimal CBF reader for XDS tables (BKGINIT/GAIN/BLANK.cbf): byte-offset compression, int32 values."""
    b = open(path, "rb").read()
    hdr = b[: b.index(b"\x0c\x1a\x04\xd5")].decode("latin1")
    import re
    nx = int(re.search(r"X-Binary-Size-Fastest-Dimension:\s*(\d+)", hdr).group(1))
    ny = int(re.search(r"X-Binary-Size-Second-Dimension:\s*(\d+)", hdr).group(1))
    n = int(re.search(r"X-Binary-Number-of-Elements:\s*(\d+)", hdr).group(1))
    data = b[b.index(b"\x0c\x1a\x04\xd5") + 4:]
    out = np.empty(n, np.int64); cur = 0; i = 0; k = 0
    mv = memoryview(data)
    while k < n:
        d = int(np.frombuffer(mv[i:i + 1], np.int8)[0]); i += 1
        if d == -128:
            d = int(np.frombuffer(mv[i:i + 2], "<i2")[0]); i += 2
            if d == -32768:
                d = int(np.frombuffer(mv[i:i + 4], "<i4")[0]); i += 4
        cur += d; out[k] = cur; k += 1
    return out.reshape(ny, nx)


def parse_colspot_lp(path):
    """Per-frame table of COLSPOT.LP: FRAME #, #STRONG PIXELS, ... (columns vary by version; keep the numeric rows)."""
    rows = []
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) >= 3 and f[0].isdigit() and all(x.replace(".", "", 1).replace("-", "", 1).isdigit() for x in f[1:3]):
            rows.append([int(f[0])] + [float(x) for x in f[1:]])
    return rows


def run_xds(d, timeout=900):
    import subprocess
    t0 = time.perf_counter()
    r = subprocess.run(["perl", "-e", f"alarm {timeout}; exec @ARGV", XDS_BIN], cwd=d, capture_output=True, text=True)
    return time.perf_counter() - t0, r.returncode, (r.stdout + r.stderr)[-2000:]


XDS_DEFAULTS = {"BACKGROUND_PIXEL": 2, "SIGNAL_PIXEL": 3, "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT": 6}


def stage_xds(only=""):
    """Primary XDS setting = XDS defaults (BACKGROUND_PIXEL 2, SIGNAL_PIXEL 3, 6 px) at both binnings; the owner's
    electron setting (4 / 7 / 5) is run as a variant.  Bin 8: single-frame runs FAIL in COLSPOT (NBKG ~500 of 1M
    pixels, NSTRONG 0 at SIGNAL_PIXEL 7, "INSUFFICIENT NUMBER OF SPOTS"; kept in xds/<id>/ as the record), so the 15
    frames of a session are run as one pseudo-sweep (INIT averages its table over them; COLSPOT is per frame anyway).
    Bin 4: the natural use, one sweep over the 120 TIFFs of mov11."""
    global COLSPOT_PARAMS
    man = manifest(only); od = OUT / "xds"; od.mkdir(exist_ok=True)
    peaks, log = [], []
    settings = [("default", XDS_DEFAULTS), ("owner", {"BACKGROUND_PIXEL": 4, "SIGNAL_PIXEL": 7, "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT": 5})]
    for tag, params in settings:
        COLSPOT_PARAMS = params
        for sid, rows in man[man.binning == 8].groupby("sid", sort=False):
            d = od / f"pseudo_{sid}_{tag}"; d.mkdir(exist_ok=True)
            ids = list(rows.id)
            for k, fid in enumerate(ids, 1):
                write_smv(d / f"frame_{k:04d}.img", np.load(FR / f"{fid}.npz")["frame"])
            (d / "XDS.INP").write_text(xds_inp("frame_????.img", 1024, D.binned_pixel_um(8) / 1000.0, len(ids), 0, rows.iloc[0].cx + 0.5, rows.iloc[0].cy + 0.5))
            dt, rc, tail = run_xds(d)
            sp = read_spot_xds(d / "SPOT.XDS") if (d / "SPOT.XDS").exists() else pd.DataFrame()
            if len(sp):
                sp = sp[(sp.frame >= 1) & (sp.frame <= len(ids))]; sp.insert(0, "id", [ids[f - 1] for f in sp.frame]); sp.insert(1, "setting", tag); peaks.append(sp)
            log.append({"id": f"pseudo_{sid}", "setting": tag, "mode": f"pseudo_sweep{len(ids)}", "n_spots": len(sp), "t_s": dt, "rc": rc, "bkginit": (d / "BKGINIT.cbf").exists()})
            print(f"{sid} pseudo-sweep [{tag}]: XDS {len(sp)} spots in {len(ids)} frames, {dt:.1f} s, rc {rc}", flush=True)
        b4 = man[man.binning == 4]
        if len(b4):
            d = od / f"mov11_sweep_{tag}"; d.mkdir(exist_ok=True)
            src = sorted(glob.glob(str(BIN4["dir"] / "*_???.tif"))); tmpl = str(BIN4["dir"] / (Path(src[0]).name[:-7] + "???.tif"))
            r0 = b4.iloc[0]
            (d / "XDS.INP").write_text(xds_inp(tmpl, 2048, 0.016, 120, 1, r0.cx + 0.5, r0.cy + 0.5))
            dt, rc, tail = run_xds(d, timeout=1800)
            sp = read_spot_xds(d / "SPOT.XDS") if (d / "SPOT.XDS").exists() else pd.DataFrame()
            n_all = len(sp)
            if len(sp):
                sp = sp[sp.frame.isin(set(b4.frame))]; sp.insert(0, "id", ["b4_mov11_%03d" % f for f in sp.frame]); sp.insert(1, "setting", tag); peaks.append(sp)
            log.append({"id": "mov11_sweep", "setting": tag, "mode": "sweep120", "n_spots": n_all, "n_on_selected": len(sp), "t_s": dt, "rc": rc, "bkginit": (d / "BKGINIT.cbf").exists()})
            print(f"mov11 sweep [{tag}]: XDS {n_all} spots in 120 frames ({len(sp)} on the 10 selected), {dt:.1f} s, rc {rc}", flush=True)
    suf = ('_' + only) if only else ''
    (pd.concat(peaks) if peaks else pd.DataFrame()).to_csv(OUT / f"peaks_xds{suf}.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / f"xds_log{suf}.csv", index=False)
    for f in (f"peaks_xds{suf}.csv", f"xds_log{suf}.csv"):
        sidecar(OUT / f, xds=XDS_BIN, settings=dict(settings), note="superxds does not build on macOS/arm64; XDS is the executable stand-in for the same algorithm; bin 8 single-frame runs fail (kept in xds/<id>/)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["select", "bgsub", "pyreyes", "xds"]); ap.add_argument("--versions", nargs="+", default=["v1", "v2.1"]); ap.add_argument("--only", default="", help="id prefix filter, e.g. b8 or b4")
    a = ap.parse_args()
    {"select": stage_select, "bgsub": lambda: stage_bgsub(a.versions, a.only), "pyreyes": lambda: stage_pyreyes(a.only), "xds": lambda: stage_xds(a.only)}[a.stage]()
