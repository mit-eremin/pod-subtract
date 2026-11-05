#!/usr/bin/env python
"""Bragg-peak intensity recovery after background subtraction: known peaks injected into real frames (peak recovery v0,
2026-11-04).

Pre-registration: notes/reports/2026-11-04_peak_recovery_injection_v0.md Sec. 1 and PREREG below (committed 040015d before any
measurement).  Pure functions in podbg.peaks.

Stages (--stages, comma separated; outputs ~/pod_data/background/peak_recovery_v0/ with sidecars; logs/, checkpoints/,
hosts/ (cached host frames by sid), private/):
  spotshape  sigma of real Bragg spots on the cached Good frames of the atlas v1 spot test -> spot_shape_v0.csv
  stream     the 3 pre-registered rows per held-out session streamed over ssh (timeout 300 cat); host frames cached
  synth      the injection check on pure synthetic frames (Sec. 1.5)
  run        fly-scan injection + every background method, per session (checkpointed)
  classic    classic bin-4 hosts
  tables, figs
Rules: single process; no HPC compute (coreutils reads only); sids only in public files (leak check).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from podbg import paths
from podbg import peaks as PK
from podbg.provenance import write_sidecar

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = paths.SUB / "peak_recovery_v0"; LOGS = OUT / "logs"; CK = OUT / "checkpoints"; HOSTS = OUT / "hosts"
FIG = ROOT / "notes" / "figures" / "peak_recovery_v0"
SSH_TIMEOUT = 300
K8 = 0.0022811
N_SHELL = 513

PREREG = {
    "date": "2026-11-04", "commit_prereg": "040015d",
    "sessions_flyscan": "reyes_drop_v0.HELD (20 held-out, bin 8, CL 420)",
    "rows": "reyes_drop_v0/m1_rows_v0.csv (3 rows per session with the most Good/Bad frames), streamed whole with timeout 300 cat",
    "classes": {"F": "film_selection.film_mask (states v1.4 film_ok), REyes No, trusted",
                "BA": "REyes No, trusted, state_v1_4 film_bar_adjacent, 0 <= bar_edge_um <= 5",
                "CN": "REyes No, trusted, Good/Bad within +/-2 positions in the row (full table), no Grid within +/-2 (adj2), not film_ok"},
    "n_host": 12, "n_host_classic": 6, "classic_min_frames": 8, "replicates": 2,
    "sigma_clip_bin8": [0.5, 2.0], "sigma_bin4_factor": 2.0,
    "box": {8: {"r_box": 3.0, "r_in": 5.0, "r_out": 9.0, "half": 10, "spacing": 24.0, "disc_mask": 9.0},
            4: {"r_box": 6.0, "r_in": 10.0, "r_out": 18.0, "half": 20, "spacing": 48.0, "disc_mask": 18.0}},
    "d_bins": {"D1": [1.00, 1.163], "D2": [0.80, 1.00], "D3": [0.50, 0.80], "D4": [0.25, 0.50]},
    "d_labels": {"D1": "0.86-1.00 A", "D2": "1.00-1.25 A", "D3": "1.25-2.00 A", "D4": "2.00-4.00 A"},
    "pb_ladder": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "placement": "s uniform in the bin, azimuth uniform; reject if any box/annulus pixel invalid (mask dilated 1, seams +/-1, 2-px border) or centre within spacing of another peak; <= 200 tries",
    "injection": "latent N ~ Poisson(mu_h) x T(K|N); dead time paralyzable TAU_PHYS_V1 (bin 8 only); K' = quantise(N' + n) at the session tie rule",
    "B_box": "sum over the box of mu_h = the host's masked shell mean inverted (true_mean_fast, session tie rule, no floor, no dead time)",
    "methods": ["O", "A", "B0", "B1", "B4", "P2", "C", "D", "E"], "classic_methods": ["O", "A", "Bc", "D"],
    "pool": {"W": 20, "W_fallback": 40, "n_min": 3, "scale_s": [0.34, 0.80]},
    "atlas": "atlas_v1b.npz pooled held-out fold (I_e, q from frame_keys on the host's masked profile) x rolling inner curve tau 300 n_prior 30 over v1c film keys, at the host's position (causal); atlas_refresh_v0._forward",
    "physics": "structure_family.fit_holdout gate+tie, S_family_flyscan_v2 (fam_parts), a_h 0.20; classic: gate, no tie",
    "truth": "reyes_drop_v0.truth_bins/truth_at at the host's ln masked_sum, host-scaled",
    "sigma": {"naive": "Poisson in stored counts", "law": "g(max(K', B)) from the response table (bin 4: g(m) = m)"},
    "within": {"X": [0.05, 0.10], "session_frac": 2.0 / 3.0, "min_peaks_session": 5},
    "honest": {"pull_sd": [0.85, 1.15], "pull_median_abs": 0.5, "pb_groups": {"low": [0.03, 1.0], "high": [10.0, 30.0]}},
    "recommend": "live methods A, B0, B1, B4, P2, C, D: smallest P/B*_5 (truth); ties smaller |median beta|, then A < B < C < D",
    "synth": {"sids": ["s0047f887", "s4fead7c7", "sc19f33b9"], "n_frames": 12,
              "pass": "(i) routes agree within max(0.01, 2 SE) in >= 90 % of cells; (ii) |E_exact - O| <= 0.01 at P/B >= 1; (iii) E_exact sigma_law pull robust SD 0.85-1.15 at P/B >= 0.3"},
}
LADDER = PREREG["pb_ladder"]; DBINS = PREREG["d_bins"]


# ============================================================================= io
def _mod(name, fn):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_M = {}


def mod(name):
    fn = {"rd": "reyes_drop_v0.py", "ar": "atlas_refresh_v0.py", "sf2": "structure_family_v2.py", "c0": "classic_v0.py", "slv": "s_level_bin4_v0.py"}[name]
    if name not in _M:
        _M[name] = _mod(f"pr_{name}", fn)
    return _M[name]


LOGFILE = None


def log(*a):
    msg = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, "a") as fh:
            fh.write(msg + "\n")


_TOK = None


def leak_check(p: Path):
    global _TOK
    if p.suffix not in (".csv", ".json", ".md", ".txt"):
        return
    if _TOK is None:
        _TOK = mod("rd").tokens()
    txt = p.read_text().lower(); hits = [t for t in _TOK if t in txt]
    if hits:
        raise SystemExit(f"LEAK in {p}: {hits[:5]}")


def save_csv(df, name, inputs, params, note):
    p = OUT / name; df.to_csv(p, index=False); leak_check(p)
    write_sidecar(p, inputs=[str(i) for i in inputs], params=params, note=note); return p


def save_json(obj, name, inputs, params, note):
    p = OUT / name; json.dump(obj, open(p, "w"), indent=1, default=float); leak_check(p)
    write_sidecar(p, inputs=[str(i) for i in inputs], params=params, note=note); return p


def save_parquet(df, name, inputs, params, note):
    p = OUT / name; df.to_parquet(p, index=False)
    write_sidecar(p, inputs=[str(i) for i in inputs], params=params, note=note); return p


# ============================================================================= spot shape
def stage_spotshape():
    import glob
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    files = sorted(glob.glob(str(paths.FRAMES / "atlas_v1" / "atlas_v1_spots_*.npz")))
    res = []
    yy, xx = np.mgrid[-6:7, -6:7]
    for f in files:
        sid = Path(f).stem.split("_")[-1]
        F = np.load(f)["frames"].astype(float)
        for img in F:
            ex = img - ndi.median_filter(img, size=15); sm = ndi.gaussian_filter(ex, 1.0)
            lab, n = ndi.label(sm > 1.0)
            if n == 0:
                continue
            tot = ndi.sum(ex, lab, range(1, n + 1))
            for k, t in enumerate(tot):
                if t < 40:
                    continue
                cy, cx = ndi.center_of_mass(np.clip(ex, 0, None), lab, k + 1); cy, cx = int(round(cy)), int(round(cx))
                if cy < 12 or cx < 12 or cy > 1011 or cx > 1011 or np.hypot(cy - 512, cx - 512) < 80:
                    continue
                c = ex[cy - 6:cy + 7, cx - 6:cx + 7]
                mfun = lambda p: p[0] * np.exp(-((yy - p[1]) ** 2 + (xx - p[2]) ** 2) / (2 * p[3] ** 2)) + p[4]
                o = least_squares(lambda p: (mfun(p) - c).ravel(), [c.max(), 0, 0, 1.0, 0], bounds=([0, -3, -3, 0.3, -5], [1e4, 3, 3, 5, 5]))
                res.append({"sid": sid, "excess_counts": float(t), "sigma_px": float(o.x[3]), "r_px": float(np.hypot(cy - 512, cx - 512)), "peak_counts": float(img[cy, cx])})
    d = pd.DataFrame(res)
    save_csv(d, "spot_shape_v0.csv", files, {"rule": "excess over a 15-px median >= 40 counts, Gaussian+const fit on 13x13, r >= 80 px"},
             "sigma of real Bragg spots (bin 8) on the cached Good frames of the atlas v1 spot test; the injection draws sigma from this table")
    log(f"spotshape: {len(d)} spots, sigma median {d.sigma_px.median():.3f}")


def spot_sigmas() -> np.ndarray:
    d = pd.read_csv(OUT / "spot_shape_v0.csv")
    return np.clip(d.sigma_px.to_numpy(float), *PREREG["sigma_clip_bin8"])


# ============================================================================= fly-scan session context
def seam_lines_mask(shape=(1024, 1024)):
    from podbg import atlas as A
    m = np.zeros(shape, bool)
    for c in A.SEAM_LINES_BIN8["cols"]:
        m[:, max(c - 1, 0):c + 2] = True
    for r in A.SEAM_LINES_BIN8["rows"]:
        m[max(r - 1, 0):r + 2, :] = True
    return m


def valid_mask_flyscan(mask_true_is_masked):
    from scipy import ndimage as ndi
    bad = ndi.binary_dilation(mask_true_is_masked, iterations=1) | seam_lines_mask(mask_true_is_masked.shape)
    bad[:2, :] = bad[-2:, :] = True; bad[:, :2] = bad[:, -2:] = True
    return ~bad


def session_hosts(sid):
    """Host table (fly-scan): positions in reyes_drop session_arrays T, with class."""
    rd = mod("rd")
    from podbg import atlas as A
    T, P, V, wts = rd.session_arrays(sid)
    st = pd.read_parquet(paths.DERIVED / "frame_states_v1_4.parquet", columns=["sid", "row", "frame", "state_v1_4"], filters=[("sid", "==", sid)])
    T = T.merge(st.drop(columns="sid"), on=["row", "frame"], how="left")
    t = A.load_session_features(sid)["tab"]
    isgb_all = t.quality.isin(["Good diffraction", "Bad diffraction"]).to_numpy(); rr = t.row.to_numpy(); nb = np.zeros(len(t), bool)
    for d in (1, 2):
        for sg in (-1, 1):
            j = np.clip(np.arange(len(t)) + sg * d, 0, len(t) - 1); nb |= isgb_all[j] & (rr[j] == rr) & (j != np.arange(len(t)))
    keep = (t.trusted_sweep == True).to_numpy() & (t.quality != "Grid").to_numpy()      # noqa: E712
    T["gb_nb"] = nb[keep]
    rows = pd.read_csv(paths.SUB / "reyes_drop_v0" / "m1_rows_v0.csv"); rows = rows[rows.sid == sid].row.tolist()
    no = (T.quality == "No diffraction").to_numpy(); inr = T.row.isin(rows).to_numpy()
    cls = {"F": no & T.film_ok.to_numpy(),
           "BA": no & (T.state_v1_4 == "film_bar_adjacent").to_numpy() & T.bar_edge_um.between(0, 5).to_numpy(),
           "CN": no & T.gb_nb.to_numpy() & ~T.adj2.to_numpy() & ~T.film_ok.to_numpy()}
    out = []
    for c, m in cls.items():
        cand = np.flatnonzero(m & inr)
        if len(cand) == 0:
            continue
        k = min(PREREG["n_host"], len(cand))
        sel = np.unique(cand[np.round(np.linspace(0, len(cand) - 1, k)).astype(int)])
        out += [{"pos": int(i), "cls": c, "row": int(T.row.iat[i]), "frame": int(T.frame.iat[i])} for i in sel]
    return pd.DataFrame(out), T, P, V, wts, rows


# ============================================================================= stream
def stream_row(hpc_path: str, keep: set):
    from podbg import mrc
    assert "'" not in hpc_path
    cmd = ["perl", "-e", f"alarm {SSH_TIMEOUT + 60}; exec @ARGV", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", paths.HPC_HOST,
           f"timeout {SSH_TIMEOUT} cat '{hpc_path}'"]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)
    got = {}
    try:
        hdr = mrc.parse_header(pr.stdout.read(mrc.HEADER_BYTES))
        if hdr.ext:
            pr.stdout.read(hdr.ext)
        for i in range(hdr.nz):
            buf = pr.stdout.read(hdr.frame_bytes)
            if len(buf) != hdr.frame_bytes:
                raise IOError(f"short read at frame {i}")
            if i in keep:
                got[i] = np.frombuffer(buf, dtype=hdr.dtype).reshape(hdr.ny, hdr.nx).copy()
    finally:
        pr.stdout.close(); rc = pr.wait(timeout=90)
    if rc not in (0, None):
        raise IOError(f"ssh cat exit {rc}")
    return got


def stage_stream():
    rd = mod("rd")
    HOSTS.mkdir(parents=True, exist_ok=True)
    from podbg import atlas as A
    for sid in rd.HELD:
        p = HOSTS / f"{sid}_hosts_v0.npz"
        if p.exists():
            continue
        t0 = time.time()
        H, T, *_ = session_hosts(sid)
        s = A.load_session_features(sid)
        frames = np.zeros((len(H), 1024, 1024), np.uint8); ok = np.zeros(len(H), bool)
        for row, g in H.groupby("row"):
            path = rd.stack_of(sid, row)
            for att in range(3):
                try:
                    got = stream_row(path, set(g.frame.tolist())); break
                except Exception as e:  # noqa: BLE001
                    log(f"  {sid} row {row} attempt {att + 1}: {e}"); time.sleep(5)
            else:
                raise IOError(f"{sid} row {row}")
            for j, fr in zip(g.index, g.frame):
                if fr in got:
                    frames[j] = got[fr]; ok[j] = True
        tab = s["tab"].set_index(["row", "frame"])
        ms = tab.loc[list(zip(H.row, H.frame))].masked_sum.to_numpy(float)
        mine = np.array([float(f[~s["mask"]].sum(dtype=np.int64)) for f in frames])
        H["addr_ok"] = ok & (mine == ms)
        np.savez_compressed(p, frames=frames, pos=H.pos.to_numpy(), cls=H.cls.to_numpy().astype(str), row=H.row.to_numpy(), frame=H.frame.to_numpy(), addr_ok=H.addr_ok.to_numpy())
        write_sidecar(p, inputs=s["paths"], params={"sid": sid, "rule": PREREG["classes"], "n_host": PREREG["n_host"], "read": "ssh timeout 300 cat, whole row, hosts kept"},
                      note="peak recovery v0 host frames (bin 8) of one held-out session")
        log(f"stream {sid}: {len(H)} hosts ({H.cls.value_counts().to_dict()}), addressing ok {int(H.addr_ok.sum())}, {time.time() - t0:.0f} s")


# ============================================================================= per-session method context (fly-scan)
class FlyCtx:
    def __init__(self, sid):
        rd = mod("rd"); ar = mod("ar")
        from podbg import atlas as A, atlas_v1 as V
        from podbg.model import forward as FW, structure_joint as SJ, fit as F
        from podbg import stray as ST
        self.sid = sid
        self.H, self.T, self.P, self.V, self.wts, self.rows = session_hosts(sid)
        s = A.load_session_features(sid); self.s = s
        self.valid = valid_mask_flyscan(s["mask"])
        self.Mb, self.lnE, lnq = rd.memberships(self.T, self.P, self.wts)
        nbk = self.T.gb_nb.to_numpy()
        P1 = self.Mb["V1"] & ~nbk; zq = rd.robust_z(lnq, lnq[P1]); self.Mb["P2"] = P1 & (zq <= 3)
        tru = self.Mb["V3"] & (self.T.recheck.astype(str).to_numpy() != "dead_row")
        self.xT, self.TT = rd.truth_bins(self.P[tru], self.V[tru], self.lnE[tru])
        ck = np.load(paths.SUB / "atlas" / "v1b" / f"{sid}_build_v1b.npz", allow_pickle=False)
        self.floor_atlas = ck["floor"]; self.flag = str(ck["gain_flag"]); self.tie = V.tie_up_for(self.flag)
        atl, meta = V.load_bundle(paths.SUB / "atlas" / "atlas_v1b.npz")
        fold = next((i for i, f in meta["folds"].items() if sid in f["held_out"]), None)
        self.atlas = atl[f"pooled_fold{fold}" if fold is not None else "pooled_all"]
        # rolling inner curve over the v1c film keys (atlas_refresh_v0 arithmetic)
        sr = V.SeamResponse.load(); nsm = ck["n_seam_rows"]
        keys = pd.read_parquet(paths.SUB / "atlas" / "v1c" / f"{sid}_keys_v1c.parquet").sort_values(["row", "frame"]).reset_index(drop=True)
        Pk = np.asarray(s["prof"][keys.prof_idx.to_numpy()], float); nk = s["row_n_shell"][keys.row_pos.to_numpy()].astype(float); ns = nsm[keys.row_pos.to_numpy()].astype(float)
        f_row = np.where(nk > 0, ns / np.maximum(nk, 1), 0.0)
        rec, mu = ar._predict_all(self.atlas, keys.I_e.to_numpy(), self.flag, self.floor_atlas, keys.q.to_numpy())
        fl = np.nan_to_num(self.floor_atlas)
        rec_c = sr.contaminate(np.nan_to_num(rec), f_row, self.flag); recn_c = sr.contaminate(np.nan_to_num(rec - fl[None, :]), f_row, self.flag)
        cr = V.inner_ratio_curve_rolling(Pk, nk, rec_c, recn_c, mu, 8, self.tie, tau=ar.TAU, update=~keys.centre_fallback.to_numpy(bool))
        self.roll_F = V.inner_correction_rolling(cr, n_prior=ar.N_PRIOR)["factor"]
        self.roll_key = keys.row.to_numpy() * 100000 + keys.frame.to_numpy()
        # physics context
        summ = pd.read_csv(paths.DERIVED / "all_sessions_bin8_summary_v1.csv").set_index("sid")
        self.st = FW.settings_from_summary(summ.loc[sid])
        tab = s["tab"]; grid = ((tab.quality == "Grid") & tab.trusted_sweep).to_numpy()
        pr = np.asarray(s["prof"][tab.prof_idx.to_numpy()[grid]], float); nsg = s["row_n_shell"][tab.row_pos.to_numpy()[grid]].astype(float)
        c = pr * nsg; okc = np.isfinite(c)
        self.floor_grid = np.where(okc, c, 0).sum(0) / np.maximum(np.where(okc, nsg, 0).sum(0), 1)
        self.stray = ST.stray_profile(sid)
        self.S_fn, self.kern, self.ff = mod("sf2").fam_parts("flyscan"); self.a_h = mod("sf2").A_H_FIXED
        self.r_min = F.R_MIN_PX.get(sid, 0)
        self.s_sh = (np.arange(N_SHELL) + 0.5) * K8
        from podbg import binning as B
        self.tab_inv = B.response_table(8, self.tie, 0.0, 0.0, mu_min=1e-4, mu_max=2000.0, n=900)
        self.g = PK.StoredVariance(8, self.tie)
        _m, _e, _ = self.tab_inv
        self.erec = lambda mu: np.interp(np.log(np.clip(mu, 1e-4, None)), np.log(_m), _e) * (np.asarray(mu) > 1e-4)
        from podbg import detector as D
        self.tau_b = D.tau_for_binning(D.TAU_PHYS_V1, 8)
        self.tab_rows = s["tab"].set_index(["row", "frame"])

    def invert(self, m):
        from podbg import binning as B
        m = np.asarray(m, float); out = np.full(m.shape, np.nan); ok = np.isfinite(m) & (m > 0)
        out[ok] = np.asarray(B.true_mean_fast(m[ok], 8, self.tie, 0.0, table=self.tab_inv).true_mean, float)
        out[np.isfinite(m) & (m <= 0)] = 0.0
        return out

    # ---- background profiles (stored counts per shell) for a host
    def pool_profile(self, pos, variant, host_prof, host_n):
        rd = mod("rd"); row = self.T.row.to_numpy(); frame = self.T.frame.to_numpy()
        pp = PREREG["pool"]
        for W in (pp["W"], pp["W_fallback"]):
            S, n = rd.window_pool(row, frame, self.Mb[variant], self.P, np.array([pos]), W)
            Sv, _ = rd.window_pool(row, frame, self.Mb[variant], self.V, np.array([pos]), W)
            if n[0] >= pp["n_min"]:
                est = np.where(Sv[0] > 0, S[0] / np.maximum(Sv[0], 1), np.nan)
                return est * PK.window_scale(host_prof, host_n, est, self.s_sh, *pp["scale_s"]), int(n[0]), W
        return None, int(n[0]), None

    def truth_profile(self, pos, host_prof, host_n):
        rd = mod("rd")
        est = rd.truth_at(self.xT, self.TT, np.array([self.lnE[pos]]))[0]
        return est * PK.window_scale(host_prof, host_n, est, self.s_sh, *PREREG["pool"]["scale_s"])

    def atlas_profile(self, row, frame, host_prof, host_n):
        from podbg import atlas_v1 as V
        P = host_prof[:V.N_RADII] if len(host_prof) >= V.N_RADII else np.pad(host_prof, (0, V.N_RADII - len(host_prof)), constant_values=np.nan)
        n = host_n[:V.N_RADII] if len(host_n) >= V.N_RADII else np.pad(host_n, (0, V.N_RADII - len(host_n)))
        kk = V.frame_keys(np.nan_to_num(P)[None], n[None], self.floor_atlas, 8, self.tie)
        _, (_, mu) = self.atlas.predict(float(kk["I_e"][0]), 8, self.flag, self.floor_atlas, return_electrons=True, q=float(kk["q"][0]))
        j = np.searchsorted(self.roll_key, row * 100000 + frame, side="left")
        F = self.roll_F[min(j, len(self.roll_F) - 1)] if j < len(self.roll_F) else self.roll_F[-1]
        rec = mod("ar")._forward((mu * F)[None], self.flag, self.floor_atlas)[0]
        return rec

    def physics_profile(self, host_prof, host_n):
        from podbg.model import forward as FW, structure_joint as SJ, structure_family as SF, fit as F
        r = np.arange(N_SHELL)
        inv = FW.invert_profile(host_prof, host_n, self.st, floor_counts=self.floor_grid, cv=0.0)
        msk = FW.fit_mask(self.s_sh, r, (float(self.H_cy), float(self.H_cx)), inv.ok, r_min=self.r_min)
        mu = np.asarray(inv.mu, float); ok = np.isfinite(mu) & np.asarray(inv.ok, bool)
        E = float(np.nansum(np.where(ok, mu * host_n, 0.0)))
        lev = SJ.tie_lever(host_prof, 8, self.st.gain_flag, 0.0, self.st.tau_binned)
        msk &= np.isfinite(lev) & np.isfinite(inv.sigma_poisson) & (inv.sigma_poisson > 0)
        if msk.sum() < 50:
            return None
        pr = SJ.Profile("host", self.s_sh[msk], (mu - self.stray)[msk], np.asarray(inv.sigma_poisson)[msk], E, 8, "archive", tie_lever=lev[msk])
        spec = SF.HoldoutSpec(mode="gate", tie=True, a_h=self.a_h, f_flat=self.ff)
        fit = SF.fit_holdout(pr, self.S_fn, self.kern, spec)
        m = full_gate_model(fit, self.s_sh, np.nan_to_num(lev), self.S_fn, self.kern, E, 8)
        mfit = full_gate_model(fit, pr.s, pr.tie_lever, self.S_fn, self.kern, E, 8)
        assert np.allclose(mfit, fit["model"], rtol=1e-6, atol=1e-9), "full-grid gate model does not reproduce the fit"
        rec = FW.recorded(np.clip(m + self.stray, 0, None), self.st, floor_counts=self.floor_grid)
        return rec, fit


def full_gate_model(fit, s, lev, S_fn, kern, E_frame, binning):
    from podbg.model import structure_joint as SJ, forward as FW, fit as F
    s = np.asarray(s, float); lam = kern.lambda_in_nm
    f2 = F.f2_C(s); I_in = kern.inelastic(s, lam); I_hc = FW.I_hydrocarbon(s); ceq = SJ.layer_c_equivalents(s)
    phi = SJ.floor_shape(s, binning); S0 = np.asarray(S_fn(s), float)
    lnb = np.log(fit["b_over_prior"]) + SJ._lnb_prior(E_frame)
    R = kern.R(s, fit["t"], lam)
    m = fit["A"] * R * (f2 * S0 + I_in * (1 + fit["a_h"] * ceq) + fit["a_h"] * I_hc) + np.exp(lnb) * phi + fit["cs"] * s ** (-SJ.P_PED) + fit["c0"]
    if fit["tie"]:
        m = m * (1.0 + fit["delta"] * np.asarray(lev, float))
    return m


# ============================================================================= placement and per-host injection
def place_peaks(rng, valid, centre, k_per_px, box, sigmas, n_shell_max=None):
    """28 peaks (P/B x d bin): list of dicts with cy, cx, s, dbin, pb, sigma."""
    ny, nx = valid.shape; half = box["half"]; out = []
    for db, (lo, hi) in DBINS.items():
        for pb in LADDER:
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
                out.append({"cy": cy, "cx": cx, "s": s, "dbin": db, "pb": pb, "sigma": float(rng.choice(sigmas))})
                break
            else:
                out.append({"cy": np.nan, "cx": np.nan, "s": np.nan, "dbin": db, "pb": pb, "sigma": np.nan})
    return out


def disc_mask(shape, peaks, radius):
    m = np.zeros(shape, bool)
    for q in peaks:
        if not np.isfinite(q["cy"]):
            continue
        iy, ix = int(round(q["cy"])), int(round(q["cx"])); R = int(np.ceil(radius)) + 1
        y0, y1 = max(iy - R, 0), min(iy + R + 1, shape[0]); x0, x1 = max(ix - R, 0), min(ix + R + 1, shape[1])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= np.hypot(yy - q["cy"], xx - q["cx"]) <= radius
    return m


def inject_frame(rng, K0, peaks, mu_prof, centre, binning, tie, tau_b, box, epc, N_known=None, erec=None):
    """Inject every peak; returns (K', per-peak geometry list).  POST HOC (after the synthetic check, reported): lam_eff =
    the law-expected oracle intensity epc sum_box [E(mu_h e^{-lam tau} + lam e^{-(mu_h + lam) tau}) - E(mu_h)], E = the
    stored mean of the response table (identity at bin 4); it is not used by any method."""
    Kp = K0.astype(np.int64).copy(); half = box["half"]; geo = []
    for q in peaks:
        if not np.isfinite(q["cy"]):
            geo.append(None); continue
        sig = q["sigma"] * (PREREG["sigma_bin4_factor"] if binning == 4 else 1.0)
        y0, x0, unit = PK.spot_pixels(q["cy"], q["cx"], sig, 1.0, half)
        rp = PK.patch_radius(y0, x0, 2 * half + 1, q["cy"], q["cx"])
        bx, an = PK.box_and_annulus(rp, box["r_box"], box["r_in"], box["r_out"])
        rc = PK.patch_radius(y0, x0, 2 * half + 1, centre[0], centre[1])
        mu_h = np.clip(PK.profile_at(mu_prof, rc), 0, None)
        B_box_e = float(mu_h[bx].sum())
        lam_box_target = q["pb"] * B_box_e
        frac = float(unit[bx].sum()); lam = unit * (lam_box_target / max(frac, 1e-12))
        sl = (slice(y0, y0 + 2 * half + 1), slice(x0, x0 + 2 * half + 1))
        Nk = None if N_known is None else N_known[sl]
        Kp[sl] = PK.inject(K0[sl], mu_h, lam, binning, tie, tau_b, rng, N_known=Nk)
        if erec is not None:
            lc = lam * np.exp(-(mu_h + lam) * tau_b); mh = mu_h * np.exp(-lam * tau_b)
            lam_eff = float(epc * (erec(mh + lc) - erec(mu_h))[bx].sum())
        else:
            lam_eff = np.nan
        geo.append({"y0": y0, "x0": x0, "box": bx, "ann": an, "rc": rc, "lam_box": float(lam[bx].sum()), "lam_tot": float(lam.sum()), "box_frac": frac, "B_box_e": B_box_e,
                    "lam_eff": lam_eff})
    return Kp, geo


def integrate_all(Kp, K0, peaks, geo, methods: dict, mh_prof, epc, g, half):
    """methods: name -> profile (stored counts per shell) or 'oracle' / 'annulus' / None (unserved)."""
    rows = []
    for k, (q, gg) in enumerate(zip(peaks, geo)):
        if gg is None:
            continue
        sl = (slice(gg["y0"], gg["y0"] + 2 * half + 1), slice(gg["x0"], gg["x0"] + 2 * half + 1))
        Kpp = Kp[sl]; K0p = K0[sl].astype(float)
        mh_box = float(np.clip(PK.profile_at(mh_prof, gg["rc"]), 0, None)[gg["box"]].sum())
        base = {"k": k, "dbin": q["dbin"], "pb": q["pb"], "s": q["s"], "sigma": q["sigma"], "lam_box": gg["lam_box"], "lam_tot": gg["lam_tot"],
                "box_frac": gg["box_frac"], "B_box_e": gg["B_box_e"], "lam_eff": gg["lam_eff"], "mh_box": mh_box, "K0_box": float(K0p[gg["box"]].sum())}
        for name, prof in methods.items():
            if prof is None:
                rows.append({**base, "method": name, "I": np.nan, "sn": np.nan, "sl": np.nan, "Bb": np.nan}); continue
            if isinstance(prof, str) and prof == "annulus":
                Bpx = None
            elif isinstance(prof, str) and prof == "oracle":
                Bpx = K0p
            else:
                Bpx = PK.profile_at(prof, gg["rc"])
            I, sn, slw, Bb = PK.integrate(Kpp, Bpx, gg["box"], gg["ann"], epc, g)
            rows.append({**base, "method": name, "I": I, "sn": sn, "sl": slw, "Bb": Bb})
    return rows


# ============================================================================= run (fly-scan)
def host_profiles(ctx, img, extra_bad, centre):
    valid = ctx.valid & ~extra_bad
    return PK.shell_means(img, valid, centre, N_SHELL)


def run_session(sid):
    ck = CK / "run" / f"{sid}.pkl"
    if ck.exists():
        return
    t0 = time.time()
    ctx = FlyCtx(sid)
    z = np.load(HOSTS / f"{sid}_hosts_v0.npz")
    sig = spot_sigmas(); box = PREREG["box"][8]; epc = 4.0
    recs, info = [], []
    for h in range(len(z["pos"])):
        if not z["addr_ok"][h]:
            continue
        pos, cls, row, frame = int(z["pos"][h]), str(z["cls"][h]), int(z["row"][h]), int(z["frame"][h])
        tr = ctx.tab_rows.loc[(row, frame)]; centre = (float(tr.cy), float(tr.cx)); ctx.H_cy, ctx.H_cx = centre
        K0 = z["frames"][h].astype(np.int64)
        mh0, n0 = host_profiles(ctx, K0, np.zeros(K0.shape, bool), centre)
        mu0 = ctx.invert(mh0)
        for rep in range(PREREG["replicates"]):
            rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 32) + 1000003 * pos + 17 * rep)
            peaks = place_peaks(rng, ctx.valid, centre, K8, box, sig)
            Kp, geo = inject_frame(rng, K0, peaks, mu0, centre, 8, ctx.tie, ctx.tau_b, box, epc, erec=ctx.erec)
            dm = disc_mask(K0.shape, peaks, box["disc_mask"])
            hp, hn = host_profiles(ctx, Kp, dm, centre)
            meth = {"O": "oracle", "A": "annulus"}
            pool_info = {}
            for v, name in (("V0", "B0"), ("V1", "B1"), ("V4", "B4"), ("P2", "P2")):
                prof, npool, W = ctx.pool_profile(pos, v, hp, hn); meth[name] = prof; pool_info[name] = (npool, W)
            try:
                meth["C"] = ctx.atlas_profile(row, frame, hp, hn)
            except Exception as e:  # noqa: BLE001
                log(f"  atlas failed {sid} {row}/{frame}: {e}"); meth["C"] = None
            fitinfo = {}
            try:
                res = ctx.physics_profile(hp, hn)
                meth["D"] = None if res is None else res[0]
                if res is not None:
                    fitinfo = {"t": res[1]["t"], "chi2_dof": res[1]["chi2_dof"], "delta": res[1]["delta"], "rms_pct": res[1]["rms_pct"]}
            except Exception as e:  # noqa: BLE001
                log(f"  physics failed {sid} {row}/{frame}: {e}"); meth["D"] = None
            meth["E"] = ctx.truth_profile(pos, hp, hn)
            rows = integrate_all(Kp, K0, peaks, geo, meth, mh0, epc, ctx.g, box["half"])
            for r_ in rows:
                r_.update({"sid": sid, "regime": cls, "row": row, "frame": frame, "rep": rep})
            recs += rows
            info.append({"sid": sid, "regime": cls, "row": row, "frame": frame, "rep": rep, "n_placed": int(sum(g is not None for g in geo)),
                         **{f"pool_n_{k}": v[0] for k, v in pool_info.items()}, **{f"pool_W_{k}": v[1] for k, v in pool_info.items()},
                         **{f"fit_{k}": v for k, v in fitinfo.items()}})
    pickle.dump({"rows": pd.DataFrame(recs), "info": pd.DataFrame(info)}, open(ck, "wb"))
    log(f"run {sid}: {len(info)} host-replicates, {len(recs)} peak-methods, {time.time() - t0:.0f} s")


def stage_run(args):
    (CK / "run").mkdir(parents=True, exist_ok=True)
    sids = args.sids.split(",") if args.sids else mod("rd").HELD
    for sid in sids:
        try:
            run_session(sid)
        except Exception:
            log(f"run {sid} FAILED:\n{traceback.format_exc()}")
            raise


# ============================================================================= synthetic check
def stage_synth():
    from podbg import binning as B
    (CK / "synth").mkdir(parents=True, exist_ok=True)
    sig = spot_sigmas(); box = PREREG["box"][8]; epc = 4.0
    allrows = []
    for sid in PREREG["synth"]["sids"]:
        ck = CK / "synth" / f"{sid}.pkl"
        if ck.exists():
            allrows.append(pickle.load(open(ck, "rb"))); continue
        ctx = FlyCtx(sid); rd = mod("rd")
        prof_rec = rd.truth_at(ctx.xT, ctx.TT, np.array([np.median(ctx.xT)]))[0]
        mu_prof = ctx.invert(prof_rec)
        exp_rec = np.full(N_SHELL, np.nan); fin = np.isfinite(mu_prof)
        exp_rec[fin] = np.asarray(B.recorded_mean(mu_prof[fin], 8, ctx.tie), float)
        centre = tuple(float(v) for v in ctx.s["centre_session"])
        yy, xx = np.indices((1024, 1024)); rr = np.hypot(yy - centre[0], xx - centre[1])
        mu_img = np.clip(PK.profile_at(mu_prof, rr), 0, None)
        rows = []
        for f in range(PREREG["synth"]["n_frames"]):
            rng0 = np.random.default_rng(500 + f)
            N = rng0.poisson(mu_img); K0 = B.quantise(N, 8, ctx.tie, rng=rng0).astype(np.int64)
            for rep in range(PREREG["replicates"]):
                for route in ("known", "latent"):
                    rng = np.random.default_rng(10000 + 97 * f + rep)                # same placement and draws order for both routes
                    peaks = place_peaks(rng, ctx.valid, centre, K8, box, sig)
                    Kp, geo = inject_frame(rng, K0, peaks, mu_prof, centre, 8, ctx.tie, ctx.tau_b, box, epc, N_known=(N if route == "known" else None), erec=ctx.erec)
                    rws = integrate_all(Kp, K0, peaks, geo, {"O": "oracle", "A": "annulus", "E_exact": exp_rec}, prof_rec, epc, ctx.g, box["half"])
                    for r_ in rws:
                        r_.update({"sid": sid, "frame": f, "rep": rep, "route": route})
                    rows += rws
        df = pd.DataFrame(rows); pickle.dump(df, open(ck, "wb")); allrows.append(df)
        log(f"synth {sid}: {len(df)} rows")
    D = pd.concat(allrows, ignore_index=True)
    D["ratio"] = D.I / D.lam_box; D["pull_l"] = (D.I - D.lam_box) / D.sl
    save_parquet(D, "synth_peaks_v0.parquet", [CK / "synth"], {"synth": PREREG["synth"]}, "synthetic-host injection check: per peak, route (known N / latent posterior), method")
    # pass rules
    rng = np.random.default_rng(0)
    cells = []
    for (m, db, pb), g in D.groupby(["method", "dbin", "pb"]):
        a = g[g.route == "known"].ratio.to_numpy(); b = g[g.route == "latent"].ratio.to_numpy()
        diff = np.median(a) - np.median(b)
        bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(200)]
        se = float(np.std(bs))
        cells.append({"method": m, "dbin": db, "pb": pb, "n": len(a), "med_known": float(np.median(a)), "med_latent": float(np.median(b)), "diff": float(diff), "se": se,
                      "agree": bool(abs(diff) <= max(0.01, 2 * se))})
    C = pd.DataFrame(cells)
    L = D[D.route == "latent"]
    mo = L[L.method == "O"].groupby(["dbin", "pb"]).ratio.median(); me = L[L.method == "E_exact"].groupby(["dbin", "pb"]).ratio.median()
    dE = (me - mo).reset_index(name="E_minus_O")
    pulls = L[(L.method == "E_exact") & (L.pb >= 0.3)].groupby(["dbin", "pb"]).pull_l.apply(PK.robust_sd).reset_index(name="pull_sd")
    v = {"i_routes_agree_frac": float(C.agree.mean()), "i_pass": bool(C.agree.mean() >= 0.9),
         "ii_max_abs_E_minus_O_pb_ge_1": float(dE[dE.pb >= 1].E_minus_O.abs().max()), "ii_pass": bool(dE[dE.pb >= 1].E_minus_O.abs().max() <= 0.01),
         "iii_pull_sd_range": [float(pulls.pull_sd.min()), float(pulls.pull_sd.max())], "iii_pass": bool(pulls.pull_sd.between(0.85, 1.15).all())}
    v["pass"] = bool(v["i_pass"] and v["ii_pass"] and v["iii_pass"])
    # POST HOC (after the pre-registered verdict was read; reported as such): the law-expected oracle intensity lam_eff.
    # (iv) mean O intensity per cell vs mean lam_eff (|diff| <= max(0.01 lam, 2 SE)); (v) E_exact pull about lam_eff,
    # sigma_law, plain SD 0.85-1.15 in cells with mean lam_eff >= 20 e (not a discrete distribution)
    L2 = L.assign(eff=L.lam_eff)
    iv = []
    for (db, pb), g in L2[L2.method == "O"].groupby(["dbin", "pb"]):
        d_ = g.I - g.eff; se = float(d_.std(ddof=1) / np.sqrt(len(d_)))
        iv.append({"dbin": db, "pb": pb, "mean_I_over_lam": float(g.I.mean() / g.lam_box.mean()), "mean_eff_over_lam": float(g.eff.mean() / g.lam_box.mean()),
                   "diff_over_lam": float(d_.mean() / g.lam_box.mean()), "se_over_lam": se / float(g.lam_box.mean()),
                   "agree": bool(abs(d_.mean()) <= max(0.01 * g.lam_box.mean(), 2 * se))})
    IV = pd.DataFrame(iv)
    E_ = L2[L2.method == "E_exact"].assign(pe=lambda x: (x.I - x.eff) / x.sl)
    V5 = E_.groupby(["dbin", "pb"]).agg(eff_mean=("eff", "mean"), pull_sd=("pe", "std"), pull_mean=("pe", "mean")).reset_index()
    V5u = V5[V5.eff_mean >= 20]
    v.update({"POSTHOC_iv_agree_frac": float(IV.agree.mean()), "POSTHOC_v_pull_sd_range_eff_ge_20": [float(V5u.pull_sd.min()), float(V5u.pull_sd.max())],
              "POSTHOC_v_pass_frac": float(V5u.pull_sd.between(0.85, 1.15).mean())})
    save_csv(IV.merge(V5, on=["dbin", "pb"]), "synth_lawcheck_v0.csv", [OUT / "synth_peaks_v0.parquet"], {"posthoc": True},
             "POST HOC synthetic check: oracle mean intensity vs the law-expected lam_eff; E_exact pulls about lam_eff")
    save_csv(C, "synth_cells_v0.csv", [OUT / "synth_peaks_v0.parquet"], {"rule": PREREG["synth"]["pass"]}, "synthetic check per cell: known vs latent route")
    save_csv(dE.merge(pulls, on=["dbin", "pb"], how="left"), "synth_exact_v0.csv", [OUT / "synth_peaks_v0.parquet"], {}, "synthetic: E_exact - O median ratio and E_exact sigma_law pull SD per cell")
    save_json(v, "synth_verdict_v0.json", [OUT / "synth_peaks_v0.parquet"], {"rule": PREREG["synth"]["pass"]}, "synthetic injection check verdict")
    log("synth verdict: " + json.dumps(v))
    log("\n" + L.groupby(["method", "pb"]).ratio.median().unstack().round(3).to_string())


# ============================================================================= classic
def stage_classic():
    from podbg.model import s_level as L, structure_family as SF, bin4_data as B4
    c0 = mod("c0"); sf2 = mod("sf2")
    (CK / "classic").mkdir(parents=True, exist_ok=True)
    prof = pickle.load(open(paths.SUB / "s_level_bin4_v0" / "checkpoints" / "profiles_v0.pkl", "rb")) if (paths.SUB / "s_level_bin4_v0" / "checkpoints" / "profiles_v0.pkl").exists() else \
        pickle.load(open(paths.SUB / "s_level_bin4_v0" / "classic_profiles_v0.pkl", "rb"))
    FRD = paths.SUB / "s_level_bin4_v0" / "frames"
    S_fn, kern, ff = sf2.fam_parts("flyscan")
    sig = spot_sigmas(); box = PREREG["box"][4]; epc = 1.0; g = PK.StoredVariance(4, None)
    for sd in sorted(FRD.iterdir()):
        sid = sd.name; ck = CK / "classic" / f"{sid}.pkl"
        if ck.exists() or sid not in prof or prof[sid].get("profile") is None:
            continue
        t0 = time.time()
        files = sorted(sd.glob("*.npz"))
        frames = [np.load(f)["frame"] for f in files]
        keep = [i for i, a in enumerate(frames) if np.issubdtype(a.dtype, np.integer)]
        frames = [frames[i] for i in keep]
        if len(frames) < PREREG["classic_min_frames"]:
            pickle.dump({"rows": pd.DataFrame(), "info": pd.DataFrame([{"sid": sid, "skipped": f"{len(frames)} frames"}])}, open(ck, "wb")); continue
        S0 = np.sum([f.astype(float) for f in frames], axis=0)
        bad, stop, sinfo = c0.bad_mask_for(S0)
        centre = tuple(float(v) for v in prof[sid]["profile"].meta["centre"])
        from scipy import ndimage as ndi
        badd = ndi.binary_dilation(bad, iterations=1); badd[:2, :] = badd[-2:, :] = True; badd[:, :2] = badd[:, -2:] = True
        valid = ~badd
        ny, nx = S0.shape; nsh = int(np.hypot(max(centre[0], ny - centre[0]), max(centre[1], nx - centre[1]))) + 2
        s_sh = (np.arange(nsh) + 0.5) * B4.K4
        profs = [PK.shell_means(f, valid, centre, nsh) for f in frames]
        hosts = np.unique(np.round(np.linspace(0, len(frames) - 1, min(PREREG["n_host_classic"], len(frames)))).astype(int))
        recs, info = [], []
        for h in hosts:
            K0 = frames[h].astype(np.int64); mh0, n0 = profs[h]
            others = [profs[j][0] for j in range(len(frames)) if j != h]
            for rep in range(PREREG["replicates"]):
                rng = np.random.default_rng(int.from_bytes(sid.encode()[1:9], "little") % (2 ** 32) + 7919 * int(h) + 31 * rep)
                peaks = place_peaks(rng, valid, centre, B4.K4, box, sig)
                Kp, geo = inject_frame(rng, K0, peaks, mh0, centre, 4, None, 0.0, box, epc, erec=lambda m: np.asarray(m, float))
                dm = disc_mask(K0.shape, peaks, box["disc_mask"])
                hp, hn = PK.shell_means(Kp, valid & ~dm, centre, nsh)
                pool = np.nanmean(np.array(others), axis=0)
                meth = {"O": "oracle", "A": "annulus", "Bc": pool * PK.window_scale(hp, hn, pool, s_sh)}
                fitinfo = {}
                try:
                    pr = L.build_profile(f"{sid}_host", [Kp], bad | dm, centre, B4.K4)
                    fit = SF.fit_holdout(pr, S_fn, kern, SF.HoldoutSpec(mode="gate", tie=False, a_h=sf2.A_H_FIXED, f_flat=ff))
                    mfit = full_gate_model(fit, pr.s, np.zeros(pr.n), S_fn, kern, pr.E_frame, 4)
                    assert np.allclose(mfit, fit["model"], rtol=1e-6, atol=1e-9)
                    meth["D"] = np.clip(full_gate_model(fit, s_sh, np.zeros(nsh), S_fn, kern, pr.E_frame, 4), 0, None)
                    fitinfo = {"t": fit["t"], "chi2_dof": fit["chi2_dof"], "rms_pct": fit["rms_pct"]}
                except Exception as e:  # noqa: BLE001
                    log(f"  classic physics failed {sid} host {h}: {e}"); meth["D"] = None
                rows = integrate_all(Kp, K0, peaks, geo, meth, mh0, epc, g, box["half"])
                for r_ in rows:
                    r_.update({"sid": sid, "regime": "classic", "row": -1, "frame": int(h), "rep": rep})
                recs += rows
                info.append({"sid": sid, "regime": "classic", "frame": int(h), "rep": rep, "n_placed": int(sum(q is not None for q in geo)), "n_pool": len(others), **{f"fit_{k}": v for k, v in fitinfo.items()}})
        pickle.dump({"rows": pd.DataFrame(recs), "info": pd.DataFrame(info)}, open(ck, "wb"))
        log(f"classic {sid}: {len(frames)} frames, {len(hosts)} hosts, {len(recs)} rows, {time.time() - t0:.0f} s")


# ============================================================================= tables
def load_all():
    R, I = [], []
    for d in ("run", "classic"):
        for p in sorted((CK / d).glob("*.pkl")):
            x = pickle.load(open(p, "rb"))
            if len(x["rows"]):
                R.append(x["rows"])
            I.append(x["info"])
    D = pd.concat(R, ignore_index=True); Info = pd.concat(I, ignore_index=True)
    D["ratio"] = D.I / D.lam_box
    D["z"] = D.I / D.sl
    D["pull_l"] = (D.I - D.lam_box) / D.sl; D["pull_n"] = (D.I - D.lam_box) / D.sn
    epc = np.where(D.regime == "classic", 1.0, 4.0)
    D["beta"] = epc * (D.K0_box - D.Bb) / (epc * D.mh_box)
    # POST HOC (added after the first session showed that the per-peak median beta is degenerate at bin-8 box sums of
    # 0-few counts): pulls about the law-expected oracle intensity lam_eff
    D["pull_eff"] = (D.I - D.lam_eff) / D.sl
    return D, Info


def q25(x):
    return float(np.nanquantile(x, 0.25)) if np.isfinite(x).any() else np.nan


def q75(x):
    return float(np.nanquantile(x, 0.75)) if np.isfinite(x).any() else np.nan


def stage_tables():
    D, Info = load_all()
    save_parquet(D, "peaks_v0.parquet", [CK / "run", CK / "classic"], {"prereg": PREREG}, "per injected peak and method: I, sigmas, lambda_box, background sums, ratio, pulls, beta")
    save_csv(Info, "hosts_v0.csv", [CK / "run", CK / "classic"], {}, "per host-replicate: placed peaks, pool sizes and windows, physics fit summary")
    ok = D[np.isfinite(D.I)]
    # oracle ratio per peak, to form ratio / oracle
    o = ok[ok.method == "O"][["sid", "regime", "row", "frame", "rep", "k", "ratio"]].rename(columns={"ratio": "ratio_O"})
    ok = ok.merge(o, on=["sid", "regime", "row", "frame", "rep", "k"], how="left")
    ok["rel_O"] = ok.I / (ok.ratio_O * ok.lam_box)
    ok["dI_rel"] = (ok.I - ok.ratio_O * ok.lam_box) / ok.lam_box          # POST HOC: the background's part of the intensity error
    keys = ["method", "regime", "dbin", "pb"]
    cell = ok.groupby(keys).agg(n=("ratio", "size"), n_sessions=("sid", "nunique"), med=("ratio", "median"), q25=("ratio", q25), q75=("ratio", q75),
                                med_relO=("rel_O", "median"), lost=("I", lambda x: float((x <= 0).mean())), below3=("z", lambda x: float((x < 3).mean())),
                                pull_l_med=("pull_l", "median"), pull_l_sd=("pull_l", PK.robust_sd), pull_n_med=("pull_n", "median"), pull_n_sd=("pull_n", PK.robust_sd),
                                beta_med=("beta", "median"), beta_q25=("beta", q25), beta_q75=("beta", q75),
                                mean_ratio=("ratio", "mean"), dI_rel_med=("dI_rel", "median"), dI_rel_mean=("dI_rel", "mean"),
                                pull_eff_med=("pull_eff", "median"), pull_eff_sd=("pull_eff", PK.robust_sd)).reset_index()
    ps = ok.groupby(keys + ["sid"]).agg(n=("ratio", "size"), med=("ratio", "median"), medO=("rel_O", "median")).reset_index()
    ps = ps[ps.n >= PREREG["within"]["min_peaks_session"]]
    save_csv(cell, "cells_v0.csv", [OUT / "peaks_v0.parquet"], {"keys": keys}, "per method x regime x d bin x P/B: median ratio, IQR, ratio/oracle, lost, below 3 sigma, pulls, beta")
    save_csv(ps, "cells_by_session_v0.csv", [OUT / "peaks_v0.parquet"], {"min_peaks": PREREG["within"]["min_peaks_session"]}, "per-session medians per cell")
    # decisions
    rows = []
    for (m, reg, db), g in cell.groupby(["method", "regime", "dbin"]):
        out = {"method": m, "regime": reg, "dbin": db}
        for X in PREREG["within"]["X"]:
            for tag, col, pcol in (("truth", "med", "med"), ("relO", "med_relO", "medO")):
                cells = {}
                for pb in LADDER:
                    c = g[g.pb == pb]
                    if not len(c):
                        continue
                    sm = ps[(ps.method == m) & (ps.regime == reg) & (ps.dbin == db) & (ps.pb == pb)][pcol].to_numpy()
                    cells[pb] = (sm, float(c[col].iloc[0]))
                out[f"pbstar_{int(X * 100)}_{tag}"] = PK.pb_star(cells, LADDER, X)
        out["beta_med"] = float(ok[(ok.method == m) & (ok.regime == reg) & (ok.dbin == db)].beta.median())
        b = ok[(ok.method == m) & (ok.regime == reg) & (ok.dbin == db)].beta
        out["beta_q25"] = q25(b); out["beta_q75"] = q75(b)
        # POST HOC: mean beta (the expected background error; the per-peak median is degenerate at bin-8 counts), and its
        # 68 % interval from a bootstrap over sessions; per-session mean beta spread
        bb = ok[(ok.method == m) & (ok.regime == reg) & (ok.dbin == db)]
        # ratio of sums (the per-peak ratio is undefined where the host's expected box background is 0 counts)
        agg = lambda x: float((x.K0_box - x.Bb).sum() / x.mh_box.sum()) if x.mh_box.sum() > 0 else np.nan
        sm = bb.groupby("sid").apply(agg).dropna()
        out["beta_mean"] = agg(bb)
        if len(sm) >= 3:
            rng = np.random.default_rng(0)
            bs = [np.mean(rng.choice(sm.to_numpy(), len(sm))) for _ in range(500)]
            out["beta_mean_sess"] = float(sm.mean()); out["beta_mean_lo"] = float(np.quantile(bs, 0.16)); out["beta_mean_hi"] = float(np.quantile(bs, 0.84))
            out["beta_sess_sd"] = float(sm.std(ddof=1)); out["n_sess"] = int(len(sm))
        rows.append(out)
    Dec = pd.DataFrame(rows)
    # honesty
    hon = []
    for (m, reg), g in ok.groupby(["method", "regime"]):
        for grp, (lo, hi) in PREREG["honest"]["pb_groups"].items():
            x = g[(g.pb >= lo) & (g.pb <= hi)]
            for sname, col in (("law", "pull_l"), ("naive", "pull_n"), ("law_about_lam_eff_POSTHOC", "pull_eff")):
                sd = PK.robust_sd(x[col]); md = float(x[col].median())
                hon.append({"method": m, "regime": reg, "pb_group": grp, "sigma": sname, "n": len(x), "pull_med": md, "pull_sd": sd,
                            "honest": bool(0.85 <= sd <= 1.15 and abs(md) <= 0.5)})
    Hon = pd.DataFrame(hon)
    # recommendation
    live = ["A", "B0", "B1", "B4", "P2", "C", "D", "Bc"]; order = {"A": 0, "B0": 1, "B1": 1, "B4": 1, "P2": 1, "Bc": 1, "C": 2, "D": 3}
    rec = []
    for (reg, db), g in Dec[Dec.method.isin(live)].groupby(["regime", "dbin"]):
        g = g.assign(star=g.pbstar_5_truth.fillna(np.inf), ab=g.beta_med.abs(), o=g.method.map(order)).sort_values(["star", "ab", "o"])
        b = g.iloc[0]
        rec.append({"regime": reg, "dbin": db, "d": PREREG["d_labels"][db], "recommended": b.method, "pbstar_5": b.pbstar_5_truth, "pbstar_10": b.pbstar_10_truth,
                    "beta_med": b.beta_med, "beta_q25": b.beta_q25, "beta_q75": b.beta_q75,
                    "runner_up": g.iloc[1].method if len(g) > 1 else None, "runner_up_pbstar_5": g.iloc[1].pbstar_5_truth if len(g) > 1 else None})
    Rec = pd.DataFrame(rec)
    save_csv(Dec, "decisions_v0.csv", [OUT / "cells_v0.csv"], {"within": PREREG["within"]}, "P/B* (smallest P/B from which every cell is within X %), truth and relative to the oracle; median beta")
    save_csv(Hon, "honesty_v0.csv", [OUT / "peaks_v0.parquet"], {"honest": PREREG["honest"]}, "pull median and robust SD per method / regime / P/B group / sigma model")
    save_csv(Rec, "recommendation_v0.csv", [OUT / "decisions_v0.csv"], {"rule": PREREG["recommend"]}, "recommended live background method per regime and d bin")
    # key table (markdown): median [q25-q75] per method x (d bin x P/B), per regime
    md = []
    for reg in ["F", "BA", "CN", "classic"]:
        md.append(f"\n### regime {reg}\n")
        sub = cell[cell.regime == reg]
        pbs = [0.1, 0.3, 1.0, 3.0, 30.0]
        hdr = "| method | " + " | ".join(f"{db} P/B {pb:g}" for db in DBINS for pb in pbs) + " |"
        md.append(hdr); md.append("|" + "---|" * (1 + len(DBINS) * len(pbs)))
        for m in ["O", "A", "B0", "B1", "B4", "P2", "Bc", "C", "D", "E"]:
            x = sub[sub.method == m]
            if not len(x):
                continue
            cells_ = []
            for db in DBINS:
                for pb in pbs:
                    c = x[(x.dbin == db) & (x.pb == pb)]
                    cells_.append(f"{c.med.iloc[0]:.3f} [{c.q25.iloc[0]:.2f}-{c.q75.iloc[0]:.2f}]" if len(c) else "-")
            md.append(f"| {m} | " + " | ".join(cells_) + " |")
    md.append("\n### POST HOC: mean beta (background error / true background under the box) per method x d bin, by regime\n")
    piv = Dec[Dec.method != "O"].pivot_table(index=["regime", "method"], columns="dbin", values="beta_mean")
    md.append(piv.round(3).to_markdown())
    (OUT / "key_table_v0.md").write_text("\n".join(md)); leak_check(OUT / "key_table_v0.md")
    write_sidecar(OUT / "key_table_v0.md", inputs=[str(OUT / "cells_v0.csv")], params={}, note="key table: median ratio [IQR] per method x d bin x P/B per regime")
    pd.set_option("display.width", 250)
    log("\n" + Rec.to_string())
    log("\n" + Dec.pivot_table(index=["regime", "method"], columns="dbin", values="pbstar_5_truth", aggfunc="first").to_string())
    log("\n" + Dec.pivot_table(index=["regime", "method"], columns="dbin", values="beta_mean").round(4).to_string())
    log("\n" + Hon[Hon.sigma == "law"].pivot_table(index=["regime", "method"], columns="pb_group", values="pull_sd").round(2).to_string())


# ============================================================================= figures
def stage_figs():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    cell = pd.read_csv(OUT / "cells_v0.csv")
    cols = {"O": "0.3", "A": "tab:orange", "B0": "tab:red", "B1": "tab:pink", "B4": "tab:purple", "P2": "tab:brown", "C": "tab:blue", "D": "tab:green", "E": "k", "Bc": "tab:red"}
    for reg in ["F", "BA", "CN", "classic"]:
        sub = cell[cell.regime == reg]
        if not len(sub):
            continue
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        for ax, db in zip(axes, DBINS):
            for m, g in sub[sub.dbin == db].groupby("method"):
                g = g.sort_values("pb")
                ax.plot(g.pb, g.med, "o-", color=cols.get(m, "c"), label=m, lw=1.2, ms=4)
                ax.fill_between(g.pb, g.q25, g.q75, color=cols.get(m, "c"), alpha=0.08)
            ax.axhline(1, color="k", lw=0.5); ax.axhspan(0.95, 1.05, color="0.9", zorder=0)
            ax.set_xscale("log"); ax.set_ylim(0, 2.0); ax.set_title(f"{reg}: d {PREREG['d_labels'][db]}"); ax.set_xlabel("P/B (box)")
        axes[0].set_ylabel("recovered / true intensity (median, IQR band)"); axes[-1].legend(fontsize=8)
        fig.tight_layout(); f = FIG / f"ratio_vs_pb_{reg}_v0.png"; fig.savefig(f, dpi=90); plt.close(fig)
        write_sidecar(f, inputs=[str(OUT / "cells_v0.csv")], params={"regime": reg}, note="median recovered/true ratio vs P/B per d bin and method")
    dec = pd.read_csv(OUT / "decisions_v0.csv")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ax, reg in zip(axes, ["F", "BA", "CN", "classic"]):
        sub = dec[(dec.regime == reg) & (dec.method != "O")]
        for k, (m, g) in enumerate(sub.groupby("method")):
            g = g.set_index("dbin").reindex(list(DBINS))
            x = np.arange(4) + (k - 4) * 0.08
            ax.errorbar(x, g.beta_mean_sess, yerr=[g.beta_mean_sess - g.beta_mean_lo, g.beta_mean_hi - g.beta_mean_sess], fmt="o", color=cols.get(m, "c"), label=m, ms=4, lw=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(4)); ax.set_xticklabels([PREREG["d_labels"][d] for d in DBINS], fontsize=8)
        ax.set_title(f"mean background bias beta (session bootstrap 68 %), {reg}")
    axes[0].set_ylabel("(true - method background) / true background under the box"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); f = FIG / "beta_by_regime_v0.png"; fig.savefig(f, dpi=90); plt.close(fig)
    write_sidecar(f, inputs=[str(OUT / "decisions_v0.csv")], params={}, note="POST HOC mean background bias beta (session-mean, bootstrap 68 % over sessions) per method, d bin and regime")
    hon = pd.read_csv(OUT / "honesty_v0.csv")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    h = hon[hon.pb_group == "low"]
    for k, (sg, mk) in enumerate((("law", "o"), ("naive", "s"))):
        x = h[h.sigma == sg].reset_index(drop=True)
        ax.scatter(np.arange(len(x)) + 0.15 * k, x.pull_sd, marker=mk, label=f"sigma_{sg}")
    lab = h[h.sigma == "law"].reset_index(drop=True)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels([f"{a}/{b}" for a, b in zip(lab.regime, lab.method)], rotation=90, fontsize=7)
    ax.axhspan(0.85, 1.15, color="0.9"); ax.set_ylabel("pull robust SD (P/B <= 1)"); ax.legend(); ax.set_yscale("log")
    fig.tight_layout(); f = FIG / "pull_sd_v0.png"; fig.savefig(f, dpi=90); plt.close(fig)
    write_sidecar(f, inputs=[str(OUT / "honesty_v0.csv")], params={}, note="pull robust SD per regime/method for the two sigma models")
    log("figs written")


# ============================================================================= main
def main():
    global LOGFILE
    ap = argparse.ArgumentParser(); ap.add_argument("--stages", default="spotshape,stream,synth,run,classic,tables,figs"); ap.add_argument("--sids", default="")
    a = ap.parse_args()
    for d in (OUT, LOGS, CK):
        d.mkdir(parents=True, exist_ok=True)
    LOGFILE = LOGS / f"peak_recovery_{time.strftime('%Y%m%d_%H%M%S')}.log"
    for st in a.stages.split(","):
        t0 = time.time(); log(f"== stage {st}")
        {"spotshape": stage_spotshape, "stream": stage_stream, "synth": stage_synth, "run": lambda: stage_run(a), "classic": stage_classic,
         "tables": stage_tables, "figs": stage_figs}[st]()
        log(f"== stage {st} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
