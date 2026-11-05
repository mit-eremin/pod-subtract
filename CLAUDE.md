# pod/bgsub: background estimation and subtraction for diffraction frames

Subproject of `pod/` (see `../CLAUDE.md` for the shared conventions). Extracted from `pod/background` on 2026-11-05
(background commit f1e93f5), where it was "item 2" / `podbg.bgsub`. This file is the durable memory of the
subproject. A cold session reads, in order: `HANDOFF.md`, this file, `ROADMAP.md`, the tail of `LOG.md`.
Owner: Dmitry Eremin (GitHub `mit-eremin`, Nelson Lab, Caltech). Repo (to be created by the owner):
`mit-eremin/pod-bgsub`.

## Goal (owner's words, 2026-11-04 re-scope + 2026-11-05 scope correction)

A very FAST, MATHEMATICALLY CORRECT, UNIVERSAL background ESTIMATION AND SUBTRACTION algorithm for diffraction frames,
better than the flat local backgrounds of XDS / DIALS (theirs are flat and lose intensity; do not copy their code).
Correctness means: unbiased background everywhere and under masked spots, honest per-pixel background variance,
correct behaviour under the detector's real count statistics, near the beam stop, grid bars and crystals, and a fast
background estimate.

- BACKGROUND ONLY. Peak integration is NOT part of the product. It is NOT for use with XDS (the owner has other
  purposes). The integrators in the code (G-ML, LC, intervals, coverage, integration speed) were the orchestrator's
  addition; they are kept, but bgsub is judged on the background only.
- Injected peaks may still be used as a TEST, comparing against an ORACLE-background subtraction, not against the true
  intensity (the bin-8 counting-law loss is a detector property, not a subtraction error).
- NO OTHER PACKAGES: do not propose DIALS, nXDS or similar; the methods are developed here. XDS was only ever a
  reference measurement.
- The forward model / simulator stays in `pod/background` (its priority 1); here it is a test bench only.

## The algorithm (v1 default; v2, v2.1 opt-in; v0 reproducible)

1. Sufficient statistics on polar cells (1-px rings x 64 sectors x 2 checkerboard halves) of the STORED counts.
2. A linear smooth background model in stored counts: fine radial cubic B-spline (rings, steep halo) + azimuthal
   harmonics m <= 6 on a coarse radial spline + four centre / ellipticity derivative columns + (v1) penumbra columns
   on the 1-4 px next to the beam stop; (v2) re-centring and a shrunken 1-px ring refinement, (v2.1) on elliptical rings.
3. Cross-fitted weighted least squares: weights = the count-law variance taken from the OTHER checkerboard half, so the
   estimator is linear in the counts and unbiased for ANY count law when the truth is in the model's span (no median,
   no Poisson assumption).
4. Spots, hot pixels and zingers excluded by one-sided count-law tests (alpha 1e-6 per pixel and per 3 x 3 window);
   spots the caller will measure are ALWAYS excluded by discs (`disc_mask`).
5. Variance: sandwich with the frame's Pearson dispersion (`res.var`, lazy `var_map`); optional local layer (v2.1 H2.1):
   the global G2.1 by default, plus a local annulus residual only where its signal is significant (T > 2 with block SEs),
   with the spot's own light removed (spill correction).

Derivations: `notes/reports/2026-11-04_bgsub_algorithm_v0.md` Sec. 1.1-1.6 (observation model, why flat/plane local
backgrounds are biased, the estimator, GL, exclusion, speed). Changes per version: `_v1.md` Sec. 2, `2026-11-05_
bgsub_algorithm_v2.md` Sec. 2, `_v2_1.md` Sec. 2. Code map: `src/bgsub/README.md`. Path mapping between the copied
reports and this repo: `notes/reports/README.md`.

## Current status and validity, judged on the BACKGROUND ONLY (2026-11-05)

v2.1 is NOT VALID by its own pre-registered verdict, by two narrow margins, both about the background:
- R1 (synthetic unbiasedness, 60 lacey cells, |beta| <= max(0.5 %, 3.2 SE)): the HYBRID H2.1 fails 1/60 cells, the
  sparse bin-8 cell (bin 8, dose x3, plain, d 1.0-1.16 A; 0.2 stored counts per box): -2.46 +- 0.72 %. The GLOBAL G2.1
  passes 0/60 (but sits at -1.98 % = 3.05 SE in that cell on those frames; on 64 fresh frames H2.1 is -0.47 +- 0.72 %,
  A276). Not power-adequate: needs a re-test with enough frames.
- R5 (speed of the background estimate, 1024^2, numba, plan cached, <= 50 ms): 54.2 ms under machine load (v1 46.3 and
  v2 50.2 ms in the same run; the elliptical ring kernels cost +2.7 ms, A278).
- Passing: R2 (count statistics / truncation, paired, 24 cells), R4 (real, better than FLAT, per regime x d bin, spots
  without host overlap: F, BA, CN, classic).
- Background error next to the stop, real frames, D0 (d 4-8 A), box-noise units, spots without host overlap:
  v2.1 H2.1 / G2.1 / FLAT: fly-scan clean film F 0.08 / 0.32 / 0.40; bar-adjacent BA 0.34 / 0.46 / 0.50; crystal
  neighbour CN 0.37 / 0.57 / 0.60; classic bin 4 1.57 / 1.58 / 2.98; classic confirmation 0.50 / 0.48 / 1.30. With ALL
  spots (host overlap included) classic D0 H2.1 7.20 vs FLAT 5.96 (sample Bragg spots in the host: an overlap, not
  background) and F 0.36 (A279). Outside D0, H2.1 = G2.1 within 0.01-0.03; synthetic plain frames 0.07-0.16 box noise vs
  FLAT 0.38-0.62.
- Known background defects: a sharp film ring narrower than the spline knots (classic sb7857cd8, 4.6 A) is NOT removed
  (34-81 % of its pixels are count-law flags, A277); structure that is neither smooth on the model's scales nor linear
  at the annulus scale is not modelled.
- OUT OF SCOPE NOW (kept in code, not product criteria): G-ML / LC intensities, intervals and their coverage (R3,
  nominal 0.66-0.69 in v2.1), integration speed (286 ms / 35 spots), R6.
- Open defect in the evaluation code: A281, the real-table merge duplicates ~10 % of rows in the v2 and v2.1 pull and
  per-P/B tables (R4 and D0 unaffected). Script: `legacy/podbg_scripts/bgsub_v2_1_tables.py` (merge on sid:frame).

## Detector coverage

| detector / mode | status |
|---|---|
| Apollo bin 8 (fly-scan, 1024^2) | well tested: real fly-scan test sessions with injected peaks in the F / BA / CN regimes (17 sessions in v0; the v1-v2.1 harness reuses them) and synthetic |
| Apollo bin 4 (classic, 2048^2) | real classic sessions (20 test + 9 confirmation), synthetic; Poisson-like (1 count = 1 e) |
| Apollo bin 16 (512^2) | SYNTHETIC ONLY; variance not honest (saturated spots on the wrong dead-time branch, g oscillating; A196) |
| CETA-D bin 2 (integrating) | NOT TESTED; needs an integrating-detector variance model (gain, read noise, no rounding) first |
| TVIPS F30 (TF30, 300 kV) | NOT TESTED; `pod/ringcal` has TVIPS readers (`ringcal.io`, local and remote header parsing) |

## Detector facts that matter (Apollo; measured in pod/background, full text there)

- STORED-COUNT LAW: each stored frame is the round-half-to-even of the 3-raw-frame electron sum / m, with m = b^2 / 16
  electrons per count (1 e at bin 4, 4 e at bin 8, 16 e at bin 16). Mean and variance of the stored value as a
  function of the electron mean: `bgsub.vendor.binning.response_table`, `recorded_pmf`; in bgsub: `CountLaw.apollo(binning,
  tie_up, dead_time_tau)`. Bin 4 is effectively Poisson; bin 8 is strongly sub-linear at sparse counts (0.003-0.02
  stored counts per pixel at high angle); bin 16 is not invertible at archive fluxes.
- TIE RULE (gain flag on): blocks with an exact half-integer sum round UP with p: 0.403 +- 0.001 on film at 0.1-1 e
  (constant above), rising at low rate (0.67 at 0.02 e, 0.57 at 0.05, 0.41 at 0.1; 0.73 on bars, 0.53 on vacuum);
  gain flag off: parity (to even). The simulator default is p = 0.403; the bgsub examples use 0.35 (the B0 inversion).
  `binning.tie_prob` holds the rate law.
- DEAD TIME: paralyzable, tau = 0.206 +- 0.015 per electron per physical pixel per raw frame (418 us); per output pixel
  per stored frame tau_b = tau / (b/2)^2 / 3 (`vendor.detector.tau_for_binning`). Ceiling 85.7 e per bin-8 px per 49-ms
  frame; 5 % loss at 15 e/px/s, 20 % at 66; past the peak the stored value FALLS (bright spot cores can sit on the wrong
  branch). Valid below 0.95 of the ceiling.
- ZERO LEVEL: archive TIFFs are MRC + 1 (A184); declare it (`CountLaw.with_offset(1)`).
- Fixed features: seams (cols 255-256, 511-512, 767-768; rows 511-512 at bin 8), row 0, column 0 and the 1-px border
  are a sub-threshold offset: MASK them (`vendor.detector.seam_lines`, `fixed_pattern_mask`). A physical beam stop
  (disc ~52 px at bin 8 + an arm to the left) is in every Apollo fly-scan frame; its arm casts a 1-4 px penumbra.
- Bar-frame floor (gain on): multi-event clusters surviving rounding, a per-session constant; additive in counts.

## Layout

```
pod/bgsub/
  CLAUDE.md  HANDOFF.md  ROADMAP.md  LOG.md
  pyproject.toml          package bgsub 0.1.x; deps numpy, scipy; extras numba (optional; 2-5x faster), harness (pandas, pyarrow), dev
  src/bgsub/              THE PRODUCT: law, core, zone, refine, integrate, spotlaw, reference, _kernels (moved from podbg.bgsub)
    config.py             data roots (own: ~/pod_data/bgsub; read-only: ~/pod_data/background); env overrides
    vendor/               VENDORED podbg modules: binning (count law), detector (dead time, seams), masks, centre, provenance
    validation/           harness (not product API): peaks (injection; vendored podbg.peaks), harness (vendored
                          bgsub_v0_common: placement, injection, evaluation), sim_geometry + sim_detectors (vendored
                          podbg.sim pieces), simbank (renders frozen simulator-v1 frames), synth (the v1/v2/v2.1 design)
  scripts/                make_simbank_v1.py (BRIDGE: the only file importing podbg, run with ../background/.venv),
                          synth_v2_1.py (synthetic re-run), check_rerun_vs_background.py, speed_v2_1.py (R5)
  tests/                  unit + regression tests (v0/v1/v2 frozen references in tests/data), simbank, vendor headers,
                          one end-to-end reproduction of a published v2.1 synthetic frame (skips without ~/pod_data)
  legacy/podbg_scripts/   the background's real-frame / table / reference scripts, verbatim (import podbg; port before use)
  notes/reports/          reports copied from background (verbatim) + new ones; notes/figures/bgsub_v*/
```

VENDORING RULE (as in ringcal): every file in `vendor/` and the vendored files in `validation/` start with a header
naming the source file, the background commit, the source sha256, the edits, and "KEEP IN STEP".
`tests/test_vendor_headers.py` warns when a background source has drifted; re-copy (or record why not) in LOG.md.
No runtime import of podbg anywhere in `src/` or `tests/` (`tests/test_simbank.py::test_no_podbg_import`).

SIMULATOR: podbg.sim pulls ~9,000 lines of the background physics model and five theory tables, so it is NOT vendored.
Instead `scripts/make_simbank_v1.py` (run once with background's venv) froze simulator-v1 radial tables + geometry for
the 704 frames of the v1/v2/v2.1 synthetic design and the 12 R5 speed frames into `~/pod_data/bgsub/simbank_v1/`;
`bgsub.validation.simbank` renders them BIT FOR BIT (sha256-checked, all 716 verified 2026-11-05) and the v2.1 synthetic
checkpoints are reproduced exactly (`scripts/check_rerun_vs_background.py`). New simulator designs (other seeds, v2/v3.1
grids, CETA/TVIPS geometries) need a new bank from background, or a request to the background project.

## Data and conventions

- Own data root: `~/pod_data/bgsub/` (versioned subfolders: `simbank_v1/`, `synth_v2_1_rerun/`, `<study>_v<k>/`). Never
  in the repo, never in Dropbox.
- Read-only inputs, by path in `bgsub.config.BG_INPUTS`: `~/pod_data/background/peak_recovery_v0/` (fly-scan hosts,
  spot shapes), `s_level_bin4_v0/` (classic bin-4 frames, 796 MB), `features/`, `atlas/v1b/`, `bgsub_v0..v2_1/` (published
  tables and checkpoints). Never write there.
- Anonymous session ids (sids, e.g. sb7857cd8) only; never real names or paths in the repo. The private key lives with
  background (`podbg.registry`, `~/pod_data/background/private/`); do not copy it.
- Provenance sidecar (`<file>.provenance.json`, `bgsub.vendor.provenance.write_sidecar`) on every produced table.
- PRE-REGISTRATION: decision rules are written and committed BEFORE any test run; test seeds / sessions are not opened
  during development; no parameter changes after a test run; failures are reported with their cells. Anomalies are
  numbered B1, B2, ... (background used A1-A290).
- Versioned outputs and versioned algorithms (`version=` switch; old versions stay reproducible with frozen references
  in `tests/data/`; freeze a new reference from a clean committed state before changing behaviour).
- Frame references, never copies (path + frame index); metadata harvested by code (metadata-first rule), never typed.
- COMPUTE ON THE MAC. NO HPC JOBS. Data reads from the Caltech HPC archive (`ssh hpc`,
  `/resnick/groups/NelsonLab/data/DBE`) only over ssh with COREUTILS (dd/cat/head; never the remote python, which
  hangs: see background/CLAUDE.md) or via sshfs. The Apollo detector PC is READ-ONLY via `ssh apollo` (WSL).
- MULTIPROCESSING HAS DIED SILENTLY under the Claude Code harness (memory-pressure kills of background shells): default
  to one worker, checkpoint per frame under `~/pod_data/bgsub/`, and VERIFY COMPLETENESS (found / planned) before
  reading any table. Long jobs: inputs, logs and checkpoints under `~/pod_data/bgsub/`, never in a scratchpad.
- Timeouts: wrap long commands with `perl -e 'alarm N; exec @ARGV' ...`.

## Git

- Path-limited commits only: `git commit -m "..." -- <paths>`; NEVER `git add -A` / `git commit -a` (concurrent agents
  share working trees; a staging error swept a pre-registration into another commit in background, A264).
- The orchestrator pushes (remote `git@github.com:mit-eremin/pod-bgsub.git` once the owner has created it; SSH, no gh
  auth on this Mac).

## Session close-out (every session)

Run `/wrap` (dated `LOG.md` entry, `ROADMAP.md` update, commit) AND update Claude's persistent memory under
`~/.claude/projects/-Users-eremin-Library-CloudStorage-Dropbox-pod/memory/` with anything a cold session needs that the
repo does not carry (in-flight jobs and their durable paths, environment breakage, decisions taken in conversation).
Do it before long or risky steps too: sessions died on 2026-10-25 and 2026-10-26 and lost their unlogged reasoning.
