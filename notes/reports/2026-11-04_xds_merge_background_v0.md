# Does better background handling improve real crystallographic data? (xds merge v0)

Date: 2026-11-04. Code: `scripts/xds_merge_v0.py`, `src/podbg/xds_io.py`, `tests/test_xds_merge.py`. Outputs:
`~/pod_data/background/xds_merge_v0/` (public tables by anonymous ids; archive paths and sample identities under
`private/`). Figures: `notes/figures/xds_merge_v0/`. Anomalies from A183. XDS, XSCALE and xdsconv run locally on the
Mac only; the archive is only read (ssh `timeout 300 cat` and rsync, bounded). No HPC compute.

Dataset ids: rotation sessions use the `x..` keys of rotation v0 (`rotation_v0/private/xds/xsession_key_v0.json`);
movies inside a session are `m0..m7` in archive order; classic snapshot sessions use their podbg sids. The sample
behind each id is in `xds_merge_v0/private/dataset_key_v0.json`.

## 0.1 Feasibility (design note, committed before any comparison is run)

### What was checked

| item | finding | how |
|---|---|---|
| local nXDS | **does not run**: the installed `nxds` / `nxds_par` (built 2025-06-24) stop with "license expired on 30-Sep-2025" | ran `nxds` in an empty directory |
| local XDS | runs: XDS VERSION Apr 16, 2026 (built 2026-06-16); one 120-frame bin-4 movie takes 5 s, XYCORR to CORRECT, 8 threads | probe run on x08 m0 (`private/runs/probe_mov11_V0`) |
| archive rotation runs | 199 Apollo CL-420 movies (rotation v0). Sessions with a usable CORRECT result and a crystal of known published structure: **x08** (organic small molecule, 8 of 12 movies with a CORRECT result in SG 16; cell 5.17-5.21 / 10.26-10.37 / 20.81-21.10 A; data to 0.78 A; archive totals CC1/2 94.9-99.3 %, Rmeas 16-33 %) and **x04** (inorganic oxide, 6 of 9 movies in the right cell, about 300 unique reflections each). x07 (protein, CL 960, 4096 x 4096, ~7 GB per movie, 30 runs, 1.8 A) is too large for this round. x06 has no CORRECT result. | rsync of the small XDS files to `private/archive_xds/` |
| frames | each x08 movie is 121 uint16 TIFFs of 8 MB (961 MB) at ~100 MB/s; values are counts + 1 (TIFF = MRC + 1, rotation v0), minimum 1, median 1 (most pixels hold 0 electrons) | rsync of m0, tifffile |
| external background source | x08's grid also has a REyes classic dif map: classic sid **s4e3c1032** (same grid, same CL 420, bin 4, 0.25-s snapshots; 1,089 snapshots: No 514, Grid 252, Poor 164, Bad 113, Good 46). Format gain-normalised float32 (Fano 1.01). Rule 4 sample check: 0 of 26 tested No frames flagged, session not flagged, so it may supply a film background. Spec v1 rule 2 + P2 drop on the raster (step 5.0 um): Grid neighbours remove 137 No frames, Grid/Good/Bad neighbours 253, leaving **261** before the high-s outlier test. | `reyes_drop_v0/private/classic_frames_v0.parquet`, `contam_sessions_v0.csv`, `podbg.model.s_level.bar_adjacent_inrow` |
| x04 external source | none: x04's session has no REyes classic dif map in the archive, so there is no No-frame pool for it | classic session map |

### How XDS treats background and counts (this decides what an "external background" can do)

- INIT sets the dark current (BLANK.cbf) to a constant OFFSET = 0 for this detector type and estimates GAIN from the
  variance/mean of background pixels of frames 1-10. On these TIFFs GAIN = 0.37, because the +1 TIFF pedestal is
  counted as signal (mean background 1.64, of which 1.00 is pedestal; variance ~0.64). See A184.
- INTEGRATE needs `XPARM.XDS`, `X/Y-CORRECTIONS.cbf`, `BKGPIX.cbf` (trusted-pixel table), `BLANK.cbf` and `GAIN.cbf`
  (checked: it stops without each of the last three). For each reflection it estimates the background from the
  background pixels of the reflection's own box, flat within the box, and the variance of each pixel as GAIN x
  value. CORRECT then rescales the sigmas with its error model (a, b) fitted to the scatter of equivalents.
- So XDS has no input for an external background or a per-pixel variance. What an external subtraction can change:
  (i) the intensities, only through background structure WITHIN a box (curvature, steps at the stop edge, ring
  shoulders), because a flat or linear background is removed by the box estimate anyway; (ii) the sigmas, because XDS
  derives the variance from the pixel values.
- Consequence for sigma propagation: a pixel keeps a correct XDS variance only if its value stays equal to its
  counts (up to the gain). Any image whose local mean is lowered by a subtraction has its variance underestimated,
  and any pedestal added to avoid negatives overstates it where the background is low. Exact sigma propagation of a
  fully subtracted image is therefore IMPOSSIBLE in XDS. The one sigma-consistent form is to subtract the external
  model and add back its own smooth (low-pass) part as the pedestal, i.e. remove only the part of the background
  that varies on the scale of a reflection box. That keeps the local mean, hence XDS's variance, unchanged to first
  order, while removing the structure the flat box estimate cannot follow.
- Validity of the images (applies to every variant):
  - integers: the subtracted field is rounded by dithered cumulative rounding along each row (each row segment's sum
    is preserved to < 1 count, no 2-D bias pattern, seeded and reproducible);
  - no negatives and the offset: the archive pedestal (+1, MINIMUM_VALID_PIXEL_VALUE = 1) is kept, and a constant
    pedestal is added where needed so that no trusted pixel falls below 1 (no clipping, by construction);
  - the trusted-pixel set, the gain, the blank table and the geometry are taken from the baseline run and reused
    unchanged (JOB = INTEGRATE CORRECT), so the variants differ ONLY in the image values. This is necessary: DEFPIX
    defines untrusted pixels from the background level (VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS on BKGINIT), so a
    flattened image would otherwise change the trusted area, which is not a background effect.
- CC1/2 in XDS is computed from a random half-split: two runs on byte-identical input (V0 and its INTEGRATE CORRECT
  re-run) give identical intensities, I/sigma and Rmeas, and CC1/2 / anomalous correlation that differ in the last
  digit (A185). The comparison therefore needs a replicate to measure the noise floor.

### Route (a): classic snapshots (stills) with nXDS, merged with XSCALE

- NOT FEASIBLE on this Mac now: the local nXDS licence expired on 2025-09-30 (XDS itself cannot integrate stills).
- Even with a licence, the case is weak for the one known-structure session (s4e3c1032): 46 Good and 113 Bad 0.25-s
  stills of a small-cell orthorhombic crystal; at 200 kV each still is close to a single reciprocal-lattice plane, so
  the orientation (and therefore the partiality) of a still is poorly determined from its own spots; a merge would
  rest on a few thousand partial observations. The snapshots are gain-normalised floats and would need rounding
  (valid at Fano 1.01).
- The background argument is the same as for XDS (nXDS also estimates a local background per spot and derives the
  variance from the pixel values), so the snapshot route would test the same thing with weaker statistics.
- What is needed: a current nXDS build (or DIALS / CrystFEL stills processing), and ideally a session with many
  hundreds of Good snapshots of a known structure. Route (a) is not run.

### Route (b): rotation movies with XDS, own background versus an external one

- FEASIBLE, and the cleaner of the two: x08 has 8 movies with archive XDS runs in SG 16 to 0.78 A, of a published
  small-molecule structure, on a grid whose classic dif map supplies a P2-clean No-frame pool from the same session,
  detector, CL and binning. Local XDS reproduces a movie in seconds.
- Mapping the pool to a movie: the pool's mean radial profile (per 0.25-s snapshot, about the pool's own centre, stop
  masked) placed about the movie's beam centre by pixel radius (same detector, CL and binning), scaled per movie
  frame to the frame's own film level. The movie frames contain the crystal and a tilted film (|alpha| up to 60 deg),
  so the scale is fitted per frame and any crystal-specific diffuse background stays for XDS's own box estimate.
- x04 cannot take the pool route (no classic map) and is not used; x07 is left for a later round (size).
- Comparison against the model: the x08 structure is published, but no coordinate file is in the archive or the repo.
  A self-derived model (solved from the baseline data) can serve only as a descriptive check. The owner should provide
  the published coordinates (and confirm the identities in `private/dataset_key_v0.json`) for a model-based verdict.
- What would test external backgrounds properly: an integration program that accepts a fixed background model with
  its own variance (e.g. DIALS `background.algorithm=gmodel`), which is not installed here.

## 0.2 Pre-registration (committed before any variant image is made or processed)

Fixed now; nothing below is changed after a variant result is seen. Machine-readable copy: `PREREG` in
`scripts/xds_merge_v0.py`, written to `xds_merge_v0/prereg_v0.json` at the first stage.

### Datasets

- x08, movies m0-m7 (the 8 movies whose archive CORRECT ran in SG 16; key in `private/dataset_key_v0.json`), all
  121 TIFFs each, DATA_RANGE / SPOT_RANGE / BACKGROUND_RANGE as in each movie's archive XDS.INP.
- A movie whose V0 run (below) does not reach CORRECT in SG 16 is dropped from every variant and the pool; the drop
  is reported. No other dataset enters.

### Processing (identical for every variant except the image values)

- XDS.INP = the movie's archive XDS.INP with only these changes: NAME_TEMPLATE to the local copy;
  SPACE_GROUP_NUMBER = 16 and UNIT_CELL_CONSTANTS = 5.20 10.30 21.00 90 90 90 (the archive CORRECT input);
  RESOLUTION_SHELLS = 10 8 5 3 2.3 2.0 1.7 1.5 1.3 1.2 1.1 1.0 0.9 0.8 (the archive XSCALE shells);
  MAXIMUM_NUMBER_OF_PROCESSORS = 8, MAXIMUM_NUMBER_OF_JOBS = 1. Every other keyword (REFINE, trusted ranges,
  BACKGROUND_PIXEL, SIGNAL_PIXEL, INCLUDE_RESOLUTION_RANGE 40-0.78, ...) as archived.
- V0 (setup run): JOB = XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT on the original TIFFs. Its XPARM.XDS,
  X/Y-CORRECTIONS.cbf, BKGPIX.cbf (trusted pixels), BLANK.cbf and GAIN.cbf are copied unchanged to every variant.
- Every compared variant, the baseline included, is then run as JOB = INTEGRATE CORRECT in its own directory with
  those files; only NAME_TEMPLATE differs.
- Pooling: XSCALE over the movies of one variant (XDS_ASCII.HKL in m0..m7 order, SPACE_GROUP_NUMBER 16, the cell
  above, the shells above, MERGE = FALSE for the statistics below, defaults otherwise); one XSCALE run per variant.

### Variants

| id | images | role |
|---|---|---|
| V0r | original TIFFs (counts + 1) | baseline: XDS's own local background |
| V1 | I' = I - R(c) + Q, c = B - G25(B) | PRIMARY external variant: subtract the external background, add back its own Gaussian low-pass (sigma 25 px) as the pedestal; sigma-consistent to first order |
| V1b | V1 with a second dither seed | replicate: noise floor of the dither and the comparison |
| V2 | I' = I - R(B) + P | full external subtraction with the minimal constant pedestal; XDS's variance is knowingly wrong (Sec. 0.1); diagnostic |

- B, the external background for movie frame f: B_f(x) = k_f p(|x - c_movie|).
  - p(r): mean radial profile (1-px bins) of the POOL MEAN image about the pool centre, stop masked
    (`podbg.masks.beamstop_mask_v1` on the pool mean), bins with < 30 % valid pixels dropped and filled by linear
    interpolation; beyond the last valid bin, the mean of the last 20 valid bins.
  - POOL: s4e3c1032 snapshots labelled "No diffraction"; drop any whose same-row, same-block raster neighbour within
    +/-2.1 raster steps is Grid, Good or Bad (spec v1 rule 2 + P2, `s_level.bar_adjacent_inrow`; Poor is not a
    drop trigger, as in the spec); then the one-sided high-s test on what remains: q = C(1.00-1.10 A^-1) /
    C(0.80-0.92 A^-1) about each frame's pool centre, drop if (ln q - median) / (1.4826 MAD) > 3. The pool mean is
    the equal-weight mean of the kept frames (float, gain-normalised units). All kept frames are read (no cap).
  - s scale for the pool and the movie: QX / (lambda D) = 0.016 / (0.0251 x 560) = 0.0011383 A^-1 per px (the XDS
    geometry of the movies; the same detector, CL 420 and bin 4 for the snapshots).
  - Centres: `podbg.centre.centre_from_symmetry` on the pool mean (stop masked) and on each movie's frame sum
    (counts, pedestal removed; mask = V0's untrusted pixels), the movie's XPARM beam point as the starting value.
  - k_f: ratio of sums over trusted pixels (V0 BKGPIX) with s in 0.30-1.20 A^-1 of (I - 1) to p; two passes, the
    second excluding pixels with I - 1 > k p + 5 sqrt(k p) + 2 (Bragg spots).
- R(.): dithered cumulative rounding along each row: R(c)_j = floor(C_j + u) - floor(C_{j-1} + u), C = cumulative
  sum of c along the row over trusted pixels (c = 0 elsewhere), u ~ U(0, 1) per row, seeded (V1 and V2: seed
  1000 + movie index; V1b: 2000 + movie index).
- Q and P: the smallest non-negative integer per movie that keeps every trusted pixel of every frame >= 1
  (MINIMUM_VALID_PIXEL_VALUE); no clipping anywhere. Untrusted pixels keep their original values (XDS ignores
  them). A value above OVERLOAD (65000) is an error that stops the variant. Q, P and the fraction of pixels changed
  are reported.

### Statistics (computed by this project's code from the scaled, UNMERGED observations, so that the random
half-split of XDS does not enter the decision)

- Per shell and pooled: Rmeas; CC1/2 by the sigma-tau method (Assmann et al. 2016: from the per-reflection means and
  within-reflection variances, deterministic); merged I/sigma(I) (inverse-variance mean of the scaled observations);
  completeness; and N_use = number of unique reflections with d 0.9-1.1 A and merged I/sigma >= 2. XDS's own CC1/2,
  Rmeas, I/sigma, ISa, anomalous correlation and SigAno are reported beside them (descriptive).
- Shells: the 14 above; OUTER = (1.1, 1.0], (1.0, 0.9], (0.9, 0.8] A; KEY = 1.1-0.9 A.
- Per movie: the same statistics from each variant's own XDS_ASCII.HKL (CORRECT-scaled).

### Decision rule (per external variant, V1 and V2 separately, against V0r)

- Noise floor from the replicate, per metric and shell: delta = |V1b - V1| (pooled). Thresholds: CC1/2
  tau_CC = max(0.02, 2 delta_CC); Rmeas tau_R = max(5 % relative, 2 delta_R); N_use tau_N = max(5 %, 2 delta_N).
- IMPROVES if all of:
  (a) in >= 2 of the 3 OUTER shells, CC1/2 rises by >= tau_CC or Rmeas falls by >= tau_R;
  (b) in no OUTER shell CC1/2 falls by >= tau_CC or Rmeas rises by >= tau_R;
  (c) (V1 only) N_use does not fall by more than tau_N;
  (d) the KEY-range CC1/2 is higher than V0r's in >= 6 of the 8 movies (>= 75 % of the movies kept).
- DEGRADES if in >= 2 of the 3 OUTER shells CC1/2 falls by >= tau_CC or Rmeas rises by >= tau_R, or (V1 only) N_use
  falls by more than tau_N.
- Otherwise NO MEASURABLE EFFECT.
- I/sigma: reported per shell for every variant; it enters the rule only through N_use and only for V1 (V2's sigmas
  are not valid by construction).

### Secondary, descriptive only (no verdict rests on them)

- Anomalous correlation and SigAno per shell (in electron diffraction these measure dynamical Friedel differences,
  not anomalous scattering).
- Model check: if SHELXT (formula C10 H16 N2 O3 S, Z 4, P212121) solves the pooled V0r data with all 15
  non-H atoms, the solution is refined isotropically with SHELXL (10 cycles, X-ray scattering factors as an
  approximation, no H) against each variant's pooled merged data with identical instructions; R1 (Fo > 4 sigma),
  R1 (all) and wR2 are reported. The model is derived from the baseline data, so it is not an independent
  reference; the published coordinates (owner) are needed for that.

### 0.3 Amendments (each committed before the statistics it affects were computed)

1. Before any CORRECT result: XDS stops with "TOO MANY RESOLUTION SHELLS" on the 14-shell RESOLUTION_SHELLS line, so
   the keyword is dropped from XDS.INP (CORRECT prints its default table). The project's own statistics and XSCALE keep
   the 14 shells. The movies have 120 TIFFs each, not 121 (the 121st directory entry was not a frame).
2. After the variant runs, before any XSCALE run or statistic: XDS INTEGRATE stopped with "AUTOMATIC DETERMINATION OF
   SPOT SIZE PARAMETERS HAS FAILED" for V1 and V1b on m5 and for V2 on m0, m5 and m6 (V0r integrated all 8). The
   pre-registration did not say how a failed variant enters. Fixed now: a failure is itself a result (reported); each
   comparison is made on the movies where V0r, V1, V1b and the compared variant all reached CORRECT (S1 = 7 movies
   for V1; S2 = 5 movies for V2), with XSCALE re-run for every variant on that set; rule (d) needs ceil(0.75 n)
   movies.

---

## 1. Result in one paragraph

- By the pre-registered rule both external-background variants **DEGRADE** the x08 data: V1 (external model,
  smooth part kept as pedestal) on 7 movies and V2 (full subtraction) on 5. In every outer shell CC1/2 falls and Rmeas
  rises well beyond the thresholds; the key-range CC1/2 is higher in only 2 of 7 movies (V1) and 1 of 5 (V2). XDS
  could not integrate the variant images of 1 (V1) and 3 (V2) of the 8 movies that the baseline integrates.
- POST HOC, the cause is not the background model: adding only the variant's constant pedestal to the original
  images, with no background removed, reproduces each variant to <= 0.003 in CC1/2 and <= 1 % in Rmeas in every
  shell, and reproduces the same INTEGRATE failures (A187). The external background, which accounts for 99 % of the
  counts in the 0.30-1.20 A^-1 annulus of each frame, has no measurable effect of its own on XDS's integrated data.
- What does move the data is the IMAGE ZERO LEVEL. At these count levels (most pixels hold 0 electrons, the
  high-resolution background is ~0.1-0.3 e per pixel per frame) XDS's integrated intensities are not invariant to an
  added constant: +3 to +6 counts raises outer-shell Rmeas by 24-30 % and lowers the mean intensity at 0.8-0.9 A
  (relative to 1.3-2.0 A) by 14 %; removing the archive's own +1 pedestal changes it by -29 % (gain 1) or x4
  (with the archive's gain table, 0.37-0.63) (A186). The archive's processing counts the +1 TIFF pedestal as signal (A184).
- Route (a), stills with nXDS, was not run: the local nXDS licence expired in 2025 (Sec. 0.1).
- Verdict: under XDS, "better background handling" by subtracting an external film model does NOT improve real
  data quality, and cannot, because XDS's own per-reflection background already absorbs any smooth background and
  every valid way of writing a subtracted image changes the zero level, to which XDS is sensitive. The lever that
  matters for Apollo data in XDS is the treatment of the zero level and gain (A184, A186). Which zero level gives the
  RIGHT high-resolution intensities can only be decided against the published structure.

## 2. Datasets and runs

| item | value |
|---|---|
| dataset | x08 (organic small molecule, published structure, P2(1)2(1)2(1); identity in `private/dataset_key_v0.json`), 8 movies m0-m7, 120 frames of 1 deg each, bin 4, CL 420 (calibrated 560 mm), lambda 0.0251 A |
| V0 / V0r | all 8 movies index in SG 16 and reach CORRECT (`v0_runs_v0.csv`) |
| external pool | s4e3c1032: 514 No -> 261 after the P2 neighbour drop (raster step 5.0 um) -> 241 after the high-s test (20 out); pool centre (1021.7, 951.3) px; 0.232 counts/px per 0.25-s snapshot (`pool_summary_v0.json`, `pool_frames_v0.csv`) |
| per-frame scale k_f | median 2.9-10.3 per movie (range 1.8-11.1): one movie frame (1 s, tilted film + crystal) holds 3-10 snapshots' worth of film background; the scaled model carries 99 % of the annulus counts |
| pedestals (rule) | V1: Q = 3-6 counts; V2: P = 24-46 counts (`variant_runs_v0.csv`) |
| XDS failures | V1 / V1b: m5; V2: m0, m5, m6 ("AUTOMATIC DETERMINATION OF SPOT SIZE PARAMETERS HAS FAILED"); V0r: none |
| sets (amendment 2) | S1 = m0-m4, m6, m7 (V1 decision); S2 = m1-m4, m7 (V2 decision) |
| timing | an XDS INTEGRATE + CORRECT run takes ~5 s; the whole comparison ran in minutes on the Mac |

## 3. Comparison tables (own statistics: sigma-tau CC1/2, Rmeas, merged I/sigma; `stats_v0.csv`)

### 3.1 Pooled, S1 (7 movies): baseline, V1 and its replicate

| shell (A) | CC1/2 V0r | CC1/2 V1 | CC1/2 V1b | Rmeas V0r | Rmeas V1 | Rmeas V1b | I/sig V0r | I/sig V1 | I/sig V1b |
|---|---|---|---|---|---|---|---|---|---|
| 1.7-1.5 | 0.976 | 0.966 | 0.966 | 24.0 | 26.0 | 26.0 | 17.73 | 17.58 | 17.64 |
| 1.5-1.3 | 0.982 | 0.976 | 0.976 | 27.8 | 32.1 | 32.1 | 14.98 | 14.80 | 14.82 |
| 1.3-1.2 | 0.976 | 0.963 | 0.963 | 27.7 | 34.5 | 34.4 | 14.84 | 14.43 | 14.49 |
| 1.2-1.1 | 0.971 | 0.946 | 0.947 | 32.8 | 42.5 | 42.5 | 13.53 | 13.02 | 13.04 |
| 1.1-1.0 | 0.974 | 0.966 | 0.966 | 33.8 | 42.1 | 41.9 | 12.84 | 12.05 | 12.07 |
| 1.0-0.9 | 0.956 | 0.931 | 0.932 | 44.4 | 57.6 | 57.6 | 10.45 | 9.25 | 9.24 |
| 0.9-0.8 | 0.873 | 0.805 | 0.803 | 62.3 | 80.8 | 81.5 | 6.72 | 5.46 | 5.45 |
| key 1.1-0.9 | 0.974 | 0.964 | 0.964 | 38.1 | 48.3 | 48.2 | N_use 415 | 413 | 413 |

Completeness is the same in every variant (419 unique reflections at 0.9-1.1 A in every variant). Below 1.7 A all variants agree within 0.01 in CC1/2 and 1-2 points in Rmeas.

### 3.2 Pooled, S2 (5 movies): baseline, V1 and V2

| shell (A) | CC1/2 V0r | CC1/2 V1 | CC1/2 V2 | Rmeas V0r | Rmeas V1 | Rmeas V2 | I/sig V0r | I/sig V1 | I/sig V2 |
|---|---|---|---|---|---|---|---|---|---|
| 1.7-1.5 | 0.969 | 0.953 | 0.944 | 22.5 | 25.1 | 31.6 | 15.14 | 14.87 | 13.27 |
| 1.5-1.3 | 0.967 | 0.954 | 0.925 | 27.7 | 31.8 | 41.1 | 12.81 | 12.62 | 10.74 |
| 1.3-1.2 | 0.966 | 0.949 | 0.872 | 27.6 | 32.2 | 46.4 | 12.65 | 12.23 | 10.31 |
| 1.2-1.1 | 0.964 | 0.948 | 0.880 | 32.0 | 38.0 | 54.9 | 11.34 | 10.94 | 8.88 |
| 1.1-1.0 | 0.967 | 0.961 | 0.898 | 32.5 | 36.8 | 57.4 | 11.01 | 10.40 | 8.19 |
| 1.0-0.9 | 0.943 | 0.925 | 0.821 | 41.7 | 50.5 | 80.7 | 9.17 | 8.22 | 5.92 |
| 0.9-0.8 | 0.825 | 0.775 | 0.476 | 60.4 | 75.1 | 137.3 | 5.82 | 4.84 | 3.14 |
| key 1.1-0.9 | 0.966 | 0.959 | 0.893 | 36.2 | 42.2 | 66.9 | N_use 414 | 406 | 357 |

### 3.3 Per movie, key range d 0.9-1.1 A (CORRECT-scaled XDS_ASCII.HKL)

| movie | CC1/2 V0r | V1 | V2 | Rmeas V0r | V1 | V2 | N_use V0r | V1 | V2 | ISa V0r | V1 | V2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0 | 0.943 | 0.940 | fail | 30.8 | 37.1 | fail | 313 | 279 | - | 10.07 | 10.49 | - |
| m1 | 0.978 | 0.981 | 0.969 | 23.0 | 24.3 | 30.7 | 211 | 186 | 146 | 9.54 | 9.55 | 9.74 |
| m2 | 0.904 | 0.890 | 0.781 | 40.0 | 45.4 | 64.1 | 287 | 258 | 168 | 5.93 | 5.82 | 6.35 |
| m3 | 0.944 | 0.943 | 0.888 | 24.3 | 26.3 | 36.0 | 286 | 274 | 234 | 8.85 | 8.60 | 9.67 |
| m4 | 0.949 | 0.951 | 0.954 | 21.9 | 22.4 | 24.2 | 185 | 167 | 155 | 7.82 | 8.51 | 10.90 |
| m5 | (V0r CORRECT OK) | fail | fail | | | | | | | | | |
| m6 | 0.955 | 0.947 | fail | 30.6 | 36.0 | fail | 293 | 255 | - | 9.41 | 9.69 | - |
| m7 | 0.878 | 0.815 | 0.746 | 47.6 | 66.4 | 134.5 | 240 | 170 | 67 | 6.31 | 5.63 | 8.57 |

Rmeas rises in every movie for both variants; CC1/2 is higher only in m1 and m4 (V1) and m4 (V2), by <= 0.005.

### 3.4 Decision (`decision_v0.json`)

| variant | set | thresholds (tau_CC / tau_R / tau_N) | outer shells better / worse | dN_use | movies with higher key CC1/2 | verdict |
|---|---|---|---|---|---|---|
| V1 | S1 | 0.02 / 5 % / 5 % (replicate differences <= 0.002 and 0.9 %) | 0 / 3 (Rmeas +24 %, +30 %, +30 %; CC -0.009, -0.024, -0.068) | -0.5 % | 2 / 7 | **DEGRADES** |
| V2 | S2 | 0.02 / 5 % / (not used) | 0 / 3 (Rmeas +77 %, +94 %, +127 %; CC -0.069, -0.121, -0.350) | -13.8 % (not in the rule) | 1 / 5 | **DEGRADES** |

The V1 / V1b replicate agrees to <= 0.002 in CC1/2 and <= 0.9 % in Rmeas per shell, so the dither and the
comparison noise are far below the effects.

### 3.5 XDS's own tables, anomalous columns and the model check (descriptive)

- XSCALE's random-split CC1/2 shows the same ordering only weakly (S1 total 98.4 / 98.7 / 98.7; the random split is
  noisy at this multiplicity, A185); XSCALE Rmeas agrees with ours (S1 outer 62.1 / 81.1 / 81.9 %).
- Anomalous correlation (XSCALE, pooled, total): S1 V0r -6 %, V1 -15 %; S2 V0r -10 %, V2 -21 %; every shell lies
  within -30 to +15 % with no resolution trend: no usable Friedel signal in any variant (dynamical Friedel
  differences are not resolved at this multiplicity). Background handling does not change that.
- Model: SHELXT solves the pooled V0r data in P2(1)2(1)2(1) (R1 0.228; 16 atoms typed C13 O2 S, i.e. N and one O
  typed as C, A190). SHELXL, isotropic, X-ray factors, identical instructions (`shelx_refine_v0.csv`):

| data | R1 (Fo > 4 sig) | n | R1 (all) | wR2 |
|---|---|---|---|---|
| S1 V0r | 0.212 | 2236 | 0.215 | 0.459 |
| S1 V1 | 0.216 | 2150 | 0.222 | 0.466 |
| S1 V1b | 0.215 | 2150 | 0.221 | 0.465 |
| S2 V0r | 0.223 | 2113 | 0.231 | 0.478 |
| S2 V2 | 0.234 | 1606 | 0.268 | 0.522 |

  The self-derived model agrees with the merging statistics (variants slightly worse), but it is derived from the
  baseline data and cannot judge the zero-level question (A186), whose effect is mostly a high-resolution fall-off
  that isotropic U values absorb.

Figures: `notes/figures/xds_merge_v0/pooled_shells_v0.png` (CC1/2, Rmeas, I/sigma by shell, S1 and S2),
`per_movie_key_v0.png`, `posthoc_pedestal_v0.png`, `pool_profile_v0.png`.

## 4. Post hoc (after the decision; labelled, no verdict rests on it)

### 4.1 A pedestal alone reproduces the variants (`posthoc_stats_v0.csv`, `posthoc_runs_v0.csv`)

PQ = original images + the V1 pedestal Q of that movie on trusted pixels; PQ2 = + the V2 pedestal P. No background
removed.

| shell (A) | Rmeas V0r | V1 | PQ | V2 | PQ2 | CC1/2 V0r | V1 | PQ | V2 | PQ2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.2-1.1 (S1 / S2) | 32.8 / 32.0 | 42.5 | 42.4 | 54.9 | 56.5 | 0.971 / 0.964 | 0.946 | 0.945 | 0.880 | 0.877 |
| 1.1-1.0 | 33.8 / 32.5 | 42.1 | 42.2 | 57.4 | 59.3 | 0.974 / 0.967 | 0.966 | 0.965 | 0.898 | 0.892 |
| 1.0-0.9 | 44.4 / 41.7 | 57.6 | 57.5 | 80.7 | 81.2 | 0.956 / 0.943 | 0.931 | 0.932 | 0.821 | 0.817 |
| 0.9-0.8 | 62.3 / 60.4 | 80.8 | 81.0 | 137.3 | 138.3 | 0.873 / 0.825 | 0.805 | 0.804 | 0.476 | 0.490 |

PQ2 also fails INTEGRATE on m0 and m6, exactly where V2 failed. The background content of V1 and V2 therefore did
nothing measurable; the whole difference is XDS's response to the raised zero level.

### 4.2 The zero level changes the high-resolution intensities (A186)

Mean scaled intensity per shell relative to the 1.3-2.0 A mean, pooled:

| data | 1.3-1.1 | 1.1-1.0 | 1.0-0.9 | 0.9-0.8 |
|---|---|---|---|---|
| S1 V0r (archive +1 pedestal, gain 0.37-0.63) | 0.499 | 0.351 | 0.165 | 0.076 |
| S1 V1 / PQ (+3..6) | 0.471 / 0.469 | 0.316 / 0.314 | 0.145 / 0.144 | 0.066 / 0.065 |
| S2 V0r | 0.477 | 0.326 | 0.155 | 0.067 |
| S2 V2 / PQ2 (+24..46) | 0.415 / 0.413 | 0.271 / 0.267 | 0.129 / 0.127 | 0.057 / 0.055 |
| S2 PM1: pedestal removed (true counts), MINIMUM_VALID 0, GAIN 1.00 | 0.435 | 0.276 | 0.120 | 0.047 |
| S2 PM1g: pedestal removed, the archive gain table (0.37-0.63) | 0.580 | 0.483 | 0.335 | 0.268 |

- PM1 (the physically consistent statement for a counting detector: 1 count = 1 e, Poisson, no offset) does not
  improve the merging statistics either (S2 key CC1/2 0.964 vs 0.966, Rmeas 38.9 vs 36.2 %) and XDS fails INTEGRATE
  on m0, m5, m6 with it (`posthoc2_*_v0.csv`). PM1g lowers Rmeas at 0.8-0.9 A (40.8 vs 60.4 %) while CC1/2 falls
  (0.666 vs 0.825) and the 0.8-0.9 A mean intensity rises 4x: a biased background, not better data.
- So the archive's high-resolution intensities, and every B factor and resolution cut derived from them, depend on how
  the zero level and the gain are declared to XDS, by tens of percent at 0.8-1.1 A. The merging statistics cannot say
  which declaration is right, because a biased but consistent background leaves equivalents consistent. Fcalc from
  the published structure can: the ratio <Iobs>/<Fcalc^2> by shell must be flat (up to an overall B) under the right
  zero level.

## 5. Anomalies

- **A183.** The archive CORRECT.LP of x08 is not produced by the archived XDS.INP: SPACE_GROUP_NUMBER and the cell are
  commented out in XDS.INP but given in CORRECT (SG 16, 5.20 10.30 21.00), with XDS of Jan 2025. Re-run locally with
  XDS Apr 2026 and the same inputs, m0 gives total I/sigma 6.39 vs 7.34, Rmeas 19.1 vs 17.1 %, CC1/2 98.6 vs 98.8.
  The archived numbers are reproducible only approximately (version or undocumented inputs).
- **A184 (lab-relevant).** XDS treats the +1 pedestal of the lab's MRC -> TIFF conversion as signal: BLANK.cbf is set
  to OFFSET 0 and INIT estimates GAIN 0.37-0.63 per movie (variance / mean of background pixels whose mean includes
  the pedestal), where a bin-4 Apollo count is one Poisson electron (gain 1). Every Apollo XDS run in the archive with
  this conversion carries the same mis-declared zero level and gain.
- **A185.** XDS / XSCALE CC1/2 and the anomalous correlation come from a random half-split: byte-identical input
  (V0 and its INTEGRATE CORRECT rerun) gives identical intensities, I/sigma and Rmeas but CC1/2 and anomalous
  correlation differing in the last digit (anomalous correlation by up to 0.04). Small-dataset comparisons of XDS's
  CC1/2 need a deterministic estimator (sigma-tau, used here).
- **A186.** At Apollo count levels (median pixel 0 e, high-resolution background 0.1-0.3 e per pixel per frame) XDS's
  integrated intensities are not invariant to an additive constant in the images: +3..6 counts lowers the 0.8-0.9 A
  mean intensity (relative to 1.3-2.0 A) by 14 % and raises outer-shell Rmeas by 24-30 %; +24..46 counts by 14-17 %
  and x2.3; removing the pedestal changes it by -29 % (gain 1) or x4 (the archive gain table). The likely mechanism is XDS's
  per-box background estimate with outlier rejection judged against a gain x value variance, which is biased when the
  counts are mostly 0 and 1 (not verified inside XDS).
- **A187.** The external film background has no measurable effect of its own: a pedestal-only image reproduces V1 and
  V2 to <= 0.003 in CC1/2 and <= 1 % in Rmeas in every shell, and the same INTEGRATE failures. Consistent with XDS's
  flat per-box background: the smooth film background is removed by XDS anyway, and its within-box curvature (ring
  shoulders, stop edge) is below the noise.
- **A188.** XDS INTEGRATE's automatic spot-size determination fails when the zero level is moved: V1 on 1/8 movies,
  V2 and PQ2 on 3/8 and 2/7, PM1 (true counts, gain 1) on 3/8; never on the archive convention (8/8).
- **A189.** The centre of point symmetry of each movie's frame sum lies (+5.6 to +7.9, +1.8 to +5.6) px (row, column) from the
  XDS beam point derived from XPARM (beam direction refined, ORGX/ORGY fixed) in all 8 movies, consistently; the
  classic pool's centre is 13-15 px from the movie centres. Irrelevant to the verdict (A187), but any shape-dependent
  background subtraction on rotation data must fix which centre is right.
- **A190.** SHELXT solves the pooled baseline in P2(1)2(1)2(1) (R1 0.228) with 16 atoms typed C13 O2 S for the
  expected C10 N2 O3 S skeleton: element typing is not reliable at this data quality with X-ray factors.

## 6. What the owner must provide or decide

1. The published coordinates of the x08 structure (CIF; identity in `private/dataset_key_v0.json`) and confirmation of
   the identities there. With them, <Iobs>/<Fcalc^2> by shell under V0r, PM1 and PQ decides which zero-level
   declaration gives the right high-resolution intensities (A186) and whether the archive's Apollo XDS data are biased
   at 0.8-1.1 A.
2. A current nXDS build (or a DIALS install) if the stills route (a) should be tested; DIALS's global background model
   (`background.algorithm=gmodel`) is also the proper test of an external background with its own variance, which XDS
   cannot take.
3. Whether to change the lab's conversion/processing convention (A184): declare the pedestal as an offset (or drop it)
   and the gain as 1 for bin-4 Apollo counting data. This must not be adopted from this report alone: the post-hoc
   PM1 run does not improve the merging statistics and makes INTEGRATE fail on 3/8 movies; the Fcalc test (item 1)
   comes first.
4. Optional, larger: x07 (protein, CL 960, 4096 x 4096) as a second known structure, once disk and time allow (~7 GB
   per movie).

## 7. Files

- Public (`~/pod_data/background/xds_merge_v0/`): `prereg_v0.json`, `pool_summary_v0.json`, `pool_frames_v0.csv`,
  `v0_runs_v0.csv`, `variant_runs_v0.csv`, `movie_sets_v0.json`, `stats_v0.csv`, `xds_tables_v0.csv`,
  `decision_v0.json`, `posthoc_runs_v0.csv`, `posthoc_stats_v0.csv`, `posthoc2_runs_v0.csv`,
  `posthoc2_stats_v0.csv`, `posthoc2_permovie_v0.csv`, `shelx_v0.json`, `shelx_refine_v0.csv`; each with a
  `.provenance.json` sidecar.
- Private: `private/dataset_key_v0.json`, `private/archive_xds/` (archive XDS files), `private/frames/` (8.4 GB of
  TIFFs), `private/pool/`, `private/pool_v0.npz`, `private/runs/` (every XDS / XSCALE / SHELX run; variant TIFFs
  deleted after integration, reproducible from the seeds).
- Reproduce: `.venv/bin/python scripts/xds_merge_v0.py prereg fetch pool v0 variants xscale stats figs shelx`, then
  `posthoc posthoc2` for Sec. 4.
