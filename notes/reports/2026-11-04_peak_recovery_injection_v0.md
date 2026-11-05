# Bragg-peak intensity recovery after background subtraction: injection into real frames, v0 (2026-11-04)

Terms of reference: measure end to end how well integrated Bragg-peak intensities are recovered after background
subtraction, with peaks of KNOWN intensity injected at the electron level into REAL host frames (clean film, bar-adjacent
film, crystal-neighbour "No" frames of the 20 held-out fly-scan sessions; classic raw-count bin-4 "No" frames), for five
families of background method. Code: `src/podbg/peaks.py` (pure functions), `scripts/peak_recovery_v0.py` (stages, single
process, checkpoints), tests `tests/test_peak_recovery.py`. Outputs with sidecars: `~/pod_data/background/peak_recovery_v0/`.
Figures: `notes/figures/peak_recovery_v0/`. Sessions are sids only. Nothing runs on HPC; archive reads are
`ssh hpc 'timeout 300 cat <file>'` (coreutils, one file per call) under a local alarm.

## 1. Pre-registration (written and committed BEFORE any injection was measured)

What I had seen before writing this section: CLAUDE.md (stored-count law, atlas v1-v1e, Stage 1 gate 1b, BAR ADJACENT v0,
HOUSEKEEPING v0, REyes DROP VALIDATION v0 and its confirmation), the reyes drop v0 report and spec v1 Sec. 9, the code of
`reyes_drop_v0.py`, `atlas`, `atlas_v1` (predict, frame_keys, rolling inner curve), `atlas_refresh_v0.py` (v1d prediction),
`atlas_v1c_eval_spots.py`, `film_selection`, `binning` (quantise, recorded_pmf, response tables), `detector` (dead time),
`model.forward` (invert_profile, recorded, render), `structure_family.fit_holdout`, `thickness_proxy_v0.group_profiles`,
`s_level.build_profile`. Data looked at: (i) host-class COUNTS in the 60 streamed rows (no profiles); (ii) the spot shape of
real Bragg spots on the 210 cached Good frames of the atlas v1 spot test (7 sessions): 3,689 spots with excess >= 40
counts, Gaussian sigma median 0.78 bin-8 px (IQR 0.65-1.01; 0.72-0.81 by radius; brighter spots broader, median 1.5 for
excess > 200 counts), totals 44-348 counts (10-99 %), peak pixels up to 46 counts. Nothing was injected and no recovery
ratio was computed. Everything below is also in `PREREG` of `scripts/peak_recovery_v0.py` (the code is the authority if the
two differ).

### 1.1 Host frames

- FLY-SCAN: the 20 held-out sessions (`reyes_drop_v0.HELD`; bin 8, CL 420). Rows: the 3 rows per session of
  `reyes_drop_v0/m1_rows_v0.csv` (the rows with the most Good/Bad frames), streamed whole with `timeout 300 cat`; only
  the host frames are kept (cached, compressed).
- Three host classes, each trusted sweep AND REyes "No diffraction", in those rows:
  - **F clean film** = `film_selection.film_mask` (states v1.4 film_ok);
  - **BA bar-adjacent** = `state_v1_4 == film_bar_adjacent` AND 0 <= bar_edge_um <= 5 (the pedestal population of A52/A71);
  - **CN crystal neighbour** = a Good/Bad frame within +/-2 positions in the row, no Grid frame within +/-2, not film_ok.
- Up to N_HOST = 12 per class per session, evenly spaced in (row, frame) order (deterministic). Class counts in those rows
  (seen): F 3-208, BA 40-79, CN 6-54 per session; classes with fewer frames take all.
- Addressing check: each kept frame's masked sum must equal the stored `masked_sum` (reported; a mismatching frame is dropped).
- CLASSIC: the cached kept "No" frames of s level bin4 v0 (raw counts, bin 4, CL 420; `s_level_bin4_v0/frames/`), every
  session with >= 8 cached frames; up to N_HOST_C = 6 hosts per session evenly spaced; the session's other cached frames are
  its pool. Session mask and centre as s level bin4 v0 (`classic_v0.bad_mask_for` on the session sum, centre from
  `s_level_bin4_v0/classic_profiles_v0.pkl`).

### 1.2 The injected peak

- Shape: isotropic 2-D Gaussian, pixel-integrated (erf), sub-pixel centre uniform; sigma drawn per peak from the measured
  real-spot sigmas (`spot_shape_v0.csv`, stage `spotshape`, the measurement described above) clipped to [0.5, 2.0] bin-8 px;
  at bin 4 sigma x 2. (No separate detector PSF: at bin 8 the Apollo's super-resolution centroiding puts the electron in its
  32-um output pixel; the measured spot width already contains whatever PSF there is.)
- Integration box: disc of radius R_BOX = 3.0 px about the true centre (pixel centres within 3.0; bin 4: 6.0). Background
  annulus for method A: 5.0 < r <= 9.0 px (bin 4: 10-18). Patch 21 x 21 (bin 4: 41 x 41).
- TRUE intensity lambda_box = expected injected electrons inside the box (before dead time). lambda_total and the box
  fraction are recorded.
- Resolution bins (s = 1/d; s = 0.0022811 (r + 0.5) at bin 8): D1 d 0.86-1.00 A (s 1.00-1.163), D2 1.00-1.25 (0.80-1.00),
  D3 1.25-2.00 (0.50-0.80), D4 2.00-4.00 (0.25-0.50). Bin 4: s = K4 (r + 0.5), the same s bins.
- P/B ladder: P/B = lambda_box / B_box in {0.03, 0.1, 0.3, 1, 3, 10, 30}; B_box = sum over the box of the host's electron
  mean mu_h(r), mu_h = the host frame's own masked azimuthal mean (stored counts) inverted per shell with
  `binning.true_mean_fast` at the session's tie rule (`atlas_v1.tie_up_for(flag)`), no floor subtraction, no dead time.
- Placement: per host and replicate, 28 peaks (7 P/B x 4 d bins); s uniform in the bin, azimuth uniform; rejected and
  redrawn (<= 200 tries; else skipped, reported) if any box or annulus pixel is masked (session mask dilated by 1, seam
  lines +/-1, 2-px border, stop) or the centre lies within 24 px (bin 4: 48) of another peak of the same replicate. Two
  replicates per host (independent positions, sigmas and draws). Seeds fixed per (sid, host, replicate).
- ELECTRON-LEVEL injection, then the stored-count law (bin 8, gain flag of the session):
  1. latent host electrons per patch pixel: N_h ~ P(N | K) proportional to Poisson(N; mu_h(r)) x T(K | N), T the stored-count
     quantiser (K = rint(N / 4); N = 4j + 2 ties to j + 1 with p = tie_up_for(flag) when the gain flag is on, parity when off);
  2. dead time (paralyzable, `detector.TAU_PHYS_V1` per e per physical px per raw frame -> tau_b per bin-8 px per stored
     frame, `tau_for_binning`): peak electrons counted n ~ Poisson(lambda_px exp(-(mu_h + lambda_px) tau_b)); host electrons
     thinned Binomial(N_h, exp(-lambda_px tau_b));
  3. K' = quantise(N_h' + n) with the same tie rule (`binning.quantise`). Pixels outside the patch are unchanged.
  - Bin 4 classic raw counts: K = N (1 count = 1 e, no rounding); peak electrons Poisson(lambda_px) added; NO dead time
    (the raw frames per classic snapshot are not harvested here; stated as a limitation).

### 1.3 Background methods (each gives B_px in stored counts on the box pixels)

Integrated intensity I = epc x sum_box (K' - B_px), epc = 4 e per count at bin 8, 1 at bin 4. Radial profiles (index i =
shell [i, i+1) px, mean radius i + 0.5) are rendered at r_px - 0.5 about the host's row centre (linear interpolation).
"Host-scaled" = multiplied by h / p, h and p the pixel-weighted mean of the host's own masked profile and of the
profile over s 0.34-0.80. The host's own profile (methods C, D and the scaling) is computed from the injected frame with
discs of radius 9 px (bin 4: 18) about every injected peak masked, seams and the session mask out (an integrator knows its
predicted spot positions).

| code | method | live? |
|---|---|---|
| O | ORACLE: B_px = the host's own pre-injection stored values (only counting law, dead time, peak noise remain) | reference |
| A | LOCAL ANNULUS: B_px = mean of K' over the annulus pixels (XDS/REyes-style summation) | yes |
| B0 / B1 / B4 / P2 | REyes POOL: equal-weight mean of the Stage 0 profiles (`profile_mean_v1`) of the pool frames of the variant, same row, 0 < \|j - i\| <= 20, fallback 40, n_min 3, host excluded; host-scaled. Variants as reyes drop v0: V0 all No, V1 spec drop, V4 drop + high-s test, P2 drop Grid and crystal neighbours + high-s test | yes (labels) |
| C | ATLAS v1d: the pooled held-out fold atlas of `atlas_v1b.npz` keyed by `frame_keys` (I_e, q) on the host's masked profile, times the causal rolling inner curve (tau 300, n_prior 30) over the session's v1c film keys in acquisition order taken at the host's position, forwarded through the response table plus the session floor (`atlas_refresh_v0._forward`) | yes |
| D | PHYSICS: `structure_family.fit_holdout` gate+tie (S_family_flyscan_v2, central kernel, a_h 0.20, f_flat of the family) on the host's masked single-frame profile inverted with `forward.invert_profile` (session Grid floor, cv 0, stray subtracted), fit points by `forward.fit_mask`; the model on all shells (+ stray) through `forward.recorded` + floor | yes (fit) |
| E | TRUTH: the session's film_ok truth (`reyes_drop_v0.truth_bins` / `truth_at`) at the host's ln(masked sum), host-scaled | reference (montage) |
| classic | O, A; Bc = mean of the session's other cached frames' profiles, host-scaled; D at bin 4 = gate (no tie) under S v2 on the host's single-frame profile (`s_level.build_profile`, rebin 2), model on all radii | |

A host that a method cannot serve (pool < 3 at W 40, failed fit) is NaN for that method (reported as a cost).

### 1.4 sigma models and metrics

- sigma_naive^2 = epc^2 (sum_box K' [+ n_box^2 sum_ann K' / n_ann^2 for A]) (stored counts treated as Poisson: an
  integrator with gain 1).
- sigma_law^2 = epc^2 (sum_box g(max(K'_px, B_px)) [+ n_box^2 sum_ann g(K'_ann) / n_ann^2 for A]); g(m) = stored-count
  variance at stored mean m from `binning.response_table` at the session's tie rule (bin 4: g(m) = m).
- Per peak: ratio = I / lambda_box; z = I / sigma_law; pull = (I - lambda_box) / sigma (both sigma models); background bias
  beta = epc sum_box (K_host - B_px) / (epc sum_box m_h(r)) = the method's background error as a fraction of the expected
  background under the box (m_h the host's recorded shell means); beta(O) = 0, and the intensity error a background error
  causes is beta / (P/B).
- Cells: method x regime (F, BA, CN, classic) x d bin x P/B. Per cell: n, median ratio, q25, q75; per-session medians
  (sessions with >= 5 peaks in the cell); fraction lost (I <= 0) and below 3 sigma (z < 3); pull median and robust SD
  (1.4826 MAD) for sigma_law and sigma_naive; beta median and IQR.

### 1.5 Checks before the real run (stage `synth`)

Pure synthetic hosts: per session of three (s0047f887, s4fead7c7, sc19f33b9), 12 frames of Poisson electrons at the
session's film_ok truth profile (inverted) through `binning.quantise`; inject (a) from the KNOWN N and (b) through the
latent posterior from K (the real-data route); methods O, A and E-exact (the exact expected stored profile). PASS iff
(i) the two routes agree: per cell (d x P/B) |median ratio(a) - median ratio(b)| <= max(0.01, 2 x the bootstrap SE of
the difference) in >= 90 % of cells; (ii) E-exact: |median ratio(E-exact) - median ratio(O)| <= 0.01 at P/B >= 1;
(iii) E-exact sigma_law pull robust SD within 0.85-1.15 at P/B >= 0.3. A failure is reported and the injection code fixed
before the real run (a fix of the injection, never of a method or a threshold).

### 1.6 Decision statements

- WITHIN-X: a cell is "within X %" for a method iff |median ratio - 1| <= X/100 AND >= 2/3 of the sessions entering the
  cell have |per-session median - 1| <= X/100. P/B*_X(method, regime, d bin) = the smallest ladder P/B such that that cell
  and every higher one are within X %. Reported for X = 5 and 10, on the ratio to TRUTH (the user's quantity) and,
  separately, on ratio(method) / ratio(O) (the background's own part).
- Statements are generated from the table in the form "method M recovers intensities within 5 % for P/B >= x at d bin
  D in regime R".
- SYSTEMATIC per regime: median beta (and IQR) per method x regime x d bin, pooled over P/B; the background-induced
  intensity error at P/B = 1 is beta.
- SIGMA HONESTY: a method/regime is honest iff the sigma_law pull robust SD lies in 0.85-1.15 AND |pull median| <= 0.5,
  pooled over d bins, separately for P/B <= 1 and P/B >= 10. The same is reported for sigma_naive.
- RECOMMENDATION per regime x d bin: among the live methods (A, B0, B1, B4, P2, C, D) the one with the smallest
  P/B*_5 (truth); ties -> the smaller |median beta|, then the simpler (A < B < C < D). Its median beta is quoted as the
  systematic error of that regime.

(Pre-registration committed before any measurement; everything below was produced afterwards.)

### 1.7 Deviations found while running (no method, threshold or host rule changed)

- SYNTHETIC CHECK (`synth_verdict_v0.json`, `synth_cells_v0.csv`, `synth_exact_v0.csv`): (i) PASSES (the latent-posterior
  route and the known-electron route agree in 95 % of cells): the injection is consistent. (ii) and (iii) FAIL as written
  (max |E_exact - O| 0.14 at P/B >= 1; E_exact pull SD 0.05-9.2), and the failure is not in the injection: the medians of
  discrete, skewed integrated intensities differ between two estimators with different noise, and at bin 8 the recovered
  intensity does not have expectation lambda_box (rounding thresholds and dead time; A173), so a pull about lambda_box cannot
  have unit width. POST HOC check added after reading the verdict (`synth_lawcheck_v0.csv`): the law-expected oracle
  intensity lam_eff = 4 sum_box [E(mu_h e^{-lam tau} + lam e^{-(mu_h+lam) tau}) - E(mu_h)] (E = stored mean of the
  response table) agrees with the mean injected-and-recovered oracle intensity in 24/28 cells (2 of the 4 are D1 P/B 0.03-0.1,
  where no count was injected in 72 peaks, 1 is 2.2 SE, 1 is 3.6 SE at D2 P/B 0.3); E_exact pulls about lam_eff have SD
  0.84-1.44 in cells with lam_eff >= 20 e. I judged the injection fit for the real run; nothing in it was changed.
- lam_eff is carried for every real peak (post hoc, used by no method) and gives a third pull, `law_about_lam_eff_POSTHOC`.
- BETA: the per-peak median beta is degenerate at bin-8 box sums of 0-few counts (the host's own box sum is mostly 0, so the
  median is -B / m_h ~ -1 for every model method, and per-peak beta is undefined where m_h = 0). POST HOC the systematic is
  reported as the ratio of sums beta = sum (K_host - B) / sum m_h per cell, with a 68 % bootstrap over session values (the
  bootstrap is over unweighted session ratios, so its interval need not contain the pooled, background-weighted value).
  The pre-registered median beta stays in `decisions_v0.csv` (`beta_med`).
- P/B*_X assumes that recovery improves monotonically with P/B. Dead time makes it fall once more above P/B ~ 3-10, so P/B*
  (truth) is undefined for EVERY fly-scan method, the oracle included (A174). The pre-registered recommendation rule then
  falls through to its tie-breaks (degenerate median beta, then simplicity) and returns "A" almost everywhere; I report it
  (`recommendation_v0.csv`) and give the post-hoc reading in Sec. 6. The post-hoc list of passing P/B cells per method is
  `posthoc_passing_pb_v0.csv`.

## 2. What ran

| item | value |
|---|---|
| fly-scan hosts | 20 sessions; F 231, BA 240, CN 216 host frames (x 2 replicates = 462 / 480 / 432); all 687 passed the addressing check |
| classic hosts | 24 raw-count bin-4 sessions with >= 8 cached frames, 6 hosts each (144, x 2 = 288) |
| peaks | 28 per host-replicate, all placed (0 skipped); 38,472 fly-scan + 8,064 classic peaks, each integrated by every method |
| box | contains a median 99.9 % (10-90 %: 89-100 %) of the injected spot |
| B_box (median electrons, fly-scan F / BA / CN; classic) | D1 2.3 / 2.1 / 3.8; 26 - D2 6.4 / 4.7 / 8.5; 63 - D3 9.3 / 7.2 / 13.3; 95 - D4 26 / 19 / 37; 282 |
| pools (W 20, fallback 40) | fell back for 0.3 / 3 / 5 / 8 % of hosts (B0 / B1 / B4 / P2); unserved 0.8 / 1.8 / 2.7 / 5.3 % |
| physics D | converged on every host; single-frame gate+tie: median chi2/dof 1.36, rms 19 %, t at the 100-nm bound in 38 % (A179); classic median chi2/dof 8.4, rms 6.7 % |
| archive reads | 60 rows, `timeout 300 cat`, 255 s; no HPC compute |
| run time | 62 s per fly-scan session, 5-6 s per classic session; single process |

## 3. Key table: recovered / true intensity, median (IQR), by method x d bin x P/B

Ratio = I / lambda_box. d bins: D1 0.86-1.00 A, D2 1.00-1.25, D3 1.25-2.00, D4 2.00-4.00. P/B 1, 3, 10, 30 shown; the full
ladder (0.03-30) is in `key_table_v0.md` / `cells_v0.csv`. Figures: `ratio_vs_pb_<regime>_v0.png`.

**F**

| method | D1 1 | D1 3 | D1 10 | D1 30 | D2 1 | D2 3 | D2 10 | D2 30 | D3 1 | D3 3 | D3 10 | D3 30 | D4 1 | D4 3 | D4 10 | D4 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O | 0.00 (0.00-0.00) | 0.46 (0.00-0.81) | 0.80 (0.60-0.99) | 0.89 (0.79-0.97) | 0.55 (0.00-1.05) | 0.83 (0.60-1.02) | 0.94 (0.82-1.03) | 0.89 (0.83-0.95) | 0.74 (0.41-1.11) | 0.93 (0.74-1.11) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.03 (0.86-1.23) | 0.98 (0.88-1.08) | 0.87 (0.81-0.94) | 0.69 (0.57-0.79) |
| A | 0.00 (0.00-0.00) | 0.49 (0.00-0.82) | 0.79 (0.59-0.98) | 0.89 (0.79-0.97) | 0.53 (0.00-1.10) | 0.81 (0.58-1.02) | 0.93 (0.82-1.03) | 0.89 (0.83-0.95) | 0.74 (0.29-1.10) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.03 (0.79-1.31) | 0.98 (0.88-1.09) | 0.88 (0.81-0.94) | 0.69 (0.57-0.79) |
| B0 | -0.06 (-0.09--0.02) | 0.48 (-0.02-0.81) | 0.79 (0.59-0.98) | 0.89 (0.79-0.97) | 0.52 (-0.09-1.04) | 0.81 (0.58-1.01) | 0.93 (0.82-1.02) | 0.89 (0.83-0.95) | 0.74 (0.32-1.06) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.02 (0.82-1.30) | 0.98 (0.89-1.09) | 0.88 (0.81-0.95) | 0.69 (0.57-0.79) |
| B1 | -0.06 (-0.08--0.02) | 0.48 (-0.02-0.81) | 0.80 (0.59-0.98) | 0.89 (0.79-0.97) | 0.52 (-0.09-1.03) | 0.81 (0.58-1.01) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.74 (0.32-1.06) | 0.93 (0.75-1.11) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.02 (0.82-1.30) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.56-0.79) |
| B4 | -0.06 (-0.08--0.02) | 0.48 (-0.02-0.81) | 0.80 (0.59-0.99) | 0.89 (0.79-0.97) | 0.52 (-0.09-1.04) | 0.81 (0.58-1.02) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.75 (0.32-1.08) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.01 (0.81-1.30) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.56-0.79) |
| P2 | -0.06 (-0.08--0.02) | 0.49 (-0.02-0.81) | 0.80 (0.60-0.99) | 0.89 (0.79-0.97) | 0.54 (-0.09-1.04) | 0.81 (0.58-1.02) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.75 (0.33-1.07) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.01 (0.81-1.30) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.56-0.79) |
| C | -0.06 (-0.08--0.01) | 0.48 (-0.02-0.81) | 0.80 (0.59-0.98) | 0.89 (0.79-0.97) | 0.53 (-0.09-1.05) | 0.81 (0.57-1.02) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.75 (0.33-1.08) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.02 (0.80-1.29) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.57-0.79) |
| D | -0.07 (-0.08--0.01) | 0.48 (-0.02-0.80) | 0.79 (0.59-0.98) | 0.89 (0.79-0.97) | 0.53 (-0.09-1.05) | 0.81 (0.58-1.02) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.76 (0.32-1.08) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.03 (0.80-1.30) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.57-0.79) |
| E | -0.06 (-0.08--0.02) | 0.48 (-0.02-0.81) | 0.80 (0.59-0.98) | 0.89 (0.79-0.97) | 0.53 (-0.08-1.05) | 0.81 (0.58-1.02) | 0.94 (0.82-1.02) | 0.89 (0.83-0.95) | 0.75 (0.32-1.08) | 0.94 (0.75-1.12) | 0.94 (0.85-1.03) | 0.86 (0.79-0.93) | 1.01 (0.81-1.29) | 0.98 (0.88-1.09) | 0.88 (0.81-0.95) | 0.69 (0.57-0.79) |

**BA**

| method | D1 1 | D1 3 | D1 10 | D1 30 | D2 1 | D2 3 | D2 10 | D2 30 | D3 1 | D3 3 | D3 10 | D3 30 | D4 1 | D4 3 | D4 10 | D4 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O | 0.00 (0.00-0.00) | 0.46 (0.00-0.82) | 0.79 (0.52-0.98) | 0.89 (0.78-0.99) | 0.00 (0.00-0.99) | 0.78 (0.49-1.06) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.60 (0.00-1.04) | 0.89 (0.65-1.09) | 0.92 (0.82-1.02) | 0.88 (0.81-0.95) | 0.98 (0.73-1.19) | 0.98 (0.86-1.09) | 0.92 (0.84-0.99) | 0.77 (0.65-0.86) |
| A | 0.00 (0.00-0.00) | 0.43 (0.00-0.84) | 0.80 (0.52-0.98) | 0.89 (0.78-0.98) | 0.40 (0.00-0.97) | 0.77 (0.49-1.05) | 0.89 (0.77-1.02) | 0.89 (0.83-0.96) | 0.64 (0.00-1.11) | 0.87 (0.64-1.08) | 0.92 (0.83-1.02) | 0.88 (0.80-0.94) | 0.92 (0.69-1.23) | 0.98 (0.85-1.11) | 0.92 (0.84-0.99) | 0.77 (0.64-0.86) |
| B0 | -0.03 (-0.06--0.01) | 0.43 (-0.01-0.82) | 0.79 (0.50-0.98) | 0.88 (0.77-0.98) | 0.40 (-0.07-0.97) | 0.78 (0.48-1.05) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.63 (-0.02-1.08) | 0.87 (0.65-1.08) | 0.92 (0.83-1.03) | 0.88 (0.81-0.95) | 0.97 (0.66-1.24) | 1.00 (0.85-1.11) | 0.92 (0.84-1.00) | 0.77 (0.65-0.86) |
| B1 | -0.03 (-0.06--0.01) | 0.43 (-0.01-0.82) | 0.79 (0.50-0.98) | 0.88 (0.77-0.98) | 0.40 (-0.07-0.98) | 0.78 (0.48-1.05) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.65 (-0.02-1.09) | 0.88 (0.65-1.09) | 0.92 (0.84-1.03) | 0.88 (0.81-0.95) | 0.98 (0.66-1.25) | 1.00 (0.85-1.11) | 0.92 (0.84-0.99) | 0.77 (0.65-0.86) |
| B4 | -0.03 (-0.05--0.01) | 0.44 (-0.01-0.82) | 0.78 (0.49-0.98) | 0.88 (0.78-0.98) | 0.40 (-0.07-0.97) | 0.78 (0.48-1.06) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.66 (-0.02-1.09) | 0.88 (0.65-1.09) | 0.92 (0.84-1.03) | 0.88 (0.81-0.95) | 0.95 (0.63-1.22) | 0.99 (0.85-1.11) | 0.92 (0.84-0.99) | 0.77 (0.65-0.86) |
| P2 | -0.03 (-0.05--0.01) | 0.43 (-0.01-0.82) | 0.78 (0.49-0.98) | 0.88 (0.77-0.98) | 0.38 (-0.07-0.97) | 0.78 (0.47-1.06) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.64 (-0.02-1.08) | 0.87 (0.64-1.08) | 0.92 (0.84-1.03) | 0.88 (0.81-0.95) | 0.95 (0.63-1.22) | 0.99 (0.85-1.11) | 0.92 (0.84-0.99) | 0.77 (0.65-0.86) |
| C | -0.04 (-0.06--0.00) | 0.44 (-0.01-0.84) | 0.79 (0.50-0.98) | 0.89 (0.78-0.98) | 0.42 (-0.07-0.97) | 0.78 (0.48-1.05) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.66 (-0.01-1.10) | 0.88 (0.65-1.09) | 0.92 (0.83-1.03) | 0.88 (0.81-0.95) | 0.94 (0.65-1.22) | 0.98 (0.84-1.11) | 0.92 (0.83-0.99) | 0.77 (0.64-0.86) |
| D | -0.04 (-0.06--0.00) | 0.44 (-0.01-0.84) | 0.79 (0.50-0.98) | 0.89 (0.78-0.98) | 0.43 (-0.07-1.00) | 0.79 (0.48-1.06) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.67 (-0.01-1.13) | 0.88 (0.65-1.09) | 0.92 (0.84-1.03) | 0.88 (0.81-0.95) | 0.97 (0.67-1.23) | 1.00 (0.86-1.11) | 0.92 (0.84-1.00) | 0.77 (0.64-0.86) |
| E | -0.03 (-0.05--0.01) | 0.44 (-0.01-0.84) | 0.79 (0.50-0.99) | 0.89 (0.78-0.98) | 0.42 (-0.07-0.98) | 0.78 (0.48-1.06) | 0.89 (0.77-1.02) | 0.90 (0.83-0.96) | 0.66 (-0.01-1.10) | 0.88 (0.65-1.09) | 0.92 (0.83-1.03) | 0.88 (0.81-0.95) | 0.92 (0.62-1.21) | 0.98 (0.84-1.10) | 0.92 (0.83-0.99) | 0.77 (0.65-0.86) |

**CN**

| method | D1 1 | D1 3 | D1 10 | D1 30 | D2 1 | D2 3 | D2 10 | D2 30 | D3 1 | D3 3 | D3 10 | D3 30 | D4 1 | D4 3 | D4 10 | D4 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O | 0.00 (0.00-0.89) | 0.70 (0.31-0.98) | 0.88 (0.75-0.99) | 0.88 (0.79-0.97) | 0.67 (0.00-1.01) | 0.90 (0.72-1.06) | 0.93 (0.83-1.01) | 0.87 (0.79-0.93) | 0.90 (0.62-1.18) | 0.95 (0.81-1.09) | 0.92 (0.84-1.00) | 0.81 (0.71-0.89) | 0.99 (0.86-1.14) | 0.96 (0.86-1.04) | 0.84 (0.76-0.92) | 0.62 (0.49-0.74) |
| A | 0.00 (0.00-0.96) | 0.70 (0.27-0.98) | 0.87 (0.75-0.99) | 0.88 (0.79-0.97) | 0.63 (0.03-1.06) | 0.91 (0.72-1.07) | 0.93 (0.83-1.01) | 0.87 (0.79-0.93) | 0.89 (0.58-1.23) | 0.93 (0.78-1.09) | 0.92 (0.84-1.00) | 0.81 (0.71-0.90) | 0.98 (0.81-1.20) | 0.96 (0.86-1.05) | 0.84 (0.75-0.91) | 0.62 (0.48-0.74) |
| B0 | -0.02 (-0.10-0.98) | 0.69 (0.27-1.02) | 0.87 (0.75-1.00) | 0.88 (0.80-0.97) | 0.61 (0.05-1.09) | 0.90 (0.71-1.07) | 0.93 (0.83-1.01) | 0.87 (0.79-0.93) | 0.90 (0.54-1.24) | 0.94 (0.78-1.09) | 0.92 (0.84-1.00) | 0.81 (0.71-0.89) | 1.00 (0.76-1.21) | 0.96 (0.85-1.06) | 0.84 (0.76-0.91) | 0.62 (0.48-0.74) |
| B1 | -0.02 (-0.09-0.99) | 0.68 (0.27-1.01) | 0.87 (0.74-1.00) | 0.88 (0.80-0.97) | 0.61 (-0.01-1.09) | 0.90 (0.71-1.08) | 0.93 (0.83-1.01) | 0.87 (0.79-0.93) | 0.91 (0.54-1.25) | 0.94 (0.78-1.09) | 0.92 (0.84-1.00) | 0.81 (0.71-0.90) | 0.99 (0.76-1.21) | 0.96 (0.86-1.07) | 0.84 (0.76-0.91) | 0.62 (0.48-0.74) |
| B4 | -0.02 (-0.07-1.04) | 0.69 (0.28-1.03) | 0.88 (0.75-1.00) | 0.88 (0.80-0.97) | 0.66 (0.10-1.13) | 0.91 (0.72-1.09) | 0.93 (0.83-1.02) | 0.87 (0.79-0.93) | 0.93 (0.58-1.29) | 0.96 (0.79-1.10) | 0.92 (0.84-1.01) | 0.81 (0.71-0.90) | 0.94 (0.68-1.16) | 0.94 (0.83-1.06) | 0.84 (0.75-0.91) | 0.62 (0.48-0.74) |
| P2 | -0.02 (-0.07-1.05) | 0.69 (0.30-1.03) | 0.89 (0.75-1.01) | 0.88 (0.80-0.97) | 0.65 (0.09-1.14) | 0.91 (0.72-1.09) | 0.93 (0.84-1.02) | 0.87 (0.79-0.93) | 0.93 (0.55-1.28) | 0.95 (0.78-1.10) | 0.92 (0.84-1.01) | 0.82 (0.72-0.90) | 0.94 (0.67-1.16) | 0.94 (0.82-1.06) | 0.84 (0.75-0.91) | 0.63 (0.48-0.74) |
| C | -0.02 (-0.07-1.02) | 0.70 (0.28-1.02) | 0.88 (0.75-1.00) | 0.88 (0.80-0.97) | 0.63 (0.11-1.09) | 0.91 (0.71-1.08) | 0.93 (0.83-1.02) | 0.87 (0.79-0.93) | 0.92 (0.58-1.26) | 0.94 (0.79-1.09) | 0.92 (0.84-1.01) | 0.81 (0.71-0.90) | 0.99 (0.81-1.19) | 0.95 (0.86-1.06) | 0.84 (0.76-0.91) | 0.62 (0.48-0.74) |
| D | -0.02 (-0.08-1.03) | 0.70 (0.29-1.03) | 0.88 (0.75-1.00) | 0.88 (0.80-0.97) | 0.65 (0.12-1.12) | 0.91 (0.72-1.09) | 0.93 (0.83-1.02) | 0.87 (0.79-0.93) | 0.94 (0.57-1.29) | 0.96 (0.80-1.09) | 0.92 (0.85-1.00) | 0.81 (0.71-0.90) | 1.00 (0.84-1.22) | 0.96 (0.87-1.06) | 0.85 (0.76-0.91) | 0.62 (0.48-0.74) |
| E | -0.02 (-0.07-1.04) | 0.71 (0.29-1.04) | 0.89 (0.75-1.00) | 0.88 (0.80-0.97) | 0.66 (0.10-1.13) | 0.91 (0.72-1.09) | 0.93 (0.83-1.02) | 0.87 (0.79-0.93) | 0.93 (0.59-1.29) | 0.95 (0.80-1.09) | 0.92 (0.85-1.01) | 0.81 (0.71-0.90) | 0.94 (0.71-1.18) | 0.94 (0.83-1.05) | 0.84 (0.75-0.91) | 0.62 (0.48-0.74) |

**classic**

| method | D1 1 | D1 3 | D1 10 | D1 30 | D2 1 | D2 3 | D2 10 | D2 30 | D3 1 | D3 3 | D3 10 | D3 30 | D4 1 | D4 3 | D4 10 | D4 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O | 1.00 (0.88-1.19) | 0.99 (0.92-1.08) | 1.00 (0.96-1.05) | 0.99 (0.97-1.02) | 1.00 (0.91-1.08) | 1.00 (0.94-1.05) | 1.00 (0.97-1.03) | 1.00 (0.98-1.02) | 1.00 (0.94-1.06) | 1.00 (0.96-1.05) | 1.00 (0.98-1.02) | 1.00 (0.99-1.01) | 1.00 (0.97-1.04) | 1.00 (0.98-1.03) | 1.00 (0.99-1.01) | 1.00 (0.99-1.01) |
| A | 1.01 (0.81-1.22) | 0.99 (0.90-1.07) | 1.00 (0.95-1.04) | 0.99 (0.96-1.02) | 0.99 (0.87-1.12) | 0.99 (0.93-1.06) | 1.00 (0.97-1.04) | 1.00 (0.98-1.02) | 1.01 (0.88-1.11) | 1.00 (0.95-1.05) | 1.00 (0.98-1.02) | 1.00 (0.99-1.01) | 1.00 (0.94-1.06) | 1.00 (0.97-1.03) | 1.00 (0.98-1.01) | 1.00 (0.99-1.01) |
| Bc | 1.02 (0.80-1.24) | 0.98 (0.89-1.08) | 1.00 (0.96-1.05) | 0.99 (0.97-1.02) | 0.99 (0.85-1.11) | 0.99 (0.93-1.06) | 1.00 (0.97-1.03) | 1.00 (0.99-1.02) | 1.01 (0.88-1.12) | 1.00 (0.95-1.05) | 1.00 (0.98-1.03) | 1.00 (0.99-1.01) | 1.00 (0.94-1.07) | 1.00 (0.97-1.03) | 1.00 (0.98-1.01) | 1.00 (0.99-1.01) |
| D | 1.03 (0.82-1.24) | 0.99 (0.89-1.08) | 1.00 (0.95-1.05) | 1.00 (0.97-1.02) | 0.98 (0.83-1.12) | 0.98 (0.92-1.06) | 0.99 (0.97-1.03) | 1.00 (0.98-1.02) | 1.02 (0.91-1.14) | 1.01 (0.96-1.06) | 1.00 (0.98-1.03) | 1.00 (0.99-1.01) | 1.03 (0.95-1.10) | 1.02 (0.98-1.04) | 1.00 (0.99-1.02) | 1.00 (0.99-1.01) |
Reading the table:

- **At bin 8 the detector, not the background, sets the recovery.** Every method, the ORACLE included, follows the same
  curve: with the exact background, a peak of box P/B 1 is recovered at a median 0.00 (D1), 0.55 (D2), 0.74 (D3) and 1.03
  (D4) on clean film; P/B 3 at 0.46 / 0.83 / 0.93 / 0.98; P/B 10 at 0.80 / 0.94 / 0.94 / 0.87; P/B 30 at 0.89 / 0.89 / 0.86 /
  0.69. Below ~4 e per pixel the stored-count law records a weak peak only where it pushes a pixel over a rounding
  threshold, and above ~20-50 e per pixel per frame the paralyzable dead time takes 10-38 % (A173). The background methods
  differ from the oracle by less than the IQR in every fly-scan cell at P/B >= 3.
- Within 5 % of truth (pre-registered WITHIN rule), fly-scan: only D4 at P/B 3 for F (every method) and for BA (A and the
  oracle); nothing at d < 2 A, nothing at P/B 30 (dead time). Within 10 %: P/B 10 at D2 (F, CN), P/B 3-10 at D3, P/B 1-3 or
  3-10 at D4 (`posthoc_passing_pb_v0.csv`).
- **Classic bin 4 (1 count = 1 e, no rounding) behaves as textbook summation:** the oracle is within 5 % from P/B 0.3 at d >= 1.25 A, from P/B 1 at D2 and from P/B 3
  at D1; the annulus A from P/B 1 at d >= 1 A and from P/B 3 at D1; the pool Bc from P/B 1 at d >= 1.25 A and P/B 3-10
  below; the physics fit D from P/B 3 (D2 from P/B 10).

## 4. Systematic background error per regime (POST HOC ratio-of-sums beta; Sec. 1.7)

beta = (true - method background) / true background under the box, in %; positive = the method's background is too LOW,
so the intensity comes out too HIGH by beta / (P/B) (in stored counts). Pooled value (68 % session bootstrap). Figure
`beta_by_regime_v0.png`.

| regime | method | D1 0.86-1.00 A | D2 1.00-1.25 A | D3 1.25-2.00 A | D4 2.00-4.00 A |
|---|---|---|---|---|---|
| F | A | +6.1 (-2.1 to +16.6) | -6.5 (-8.2 to -4.0) | -1.6 (-4.4 to +0.2) | -1.8 (-3.3 to -1.5) |
| F | B0 | -16.6 (-20.4 to -2.0) | -10.7 (-12.2 to -8.6) | -2.5 (-4.8 to -1.1) | +1.4 (-0.3 to +1.7) |
| F | B1 | -10.5 (-13.8 to +6.4) | -9.8 (-11.2 to -7.4) | -2.0 (-4.4 to -0.5) | +1.3 (-0.2 to +1.7) |
| F | B4 | +4.5 (+0.8 to +18.1) | -7.0 (-8.8 to -4.7) | -0.2 (-2.8 to +1.1) | -0.6 (-1.8 to -0.4) |
| F | P2 | +5.9 (+1.7 to +19.1) | -7.1 (-8.9 to -4.6) | -0.0 (-2.4 to +1.5) | -1.1 (-2.5 to -0.9) |
| F | C | +0.7 (+1.5 to +18.9) | -7.4 (-9.6 to -4.9) | -0.6 (-2.8 to +1.3) | -0.4 (-2.1 to -0.2) |
| F | D | -8.1 (-9.4 to +10.3) | -4.8 (-7.2 to -2.5) | +2.2 (-0.1 to +3.7) | +0.5 (-0.9 to +1.0) |
| F | E | +0.4 (-1.7 to +15.2) | -9.0 (-10.4 to -6.4) | -0.7 (-3.1 to +1.0) | -0.5 (-1.9 to -0.2) |
| BA | A | +6.8 (+5.9 to +21.1) | +5.5 (-3.4 to +4.7) | +0.8 (-3.6 to +2.8) | -2.2 (-3.4 to -1.5) |
| BA | B0 | +14.7 (+5.1 to +26.7) | +8.1 (-4.2 to +6.8) | +3.1 (-1.9 to +5.0) | -0.3 (-1.9 to +1.9) |
| BA | B1 | +25.2 (+13.5 to +36.4) | +11.0 (-2.6 to +9.2) | +4.4 (-1.0 to +5.7) | -1.3 (-2.7 to +1.0) |
| BA | B4 | +54.6 (+37.7 to +57.2) | +21.0 (+3.6 to +17.1) | +9.7 (+3.7 to +10.4) | -8.9 (-10.4 to -4.8) |
| BA | P2 | +60.3 (+40.1 to +59.3) | +22.5 (+5.5 to +19.7) | +9.6 (+4.1 to +10.6) | -9.8 (-11.4 to -5.4) |
| BA | C | +47.2 (+31.2 to +50.4) | +15.2 (+1.3 to +12.5) | +10.9 (+4.2 to +11.2) | -4.5 (-5.4 to -2.8) |
| BA | D | +50.3 (+28.3 to +49.7) | +26.9 (+11.0 to +23.0) | +18.0 (+9.7 to +16.8) | +0.3 (-0.7 to +1.5) |
| BA | E | +56.7 (+41.2 to +58.5) | +20.4 (+3.8 to +15.7) | +11.4 (+5.5 to +11.8) | -10.7 (-11.6 to -6.5) |
| CN | A | +11.5 (+3.0 to +12.3) | +3.1 (+2.0 to +5.8) | -1.8 (-5.0 to -0.7) | -1.7 (-2.7 to -1.7) |
| CN | B0 | +31.1 (+18.2 to +30.8) | +11.8 (+7.1 to +13.7) | +3.4 (+0.5 to +4.3) | -11.1 (-14.1 to -9.5) |
| CN | B1 | +34.4 (+24.2 to +36.1) | +12.3 (+8.3 to +15.1) | +3.8 (+1.2 to +4.9) | -11.3 (-14.9 to -9.8) |
| CN | B4 | +76.4 (+48.3 to +63.0) | +30.0 (+19.0 to +27.6) | +14.2 (+7.1 to +12.9) | -30.2 (-32.5 to -21.4) |
| CN | P2 | +79.1 (+50.7 to +65.1) | +30.9 (+20.5 to +29.8) | +15.3 (+8.3 to +14.1) | -32.9 (-34.3 to -23.3) |
| CN | C | +59.3 (+34.5 to +48.5) | +13.9 (+7.4 to +16.1) | +8.1 (+2.3 to +8.2) | -3.4 (-5.8 to -2.9) |
| CN | D | +63.7 (+36.3 to +50.8) | +23.5 (+15.0 to +25.4) | +15.0 (+8.6 to +15.9) | +0.8 (-0.3 to +1.2) |
| CN | E | +74.7 (+44.8 to +60.3) | +28.9 (+16.6 to +26.1) | +13.3 (+5.7 to +11.9) | -26.0 (-28.1 to -18.0) |
| classic | A | -1.1 (-2.3 to -0.9) | -0.2 (-0.6 to -0.1) | -0.2 (-0.8 to -0.1) | -0.7 (-1.0 to -0.6) |
| classic | Bc | +0.9 (-1.3 to +0.8) | +0.4 (-0.2 to +0.5) | +0.5 (-0.1 to +0.6) | -0.1 (-0.7 to -0.0) |
| classic | D | +1.7 (-1.9 to +2.6) | +0.4 (-2.6 to +0.1) | +2.6 (+1.5 to +3.3) | +3.1 (+1.3 to +3.0) |
- **Clean film (F):** every radial background (atlas C, truth E, pools B4/P2, physics D) is within -1 to +2 % at d >= 1.25 A
  and -5 to -9 % at d 1.00-1.25 A; at d < 1 A the box background is 0.5-1 count and beta is noise-dominated (session SD
  0.4). B0/B1 (pools containing bar- and crystal-adjacent frames) are the worst at d < 1.25 A (-10 to -17 %). The local
  annulus A: -2 % at d >= 1.25 A, -6.5 % at 1.0-1.25 A (part of it is spot tail leaking into the annulus at high P/B).
  Every radial method AND the montage truth read 5-9 % high at d 1.0-1.25 A on these hosts (A178).
- **Bar-adjacent (BA, 0-5 um from a bar):** every RADIAL background is too low at high angle: +47 to +60 % at d < 1 A,
  +15 to +27 % at 1.0-1.25 A, +10 to +18 % at 1.25-2 A (atlas, truth, P2, physics alike; this is the one-sided pedestal of
  A71, which no radial model carries). The local annulus A is +7 / +5.5 / +1 / -2 %. B0/B1 do better than P2 here because
  their pools contain bar-adjacent frames that carry the same pedestal.
- **Crystal neighbours (CN):** the same picture, stronger: radial +59 to +79 % (D1), +14 to +31 % (D2), +8 to +15 % (D3);
  annulus +11.5 / +3 / -2 / -2 %. At d 2-4 A the drop-based pools and the truth over-predict by 26-33 % (A180), the
  frame-keyed atlas (-3 %) and physics fit (+1 %) do not.
- **Classic bin 4:** A -0.2 to -1.1 %, Bc -0.1 to +0.9 %, D +0.4 to +3.1 %.
- Translated to intensities (fly-scan): on clean film a P/B-1 peak at d 1.0-1.25 A carries -5 to -9 % from the background
  (P/B 3: -2 to -3 %); next to a bar or a crystal a radial background adds +15 to +30 % at P/B 1 (+5 to +10 % at P/B 3)
  at d 1.0-1.25 A and +50 to +80 % at P/B 1 below 1 A.

## 5. Are the sigmas honest? (`honesty_v0.csv`, figure `pull_sd_v0.png`)

Pull robust SD (median), pooled over d bins:

| sigma | P/B group | F radial (B*, C, D, E) | F A | BA radial | BA A | CN radial | CN A | classic A / Bc / D |
|---|---|---|---|---|---|---|---|---|
| law, about lambda_box (pre-registered) | <= 1 | 0.92-0.94 (median -0.50) | 1.05 (-0.75) | 0.92-0.94 (-0.53) | 1.02 (-0.91) | 1.03-1.12 (-0.40 to -0.46) | 1.05 (-0.58) | 0.92 / 0.98 / 1.06 (~0) |
| law, about lambda_box | >= 10 | 1.85 (-1.42) | 1.82 | 1.67-1.69 (-1.2) | 1.65 | 2.32-2.34 (-1.75) | 2.34 | 1.01 / 1.02 / 1.03 |
| law, about lam_eff (POST HOC) | >= 10 | 1.04-1.05 | 1.04 | 1.06-1.07 | 1.07 | 1.04-1.07 | 1.04 | 1.01-1.03 |
| law, about lam_eff | <= 1 | 0.70-0.71 | 0.95 | 0.61-0.63 | 0.96 | 0.86-0.92 | 1.01 | 0.92-1.06 |
| naive (Poisson in stored counts) | >= 10 | 1.01 | 1.01 | 0.91-0.93 | 0.92 | 1.26 | 1.27 | 1.01-1.04 |
| naive | <= 1 | 0.75-0.76 | 1.44 | 0.77-0.81 | 1.50 | 0.79-0.89 | 1.26 | 1.06-1.25 |

- Pre-registered verdict: honest (SD 0.85-1.15 and |median| <= 0.5) for every classic method at both P/B groups; on
  fly-scan only at P/B <= 1 for the radial methods on CN and for D on F; nowhere at P/B >= 10.
- What the numbers mean: at bin 8 the COUNTING part of sigma_law is right (pulls about the law-expected intensity have SD
  1.04-1.07 at P/B >= 10), but the recovered intensity is BIASED relative to the true one by the counting law and dead
  time, and that bias varies with spot width and local rate, so the pull about the true intensity is 1.7-2.3 wide and
  centred at -1.2 to -1.8 (A176). The naive "stored counts are Poisson" sigma is twice the counting sigma at high counts
  (a count is 4 electrons); its pull SD near 1 at P/B >= 10 is that factor 2 cancelling the law spread by coincidence.
- At low P/B the model-background sigma_law is 30-40 % too large about lam_eff (the plug-in g(max(K', B)) over-counts
  sub-count backgrounds) and the annulus sigma is right on average, but for D1 boxes whose annulus is all zeros the annulus
  sigma has no background term, so A calls 39-57 % of P/B-1 peaks at d < 1 A "3 sigma" versus 21-38 % for the oracle and
  ~0 % for the radial methods (A177).
- Lost / below 3 sigma: at P/B 1 the radial methods put 97-100 % (D1-D2) and 91-95 % (D3) of fly-scan peaks below 3 sigma,
  the oracle 21-64 %; at P/B 3, 72-95 % (D1), 22-66 % (D2), 13-40 % (D3), 0-4 % (D4) versus 47-55 / 23-51 / 9-31 / 0-2 % for
  the oracle. Classic: below 3 sigma 45-48 % at D1 P/B 1, <= 3 % from P/B 3 at d >= 1 A.

## 6. Verdict and recommendation

Pre-registered outputs: P/B*_5 (truth) is undefined for every fly-scan method (A174); the pre-registered recommendation
(`recommendation_v0.csv`) therefore reduces to its tie-breaks and names A for every fly-scan cell except D4 (B0 on F and
BA, D on CN); classic: A at d < 1.25 A, Bc at d >= 1.25 A, all within 5 % from P/B 1-3. The decision statements the rule
generates (classic): "A recovers intensities within 5 % for P/B >= 3 at d 0.86-1.00 A and for P/B >= 1 at d >= 1.00 A in
regime classic"; "Bc ... P/B >= 10 / 3 / 1 / 1 at D1 / D2 / D3 / D4"; "D ... P/B >= 3 / 10 / 3 / 3". Fly-scan: "no method
recovers intensities within 5 % at d < 2 A at any P/B; at d 2-4 A every method does so at P/B 3 only (F), A and O also on
BA".

Reading (post hoc, from Secs. 3-5):

1. **Bin-8 fly-scan intensities are limited by the stored-count law and dead time, not by the background.** With a perfect
   background, a box P/B 1 peak loses 45-100 % at d < 1.25 A and the strongest lose 10-38 %. No background choice can
   fix this; an integrator must model the counting law (e.g. predict E[stored | background + peak] per pixel, as lam_eff
   does) or the data must be taken at bin 4. Use bin 4 (or classic raw counts) when intensities at d < 1.25 A matter.
2. **Background choice by regime (bin 8):**
   - clean film: any radial background is adequate (|beta| <= 2 % at d >= 1.25 A, 5-9 % at 1.0-1.25 A); the live choice is
     the atlas v1d (no labels needed, -0.4 / -0.6 / -7.4 % at D4 / D3 / D2); the local annulus is equivalent within noise.
   - bar-adjacent and crystal-neighbour frames (and therefore crystal frames, whose surroundings are both): at d < 2 A use
     the LOCAL ANNULUS (beta +1 to +12 %, vs +8 to +80 % for any radial model, atlas, physics fit, montage truth or P2 pool);
     at d 2-4 A the frame-keyed atlas or the physics fit (|beta| <= 5 %); do not use host-scaled drop pools (P2, B4) there
     on crystal neighbours (-30 %).
   - the REyes drop rules (P2, B4) that make the pool SHAPE right (drop validation v0) do not make the background under a
     peak right next to a crystal: the error is local, and a radial pool cannot carry it.
3. **Classic bin 4:** summation works: the annulus or the session pool gives |beta| <= 1 % and intensities within 5 % from
   P/B 1 (d >= 1 A); sigma (Poisson) is honest. The single-frame physics fit is 2-3 % worse and not needed.
4. **Sigma at bin 8:** report sigma from the stored-count variance (sigma_law), not from Poisson in stored counts (2x too
   large at high counts), and add the counting-law systematic (the lam_eff vs lambda gap, Sec. 5) as a separate term;
   without it the pulls about the true intensity are 1.7-2.3 wide.

Systematic intensity error by grid regime (background part, P/B 1; P/B 3 is one third):

| regime | d 0.86-1.00 A | d 1.00-1.25 A | d 1.25-2.00 A | d 2.00-4.00 A | best method |
|---|---|---|---|---|---|
| clean film | noise-dominated (+-40 % session SD) | -5 to -9 % (radial), -6.5 % (A) | -1 to +2 % | -2 to +1 % | atlas v1d (C) or any radial |
| bar-adjacent | +47 to +60 % radial; +7 % A | +15 to +27 % radial; +5.5 % A | +10 to +18 % radial; +1 % A | -10 to +0 % | A below 2 A; D/C above |
| crystal neighbour | +59 to +79 % radial; +11.5 % A | +14 to +31 % radial; +3 % A | +8 to +15 % radial; -2 % A | -33 to +1 %; A -2 %, D +1 % | A below 2 A; D/C above |
| classic bin 4 | -1 % (A), +1 % (Bc) | 0 % | 0 % | -1 to 0 % | A or Bc |

## 7. Anomalies

- A173: at bin 8 the recovered intensity with the exact background (oracle) is far from the injected one: box P/B 1 peaks at
  d < 1.25 A are recorded at a median 0-0.55 (mean 0.25-0.68), and box P/B 30 peaks lose 10-38 % (the most at d 2-4 A,
  where the background and hence the peak is brightest: 0.62-0.77). The first is the rounding threshold (a weak peak only
  registers where a pixel is near N = 2, 6, 10 e), the second the paralyzable dead time. The law-expected lam_eff
  reproduces the oracle in the synthetic check (24/28 cells). Consequence: bin-8 intensities need a counting-law model in
  the integrator, whatever the background.
- A174: recovery is non-monotone in P/B (rises with the threshold, falls with dead time), so the pre-registered P/B*
  ("this and every higher P/B within X %") is undefined for every fly-scan method, the oracle included.
- A175: per-peak median beta is degenerate at bin-8 counts (box host sums mostly 0); ratio-of-sums beta used post hoc.
- A176: sigma_law pulls about the TRUE intensity have SD 1.7-2.3 at P/B >= 10 for every method (oracle 1.64-2.36), about the
  law-expected intensity 1.04-1.07: the excess is the counting-law/dead-time loss varying with spot width (sigma 0.5-2 px)
  and rate, i.e. an unmodelled systematic, not counting noise.
- A177: the local annulus on an all-zero D1 annulus has zero background variance, so A declares 39-57 % of P/B-1 peaks at
  d < 1 A "above 3 sigma" (oracle 21-38 %, radial methods ~0 %): spurious significance at sub-count backgrounds.
- A178: on the clean-film hosts (film_ok frames in the 3 crystal-richest rows per session) every radial method and the
  montage truth predict 5-9 % MORE background than the host has at d 1.0-1.25 A (beta -4.8 to -9.0 %, session bootstrap
  excluding 0 for E, C, P2, B4), while d >= 1.25 A is within 2 %. Either the film in crystal-rich rows is lower at high
  angle than the session's film_ok (a row/position effect inside a session), or the host-scaling window 0.34-0.80 transfers
  a shape difference. Not tested.
- A179: single-frame physics fits (gate+tie, S v2) put t at the 100-nm bound in 38 % of fly-scan hosts (rms 19 %, counting-
  limited chi2/dof 1.4): one bin-8 frame does not constrain t; the fitted background is still as good as the atlas on film
  (|beta| <= 5 % at d >= 1 A) but worse next to bars and crystals (D2 +23 to +27 %).
- A180: on crystal-neighbour hosts at d 2-4 A the host-scaled drop pools (B4 -30 %, P2 -33 %) and the montage truth (-26 %)
  over-predict the background, whereas B0/B1 (-11 %), the frame-keyed atlas (-3 %) and the physics fit (+1 %) do not:
  scaling a clean-film shape to the host over s 0.34-0.80, a window lifted by the crystal-neighbour excess (A133), pushes the
  low-angle prediction up. Any host-scaled clean-film background inherits this on frames with a high-angle excess.

## 8. Files

- Code: `src/podbg/peaks.py`, `scripts/peak_recovery_v0.py`, `tests/test_peak_recovery.py` (9 tests).
- Outputs (`~/pod_data/background/peak_recovery_v0/`, sidecars on each): `spot_shape_v0.csv`, `hosts/<sid>_hosts_v0.npz`,
  `synth_*`, `peaks_v0.parquet` (per peak x method), `hosts_v0.csv`, `cells_v0.csv`, `cells_by_session_v0.csv`,
  `decisions_v0.csv`, `honesty_v0.csv`, `recommendation_v0.csv`, `key_table_v0.md`, `posthoc_passing_pb_v0.csv`.
- Figures (`notes/figures/peak_recovery_v0/`): `ratio_vs_pb_{F,BA,CN,classic}_v0.png`, `beta_by_regime_v0.png`,
  `pull_sd_v0.png`.
