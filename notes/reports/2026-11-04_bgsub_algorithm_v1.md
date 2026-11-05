# bgsub v1: the near-stop block, an expected-error local switch, the information limit of the stored-count ML, and rules with power (2026-11-04)

Terms of reference: fix the failures of bgsub v0 (`2026-11-04_bgsub_algorithm_v0.md`, A193-A199) and make the algorithm
VALID by its own pre-registered rules; OWNER RE-SCOPE item 2 (priority 2). The v0 API stays backward compatible: every
public function takes `version="v0"` and reproduces v0 (regression test against outputs frozen at 07a06b9 before any
v1 change, c0ffb8c). No other packages; XDS only as a reference. The simulator is used read-only, pinned to
`version="v1"`. Anonymous sids only; nothing runs on HPC.

Code: `src/podbg/bgsub/` (new `zone.py`; v1 paths in `core.py`, `integrate.py`, `spotlaw.py`), tests
`tests/test_bgsub_v1.py`, scripts `scripts/bgsub_v1_*.py`. Outputs with provenance sidecars:
`~/pod_data/background/bgsub_v1/`. Figures: `notes/figures/bgsub_v1/`. Anomalies are numbered from A233.

## 1. Stage 0: what was looked at before the pre-registration (DEV data and the model only)

Allowed inputs: the v0 report and tables (already published, including the v0 TEST results at the level reported
there), the three fly-scan DEV sessions of v0 (s0047f887, s4fead7c7, sc19f33b9; their 108 host frames without injected
peaks), the simulator v1, and Monte Carlo under the model law. NO v0 TEST-session frame was opened for v1 design. One
v0 TEST table (`real_test_peaks_v0.parquet`) was re-tabulated by session for D0 only (Sec. 1.1), which is a re-reading of
a published v0 result; no design parameter was set from it.

### 1.1 A194 (D0 next to the beam stop)

- (a) Re-tabulating the published v0 test rows by session: the D0 failure of G is carried by TWO of 17 fly-scan
  sessions (beta -7.3 % with error 3.5x the box noise in one, -4.5 % in another); in the other 15 G's D0 error is 0.0-0.53
  of the box noise versus FLAT's 0.35-0.75. The failure is a property of some sessions, not of every frame.
- (b) DEV hosts (`scripts/bgsub_v1_diag.py`, `polar_resid_*.npz`): the v0 residual K - G on a polar grid near the stop
  shows a SHADOW (penumbra) next to the stop ARM: relative to a fit that excludes d <= 5 px of the stop, the stored
  counts are -43 to -54 % at d = 1 px, -21 to -34 % at d = 2, -6 to -15 % at d = 3 (two sessions; -10 / -1 / 0 % in the
  third), while next to the DISC rim there is none (+0 to +6 %). With the penumbra pixels inside the fit, v0's
  smooth model is pulled down by ~1 % over d 4-30 px (r < 130): the sign and size of v0's D0 beta (-0.6 to -1.5 %).
  Excluding d <= 5 px from the fit leaves no near-stop structure above the noise in two sessions; the third keeps a
  smooth m ~ 1-2 pattern of +-3-4 % at r 80-140 px.
- (c) Candidate models scored on the DEV hosts by the box-level error against the frame itself (`bgsub_v1_d0dev.py`,
  boxes in s 0.13-0.16 / 0.16-0.19 / 0.19-0.25 / 0.25-0.5 with their discs excluded from the fit; error/noise =
  sqrt(max(mean (B - K)^2 - mean sum g, 0) / mean sum g)):

  | variant | 0.13-0.16 | 0.16-0.19 | 0.19-0.25 | 0.25-0.50 |
  |---|---|---|---|---|
  | v0 G | 0.28 | 0.32 | 0.58 | 0.34 |
  | FLAT | 0.39 | 0.47 | 0.73 | 0.48 |
  | penumbra (8 azimuthal sectors x 6 knots) | 0.60 | 0.29 | 0.59 | 0.34 |
  | penumbra + near-zone harmonics m <= 12 on 16-px spline | 1.26 | 1.74 | 1.02 | 0.35 |
  | penumbra + m <= 8 on 8-px spline | 0.80 | 0.91 | 0.81 | 0.42 |
  | near-zone m <= 12 only | 0.70 | 0.67 | 0.72 | 0.44 |

  On DEV, v0 G is already better than FLAT at D0 and every richer azimuthal model ADDS variance (120-240 extra columns)
  without removing error: the near-zone harmonics are NOT adopted (kept as an option, `zone_kw={"M_near": ...}`, default
  0). The 8-sector penumbra was under-determined (its coefficients scattered +-0.3); the shadow has two classes (arm,
  disc), so the adopted penumbra block has 3 classes (disc, the two sides of the arm) x 6 knots. On an analytic frame
  with a 50 % arm shadow it removes the bias at d 1-2 px (+36 % in v0 -> -1.8 %) and is unbiased on shadow-free frames
  (100 frames: +1.8 +- 1.2 % at d 1-2, within 2 SE at every distance); its price is variance at d < 3 px.

### 1.2 A197 (G-ML low at P/B <= 1 on sparse bin-8 frames)

DEV host statistics: at D2 (d 1.0-1.25 A, r 400-440 bin-8 px) the archive fly-scan frames store 0.003-0.007 counts per
pixel (mu0 = 0.14-0.21 e per bin-8 pixel; at r 330-380, 0.009-0.02 counts). The stored values are 0 / 1 only and their
frequencies match the model law exactly. Monte Carlo of one box under the MODEL law with the EXACT background
(`scripts/bgsub_v1_ml_mc.py`, `ml_mc_v1.csv`; r_box 3, 29 px, sigma 0.8 / 1.2 / 1.6 px, tie 0.35, TAU_PHYS_V1):

| bin | mu0 (e/px) | b (counts) | P/B 1: ML mean ratio | LC | Firth | CR relative SD | P/B 3: ML |
|---|---|---|---|---|---|---|---|
| 8 | 0.15 | 0.004 | 0.66 +- 0.05 | 0.71 | 0.61 | 1.21 | 0.92 |
| 8 | 0.30 | 0.015 | 0.89 +- 0.03 | 0.92 | 0.83 | 0.67 | 1.02 |
| 8 | 0.60 | 0.057 | 1.02 +- 0.02 | 1.03 | 0.98 | 0.41 | 1.00 |
| 4 | 0.15 | 0.15 | 0.99-1.01 | | | 0.63 | 1.00 |

So the v0 real-host bias is reproduced by the model itself: it is an INFORMATION limit, not a code defect, not
host heterogeneity and not the dead-time branch. Source: at sparse bin-8 counts the stored excess responds CONVEXLY to the
spot (an electron is stored only when a pixel crosses the N / 4 rounding threshold), so every estimator that inverts
the response is a concave function of a nearly sufficient statistic (Jensen: E[L^-1(I)] <= L^-1(E I) = lambda), and the
likelihood's lower boundary (lambda >= -min mu_p / f_p) takes the all-zero boxes. Firth's first-order bias correction
(Jeffreys penalty) makes it WORSE, i.e. the regime is not asymptotic. Over the whole MC grid the ML mean is within 2.3 %
whenever the Cramer-Rao relative SD of the box intensity is <= 0.45 and biased by 5-60 % above it; at bin 4 (Poisson,
linear response) it is unbiased at every dose. A joint ML over n equivalent observations (pooling BEFORE inverting) at
mu0 0.15 e, P/B 1: 0.49 (n 1), 0.82 (n 3), 0.93 (n 10); at mu0 0.30: 0.89 / 0.96 / 0.99.

### 1.3 A198, R1, R3 (from the v0 tables)

- A198: the chi^2 probe test has power ~1 against misfits far below the noise of one spot box with ~10^3 probes, so
  it switches on 60-93 % of real frames although switching costs variance. A p value answers the wrong question.
- R1 (A193): 120 cells each tested at 2 SE have a family false-fail rate near 1 (expected 6 false fails at 5 % each);
  6 frames per cell gave SEs up to 1.8 %. The rule had neither the right multiplicity nor the power.
- R3: the v0 failures were (i) bin 16 (declared out of scope in v0), (ii) the first-order summation variance
  sum [g(B) + g'(B)(K - B)] (A196: g' at the background, 0.9, is far from g' at the spot pixels, 0.25, at bin 8), (iii) the
  ML pulls at P/B <= 1 on sparse frames, which is A197's information limit (pull robust SD 0.05-0.6 where most boxes
  are all-zero), (iv) bin-4 ML pull means -0.18 / -0.20 at low P/B: with a sigma evaluated at the estimate,
  E[pull] = -d sigma / d lambda to first order for ANY sigma that is a function of the estimate alone (MC: -0.1 to -0.2
  at the information limit, -0.7 to -2.1 below it).

## 2. Pre-registration (written and committed BEFORE any v1 evaluation run)

### 2.1 Fix 1: the near-stop block (method G1 = the v1 global model)

The v0 design plus PENUMBRA columns R(r) phi_k(d) [class j] (`zone.py`): d = Euclidean distance to the beam stop (the
connected components of the mask, opened with a (2 x 2 + 1)^2 square at 1k to drop seams and borders, that reach within
0.2 N of the centre; or a user-given `stop`), phi_k linear hats on d = 1..6 px x N / 1024, j = disc / arm side + / arm
side - (the disc is the stop opened with a disc of 0.6 x its largest inscribed radius; a pixel takes the part of its
nearest stop pixel and the side of the arm as seen from the centre), R(r) = the radial (m = 0) profile of the OTHER
checkerboard half (cross-fitting, as v0's derivative columns). Cell values of the hats are means over the USED pixels.
Penalties: ridge 1e-4 x the block's mean information, second differences along d within a class (lam_rel). The pilot
(OLS) fit leaves the penumbra pixels out; the two weighted rounds include them with their columns. Linear in K, so
v0's unbiasedness theorem holds unchanged; the box variance includes the new columns (exact sandwich). Near-zone
harmonics: option, off. Speed: ring-aggregated cross terms, no dense rows.

### 2.2 Fix 2: the expected-squared-error switch (method H = AUTO v1)

Derivation. At a probe position j (spot-free, box and annulus valid and unflagged) let Lb_j = mean over the box of
(K - G) and La_j = mean over the annulus of (K - G). Write the residual structure delta = b - G (truth minus the global
estimate; it contains G's misfit AND G's estimation error) and the count noise n: Lb = delta_b + n_b, La = delta_a + n_a,
with n_b, n_a independent (disjoint pixels) and of zero mean. For a spot box, the error of G's box-mean background is
-delta_b and that of GL (G plus the annulus residual) is delta_a - delta_b + n_a. Hence

    MSE_G  = E[delta_b^2],        MSE_GL = E[(delta_a - delta_b)^2] + s_a^2,
    E[Lb^2] = E[delta_b^2] + s_b^2,   E[(La - Lb)^2] = E[(delta_a - delta_b)^2] + s_a^2 + s_b^2
    =>  MSE_GL - MSE_G = E[(La - Lb)^2] - E[Lb^2]                                    (exact; no variance model)

GL is chosen where its estimated expected squared error is smaller: D_z = mean_j Lb_j^2 - mean_j (La_j - Lb_j)^2 > 0,
i.e. where the local-structure bias of G under a box exceeds the extra variance the annulus costs. No p value, no
threshold, no count-law assumption (the noise terms cancel). Zones: radial bands r in [r_lo 1.5^k, r_lo 1.5^(k+1)) per
frame (frame-defined, not the evaluation bins); probes on a grid of stride r_out (overlapping annuli are allowed: only
means of the two statistics enter), excluded within r_excl + r_out of any spot to be integrated and wherever a box or
annulus pixel is masked or flagged; a zone with < 10 probes keeps G. H uses GL's background and variance in the zones
that switch and G1's elsewhere; H-ML is the ML on H's background.

### 2.3 Fix 3: G-ML is bounded, not changed

The v1 ML estimator is v0's (Firth and LC are not better, Sec. 1.2). Added: for every -ML intensity the INFORMATION
LIMIT lam_min5 (electrons), the smallest box intensity at which the Cramer-Rao relative SD of the box intensity is
0.45, computed from the expected Fisher information under the declared law, the background and the profile (no data
of the spot), so it can be used without selection bias in an evaluation and as a detection-limit statement by users;
and `pooled_ml` (one joint likelihood over equivalent observations). The bound claimed: |mean bias| <= 5 % for
lambda >= lam_min5 (from 1.2: <= 2.3 % on the MC grid). Target of the task (|bias| <= 5 % at P/B 1, D2, bin 8) is
tested as R6a and expected to FAIL at archive dose; the acquisition that meets it is reported (bin 4, or the bin-8 dose
at which lam_min5 <= the P/B-1 intensity).

### 2.4 Fix 5: summation variance from the profile-model expectation

With spot sigmas given, the box variance of a summation is V_K = phi sum_box g(E_p), E_p = law.E(mu_eff(mu_p,
max(lambda^, 0) f_p)) with lambda^ the method's ML estimate (the stored-count variance at the model's expected value of
each pixel, A196); without sigmas the v0 first-order form stays.

### 2.5 Evaluation design

SYNTHETIC (simulator v1, `version="v1"`, read-only):
- Frames: `simulate_frames(grid, dose = f x DOSE0_V1, binning = b, beam_stop = True, return_expected = True, seed = s,
  version = "v1")`, one call per frame (fresh latents: mass field, square factor, dose, centre jitter 0.5 bin-8 px ->
  frame-to-frame variation). The stored frame is drawn from the (possibly structured) expected electrons with the
  simulator's own chain (Poisson at the dead-time-thinned mean, `sim.detectors.quantise_fast`, tie 0.403, gain on), so
  the host electrons are known and peaks are injected with `podbg.peaks.inject(N_known=...)`.
- Structures: PLAIN (the simulator as is) and STRUCTURED = sim `ellipticity = 1.008`, the v0 m = 1 bar pedestal
  (0.15 x ring-free profile x ramp s 0.3 -> 0.6 x max(0, cos(phi - phi_b))), and an ARM shadow
  T(d) = 1 - 0.5 exp(-(d - 1) / (N / 1024)) on pixels whose nearest stop pixel is on the arm (d from the estimator's
  mask; the DEV measurement of 1.1 b). Mask = the simulator's stop dilated by 1 px + a 2-px border.
- Cells: lacey_utc, bins 4 and 8, f in {1, 3, 10} (f = 1 is archive sparsity: mu 0.10 e per bin-8 px at D2),
  {plain, structured}: 2 x 3 x 2 = 12 frame cells x 24 frames (FIX 4: 24 frames per cell, pre-registered) = 288
  frames. Robustness (labelled UNTESTED AGAINST DATA): quantifoil_r22 and formvar_carbon_tedpella, bin 8, f in {1, 3},
  {plain, structured}, 8 frames per cell = 64 frames.
- Spots: as v0 (`bgsub_v0_common.place_peaks`): 7 P/B {0.03 .. 30} x 5 d bins (D0-D4), sigma from
  `spot_shape_v0.csv`, centre known to the estimator to +-0.5 px. Seeds: dev 0-99 (debugging only), test 30000+ (run
  once). Spot-free twins (same frame without spots) for R2.
- Methods on the same frames: v0 (G, GL, AUTO chi^2, FLAT, PLANE, ROBUST, G-ML, AUTO-ML) and v1 (G1, GL1, H, G1-ML,
  H-ML).

REAL (the v0 harness, the same TEST sessions, hosts, replicates and SEEDS as v0, so the injected frames are identical
and v0 and v1 are compared pairwise): 17 fly-scan sessions (F / BA / CN, bin 8) and 20 classic bin-4 sessions; dev =
v0's dev sessions (debugging only). Truth: the oracle K0 and lambda_box / lambda_eff.

### 2.6 Rules (v1; pre-registered)

- R1 (fix 4) SYNTHETIC UNBIASEDNESS: in every lacey cell (binning x f x structure x d bin = 60 cells; pooled over P/B;
  24 frames), |beta| <= max(0.005, 3.2 SE) for G1 AND for H (SE: frame-cluster bootstrap of the ratio of sums;
  3.2 = two-sided Bonferroni over 60 cells at family level 0.08).
- R2 COUNT STATISTICS / TRUNCATION (unchanged from v0): spot-free twins, per binning x f x 4 radial bands, pixel-pooled
  |bias of G1| <= max(0.3 %, 2 SE) and |masking on - off| <= 0.1 %.
- R3 (fix 5) SIGMA HONESTY: synthetic lacey, per binning (4, 8) x {P/B <= 1, P/B >= 10}: pull robust SD in [0.9, 1.1]
  and |pull mean| <= 0.1, for H summation (about lambda_eff) and H-ML (about lambda_box), on spots with
  lambda_box >= lam_min5 (the information limit of 2.3, data-free). Also reported on ALL spots (v0's form) and on real
  frames (bounds [0.85, 1.15], 0.25).
- R4 REAL, BETTER THAN FLAT (v0's clauses, applied to H): per regime x d bin: |beta_H| <= max(0.01, 2 SE) in >= 90 % of
  cells, rmse_bg(H) < rmse_bg(FLAT) in every cell, |beta_H| <= |beta_FLAT| + 2 SE in every cell. Also reported for G1.
- R5 SPEED: median G1 fit + map on a 1024^2 frame (numba, plan cached, >= 10 frames) <= 50 ms on this Mac.
- R6 (fix 3; reported, not in the verdict, as v0's R6): (a) the task target: |G1-ML mean ratio - 1| <= 5 % at P/B 1,
  D2, bin 8, every f (expected to FAIL at f = 1); (b) the bound: in every lacey cell with P/B >= 1, D1-D4, bins 4 / 8,
  among spots with lambda_box >= lam_min5, |G1-ML ratio - 1| <= 5 % (cells with >= 20 such spots).
- A2 (fix 2; reported): H's GL fraction per regime and its background MSE against G1 and GL1 per regime x d bin.
- VERDICT: bgsub v1 is VALID iff R1, R2, R3 (synthetic) and R5 pass and R4 passes in F and classic. BA and CN are
  reported. No parameter changes after a test run; failures are reported with their cells.

(Pre-registration ends here; everything below was produced afterwards.)

### 2.7 Amendment 1 (DEV only, before any test run)

The dev runs (synthetic seeds 0-63, 4 frames per lacey cell; real dev sessions of v0) were used to debug the pipeline.
One change, a defect of the penumbra block rather than a tuning to a score: with hats on d = 1..6 px the last hat
reaches 7 px, i.e. into the pixels of boxes placed >= 9 px from the mask, and its noise made G1 WORSE than v0 next to
the stop on the real dev hosts (D0 s < 0.19, error/noise F 0.45 vs 0.13, BA 0.73 vs 0.29). The measured shadow is zero
beyond d = 3-4 px (1.1 b), so the hats now cover d = 1..4 px (reach < 5 px): `bgsub_v1_d0dev.py` errors 0.30 / 0.20 /
0.57 / 0.32 (v0 0.28 / 0.32 / 0.58 / 0.34), real dev D0 F 0.16 / BA 0.34 / CN 0.32 (v0 0.13 / 0.29 / 0.28). Observed
on dev and NOT acted on: the 4-frame cells give bootstrap SEs that are too small (a D1 cell at -2.3 +- 0.3 % was
-0.7 +- 0.7 % with 8 frames; A193 again), R2 failed 2 / 24 cells, H switches on 2-4 % of real dev spots (v0 AUTO
62-93 %), 0-2 % of plain and 13-18 % of structured synthetic spots.

## 3. Completeness of the test runs (checked before any rule was read)

The runs used `multiprocessing` (6 workers each). Every planned unit was checked against its checkpoint:
- SYNTHETIC (seeds 30000+, run once): 352 / 352 planned frames (lacey 12 cells x 24, robustness 8 cells x 8), each with
  35 injected spots; no missing or extra checkpoint.
- REAL: 37 / 37 test sessions (17 fly-scan, 20 classic), 1,328 host-replicates and 48,930 spots per method: the same
  counts as v0's test run, per regime (F 13,650, BA 14,280, CN 12,600, classic 8,400). Joined row by row to v0's
  rows: the injected intensities and the oracle K0 agree exactly (max |diff| 0), and v0's G and FLAT backgrounds are
  reproduced bit for bit, so the v0 / v1 comparison below is paired on identical frames.

## 4. Synthetic results (simulator v1; tables `synth_test_{cells,trunc,pulls,ml}.csv`, rules `synth_test_rules.json`)

### 4.1 Background (lacey; beta %, error / box noise)

| bin | struct | method | D0 | D4 | D3 | D2 | D1 | error D0-D1 |
|---|---|---|---|---|---|---|---|---|
| 4 | plain | v0 G / v1 G1 / v1 H | -0.04 / -0.05 / -0.02 | | | | +0.08 / +0.09 / +0.09 | 0.07-0.11 (all three) |
| 4 | structured | v0 G | -0.18 | | | -0.17 | +0.10 | 0.07-1.18 (D4 1.18) |
| 4 | structured | v0 AUTO | +0.49 | | | -0.63 | +0.72 | 0.39-0.71 |
| 4 | structured | v1 H | +0.37 | | | -0.12 | +0.11 | 0.07-0.70 (D4 0.70) |
| 8 | structured | v0 G / v1 H | -0.11 / +0.15 | | -0.29 / -0.22 | -0.13 / -0.08 | +0.65 / +0.65 | v0 G D4 1.08, H D4 0.53 |
| 8 | structured | FLAT / v0 AUTO | +0.43 / +0.37 | | | +0.82 / +1.04 | +4.37 / +3.89 | 0.45-0.56 |

Background error in box-noise units, lacey plain: G1 and H 0.07-0.11 at both bins (FLAT 0.38-0.83). The structured
frames contain a bar pedestal whose ramp and half-cosine have kinks the model's span does not hold: at D4
(s 0.25-0.5) v0 G and G1 err by 1.08-1.18 box noise and H by 0.53-0.70 (GL in that zone). Robustness presets (untested
against data): quantifoil and formvar behave like lacey (G1 / H error 0.06-0.12 plain; structured D4 0.38-0.78);
the quantifoil structured D1 cell is +3.8 % for G1 and -14.4 % for FLAT.

### 4.2 Rules

| rule | v0 (its own pre-registration, v0 test) | v0 on the v1 frames (v1 rule) | v1 |
|---|---|---|---|
| R1 unbiasedness, 60 cells, 24 frames, max(0.5 %, 3.2 SE) | FAIL 7 / 120 (6 frames, 2 SE) | G PASS (0 fails; 3 at 2 SE) | G1 PASS (0; 2 at 2 SE), H PASS (0; 2 at 2 SE); median SE 0.20 % |
| R2 truncation / count law, 24 cells | PASS 48 / 48 | (not run for v0) | FAIL 1 / 24: bin 8, f 3, band 450-724 px, +0.49 +- 0.24 % (A233) |
| R3 sigmas (H, H-ML; lambda >= lam_min5) | FAIL | v0 G summation bin 8 high P/B rsd 0.80 | H PASS (rsd 0.97-1.01, mean -0.05..0.00); H-ML FAIL: bin 8 P/B <= 1 mean -0.157 (rsd 1.02; A234) |
| R6a ML target P/B 1, D2, bin 8 | (A197) | | FAIL: 0.88 +- 0.15 (f 1), 0.96 +- 0.07 (f 3), 0.99 +- 0.03 (f 10) |
| R6b ML above lam_min5 | | | FAIL 8 / 88 cells (0.89-1.08; n 30-48 spots per cell) |
| A2 AUTO switch fraction (lacey plain / structured) | | 0.21 / 0.86 | 0.007 / 0.19 |

Pulls (all spots / informative spots, bin 8, P/B <= 1): H-ML mean -0.16 / -0.16; at bin 4 -0.51 / -0.05: the
information limit removes the bin-4 artefact (spots below lam_min5) and not the bin-8 one. The v1 summation variance
(Sec. 2.4) fixes A196: v0 G summation pull rsd at bin 8, P/B >= 10, 0.80 -> v1 1.01-1.03.

## 5. Real-frame results (the v0 test frames; `real_test_{cells,d0,pulls,cells_pb}.csv`, `real_test_rules.json`)

### 5.1 Background vs the oracle: beta % (v1 H's SE) and error / box noise

| regime | method | D0 4-8 A | D4 2-4 A | D3 1.25-2 A | D2 1-1.25 A | D1 0.86-1 A |
|---|---|---|---|---|---|---|
| F | v0 G | -0.70 / 0.88 | +0.02 / 0.14 | -1.33 / 0.00 | -0.69 / 0.22 | -2.95 / 0.10 |
| F | v1 G1 | -0.42 / 0.87 | +0.07 / 0.14 | -1.30 / 0.00 | -0.65 / 0.22 | -3.09 / 0.11 |
| F | v1 H | -0.18 (0.27) / 0.27 | +0.30 (0.76) / 0.00 | -1.30 (2.0) / 0.00 | -0.65 (3.9) / 0.22 | -3.09 (8.7) / 0.11 |
| F | FLAT | +0.21 / 0.39 | +1.87 / 0.38 | +0.30 / 0.33 | +2.22 / 0.48 | +0.55 / 0.41 |
| BA | v0 G / v1 H / FLAT | -0.58 / -0.32 / +0.16; 0.41 / 0.43 / 0.51 | 0.33 / 0.33 / 0.55 | 0.37 / 0.37 / 0.52 | 0.12 / 0.12 / 0.37 | 0.27 / 0.27 / 0.53 |
| CN | v0 G / v1 H / FLAT | -0.98 / -0.27 / +0.43; 1.18 / 0.76 / 0.59 | 0.43 / 0.37 / 0.55 | 0.29 / 0.29 / 0.58 | 0.00 / 0.00 / 0.31 | 0.39 / 0.39 / 0.63 |
| classic | v0 G / v1 H / FLAT | -1.51 / -0.77 (0.53) / +0.48; 7.47 / 6.39 / 5.96 | 0.64 / 0.63 / 0.86 | 0.29 / 0.28 / 0.67 | 0.36 / 0.36 / 0.65 | 0.34 / 0.34 / 0.51 |

D0 split (error / noise, s 0.13-0.19 | 0.19-0.25): F v0 G 1.29 | 0.43, G1 1.29 | 0.43, H 0.30 | 0.26, FLAT 0.28 | 0.45; CN
v0 G 1.76 | 0.50, H 1.15 | 0.32, FLAT 0.63 | 0.57. The penumbra block does NOT move the test-set D0 failure (A235): the
expected-error switch does, by using GL in the 2 % of spots whose zone asks for it. H switches on 2.0-4.6 % of real
spots (v0 AUTO 60-93 %, A198 resolved). Figure: `notes/figures/bgsub_v1/real_background_error_v0_v1.png`.

### 5.2 R4 (v0 clauses) and intensities

| R4 | v0 G (v0 report) | v1 G1 | v1 H |
|---|---|---|---|
| F | FAIL (D0 rmse) | FAIL (D0 rmse 0.87 vs 0.39) | PASS |
| BA | PASS | PASS | PASS |
| CN | FAIL (D0) | FAIL (D0) | FAIL (D0 rmse 0.76 vs FLAT 0.59) |
| classic | FAIL (D0 rmse and beta) | FAIL | FAIL (D0 rmse 6.39 vs FLAT 5.96; beta now within 2 SE, A236) |

Every beta clause passes for H in every regime (100 % of cells within max(1 %, 2 SE); never worse than FLAT + 2 SE).
Intensities (F, ratio of sums): summation about lambda_eff 0.98-1.04 at P/B >= 0.3 for G and H; H-ML at D2 P/B 1 / 3 /
10: 0.85 / 0.97 / 1.02 (v0 G-ML identical); D1 0.49 / 0.86 / 1.01. Only 3 % (D1) and 13 % (D2) of fly-scan P/B-1 spots
are above lam_min5; above it H-ML gives 0.89-1.05 at P/B 1 and 0.98-1.00 at P/B >= 3. Real pulls on informative spots:
H-ML rsd 0.98-1.08, mean -0.09..+0.03 (all spots: rsd 0.59-0.77 at P/B <= 1, the information limit).

## 6. Speed (R5; `speed_v1.json`, plan cached, machine load ~5 from parallel agents)

| frame | v0 numba | v1 numba | v1 numpy | v1 plan |
|---|---|---|---|---|
| 512^2 (bin 16) | 15.6 ms | 18.9 ms | 34.1 ms | 49 ms |
| 1024^2 (bin 8) | 44.5 ms | 42.0 ms | 112 ms | 216 ms |
| 2048^2 (bin 4) | 116 ms | 115 ms | 472 ms | 1.1 s |

R5 PASS (42.0 ms; the penumbra cross terms are aggregated by ring through a sparse matrix, no dense rows).

## 7. Verdict (pre-registered rules)

| rule | v0 | v1 |
|---|---|---|
| R1 | FAIL 7 / 120 | PASS (G1 and H, 0 / 60) |
| R2 | PASS | FAIL 1 / 24 (A233) |
| R3 | FAIL | FAIL (H PASS; H-ML bin 8 low P/B pull mean -0.157, A234) |
| R4 F / classic | FAIL / FAIL | PASS / FAIL (D0 rmse only, A236) |
| R4 BA / CN (reported) | PASS / FAIL | PASS / FAIL (D0 rmse) |
| R5 | PASS 42.7 ms | PASS 42.0 ms |
| R6a / R6b (reported) | FAIL | FAIL / FAIL 8 of 88 |

**bgsub v1 is NOT VALID by its pre-registered verdict**: it fails R2 in one cell, R3 for H-ML in one cell and R4 in
classic D0. What the evidence supports: (i) the expected-squared-error switch replaces the p-value switch (2-5 % of spots
instead of 60-93 %) and turns the fly-scan F D0 failure into a pass (0.27 vs FLAT 0.39, v0 0.88); (ii) with 24 frames
per cell and a multiplicity-correct threshold R1 passes for v0 G as well, so v0's R1 failure was the rule's power
(A193 confirmed); (iii) A196 is fixed (summation sigmas honest at high P/B); (iv) A197 is an information limit, shown under
the model law and bounded by lam_min5, not removable by any estimator tested; bin 4 or ~3-4x the bin-8 dose at
d 1-1.25 A meets the task's target; (v) the penumbra columns model a real shadow (40-50 % at 1 px from the arm) but are not
what fixes D0. No parameter was changed after a test run.

## 8. Limits and next steps

- CN and classic D0: zones where neither G1 nor GL is good (A195 for classic; CN D0a 1.15 vs FLAT 0.63). A per-spot
  (not per-zone) MSE comparison, or a continuous weight w* = E[La Lb] / E[La^2] (derivable from the same probes), is the
  next candidate; pre-register before testing.
- H-ML pulls at bin 8 low P/B: any sigma evaluated at the estimate has E[pull] = -d sigma / d lambda; an honest pull mean
  needs a sigma from data independent of the spot (A234).
- Information limit: below lam_min5 report pooled (`pooled_ml`) intensities or acquire at bin 4.
- R2 outer band at bin 8: +0.2-0.5 % at b 0.006-0.06 counts in the frame corners (A233), likely the harmonic ridge in
  partly covered rings; small but systematic.
- The near-zone harmonics are an option; the dev data did not ask for them.

## 9. Anomalies

- A233: G1 on spot-free bin-8 frames reads +0.15 to +1.7 % high in the outermost band (450-724 px, frame corners,
  b 0.0008-0.06 counts) at every dose (2.7 SE at f 10; R2 fails at f 3, +0.49 +- 0.24 %); not at bin 4.
- A234: H-ML (and G-ML) pull mean -0.16 at bin 8, P/B <= 1, on spots above the information limit while the mean ratio
  is 0.98: the sigma-at-the-estimate correlation (Sec. 1.3 iv); bin 4 is clean above the limit (-0.05).
- A235: the penumbra block does not change the real test D0 error (F 0.87 vs v0 0.88, CN 1.17 vs 1.18): the test-set D0
  failure (carried by two sessions, Sec. 1.1 a) is not the stop shadow; the expected-error switch removes it in F.
- A236: classic D0 (A195): every method 6-10x the box noise; H 6.39 vs FLAT 5.96; H's beta is within 2 SE (-0.77 +-
  0.53 %) where v0 G's was not.
- A237: Firth's Jeffreys-penalised ML is MORE biased than the ML at sparse bin-8 counts (0.61 vs 0.66 at mu0 0.15 e,
  P/B 1): the bias is not the O(1/I) term.
- A238: v1 H summation pulls on real BA spots at P/B >= 10 have mean -0.20 (rsd 1.01): the profile-model variance
  is slightly large there (bar-adjacent spots).
- A239: v0 G passes R1 on the v1 synthetic frames (0 / 60 at 3.2 SE, 3 at 2 SE): A193's "chance" reading is confirmed.
- A240: on the DEV sessions v0 G is better than FLAT at D0 (0.28 vs 0.39); the v0 D0 failure lives in two test sessions,
  so dev-only design could not see it; the hybrid was chosen because it is principled, not because dev demanded it.
