# bgsub v2: re-centring, a ring refinement, a continuous local weight and test-inversion intervals (2026-11-05)

Terms of reference: fix the failures of bgsub v1 (`2026-11-04_bgsub_algorithm_v1.md`, A233-A240) and make the algorithm
VALID by its own pre-registered rules; OWNER RE-SCOPE item 2 (priority 2). `version="v0"` and `version="v1"` stay
reproducible (regression tests against outputs frozen before any v2 change: v0 at 07a06b9, v1 at 092ece9). No other
packages; XDS only as a reference. The simulator is used read-only, pinned to `version="v1"` (the v1 design, the same
frames). Anonymous sids only; nothing runs on HPC.

Code: `src/podbg/bgsub/` (new `refine.py`; v2 paths in `core.py`, `integrate.py`, `_kernels.py`), tests
`tests/test_bgsub_v2.py`, scripts `scripts/bgsub_v2_*.py`. Outputs with provenance sidecars:
`~/pod_data/background/bgsub_v2/`. Figures: `notes/figures/bgsub_v2/`. Anomalies are numbered from A263.

## 1. Stage 1: what was looked at before the pre-registration

Allowed inputs: the v1 report and its published tables (TEST results at the level reported there, re-tabulated where
said), the v0/v1 DEV sessions (3 fly-scan, 4 classic) and dev synthetic seeds, Monte Carlo under the model law, and
-- because the terms of reference ask first to find out WHAT the classic D0 excess is -- the classic TEST host frames
and one fly-scan CN test session, inspected for diagnosis (Sec. 1.2-1.3). Classic D0 and CN D0 are therefore no longer
blind tests in v2; a classic CONFIRMATION set that no earlier run used (the 9 cached sessions with 4-7 raw frames) is
added and reported (Sec. 3.2).

### 1.1 A233 (R2 corner band at bin 8): not an estimator bias

Decomposition of the published v1 spot-free twins (`synth_test_bands.csv`): the R2 statistic bg - E[K]law splits into
the estimator-minus-frame part bg - K and the frame's own deviation K - E[K]law. In the failing cell (bin 8, f 3,
450-724 px) bg - K = +0.11 +- 0.065 % and K - E = +0.38 +- 0.22 %; at f 10, +0.004 % and +0.235 % (2.7 SE). The
estimator tracks the frame; the "bias" is in the raw counts. Checks:
- The law table is exact: `CountLaw.E` vs the exact recorded PMF mean differs by <= 1.3e-4 (relative) at mu 0.3-2 e;
  a 2e7-draw MC of `sim.detectors.quantise_fast` gives -0.18 / -0.38 / -0.04 % (|z| <= 1.9) at mu 0.15 / 0.25 / 0.5 e.
- 160 FRESH bin-8 f-3 frames (seeds 90000+): corner K - E = -0.13 +- 0.16 % (z -0.9).
- 80 fresh frames per dose, estimator minus frame (bg - K) in the corner band: f 1 +0.07 +- 1.0 %, f 3 -0.14 +- 0.08 %,
  f 10 -0.006 +- 0.007 %; the innermost band (55-110 px) carries a tiny significant +0.02 to +0.05 % (A263).
So A233 is the frames' sampling deviation flagged by a 2-SE rule over 24 cells (expected false fails 1.1) whose SE is
dominated by the frame's own count noise. Neither the spline boundary, the sparse-count weights nor the corner masking
produces a measurable bias (<= 0.14 % at every dose). The fix is to the RULE's statistic (Sec. 2.4, R2 v2): the paired
difference bg - K has expectation exactly E[bg] - E[K] (the bias against the true expected stored value, with no law
table), and its SE is not dominated by the frame noise; multiplicity as R1 (Bonferroni over the 24 cells).

### 1.2 Classic D0 (A195 / A236): three different things, one irreducible

`scripts/bgsub_v2_diag_classic.py` (`diag/classic_d0_v2.csv`) and `bgsub_v2_diag_classic_fig.py`
(`notes/figures/bgsub_v2/classic_d0_residuals.png`): the v1 G residual in D0 of every classic session (the harness
hosts). The v1 TEST D0 failure (every method 6-10x the box noise, pooled) is carried by THREE of 20 sessions (published
v1 rows re-tabulated by session: sb7857cd8 FLAT 14.1 / H 14.8 / G1 17.6; s8975a6a4 4.4 / 8.7 / 8.7; s768d180b 0.9 /
3.9 / 4.0; the other 17 sessions 0.0-2.0, H <= 1.4):
- sb7857cd8 (5 of 6 hosts): a SHARP RING at d 4.60 A (s 0.218; FWHM ~2-3 bin-4 px), with a weaker one inside, present in
  every host (map correlation 0.20 between hosts), excess pixel variance up to 40x Poisson at the ring: the ordered
  4.6-A hydrocarbon component of the film (atlas v1c's 8-px sharp component, here at bin 4), narrower than the fine
  radial spline (step 8 px at N = 2048). Background, reducible: the radial model lacks resolution.
- s8975a6a4 (all hosts): compact BRAGG SPOTS of the sample (d 4.7 A and others; 6-45 clusters per host, frame-specific,
  correlation -0.01): sample diffraction in REyes "No" frames (A43 / A139). Not background: a predicted reflection
  overlapping an unindexed sample reflection is an overlap. IRREDUCIBLE for any background estimator.
- s768d180b host 0 (1 of 6 hosts): a one-sided crescent and ring next to the stop: the beam centre of that frame is ~20-30
  px off the session centre the harness supplies (the derivative columns imply 17 px, still 15 px after one linear step).
  Reducible by re-centring.
The other 17 test and 4 dev sessions: pixel dispersion 1.00-1.02, box-scale excess 0.20-0.47 box noise, no clusters.

### 1.3 CN D0 (v1 H 0.76 vs FLAT 0.59)

Per session (published v1 rows): one session (s4417bac3) carries it (G1 4.4, H 2.6, FLAT 0.0; G1 dispersion 1.73); the
other 16 have H 0.0-0.74 <= FLAT. Its CN hosts (5 of 12) show a hard-edged bright polygon next to the stop arm, 100 px
wide, frame-specific (`/tmp` look, not kept): a stop-geometry / shadow change against the session mask, smooth at the
annulus scale, outside the global model's span. A zone-level switch (v1 H) uses GL on whole radial zones; a local
decision is needed.

### 1.4 A234 / A238 (pulls) and the local weight (dev and Monte Carlo)

- A238 re-tabulated (published v1 rows, P/B >= 10): the BA mean pull -0.28 is a TAIL statistic (median -0.03); it comes
  from D1 / D2 spots of one gain-off session (s3912de77: mean -3.7, median -0.05) whose box background is 0.01 counts: the
  summation sigma, evaluated at the spot's own ML estimate, collapses to the background-only value when the ML returns
  ~0 for a spot that left almost no stored counts. The ratio of sums is 0.9998: no intensity bias. Same mechanism as A234.
- MC under the model law (bin 8, 29-px box, 3,000 boxes per cell; `/tmp/mc_lr.py`, numbers here): pulls of the ML with
  sigma at the ESTIMATE have mean -0.24 / -0.14 / -0.09 / -0.08 at (mu0 0.3, P/B 1) / (0.6, 1) / (0.3, 3) / (1.0, 1); with
  sigma at the TRUE lambda (the statistic whose |.| <= 1 is the event "the test-inversion interval covers the truth")
  -0.14 / -0.03 / -0.00 / -0.01, rsd 0.94-1.08; the signed-root likelihood ratio -0.12 / -0.06 / -0.03 / -0.04. The
  remaining -0.14 at (0.3, 1) is the information-limit bias (ratio 0.91; lambda 8.7 below lam_min5 13.6).
- Re-centring threshold (`bgsub_v2_diag_centre.py`, dev hosts only): implied shifts (bin-8 px) F / BA / CN median
  0.42-0.48, max 1.27-1.49; classic median 1.04, max 7.8. Threshold 2 bin-8 px (2 x N / 1024).
- Local-weight neighbourhood (`bgsub_v2_dev_weight.py`, `diag/dev_weight_v2.csv`; dev real hosts + 48 dev synthetic
  frames): 'disc' and radius-matched 'ring' neighbourhoods at R_nb 3 / 6 / 12 r_out differ little; ring-12 is best or
  near-best on the structured synthetic D4 cells (0.34-0.42 vs G 0.71-0.93) and the dev F D0 (0.52 vs G 0.58), and
  costs 0.08-0.15 box noise on plain frames (G 0.06-0.15). Adopted: ring, R_nb = 12 r_out, |dr| <= 2 r_out.

## 2. Pre-registration (written and committed BEFORE any v2 test run)

### 2.1 Fix 1a, 1b: re-centring and the ring refinement (method G2 = the v2 global model)

G2 = G1 (the v1 fit, unchanged) followed by `refine.py`:
- RE-CENTRING: the v1 centre-error columns R'(r) cos / sin phi imply a centre error (dy, dx) = -(their coefficients). If
  |(dy, dx)| > 2 x N / 1024 px, Gauss-Newton on the plan centre (re-plan + refit; <= 5 rounds; a step that under-shoots
  is accelerated by 1 / (1 - q), q = the ratio of successive implied errors, capped at 4x; a round is kept only if the
  Pearson dispersion does not rise; stop below 0.5 x N / 1024). Below the threshold nothing is re-planned (the
  first-order columns absorb the error in G).
- RING REFINEMENT about c' = centre + the remaining implied shift: per 1-px ring of r' = |p - c'| and checkerboard half,
  rho_A, rho_B = mean of K - G over the USED pixels; rhobar = (rho_A + rho_B) / 2. Since the halves have the same
  expectation delta and independent noises, E[rho_A rho_B] = delta^2 and E[rhobar^2] = delta^2 + Var(rhobar) EXACTLY
  (no variance model), so the MSE-optimal shrinkage w = delta^2 / (delta^2 + v) is estimated by
  w^_i = clip(sum_W rho_A rho_B / sum_W rhobar^2, 0, 1) over 5 rings (W = 2), and c_i = w^_i rhobar_i. Only rings with
  >= 50 expected stored counts and >= 20 used pixels per half (Gaussian ring means; at sparse counts the skewed noise
  would make the positive-part estimator non-mean-zero). Var(c_i) = w_i^2 vbar_i (window mean of (rho_A - rho_B)^2 / 4)
  enters the box variances. If |c_i| exceeds 1 pixel SD in some ring the count-law flags are recomputed against G + c
  (the ring's own pixels were flagged against G) and the refinement repeated (<= 2x).
- Bias and variance: c is a linear function of the residuals times a weight from the same rings; with Gaussian ring
  means the joint law of (rho_A, rho_B) is symmetric under sign change when delta = 0, so E[c] = 0 there; for |delta|
  >> sqrt(v), w -> 1 and c -> rhobar (unbiased). Its cost on a frame without sub-knot structure is w^2 v per ring, i.e.
  ~ 0.008 E[w^2] of the box variance at bin 8 (ring means are over 600-6,000 pixels). In between, the bias is
  -(1 - w) delta with w ~ delta^2 / (delta^2 + v): the James-Stein trade.

### 2.2 Fix 1c: the continuous local weight (method H2 = AUTO v2)

For spot k: B_k = G2_box + w_k nb La_k, La_k = mean of K - G2 over the spot's unflagged annulus, and
w_k = clip(sum_{j in N_k} La_j Lb_j / sum_{j in N_k} La_j^2, 0, 1) over the spot-free probes j (box r_box and annulus
r_in-r_out fully valid and unflagged, centres more than r_excl + r_out from every spot, stride r_out / 2) in the
radius-matched neighbourhood N_k = {|r_j - r_k| <= 2 r_out, arc distance <= 12 r_out}, widened x2, x4, x8 until >= 20
probes (else the 20 nearest).
- Derivation: with delta = b - G2 (misfit + estimation error of G2) and independent count noises n_a, n_b (disjoint
  pixels), the box-mean error of B_k is w(delta_a + n_a) - delta_b, so MSE(w) = E delta_b^2 - 2 w E[delta_a delta_b] +
  w^2 (E delta_a^2 + s_a^2), minimised by w* = E[delta_a delta_b] / (E delta_a^2 + s_a^2) = E[La Lb] / E[La^2] (the
  probe noises cancel in the numerator and appear correctly in the denominator; no variance model). MSE(w*) <=
  min(MSE(0), MSE(1)): G2 and GL2 are special cases.
- Bias: the probes never overlap spot k's box or annulus, so w^_k is independent of the spot's own counts given the
  background field; E[B_k - b_k | delta, w] = w delta_a - delta_b, zero when the field is locally stationary with zero
  mean (as G2's span holds the truth: R1). Variance: estimation noise of w^ around 0 on clean frames, SD ~
  sqrt(na / (nb n_eff)); the clipped E[w^2] costs E[w^2] nb / na of the box variance (dev: 0.05-0.10 mean w on plain
  frames). The reported box variance is (1 - w)^2 Var(G2 box) + w^2 Var(GL annulus term).
H2-ML is the ML on H2's background.

### 2.3 Fixes 3 and 4: intervals by test inversion, sigma at the hypothesised lambda

For every summation method (with spot sigmas) and every -ML, v2 reports `ci_lo`, `ci_hi`: the 68.3 % interval
{theta : |estimate - E_theta| <= sigma(theta)} with sigma evaluated at the HYPOTHESISED intensity from the declared law,
the background and the profile (summation: E_theta = law-expected stored excess L(lambda) (electrons), sigma^2 = epc^2
(phi sum_box g(E_p(lambda)) + Var B); ML: E = lambda, sigma^2 = 1 / I_exp(lambda) + sigma_B^2). No data of the spot enters
sigma. The pull whose |.| <= 1 is exactly coverage is z = (estimate - truth) / sigma(truth); the evaluation computes it
through `integrate_spots(..., lam_eval=lambda_box)` -> `sigma_at`. `sigma` (at the estimate) is kept for compatibility.
No other change to the ML or the summation estimates.

### 2.4 Rules (v2; pre-registered)

- R1 SYNTHETIC UNBIASEDNESS (as v1): every lacey cell (60), |beta| <= max(0.005, 3.2 SE) for G2 AND H2.
- R2 v2 COUNT STATISTICS / TRUNCATION (fix 2): spot-free twins, per binning x f x band (24 cells, plain + structured
  pooled, 48 frames): the PAIRED bias Delta = mean over pixels of (bg - K) / mean b_true, |Delta| <= max(0.3 %, 3.1 SE)
  (frame SE; 3.1 = two-sided Bonferroni over 24 cells at family 0.05) AND |masking on - off| <= 0.1 %. The v1 form
  (bg - E[K]law, 2 SE) and K - E are reported.
- R3 v2 SIGMA HONESTY (fixes 3, 4): synthetic lacey, per binning (4, 8) x {P/B <= 1, P/B >= 10}, spots with
  lambda_box >= lam_min5: z = (I - truth) / sigma_at(truth) has robust SD in [0.9, 1.1] and |mean| <= 0.1, for H2
  summation (truth lambda_eff) and H2-ML (truth lambda_box). Reported: the v1 (Wald) form for v1 and v2, the coverage of
  [ci_lo, ci_hi], all spots, and the real frames (bounds [0.85, 1.15], 0.25), including BA P/B >= 10 (A238).
- R4 v2 REAL, BETTER THAN FLAT (v1 clauses) for H2, per regime x d bin: |beta| <= max(0.01, 2 SE) in >= 90 % of cells,
  rmse_bg(H2) < rmse_bg(FLAT) in every cell, |beta| <= |beta_FLAT| + 2 SE in every cell, on spots WITHOUT a host overlap.
  HOST OVERLAP (method-independent, from the spot-free host alone; `bgsub_v2_real.overlap_flags`): z = (K0 - shell mean)
  / sqrt(g(shell mean)); 8-connected clusters of z > 3 with >= 4 px (bin 8) / >= 8 px (bin 4) and peak z >= 6 are sample
  diffraction; a spot overlaps if such a cluster lies within r_box + 2 (x 8 / binning) px of its centre. All-spot R4 and
  the overlap fractions are reported. FLAT is computed on the same spots (the v1 run's FLAT on the same frames).
- R5 SPEED: median G2 fit + map (re-centring not triggered, ring refinement on) on a 1024^2 frame (numba, plan cached,
  >= 10 frames) <= 50 ms on this Mac.
- R6 (reported, as v1): (a) |G2-ML ratio - 1| <= 5 % at P/B 1, D2, bin 8; (b) above lam_min5.
- VERDICT: bgsub v2 is VALID iff R1, R2, R3 (synthetic) and R5 pass and R4 passes in F and classic (BA and CN
  reported; the classic confirmation set reported). No parameter changes after a test run; failures are reported with
  their cells.

### 2.5 Evaluation design

- SYNTHETIC: the v1 design exactly (`bgsub_v1_synth.cells` / `make_frame`, seeds 30000+, simulator `version="v1"`):
  lacey bins 4 / 8 x f {1, 3, 10} x {plain, structured} x 24 frames, robustness quantifoil / formvar bin 8 x f {1, 3} x
  8 frames (UNTESTED AGAINST DATA); 35 spots per frame; v1 (G, GL, AUTO, FLAT) and v2 (G, GL, AUTO) on identical frames.
  A v2 robustness row against simulator v2/v3 is NOT run (time).
- REAL: the v0/v1 harness, same sessions, hosts, replicates and seeds (17 fly-scan, 20 classic), v1 and v2 on each
  injected frame; the classic confirmation set (9 sessions with 4-7 cached raw frames, never used before) reported.
- Workers: multiprocessing (spawn), completeness checked against the planned units before any rule is read; the v1
  outputs of this run are compared with the published v1 test rows (same frames => identical v1 numbers).

(Pre-registration ends here; everything below was produced afterwards.)

Commit of record: the pre-registration, the v2 code, scripts and tests were committed at 01:32 on 2026-11-05 BEFORE any
v2 test run, but a concurrent agent's `git commit` swept the files staged here into its own commit 5609c14 ("Operator
sheet: optional C2 aperture-ratio block ..."); the content is exactly this section and the code as run (A264). No test
checkpoint existed at that time (only `checkpoints/real_dev`, `checkpoints/synth_dev`).

## 3. Completeness of the test runs (checked before any rule was read)

All runs used `multiprocessing` (spawn, 6 workers); every planned unit was checked against its checkpoint and the
counts written to the log and the sidecars:
- SYNTHETIC (seeds 30000+, run once): 352 / 352 planned frames (lacey 12 cells x 24, robustness 8 cells x 8), 35 spots
  each. The v1 rows of this run equal the published v1 test rows bit for bit (12,320 spots per method; max |dB| = max
  |dI| = 0 for G, AUTO, GL, G-ML, AUTO-ML, FLAT), so v1 / v2 are paired on identical frames.
- REAL: 37 / 37 test sessions (17 fly-scan, 20 classic), 48,930 spots per method (F 13,650, BA 14,280, CN 12,600, classic
  8,400: v0's and v1's counts); the v1 G, AUTO, AUTO-ML, FLAT and oracle values equal the published v1 rows (max |diff| 0).
  CONFIRMATION: 9 / 9 classic sessions, 3,430 spots per method. No worker failed (`FAILED` count 0 in the logs).

## 4. Synthetic results (simulator v1; `synth_test_{cells,trunc,pulls,ml}.csv`, `synth_test_rules.json`)

### 4.1 Background, lacey (beta %, error / box noise; v1 H = AUTO v1, v2 H2 = AUTO v2)

| bin | struct | d bin | v1 G1 | v1 H | v2 G2 | v2 H2 | FLAT |
|---|---|---|---|---|---|---|---|
| 4 | plain | D0-D4 error | 0.07-0.15 | 0.07-0.19 | 0.07-0.16 | 0.08-0.16 | 0.41-0.58 |
| 4 | structured | D0 / D4 error | 0.50 / 0.84 | 0.50 / 0.54 | 0.50 / 0.84 | 0.43 / 0.49 | 0.61 / 0.56 |
| 8 | plain | D0-D4 error | 0.07-0.12 | 0.07-0.13 | 0.07-0.12 | 0.08-0.13 | 0.38-0.62 |
| 8 | structured | D0 / D3 / D4 error | 0.34 / 0.13 / 0.97 | 0.34 / 0.19 / 0.51 | 0.35 / 0.13 / 0.97 | 0.31 / 0.14 / 0.45 | 0.59 / 0.47 / 0.54 |
| 8 | plain | D1 beta | +1.13 | +1.13 | +0.42 | +0.49 | -5.57 |
| 8 | structured | D3 beta | -0.30 | -1.55 | -0.31 | -0.13 | -0.21 |

H2 lowers the structured errors further than v1 H (D4 0.45-0.49 vs 0.51-0.54; the bar-pedestal kinks) and costs
0.01-0.03 box noise on plain frames (mean w 0.06-0.07; v1 H switched on 0.7 % of plain spots). Robustness presets
(untested against data): quantifoil / formvar plain errors G2 0.07-0.15, H2 0.08-0.16; structured D4 H2 0.34-0.36 (v1 H
0.39-0.44). Re-centring triggered on 1 of 352 frames (0.3 %).

### 4.2 Rules (pre-registered, Sec. 2.4)

| rule | v1 on these frames | v2 |
|---|---|---|
| R1 unbiasedness, 60 cells, max(0.5 %, 3.2 SE) | G1 PASS 0 / 60, H PASS 0 / 60 | G2 PASS 0 / 60 (3 at 2 SE); **H2 FAIL 2 / 60** (A265) |
| R2 v2 paired bias, 24 cells, max(0.3 %, 3.1 SE) | PASS 0 / 24 (v1 form 1 / 24) | PASS 0 / 24 (max 0.45 % at b 0.0008, SE 1.2 %; v1 form 1 / 24) |
| R3 v2 sigma at truth, H2 and H2-ML, informative | (v1 Wald form: H PASS, H-ML FAIL bin 8 low -0.155) | PASS: means -0.08..+0.04, rsd 0.96-1.02 |
| R6a ML target P/B 1, D2, bin 8 | FAIL | FAIL 0.88 / 0.96 / 0.99 at f 1 / 3 / 10 (unchanged, information limit) |
| R6b ML above lam_min5 | FAIL 8 / 88 | FAIL 8 / 88 (unchanged) |

R1 fails for H2 (A265): bin 4, f 10, structured, D0: +0.70 +- 0.18 % (G2 +0.09 %, GL +0.66 %, v1 H +0.49 +- 0.24 %): the
weight (mean 0.66) carries GL's bias where box and annulus straddle the arm-shadow non-linearity; bin 8, f 3, plain,
D1 (b 0.01 counts per pixel): -3.2 +- 0.8 % (G2 -2.0 +- 0.65 %, inside 3.2 SE): at sparse counts the probe noise grows
with the local level, so w^ is larger where the local residual is negative and E[w La] < 0 (-1.2 % of a 0.3-count box).
w* minimises the MSE, not the bias; the rule asks for both.

R2 in detail: the paired statistic bg - K is 0.00-0.02 % at bin 4 and <= 0.03 % at bin 8 (bands below 450 px), 0.11 +-
0.07 % (f 3) and 0.45 +- 1.2 % (f 1, b 0.0008) in the corner band; the v1 form (bg - E[K]law) still fails one cell
(bin 8, f 3, 450-724: +0.49 +- 0.24 %), where the raw frames themselves sit at +0.38 +- 0.22 % (A233 resolved as a
sampling deviation, Sec. 1.1).

R3 in detail (informative spots, lambda_box >= lam_min5):

| method | bin | P/B | v1-form Wald mean (v1 H / v2 H2) | v2 sigma-at-truth mean, rsd | coverage of [ci_lo, ci_hi] |
|---|---|---|---|---|---|
| H summation | 4 | <= 1 | -0.003 / -0.003 | +0.030, 0.96 | 0.68 |
| H summation | 4 | >= 10 | -0.028 / -0.028 | -0.001, 0.97 | 0.59 (A268) |
| H summation | 8 | <= 1 | -0.050 / -0.050 | -0.024, 1.01 | 0.68 |
| H summation | 8 | >= 10 | -0.016 / -0.016 | +0.011, 1.02 | 0.57 (A268) |
| H-ML | 4 | <= 1 | -0.053 / -0.054 | +0.036, 0.98 | 0.68 |
| H-ML | 4 | >= 10 | -0.036 / -0.036 | +0.019, 0.97 | 0.69 |
| H-ML | 8 | <= 1 | **-0.155** / -0.156 | -0.076, 1.01 | 0.66 |
| H-ML | 8 | >= 10 | -0.036 / -0.037 | +0.020, 1.01 | 0.68 |

A234 is resolved as a property of the pull definition: the same estimates, judged by the interval the user receives
(sigma at the hypothesised lambda), are honest; the remaining -0.08 at bin 8, P/B <= 1 is the information-limit bias
(ratio 0.96-0.98 just above lam_min5). On ALL spots (incl. below lam_min5) H2-ML bin 8 low: -0.08, rsd 0.93, coverage
0.70 (v1 Wald -0.165).

## 5. Real-frame results (the v0/v1 test frames; `real_test_{cells,d0,d0_by_session,pulls,cells_pb}.csv`, `real_test_rules.json`)

### 5.1 Background error / box noise, D0 (d 4-8 A), v1 vs v2 paired

| regime | spots | v1 G1 | v1 H | v2 G2 | v2 H2 | FLAT |
|---|---|---|---|---|---|---|
| F | all | 0.88 | 0.29 | 0.87 | **0.08** | 0.40 |
| F | no host overlap | 0.32 | 0.16 | 0.31 | **0.03** | 0.40 |
| BA | all / no overlap | 0.51 / 0.50 | 0.43 / 0.42 | 0.48 / 0.46 | **0.35 / 0.33** | 0.52 / 0.50 |
| CN | all | 1.17 | 0.77 | 1.17 | **0.46** | 0.60 |
| CN | no overlap | 0.58 | 0.43 | 0.57 | **0.34** | 0.59 |
| classic | all | 7.49 | 6.39 | 7.21 | 6.30 | 5.96 |
| classic | no overlap | 1.94 | 2.09 | **1.57** | 1.76 | 2.97 |
| classic CONFIRM | all = no overlap | 0.66 | 0.57 | **0.50** | **0.50** | 1.30 |

Outside D0 H2 equals H within 0.01-0.04 everywhere (F 0.00-0.24, BA 0.14-0.37, CN 0.00-0.31, classic 0.27-0.34 without
overlap; FLAT 0.32-0.67). Beta: every H2 cell within max(1 %, 2 SE) (D0: -0.3 / +0.1 / -0.0 / +0.0 % for BA / CN / F /
classic). Figure: `notes/figures/bgsub_v2/real_d0_v1_v2.png`.

By session (`real_test_d0_by_session.csv`): CN s4417bac3 (the stop polygon) H 2.64 -> H2 1.07 (all spots), 0.84 -> 0.00
without overlap; F s4417bac3 0.75 -> 0.00. Classic: s768d180b (centre error) H 3.85 -> H2 0.49, G 4.03 -> G2 0.86
(re-centred); se94c6af9 0.70 -> 0.00, sc4a8268c 0.84 -> 0.29; sb7857cd8 (sharp ring) H 14.8 -> 14.95 all spots, 4.84
-> 4.29 without overlap, G 4.45 -> 3.79 (A267); s8975a6a4 (sample Bragg spots) 8.65 -> 6.65 all spots, 1.07 -> 1.02
without overlap: the classic all-spot D0 failure is these two sessions.

Host overlap (pre-registered flag): 0.08 % (F, BA), 0.16 % (CN), 0.29 % (classic) of spots, but they carry most of the
pooled D0 error: removing them also makes v1 pass R4 in every regime (A266). Part of the flagged spots are not sample
Bragg spots but the s4417bac3 stop polygon.

### 5.2 R4 and intensities

| R4 (no overlap; the verdict's form) | v1 G1 | v1 H | v2 G2 | v2 H2 |
|---|---|---|---|---|
| F | PASS | PASS | PASS | PASS |
| BA | PASS | PASS | PASS | PASS |
| CN | PASS | PASS | PASS | PASS |
| classic | PASS | PASS | PASS | PASS |
| classic CONFIRM | PASS | PASS | PASS | PASS |
| (all spots, reported) F / BA / CN / classic | F / P / F / F | P / P / F / F | F / P / F / F | **P / P / P / F** |

Intensities are unchanged by v2 (H2-ML at d 1.0-1.25 A, F: 0.83 / 0.97 / 1.01 at P/B 1 / 3 / 10, v1 identical to 0.001;
summation about lambda_eff 0.99-1.05). Real pulls (informative spots, sigma at truth; `real_pulls_v1_v2.png`): H2
summation means -0.02..+0.06, rsd 0.95-1.04, coverage 0.64-0.69; H2-ML -0.02..+0.06, rsd 0.97-1.07, coverage 0.65-0.69, in
every regime and P/B class. A238 FIXED: BA P/B >= 10, v1 H Wald mean -0.20 (all spots -0.28; median -0.03) -> v2 -0.02
(rsd 1.00); on all spots including the degenerate gain-off D1 / D2 boxes, -0.02. Below lam_min5 the ML pulls stay
narrow (rsd 0.62-0.80, the information limit; means -0.12..-0.15).

## 6. Speed (R5; `speed_v2.json`; load from parallel agents not controlled, v1 measured in the same run)

| frame | v1 numba | v2 numba | v2 numpy | plan |
|---|---|---|---|---|
| 512^2 (bin 16) | 18.8 ms | 20.7 ms (re-centring triggered on these frames, A271) | 36.5 ms | 51 ms |
| 1024^2 (bin 8) | 45.0 ms | **46.8 ms** | 112 ms | 204 ms |
| 2048^2 (bin 4) | 119 ms | 135 ms | 448 ms | 1.0 s |

R5 PASS (46.8 ms <= 50). The ring refinement costs ~2 ms at 1k (two numba passes). Integration of 35 spots at bin 8
(G, GL, H, -ML, intervals): v1 231 ms, v2 441 ms; the interval inversion doubles it (A270; not in R5).

## 7. Verdict (pre-registered rules)

| rule | v1 (own rules) | v2 |
|---|---|---|
| R1 | PASS | **FAIL**: H2 2 / 60 (G2 PASS 0 / 60) |
| R2 | FAIL 1 / 24 (v1 form) | PASS 0 / 24 (v2 paired form) |
| R3 | FAIL (H-ML Wald) | PASS (sigma at truth) |
| R4 F / classic | PASS / FAIL | PASS / PASS (no overlap); all spots PASS / FAIL |
| R4 BA / CN (reported) | PASS / FAIL | PASS / PASS (all spots too) |
| R5 | PASS 42.0 ms | PASS 46.8 ms |
| R6a / R6b (reported) | FAIL / FAIL | FAIL / FAIL (unchanged) |

**bgsub v2 is NOT VALID by its pre-registered verdict**: R1 fails for H2 in 2 of 60 cells. Everything else in the
verdict passes. What the evidence supports:
(i) the continuous local weight is the best background next to the beam stop on real frames (D0 F 0.03-0.08, BA 0.33,
CN 0.34-0.46 box noise vs FLAT 0.40-0.60; the CN failure of v1 is gone even with all spots), but as an MSE-optimal
estimator it is not unbiased in every synthetic cell (bias up to -3 % at 0.01 counts per pixel, +0.7 % across the arm
shadow);
(ii) G2 passes R1, R2 and R4 (no overlap) in every regime; re-centring removes the gross per-frame centre errors (classic
s768d180b 4.0 -> 0.9) and fires on 0.2-2.4 % of fly-scan frames and 14 % of classic frames (session centres);
(iii) A233 was the frames' own sampling deviation, not an estimator bias (corner band <= 0.14 % on 80 fresh frames per
dose); A234 and A238 were the pull definition: with sigma at the hypothesised intensity (the interval users receive)
the pulls are honest on synthetic and real frames;
(iv) the classic D0 excess is three things: a sharp film ring narrower than the radial knots (partly fixed, A267), a
per-frame centre error (fixed), and sample Bragg spots in "No" frames (irreducible; floor below);
(v) R4's pass rests partly on the pre-registered overlap exclusion, under which v1 passes too (A266).
No parameter was changed after a test run.

The irreducible floor (classic D0, sample diffraction): s8975a6a4 every method 4.4-8.7 box noise with all spots
(1.0-1.4 without its 7 flagged overlaps), i.e. one of 20 classic sessions (5 %) whose "No" frames carry Bragg spots; in
that session a predicted reflection at d 4-8 A has a background error of order its own box noise times 5-9 unless the
overlap is detected and flagged. The detector used here (z > 3 clusters, peak >= 6) flags 7 of its 84 D0 spots.

## 8. Limits and next steps

- H2 bias: shrink w toward 0 by its own estimation error (w = max(0, (S_ab - k SE) / S_aa)) and compute the probe
  products in a variance-stabilised scale at sparse counts; or keep G2 as the default background and use H2 only where
  the local MSE gain is significant. Pre-register.
- Sharp rings (A267): the ring refinement needs the ellipticity of the sharp component (take e, phi0 from the d2
  columns) and an un-flagged ring estimate (robust ring means instead of re-flagging); or a finer fine-spline step where
  a ring-sharpness test fires.
- The overlap flag mixes sample Bragg spots with other compact strong structure (A266); a cleaner sample-diffraction
  detector (shape, Friedel pairs) would make the R4 exclusion purely physical.
- Intervals: summation coverage at P/B >= 10 0.57-0.59 on synthetic frames (A268); integration cost 2x (A270).
- R6 (information limit) unchanged: bin 4 or pooled ML below lam_min5.

## 9. Anomalies

- A263: the estimator-minus-frame bias in the innermost synthetic band (55-110 px, next to the stop) is +0.01 to +0.04 %
  at 3-7 SE for both v1 and v2 (paired SE 0.002-0.01 %): tiny, real, unexplained (penumbra columns or the arm shadow
  of the structured frames are candidates).
- A264: the pre-registration commit was swept into a concurrent agent's commit 5609c14 (shared index); content and time
  (01:32, before any test checkpoint) verified.
- A265: H2 fails R1 in 2 / 60 synthetic cells (bin 4 f 10 structured D0 +0.70 +- 0.18 %; bin 8 f 3 plain D1 -3.2 +- 0.8 %):
  MSE-optimal weighting carries GL's bias across the arm-shadow non-linearity, and at sparse counts w^ correlates with
  the local level through the mean-variance relation.
- A266: the pre-registered host-overlap flag (0.08-0.29 % of spots) removes most of the pooled D0 error of EVERY method;
  under it v1 also passes R4 everywhere. It flags the s4417bac3 stop-polygon spots as well as sample Bragg spots.
- A267: the sharp 4.6-A ring of sb7857cd8 is only half removed (ring-zone excess variance 6.8 -> 3.0 x Poisson; mean
  residual still +22 counts at the ring peak with c = 12.5): ring pixels remain flagged after two re-flag rounds and the
  ring carries an m = 2 pattern (sector means -2 to +43 counts; the 0.4 % Arctica ellipticity on a 2-px-wide ring), which
  a radial correction cannot hold.
- A268: summation interval coverage at P/B >= 10 is 0.57-0.59 on synthetic frames although the sigma-at-truth pulls have
  rsd 0.97-1.02: heavy tails (structured-frame background errors; dead-time branch, where L(lambda) is not monotone and
  the interval is truncated). ML coverage 0.66-0.69.
- A269: re-centring fires on 13.8 % of classic test frames and 32 % of confirmation frames (session-level centres), 2.4 %
  of BA frames (an m = 1 bar pedestal can mimic a centre error), 0.2-0.5 % of F / CN frames.
- A270: integration with intervals costs 441 ms per 35-spot bin-8 frame (v1 231 ms).
- A271: re-centring triggers on the bin-16 simulator speed frames (threshold 1 px at N 512, where the sim centre jitter
  and the coarse spline give implied shifts above it).
