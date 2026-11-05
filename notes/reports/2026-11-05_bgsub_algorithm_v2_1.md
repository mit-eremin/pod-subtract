# bgsub v2.1: a gated, spill-corrected local weight, elliptical ring term and exact interval images (2026-11-05)

Terms of reference: make bgsub VALID by its own pre-registered rules after v2 (`2026-11-05_bgsub_algorithm_v2.md`,
A263-A271), where only R1 failed for H2 (2 of 60 cells). Fixes: (1) a local/global choice unbiased enough for R1 that
keeps v2's D0 gains; (2) A267, the sharp 4.6-A ring only half removed; (3) A268, summation interval coverage 0.57-0.59 at
P/B >= 10; (4) A270, integration cost; (5) keep R2-R5. `version="v0"`, `"v1"`, `"v2"` stay reproducible (v2: regression
test against outputs frozen at 5e4a90c, bit for bit; v0, v1 as before). No other packages, no HPC, anonymous sids. The
simulator is used read-only and pinned to `version="v1"` (the v1 design, the same frames).

Code: `src/podbg/bgsub/` (`version="v2.1"` in `make_plan`, `estimate_background`, `integrate_spots`), tests
`tests/test_bgsub_v2_1.py`, scripts `scripts/bgsub_v2_1_*.py`. Outputs with provenance sidecars:
`~/pod_data/background/bgsub_v2_1/`. Figures: `notes/figures/bgsub_v2_1/`. Anomalies are numbered from A273.

## 1. Stage 1: development (before the pre-registration)

Allowed inputs: the v2 report and its published tables; FRESH synthetic frames of the v1 design with seeds 70000+
(never used by any run: 288 frames, the 12 lacey cells x 16, plus 48 more in each of the two v2 failure cells;
`bgsub_v2_1_dev_weight.py`, `_eval.py`); the 7 real DEV sessions (3 fly-scan, 4 classic) of the v0-v2 harness
(`bgsub_v2_1_dev_weight_real.py`, `bgsub_v2_1_real.py --set dev`); the v2 dev synthetic seeds; toy Monte Carlo. No test
frame (synthetic seeds 30000+, the 37 real test sessions, the 9 confirmation sessions) was opened for v2.1 before
Sec. 2 was committed. Note: the confirmation set and the sharp-ring session sb7857cd8 (a TEST session) were already
inspected by v2's stage 1, so they are not blind for fix 2 (the m = 2 pattern of A267 came from there).

### 1.1 A265 revisited: what the two H2 failures were

On the 288 fresh frames H2 (v2's AUTO) passes R1 in all 60 cells (max |beta| / SE 3.05, 0 cells beyond 2 SE above the
0.5 % floor). In the two cells that failed on the test frames: bin 4, f 10, structured, D0 (the "arm cell") +0.26 +-
0.09 % (test +0.70 +- 0.18), and bin 8, f 3, plain, D1 (the "sparse cell") -0.44 +- 0.78 % (test -3.2 +- 0.8; G2 on the
test frames -2.0 +- 0.65, on the fresh frames -0.45 +- 0.69). So each failure is a small real bias plus a ~2.5-SE
realisation of the test frames, and in the sparse cell the realisation is shared with G2.

The arm-cell bias is NOT the arm shadow (v2's reading in A265 is wrong, A273): split by the distance of the spot's annulus
to the stop, 0.247 of the 0.265 % comes from spots more than 20 px from the stop, where the true misfit of G2 is zero in
both box and annulus (mean +-0.01 % of the box background). By P/B, 0.19 % comes from the P/B 10 and 30 spots: it is the
spot's OWN light in the annulus (pixel-integrated Gaussian tails beyond r_in = 5/3 r_box). At bin 4, f 10 the core of a
P/B-30 spot is past the dead-time maximum, so the stored box excess is small while the tails (not saturated) are not:
a tail estimate from the stored box excess predicts 0.001 % (wrong), one from the box intensity in electrons removes it.
With the electron-level spill subtracted (truth lambda), H2's arm-cell bias falls to +0.03 % (GL +0.19 -> -0.14 %).

### 1.2 Candidate local corrections (fresh synthetic frames; real dev D0)

Per spot k with the v2 probe neighbourhood N_k (n probes, box / annulus residual means b_j, a_j): S_ab = sum a_j b_j,
S_aa = sum a_j^2; SE of S_ab either naive (independent probes) or from BLOCK sums over 2 r_out x 2 r_out blocks (probes
at stride r_out / 2 overlap; the naive SE is ~4x too small). Background error in box-noise units, R1 bias (%):

| rule | R1 fails (3.2 SE / 2 SE) | arm cell | sparse cell | plain err | structured err (D0 / D4) | real D0 BA / CN / F / classic |
|---|---|---|---|---|---|---|
| G2 | 0 / 2 | -0.03 +- 0.13 | -0.45 +- 0.69 | 0.095 | 0.341 (0.430 / 0.992) | 0.447 / 0.477 / 0.564 / 0.332 |
| H2 (v2) | 0 / 0 | +0.26 +- 0.09 | -0.44 +- 0.78 | 0.104 | 0.213 (0.309 / 0.438) | 0.438 / 0.439 / 0.455 / 0.315 |
| JS, naive SE, k = 2 | 0 / 1 | +0.23 | -0.39 | 0.095 | 0.214 | - |
| JS, block SE, k = 1 / 2 | 0 / 1-2 | +0.17 / +0.09 | -0.44 / -0.38 | 0.096 / 0.095 | 0.216 / 0.246 | 0.436 / 0.451 / 0.486 / 0.322 (k 1, +spill) |
| gate, block SE, z = 2 / 3 | 0 / 0-1 | +0.23 / +0.21 | -0.47 / -0.18 | 0.098 / 0.096 | 0.212 / 0.226 | 0.435 / 0.459 / 0.469 / 0.336 (z 2) |
| local affine (intercept + slope) | 0 / 1 | +0.15 | -0.49 | 0.119 | 0.203 | 0.464 / 0.460 / 0.489 / 0.341 |
| local affine, slope JS block k = 2 | 0 / 0 | +0.05 | -0.45 | 0.112 | 0.209 | 0.467 / 0.482 / 0.563 / 0.361 |
| H2 + spill (electron-level, truth lambda; bin 4 arm cell) | - | +0.03 | - | - | - | - |
| **gate z = 2 + spill (ML lambda) = H2.1** | **0 / 2** | **+0.11 +- 0.08** | **-0.47 +- 0.72** | **0.097** | **0.204 (0.305 / 0.423)** | **0.438 / 0.367 / 0.460 / 0.335** (harness) |

(Real D0: the dev harness rows with the host as reference, no host overlap; the last row from the full dev harness,
where v2 H2 gives 0.442 / 0.346 / 0.448 / 0.314 and G2 0.450 / 0.391 / 0.501 / 0.329.)

Readings. (i) The local-affine forms (an intercept m_b = neighbourhood mean of b_j) remove most of the structured bias
but cost 0.02 box noise on every plain frame and lose the real D0 gains: the real local structure is not neighbourhood-
wide. (ii) Shrinking the slope by its block SE kills it on real frames (mean w 0.004-0.013 at k = 2). (iii) A
significance GATE at z = 2 block SE keeps G2 where nothing is seen (P(on) 2.4-2.6 % on plain frames, i.e. the nominal
one-sided 2.3 %) and the full v2 weight where structure is real (P(on) 26-29 % on structured frames); with the spill term
it has the lowest structured error of all candidates and keeps 80-90 % of v2's real F / CN D0 gain (classic D0: no gain
left over G2).

### 1.3 Fix 2 development: a robust free m = 2 ring term fails on real frames

First implementation: per 1-px ring a robust (Huber, 4 x a MAD scale floored at the count law) weighted LS fit of
c0 + c2c cos 2phi + c2s sin 2phi to 32 sector means over ALL unexcluded pixels (no count-law flags), each component
shrunk by its cross-half James-Stein weight. Toy elliptical sharp ring: ring-zone dispersion v2 1.37 -> 1.09. On real
dev hosts it broke the background next to the stop (D0 G error, box-noise units, fly-scan BA session sc19f33b9, 168
spots): v2 0.441; robust free m = 2 0.754; robust without m = 2 0.542; with the count-law flags (used pixels) and m = 2
0.483; flags without m = 2 0.428; flags, m = 2 only on rings with >= 30 of 32 sectors 0.459, and penumbra pixels excluded
0.446. Two separate defects: the unflagged pixels leak (hot pixels, multi-event clusters, the dilated flag rims), which a
sector-level Huber does not hold; and a free per-ring m = 2 picks up local azimuthal structure next to the stop and
spreads it round the ring. Adopted instead (no new parameters): the v2 ring refinement on ELLIPTICAL rings
r' = |p - c'| (1 + ea cos 2phi' + eb sin 2phi'), (ea, eb) = the fit's own ellipticity columns (the d2 columns
r R'(r) cos / sin 2phi give b ~ R(r (1 + ea cos 2phi + eb sin 2phi)) to first order), with skipped ring means re-centred
on G + c (re-flag against G + c, re-estimate, up to 3 rounds, stop when c changes by < 0.1 pixel SD). Toy: 1.37 ->
1.10; sc19f33b9 BA 0.441 -> 0.438; s0047f887 CN 0.304 -> 0.310; dev classic ring zones unchanged to 3 decimals
(`ring_dev.csv`); full dev harness G2.1 = G2 within 0.003 box noise in every regime x d bin.

### 1.4 Fix 3: why the summation intervals under-covered (A268)

On the v2 dev synthetic rows (informative spots, P/B >= 10, 480 spots): coverage 0.590 while |z| <= 1 (sigma at truth)
holds for 0.690; 48 spots had |z| <= 1 and were NOT covered, none the reverse. Cause (bin 4 dead time, cores past the
maximum of L(lambda) = law-expected stored excess): (i) v2 returned E at the END POINTS of the lambda set; where L is not
monotone inside the set the end points can be close in E (degenerate intervals, ci_lo = ci_hi), while the image of the
set is wide; (ii) the lambda grid (LC estimate +- 10 LC-sigma) does not reach the dead-time maximum when the LC sits on
the rising branch; (iii) the E-image is a model quantity (estimated background) while lam_eff, the evaluation truth, is
computed with the true background; near saturation a 1-3 % background difference moves the model's maximum of L by
more than sigma (checked on one spot: with the TRUE background the model reproduces lam_eff to 1e-13). (iii) is a
property of the evaluation statistic, not of the interval: the coverage that defines a confidence set is
P(lambda_true in C) for the parameter lambda (the box intensity). Fix: the image [min E, max E] over C with linearly
interpolated crossings, a grid of 41 linear points over centre +- 8 sigma plus 24 geometric points up to 4 / (tau
f_max) (where the brightest pixel is 4x past its dead-time peak), and the lambda interval [min C, max C]
(`ci_lam_lo`, `ci_lam_hi`, `ci_open` when C reaches the grid end). Dev (4 frames per cell; v2.1 H2.1 summation /
H2.1-ML coverage of lambda): bin 4 P/B >= 10 0.62 / 0.63 (rsd 1.08 / 1.06 on these 240 spots), bin 4 <= 1 0.72 / 0.68,
bin 8 >= 10 0.72 / 0.72, bin 8 <= 1 0.64 / 0.66; mismatches between "|z| <= 1" and "covered" 48 -> 12 (all case iii).

### 1.5 Fix 4: where the 441 ms went

Profile of `integrate_spots` (35 spots, bin 8, v2): probes 120 ms (numpy fancy indexing over ~65,000 probe centres x
230 offsets), intervals 69 ms, the v1 part (LC, ML, box sums) ~230 ms. v2.1: one numba pass for the probe sums
(16 ms; same probes and pixel sets, equal to 1e-12, test), 65-point interval grids (19 ms for five inversions), plus the
G-ML pre-pass of the spill term (~25 ms). Dev timing: v1 228 ms, v2 444 ms, v2.1 264-290 ms (same frame, same session).

## 2. Pre-registration (written and committed BEFORE any v2.1 test run)

### 2.1 Fix 1: H2.1 = G2.1 + a gated, spill-corrected local weight

For spot k (box b, n_b pixels; annulus a, n_a unflagged pixels), with the v2 probe neighbourhood N_k (lw_select:
radius-matched, |r_j - r_k| <= 2 r_out, arc <= 12 r_out, widened to >= 20 probes) and blocks of 2 r_out x 2 r_out:

    T_k = S_ab / SE_B(S_ab),   S_ab = sum_{j in N_k} a_j b_j,   SE_B^2 = nB / (nB - 1) sum_blocks (A_B - n_B mean(ab))^2
    w_k = clip(S_ab / S_aa, 0, 1) if T_k > z = 2, else 0          (the v2 weight w*, only where its signal is significant)
    s_k = mean_{p in a} [ E(mu_eff(mu_p, lambda_G f_p)) - E(mu_p) ]   (the spot's own light in the annulus)
    B_k = G2.1_box + w_k n_b (La_k - s_k),    Var(B_k) = (1 - w_k)^2 Var(G_box) + w_k^2 Var(La_k) n_b^2

La_k = mean of K - G2.1 over the unflagged annulus; f the box-normalised pixel-integrated Gaussian profile (the
integrator's `sigmas`); mu_p = law.E_inv(G2.1_p); lambda_G = the stored-count ML box intensity on the G2.1 background
(G2.1 excludes the spot discs, so lambda_G does not involve the annulus: no circularity). Without `sigmas`, s_k = 0.
The same spill term enters v2.1's GL (w = 1). H2.1-ML is the ML on H2.1's background.

Bias. Condition on G2.1 (misfit field delta = b - G, delta_b,k and delta_a,k its box / annulus means at spot k). The
probes never overlap spot k's box or annulus, so (T_k, w_k) are independent of the spot's own count noise n_a,k, and
E[n_a,k] = 0 (annulus pixels lie inside the excluded disc and are never flagged). With the spill term exact,

    E[B_k - b_k | G] / n_b = E[1(T_k > z) w_k] delta_a,k - delta_b,k  =:  E[w~_k] delta_a,k - delta_b,k.

Averaging over G2.1's randomness where G2.1 holds the truth in its span (E delta = 0 at every location; R1 for G2.1),
bias_k = Cov(w~_k, delta_a,k) (both deltas have mean 0). (a) No local structure (plain frames): w~ depends on the probe
residuals only through T and the ratio S_ab / S_aa, both invariant under (delta, noise) -> -(delta, noise); for a
sign-symmetric error law the covariance vanishes exactly, and in general (Cauchy-Schwarz, w~ <= 1)

    |bias_k| <= sqrt(P(T_k > z)) rms(delta_a,k).

Dev values: P(T > 2) = 0.024 (bin 4) / 0.026 (bin 8) on plain frames, rms(n_b delta_a) = 0.083 / 0.077 box noise ->
|bias| <= 0.013 / 0.012 box noise, i.e. <= 1.3 SE of a dev R1 cell (median cell SE 0.010 box noise): R1 holds BY BOUND
in every plain cell (the bound is the worst case; symmetry makes the first-order term zero). v2's H2 has the same
expression with P = 1 (w* in (0, 1] on every spot): bound 0.08 box noise, 8 SE, which the sparse-cell realisation used.
(b) Local structure (structured frames, real frames): w~ is mostly on (P 0.26-0.29); the remaining bias is
Cov(w~, delta_a) plus the curvature of the misfit between box and annulus (delta_a,k - delta_b,k in expectation); the
dominant systematic term of v2 (the spot's own light, +0.19 of +0.26 % in the arm cell) is removed by s_k. There is no
bound here; the dev evidence is 0 / 60 cells beyond 3.2 SE and 2 / 60 beyond 2 SE above the floor (max |z| 2.58) on 288
fresh frames.
Variance: on plain frames E[w~^2] n_b / n_a of the box variance (P(on) 2.5 %: the dev plain error 0.097 vs G2 0.095);
where on, as v2.

### 2.2 Fix 2: G2.1 = G1 -> re-centring (v2) -> the elliptical ring refinement

As v2 Sec. 2.1, with rings of r' = |p - c'| (1 + ea cos 2phi' + eb sin 2phi'), (ea, eb) = mean over the halves of the
fit's d2 coefficients (`refine.implied_ellipticity`), and up to 3 re-flag rounds against G + c (stop when c changes by
< 0.1 pixel SD). The estimator per ring is unchanged (cross-half product over the window, positive-part James-Stein), so
v2's bias and variance statements carry over: with Gaussian ring means E[c] = 0 where delta = 0 (sign symmetry), -(1 - w)
delta in between, -> delta for |delta| >> sqrt(v); cost w^2 v per ring. (ea, eb) are estimated from the same frame but
enter only the ring geometry (which pixels are averaged together), not the weights of the counts.

### 2.3 Fixes 3-4: intervals

`integrate_spots(version="v2.1")`: as v2 Sec. 2.3 (sigma at the hypothesised lambda, no data of the spot in sigma) with
the confidence set C = {lambda : |est - E(lambda)| <= sigma(lambda)} on the Sec. 1.4 grid; `ci_lo`, `ci_hi` = the image of
C in the estimate's units (summation: law-expected stored excess, electrons; ML: lambda); `ci_lam_lo`, `ci_lam_hi` =
[min C, max C] (summation); `ci_open` = C reaches the grid's upper end. Probe sums by one compiled pass. The ML, the
summation estimates and sigma (at the estimate) are unchanged.

### 2.4 Rules (v2.1; pre-registered)

- R1 SYNTHETIC UNBIASEDNESS (as v1, v2): every lacey cell (60), |beta| <= max(0.005, 3.2 SE) for G2.1 AND H2.1.
- R2 (v2 form) COUNT STATISTICS / TRUNCATION: spot-free twins, 24 cells, paired bias of G2.1 |Delta| <= max(0.3 %,
  3.1 SE) AND |masking on - off| <= 0.1 %.
- R3 v2.1 SIGMA HONESTY AND COVERAGE: synthetic lacey, bin 4 / 8 x {P/B <= 1, P/B >= 10}, spots with lambda_box >=
  lam_min5 (of H2.1-ML): for H2.1 summation (truth lambda_eff) and H2.1-ML (truth lambda_box), z = (I - truth) /
  sigma_at(truth) has |mean| <= 0.1 and robust SD in [0.9, 1.1], AND the coverage of the confidence set for the box
  intensity, P(lambda_box in [ci_lam_lo, ci_lam_hi]) (summation) / P(lambda_box in [ci_lo, ci_hi]) (ML), is in
  [0.63, 0.73] (the 1-sigma Gaussian coverage for an rsd of 0.9-1.1). Reported: the E-image coverage against lam_eff
  (v2's statistic), all spots, the real frames.
- R4 (v2 form) REAL, BETTER THAN FLAT for H2.1, per regime x d bin, on spots WITHOUT a host overlap (v2's
  pre-registered flag, unchanged): |beta| <= max(0.01, 2 SE) in >= 90 % of cells, rmse_bg(H2.1) < rmse_bg(FLAT) in every
  cell, |beta| <= |beta_FLAT| + 2 SE in every cell. All-spot R4 reported with it (A266). FLAT on the same spots.
- R5 SPEED: median G2.1 fit + map (plan cached, ring refinement on, re-centring not triggered) on a 1024^2 frame,
  numba, >= 15 frames, <= 50 ms on this Mac, v1 and v2 timed in the same run; measured when the 1-min load average is
  below 4 (other agents share the machine; the load is recorded), else repeated up to 3 times and the lowest-load run
  reported. Integration per 35-spot bin-8 frame reported against v1 in the same run (fix 4 target: <= 1.3x v1; not in
  the verdict).
- R6 (reported, as v1/v2).
- R7 SHARP RING (fix 2; reported, not in the verdict): on the spot-free hosts of sb7857cd8 (`bgsub_v2_1_ring.py`),
  median over hosts of the ring-zone dispersion (|s - 1/4.60 A| <= 3 bin-4 px) of K - G2.1 against the count law <= 0.5 x
  G2's AND the range of the 16 ring-zone sector means <= 0.5 x G2's; NO HARM: in every other classic test and
  confirmation session the ring-zone and D0 dispersions of G2.1 <= 1.02 x G2's (median over hosts); and in the real test
  run, sb7857cd8 D0 error (no overlap) of H2.1 <= H2's and of G2.1 <= G2's.
- VERDICT: bgsub v2.1 is VALID iff R1, R2, R3 (synthetic) and R5 pass and R4 passes in F and classic (no overlap). BA
  and CN R4, all-spot R4, the classic confirmation set, R6 and R7 are reported. No parameter changes after a test run;
  failures are reported with their cells.

### 2.5 Evaluation design

- SYNTHETIC: the v1 / v2 design exactly (`bgsub_v1_synth.cells` / `make_frame`, seeds 30000+, simulator
  `version="v1"`): lacey bins 4 / 8 x f {1, 3, 10} x {plain, structured} x 24 frames, robustness quantifoil / formvar
  bin 8 x f {1, 3} x 8 frames (untested against data); 35 spots per frame; v2 (G, GL, AUTO, FLAT) and v2.1 (G, GL,
  AUTO) on identical frames (`bgsub_v2_1_synth.py --set test`). The v2 rows of this run are compared with the published
  v2 rows (same frames => identical numbers).
- REAL: the v0-v2 harness, same sessions, hosts, replicates and seeds (17 fly-scan, 20 classic) and the 9-session
  classic confirmation set (`bgsub_v2_1_real.py --set test|confirm`), v2 and v2.1 on each injected frame, v2's host-overlap
  flag; v2 rows compared with the published v2 rows.
- R7: `bgsub_v2_1_ring.py --set test|confirm`. R5: `bgsub_v2_1_speed.py`.
- Workers: multiprocessing (spawn, 4-6 workers; every job wrapped so a failure is logged as FAILED); completeness checked
  against the planned units (frames, sessions) from the checkpoints before any rule is read, and reported.

(Pre-registration ends here; everything below was produced afterwards.)

Commit of record: the pre-registration (this file up to here), the v2.1 code, tests and harness scripts were committed
in fd2d10f and 4d6a251 (10:10, 2026-11-05) before any test run; README and the figure script in 19b4a6f. No parameter
was changed after a test run.

## 3. Completeness of the test runs (checked before any rule was read)

All runs: multiprocessing (spawn, 5 workers), every job wrapped (a failure is logged as FAILED).
- SYNTHETIC (seeds 30000+): 352 / 352 planned frames (lacey 12 cells x 24, robustness 8 cells x 8), 35 spots each,
  0 FAILED. The v2 rows of this run equal the published v2 test rows bit for bit (12,320 spots per method; max |dB| =
  max |dI| = 0 for G, AUTO, GL, G-ML, AUTO-ML).
- REAL: 37 / 37 test sessions (17 fly-scan, 20 classic), 48,930 spots per method (F 13,650, BA 14,280, CN 12,600,
  classic 8,400), 0 FAILED; CONFIRMATION 9 / 9 sessions, 3,430 spots, 0 FAILED. The v2 G, AUTO, G-ML, AUTO-ML rows equal the
  published v2 rows in the same order, max |dB| = max |dI| = 0 (48,930 and 3,430 spots), and v2.1's FLAT equals the
  published FLAT (max |dB| 0).
- R7: 20 test + 9 confirmation classic sessions, all hosts (`ring_test.csv`, `ring_confirm.csv`).
- R5: one attempt (load 3.1-3.6 < 4 at the start).

## 4. Synthetic results (`synth_test_{cells,trunc,pulls,ml}.csv`, `synth_test_rules.json`)

### 4.1 Rules, v2 vs v2.1 on identical frames

| rule | v2 | v2.1 |
|---|---|---|
| R1 G, 60 cells | PASS 0 / 60 (2 at 2 SE) | PASS 0 / 60 (2 at 2 SE); max |G2.1 - G2| in beta 0.12 % |
| R1 H, 60 cells | FAIL 2 / 60: arm cell +0.70 +- 0.19 %, sparse cell -3.16 +- 0.79 % | **FAIL 1 / 60: sparse cell -2.46 +- 0.72 %** (arm cell +0.34 +- 0.12 %: PASS) |
| R2 paired, 24 cells | PASS (max 0.45 %; v1 form 1 / 24) | PASS (max 0.45 %; v1 form 1 / 24) |
| R3 pulls (sigma at truth) | PASS | PASS: means -0.08..+0.04, rsd 0.97-1.02 |
| R3 coverage in [0.63, 0.73] | (summation 0.578 / 0.567 at P/B >= 10) | **PASS**: summation 0.681 / 0.678 / 0.667 / 0.686, ML 0.688 / 0.688 / 0.674 / 0.664 (bin 4 high / low, bin 8 high / low) |
| R6a / R6b (reported) | FAIL / FAIL 8 / 88 | FAIL (G-ML 0.88 / 0.96 / 0.99 at f 1 / 3 / 10) / FAIL 8 / 88 (unchanged, information limit) |

R1 for H2.1 fails in the sparse cell (bin 8, f 3, plain, D1; box background 0.2 stored counts): G2.1 there is -1.98 +-
0.65 % (3.05 SE, the same realisation as G2's -1.99 %), and 5 of the 168 spots have the gate on (w 0.22-0.42); their
corrections (-0.075 to +0.014 counts, four of five negative) move beta by -0.48 % (A276). The bias bound of Sec. 2.1
holds (it bounds the expectation; this is a 1.5-SE draw of the gated term added to a 3.05-SE draw of G); the rule does
not distinguish the two, and it fails as written. The arm cell is fixed by the spill term (+0.70 -> +0.34 %, limit 0.5 %).
Figure: `notes/figures/bgsub_v2_1/synth_r1_v2_v21.png`.

### 4.2 Background error (box-noise units, lacey test frames)

| bin | struct | v2 G2 | v2.1 G2.1 | v2 H2 | v2.1 H2.1 | FLAT |
|---|---|---|---|---|---|---|
| 4 | plain D0-D4 | 0.07-0.16 | 0.07-0.16 | 0.08-0.16 | 0.07-0.16 | 0.41-0.58 |
| 8 | plain D0-D4 | 0.07-0.12 | 0.07-0.12 | 0.08-0.13 | 0.07-0.12 | 0.38-0.62 |
| 4 | structured D0 / D4 | 0.50 / 0.84 | 0.51 / 0.84 | 0.43 / 0.49 | **0.36 / 0.39** | 0.61 / 0.56 |
| 8 | structured D0 / D4 | 0.35 / 0.97 | 0.35 / 0.97 | 0.31 / 0.45 | **0.26 / 0.40** | 0.60 / 0.54 |

H2.1 removes v2's plain-frame cost (mean w 0.010 vs 0.063 on plain lacey) and lowers the structured error further
(spill term). Robustness presets (untested against data): mean w 0.012-0.019 plain, 0.13-0.14 structured.

### 4.3 Coverage (fix 3)

The coverage of the confidence set for the box intensity is nominal in every cell for summation and ML (table above;
`synth_coverage_v2_v21.png`). The E-image coverage against lam_eff (v2's statistic) is 0.671 / 0.677 / 0.655 / 0.684
for the summation: also nominal; v2's 0.57-0.58 was the end-point error, not tails.

## 5. Real results (`real_test_{cells,d0,d0_by_session,pulls}.csv`, `real_test_rules.json`, `real_confirm_*`)

### 5.1 D0 (d 4-8 A) background error, box-noise units, v2 vs v2.1 paired

| regime | spots | v2 G2 | v2.1 G2.1 | v2 H2 | v2.1 H2.1 | FLAT |
|---|---|---|---|---|---|---|
| F | no host overlap | 0.31 | 0.32 | 0.03 | 0.08 | 0.40 |
| F | all | 0.88 | 0.88 | 0.08 | 0.36 | 0.40 |
| BA | no overlap / all | 0.46 / 0.48 | 0.46 / 0.47 | 0.33 / 0.35 | 0.34 / 0.35 | 0.50 / 0.52 |
| CN | no overlap / all | 0.57 / 1.17 | 0.57 / 1.17 | 0.35 / 0.47 | 0.37 / 0.50 | 0.60 / 0.60 |
| classic | no overlap | 1.57 | 1.58 | 1.76 | **1.57** | 2.98 |
| classic | all | 7.21 | 7.20 | 6.30 | 7.20 | 5.96 |
| classic CONFIRM | all = no overlap | 0.48 | 0.48 | 0.51 | 0.50 | 1.30 |

H2.1 keeps most of v2's D0 gain on clean frames (F 0.31 -> 0.08 vs v2 0.03; BA and CN within 0.01-0.03 of v2), is
better than v2 on classic without overlap (1.76 -> 1.57) and equal on the confirmation set. With all spots F rises to
0.36 (v2 0.08): one of the 4 F spots with a host overlap has the gate off (A279). Outside D0, H2.1 = G2.1 within
0.01-0.03. Figure: `real_d0_v2_v21.png`.

### 5.2 R4 and intervals

| R4 | v2 H2 | v2.1 H2.1 |
|---|---|---|
| no overlap (verdict form) F / BA / CN / classic | P / P / P / P | **P / P / P / P** |
| all spots F / BA / CN / classic | P / P / P / F | P / P / P / F (classic D0 7.20 vs FLAT 5.96; s8975a6a4, sb7857cd8) |
| confirmation (no overlap = all) | P | P |

Real pulls (informative spots, sigma at truth) for H2.1: means -0.02..+0.09, rsd 0.93-1.08; coverage of the box-intensity
set 0.65-0.70 (summation) and 0.65-0.69 (ML) in every regime x P/B class; confirmation 0.65-0.70.

### 5.3 R7: the sharp ring (fix 2)

| quantity (median over hosts) | v2 G2 | v2.1 G2.1 | rule |
|---|---|---|---|
| sb7857cd8 ring-zone dispersion (all valid pixels) | 16.2 | 15.1 | <= 8.1: FAIL |
| sb7857cd8 16-sector range (counts / px) | 39.4 | 39.6 | <= 19.7: FAIL |
| sb7857cd8 ring-zone mean residual (counts / px) | +13.4 | +12.7 | - |
| no harm: max ratio v2.1 / v2 over the 28 other sessions (ring zone / D0) | - | 1.0018 / 1.0008 | <= 1.02: PASS |
| sb7857cd8 D0 error, no overlap, H / G | 4.29 / 3.79 | 3.81 / 3.81 | H2.1 <= H2 PASS; G2.1 <= G2 FAIL (+0.02) |

The elliptical ring term does not remove the sharp ring of sb7857cd8 (A277). The fit's ellipticity is -0.14 to -0.31 %
(the 0.4 % Arctica distortion), so the geometry is right; the failure is the selection: 34-81 % of the ring-zone pixels
are count-law flags in every host, the skipped ring means of the unflagged rest sit near G, the correction stays below
one pixel SD in 3 of 5 sharp hosts and the re-flag rounds never start (post hoc: forcing the re-flag rounds changes
nothing, dispersion 47.7 -> 48.5, 37.6 -> 37.6, 17.2 -> 17.4). A skipped mean has a breakdown point below the flagged
fraction here; the robust alternative that does not use the flags failed elsewhere (Sec. 1.3). The no-harm clauses hold.

## 6. Speed (R5; `speed_v2_1.json`, one attempt at load 3.1-3.6, v1 and v2 in the same run)

| frame | v1 | v2 | v2.1 | rule |
|---|---|---|---|---|
| 512^2 (bin 16) | 19.6 ms | 20.3 ms | 21.1 ms | - |
| 1024^2 (bin 8) | 46.3 ms | 50.2 ms | **54.2 ms** | <= 50 ms: **FAIL** |
| 2048^2 (bin 4) | 123 ms | 130 ms | 142 ms | - |
| integration, 35 spots, bin 8 | 233 ms | 428 ms | **286 ms (1.23x v1)** | fix-4 target <= 1.3x v1: met |

Post hoc breakdown (same machine): the elliptical ring kernels take 2.6 ms (accumulation, v2 1.4 ms) and 2.5 ms
(application, v2 1.0 ms), i.e. +2.7 ms; the rest is load (v2 itself measured 46.8 ms in its own run and 50.2 ms here).
A278.

## 7. Verdict (pre-registered rules)

| rule | v2 | v2.1 |
|---|---|---|
| R1 G / H | PASS / FAIL 2 / 60 | PASS / **FAIL 1 / 60** (sparse cell -2.46 +- 0.72 %) |
| R2 | PASS | PASS |
| R3 (pulls + coverage) | PASS (coverage not in the v2 rule; 0.57-0.58) | PASS (coverage 0.66-0.69) |
| R4 F / classic (no overlap) | PASS / PASS | PASS / PASS |
| R5 | PASS 46.8 ms (own run) | **FAIL 54.2 ms** (v2 50.2 ms in the same run) |
| R4 BA / CN, all spots, confirmation (reported) | P, P / F (classic) / P | P, P / F (classic) / P |
| R6 (reported) | FAIL / FAIL | FAIL / FAIL |
| R7 (reported) | - | sharp ring FAIL; no harm PASS |

**bgsub v2.1 is NOT VALID by its pre-registered verdict**: R1 fails for H2.1 in 1 of 60 cells and R5 fails (54.2 ms
against 50 ms). What changed from v2, and what the evidence supports:
(i) Fix 1: the arm-cell failure of v2 was the spot's own light in the annulus, not the arm shadow (A273); the spill term
removes it (+0.70 -> +0.34 %). The gate makes H2.1 = G2.1 on frames without local structure (P(on) 2.5 %, plain error
equal to G2.1) and bounds the bias there by sqrt(P) x rms(annulus misfit); with local structure H2.1 is the most accurate
background tested (structured D4 0.39-0.40 vs v2 0.45-0.49 vs G 0.84-0.97) and keeps most of the real D0 gain (F 0.08,
BA 0.34, CN 0.37, classic 1.57 without overlap). The remaining R1 failure sits in the one cell where G2.1 itself is at
3.05 SE on these frames; five gated-on spots tip it.
(ii) Fix 2: the elliptical ring term is harmless everywhere (ratios <= 1.002) and follows elliptical sub-knot rings on
synthetic frames, but does not remove the sb7857cd8 ring: the count-law flags take most of the ring's pixels (A277).
(iii) Fix 3: the intervals are now the image of the confidence set; coverage is nominal (0.66-0.69 synthetic, 0.65-0.70
real) for summation and ML.
(iv) Fix 4: integration 428 -> 286 ms (1.23x v1).
(v) R5: G2.1 is 4 ms slower than G2 (elliptical kernels) on a machine where v2 itself measured 50.2 ms.
No parameter was changed after a test run.

## 8. Limits and next steps (not pre-registered)

- R1 in sparse cells: the gate fires at its nominal rate on noise, and at 0.2 counts per box five firings are enough to
  move a cell by 0.5 %. Options: a gate level that scales with the count level (z from the exact null of T at sparse
  counts, e.g. by a sign-flip permutation of the probe products), or no local correction below a box level where G2.1's
  error is already below 0.1 box noise.
- The sharp ring needs an unflagged first pass restricted to the ring zone (a high-breakdown ring estimate, e.g. a
  median over sectors of unflagged-and-flagged pixels within |r' - r0| <= 2 px), followed by the re-centred skipped mean;
  Sec. 1.3 shows the full-frame robust version is not acceptable.
- R5: fuse the elliptical accumulation and application into the existing passes (precomputed r' map per plan) to recover
  ~2.5 ms; and measure R5 on an otherwise idle machine.
- R4's pass still rests on the pre-registered host-overlap exclusion (A266), now also for one F spot (A279).

## 9. Anomalies

- A273: A265's arm-cell reading is wrong. 0.25 of the 0.27 % H2 bias in that cell (fresh frames) sits > 20 px from the
  stop where G2's misfit is zero, 0.19 % from P/B 10-30 spots: the spot's own Gaussian tail in the annulus, large
  because the core is past the dead-time maximum (a tail predicted from the stored box excess is 100x too small; from the
  electron intensity it is right). With the tail removed H2's arm-cell bias is +0.03 %.
- A274: the overlapping probes of the local weight are far from independent: block (2 r_out) standard errors of S_ab are
  ~4x the naive ones; a JS shrinkage by the block SE removes the slope on real frames (mean w 0.004-0.013), and local-affine
  forms (with an intercept) lose the real D0 gain: real local structure is not neighbourhood-wide.
- A275: a ring term with robust (Huber) sector means over unflagged pixels and a free per-ring m = 2 breaks the background
  next to the stop on real frames (BA D0 0.441 -> 0.754 on sc19f33b9); both parts contribute (unflagged leakage; local
  azimuthal structure projected round the ring).
- A276: H2.1 fails R1 in the sparse cell (bin 8, f 3, plain, D1): G2.1 -1.98 +- 0.65 % on these frames plus -0.48 % from 5
  of 168 gated-on spots (four negative corrections). On 64 fresh frames of the same cell H2.1 is -0.47 +- 0.72 %.
- A277: the sb7857cd8 ring is not removed by v2.1: 34-81 % of its ring-zone pixels are flagged; the skipped ring mean breaks
  down; forced re-flag rounds change nothing (post hoc).
- A278: R5 fails at 54.2 ms; the elliptical kernels cost +2.7 ms over v2's circular ones; v2 measured 50.2 ms in the same
  run (46.8 ms in its own).
- A279: all-spot F D0 error 0.08 (v2) -> 0.36 (v2.1): 1 of the 4 F spots with a host overlap has the gate off (the overlap
  is compact, the probes do not see it).
- A280: part of A268 was the evaluation statistic: the E-image of the interval is a model quantity (estimated background),
  lam_eff uses the true background; near the dead-time maximum the two differ by more than sigma. The coverage of the
  set for lambda (the parameter) is the correct statistic and is nominal.
- A281: the real-table scripts of v2 and v2.1 merge lam_min5 on cluster = sid:frame, which is not unique for fly-scan hosts
  (the same frame index in different rows): about 10 % of rows are duplicated in the merged table used for the real pull
  and per-P/B tables (informative / all subsets). R4 cells and D0 tables are computed before the merge and are unaffected;
  the v2.1 identity check was redone on the unmerged rows (exact). The duplicated rows are paired spots of other hosts, so
  the pull means move little, but the real pull tables of v2 and v2.1 carry this defect.
- A282: `info_v2.1_ring2_n` in the harness logs is always 0 (the m = 2 counter of the abandoned stage-1 ring term); no
  table uses it.
