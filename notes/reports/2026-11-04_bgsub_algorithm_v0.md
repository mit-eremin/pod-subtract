# Fast, unbiased, universal background estimation for single diffraction frames: bgsub v0 (2026-11-04)

Terms of reference: OWNER RE-SCOPE 2026-11-04 item 2 (CLAUDE.md, Goal). Design, derive, implement and validate a fast
background estimator for single frames (snapshots, stills, rotation frames) that is unbiased under Bragg spots, has honest
variances under the detector's real count statistics, excludes spots / hot pixels / zingers without bias, needs only the
frame, a mask and the geometry, and runs well under 50 ms per 1k x 1k frame. Compared against generic reference versions
(written here, not copied) of the local flat (annulus mean), local plane and robust local estimates.

Code: `src/podbg/bgsub/` (new subpackage), tests `tests/test_bgsub*.py`, scripts `scripts/bgsub_v0_*.py`. Outputs with
provenance sidecars: `~/pod_data/background/bgsub_v0/`. Figures: `notes/figures/bgsub_v0/`. Anonymous sids only. Nothing
runs on HPC. Anomalies are numbered from A193.

## 1. Design and pre-registration (written and committed BEFORE any evaluation run)

What I had seen before writing this section: CLAUDE.md (Goal and re-scope, detector facts incl. the stored-count law, the
tie rule and dead time, beam stop, geometry incl. k_x / k_y = 1.008, BAR ADJACENT v0, REyes DROP VALIDATION v0, ATLAS
REFRESH v0, S LEVEL BIN-4 v0 / A134), the code of `podbg.peaks`, `podbg.binning` (quantise, response tables),
`model.forward` (Settings, ForwardModel, render), `atlas.load_session_features`, the pre-registration of peak recovery v0
(`2026-11-04_peak_recovery_injection_v0.md` Sec. 1) and the parts of `scripts/peak_recovery_v0.py` that place, inject and
integrate peaks. Data looked at: only file listings of the cached host frames and one radial forward-model profile
(`S_family_flyscan_v2`, t 20 nm) to size the synthetic dose ladder. No background estimate of any frame has been computed.

### 1.1 Notation and the observation model

Pixel p of a frame stores an integer K_p. Its latent electron number N_p ~ Poisson(mu_p) (dead time folded into mu as the
paralyzable mean-field survival, as `forward.render`); the detector maps N to K by a known law Q: K = N at bin 4 (and for
any plain counting detector), K = rint_even(N / m) with m = b^2 / 16 e per count and the gain-flag tie rule at bin 8 / 16
(`binning.quantise`). Write

    b_p = E[K_p]                 (the background in STORED counts, the quantity a summation integrator subtracts)
    g(b) = Var[K_p | E K_p = b]  (the stored-count variance function; Poisson: g(b) = b; Apollo bin 8: from
                                  binning.response_table at the session tie rule; integrating detector: g(b) = G b + r^2)

At bin 8 K is NOT Poisson: g(b) ~ b for b << 1 (K is nearly Bernoulli) and g(b) -> b / m + 1/12 for b >> 1. Nothing
below assumes Poisson for unbiasedness; g enters only through weights (efficiency), thresholds and variances.

A spot adds lambda_p expected electrons; integration over a box with an estimated background B_p gives
I = epc sum_box (K_p - B_p), epc = electrons per count. Background error and intensity error are tied exactly:
I - I_oracle = -epc sum_box (B_p - K0_p), where K0 is the pre-spot frame. The relative background bias
beta = sum_box (E B_p - b_p) / sum_box b_p causes an intensity bias beta / (P/B) (P/B = lambda_box / B_box), so at
P/B = 0.1 a 1 % background bias is a 10 % intensity bias: this is why flat local backgrounds "lose intensity".

### 1.2 Why flat / plane local backgrounds are biased (requirement 1, derivation)

Expand b about the spot centre: b(x) = b0 + grad b . x + 1/2 x^T H x + O(|x|^3). Averaging over an annulus
r_in < |x| <= r_out (uniform in angle) kills the odd terms and gives E[B_flat] = b0 + (1/4) tr(H) <rho^2>_ann, with
<rho^2>_ann = (r_out^2 + r_in^2) / 2. The box mean of the truth is b0 + (1/4) tr(H) <rho^2>_box with <rho^2>_box =
r_box^2 / 2. Hence

    beta_flat = (1/4) lap(b) (<rho^2>_ann - <rho^2>_box) / b0         (to second order)

For the pre-registered peak-recovery boxes at bin 8 (r_box 3, annulus 5-9 px): <rho^2>_ann - <rho^2>_box = 53 - 4.5 =
48.5 px^2. A radial profile b ~ r^-n has lap(b) / b = n^2 / r^2, so at the inner halo (n ~ 2-3, r 60 px) beta_flat =
12.1 n^2 / r^2 = 1.3-3.0 %; at a ring flank the second derivative of a ring of relative contrast c and width sigma_r
gives beta ~ 12 c / sigma_r^2 (0.1-0.3 % for c 0.1, sigma_r 20 px), and at the ring crest it is negative. A plane fit
removes the gradient (which the symmetric annulus already removes on average) but NOT the curvature: beta_plane =
beta_flat to this order. Ellipticity, the m = 1 pedestal and centre errors add no bias to first order for a symmetric
annulus; masked sectors (stop arm, seams, neighbour spots) break the symmetry and turn the gradient into a first-order
bias grad b . <x>_ann. A robust estimate (median, Huber) additionally carries a distribution bias: the median of a
skewed count distribution is not its mean (A134: per-shell median biased low at bin-8 counts); the Huber M-estimate is
consistent only if its Fisher-consistency correction is computed under the TRUE count law, which at bin 8 is not
Poisson. All three are unbiased only where the background is locally linear and the count law symmetric.

### 1.3 The estimator (method G: global linear smooth model in stored counts)

b is modelled as a LINEAR combination of smooth basis functions in polar coordinates about the beam centre c:

    b(r, phi) = sum_k a_k B_k(r)                                         (fine radial cubic B-spline, knot step h_r)
              + sum_{m=1..M} sum_j A_j(r) [alpha_jm cos m phi + beta_jm sin m phi]   (harmonics, coarse radial spline, step H)
              + d_1c R'(r) cos phi + d_1s R'(r) sin phi                  (centre-error columns, fine radial structure)
              + d_2c r R'(r) cos 2phi + d_2s r R'(r) sin 2phi            (ellipticity columns, fine radial structure)

R(r) is the pilot radial profile (from the other half of the data, Sec. 1.3.3). The last four columns are the first-order
Taylor terms of b(|x - c - delta|) and of b(r (1 + e cos 2(phi - phi0))): a centre error delta or an ellipticity e moves
the pattern by R'(r) delta . u and r R'(r) e cos 2(phi - phi0), whose radial structure follows the rings and the halo
at full resolution, which a coarse harmonic spline cannot. The harmonics carry everything that is smooth in phi and on
scales of H in r: the bar-side m = 1 pedestal, residual ellipticity, illumination tilt, stray. The fine radial spline
carries rings and the steep inner halo. All basis functions vary on scales >= h_r in r and >= 2 pi r / M in phi, much
larger than a spot, and none is local enough to absorb a spot (an excluded spot leaves a hole of ~30 px in a support of
>= 10^3 px).

Pre-registered defaults (N = the larger frame dimension in px; all in output pixels):
- h_r = max(3, N / 256) px (4 px at 1k, 8 at 2k, 16 at 4k); cubic, uniform knots from r_min to r_max of the valid
  pixels; basis functions with fewer than 20 valid pixels in their support are dropped.
- H = 12 h_r (48 px at 1k); M = 6 harmonics (12 trig functions).
- Ellipticity / centre columns on (4 columns). If `geometry` carries an ellipse (e, phi0) or anisotropic pixel scales
  the radius is the elliptical radius and the columns correct the remainder.
- Penalty: second-difference P-spline penalty on the harmonic coefficients along r and on the fine radial coefficients,
  lambda = 1e-4 x the mean diagonal of the information matrix of that block (identifiability where shells are partly
  masked, e.g. frame corners and the stop arm). The induced bias is lambda (D^T D) a, quantified in Sec. 2.

#### 1.3.1 Unbiasedness theorem (requirement 1, conditions)

Let X be the design (rows x_p), W a diagonal weight matrix and E the excluded set (mask, flagged pixels, the integrated
spots' discs). The estimator a^ = (X_U^T W X_U + lambda P)^-1 X_U^T W K_U on the used pixels U is LINEAR in K. If (C1)
the truth lies in the span, b = X a, (C2) W and U are independent of the K_U they weight (W from the other half of the
data, U from the mask and from predicted spot positions), then E[a^ | W, U] = a - lambda (X^T W X + lambda P)^-1 P a and
E[b^_p] = b_p up to the penalty term, for EVERY count law, because only E[K_p] = b_p is used. There is no Poisson
assumption, no median and no log: the A134 bias cannot occur. Violations and their sizes:
- (C1) approximation error of a cubic spline: |b - Pi b| <= (5/384) h_r^4 |b''''| (de Boor); for b ~ r^-3 this is
  (h_r / r)^4 x 360 x 5 / 384 = 4.7 (h_r / r)^4 relative: 2e-4 at r = 60 px with h_r = 4. For a Gaussian ring of width
  sigma_r: ~ 0.04 (h_r / sigma_r)^4 of the ring amplitude: 6e-5 for sigma_r 20 px. Azimuthal structure beyond m = 6 or
  local on scales < H is NOT in the span: this is the residual the GL variant (Sec. 1.4) addresses.
- (C2) flagged pixels depend on their own values: truncation bias, bounded in Sec. 1.5.
- Weak spots that are neither excluded nor flagged leak into the model by their hat value: the background under a
  spot of lambda electrons rises by lambda h_box, h_box = sum_box of the hat matrix rows ~ n_box x (1 / (2 pi r 1.5 h_r)
  + (2M + 1) / (2 pi r 1.5 H)) (~2.5 % of the spot at r = 60 px, 0.5 % at r = 300). The integration helper therefore
  always excludes the discs of the spots it integrates (an integrator knows its predicted positions); "blind" use
  (no positions) is reported separately.
- Pixel-level evaluation of a model fitted on polar cells (Sec. 1.6): the within-cell variation of b contributes
  b'' / 24 (radial cell width 1 px) and is exactly zero for the trig terms because cell means of cos m phi / sin m phi
  over the static valid pixels are used: 1e-4 relative at r = 60.

#### 1.3.2 Weights, variance and the count law (requirement 2)

Weights W_p = 1 / g(b~_p) (b~ a pilot fit, floored at 1e-3 x its median) give the efficient (quasi-likelihood) linear
estimator; any positive weights keep it unbiased. Variance of the estimate (sandwich, exact for a linear estimator):

    Cov(a^) = F^-1 (X^T W Sigma W X) F^-1,  F = X^T W X + lambda P,  Sigma = diag(phi g(b^_p))

with phi the Pearson dispersion of the frame on polar cells, phi = sum_c (S_c - n_c b^_c)^2 / (n_c g(b^_c)) / (n_cells
- p). phi = 1 when g is the right law; phi > 1 absorbs pixel correlation (an integrating detector's PSF, split
events), an unknown gain, or unmodelled structure, so the variances self-calibrate on any detector. Var(b^_p) = x_p^T
Cov x_p. For a box the background errors are FULLY correlated: Var(sum_box B) = s^T Cov s with s = sum_box x_p, ~
n_box^2 Var(b^) and not n_box Var(b^). Integrated intensity (box excluded from the fit, so cov(K_box, B_box) = 0):

    sigma_I^2 = epc^2 [ V_K + s^T Cov s ],   V_K = max( sum_box [g(B_p) + g'(B_p)(K_p - B_p)], (1/2) sum_box g(B_p) ) x phi

V_K is the first-order unbiased estimate of sum_box g(b_p + lambda_p): E[g(B) + g'(B)(K - B)] = g(b) + g'(b) lambda +
O(lambda^2 g''). The plug-in sum g(K_p) is NOT used: at bin 8 and b < 1 it is biased low by up to ~40 % because g is
concave and K is mostly 0 / 1 (Jensen). The floor only matters when the box records almost nothing. For comparability
the peak-recovery "sigma_law" (g(max(K, B))) is reported too.

#### 1.3.3 Cross-fitting (makes C2 hold exactly for the weights and the derivative columns)

Pixels are split into the two checkerboard halves A ((y + x) even) and B (odd). Half A is fitted with weights and
derivative columns R' computed from the fit of half B in the previous iteration, and vice versa; the estimate is the
average of the two fits, Cov = (Cov_A + Cov_B) / 4. Iteration 0 uses unit weights (ordinary least squares on pixels,
exactly unbiased) and no derivative columns.

### 1.4 Variant GL (global + local residual)

B_box^GL = sum_box G_p + (n_box / n_ann) sum_ann (K_p - G_p), the annulus as for the flat reference (other spots,
flagged pixels and the mask excluded). The global model removes the curvature and the azimuthal structure; the local
residual absorbs structure that the global model does not contain (diffuse scattering near spots, local shadows) under
the much weaker condition that the RESIDUAL is locally linear. Its bias is (1/4) lap(b - G) (<rho^2>_ann -
<rho^2>_box); its variance is that of the flat annulus, epc^2 (n_box^2 / n_ann) phi g. G is the default (lower
variance, unbiased under C1); GL is the fallback where C1 is doubtful. The choice is made by the dev-set rule D1 below,
not by the test set.

### 1.5 Exclusion of spots, hot pixels and zingers (requirement 3)

Iterative masking under the count law, positive side only (spots, hot pixels and zingers only add counts):
1. Robust pilot: per 1-px shell, the mean of pixels below the shell mean + 10 sqrt(shell mean) + 10 (two passes);
   pilot radial profile only (it sets no estimate, only the first flags and weights).
2. Fit G (Sec. 1.3); then flag pixel p if (a) K_p >= k*_px(b^_p), the smallest k with P(K >= k | b^_p) <= alpha_px under
   the count law (bin 8/16: the recorded PMF at the tie rule; Poisson; generic: Normal), or (b) the 3 x 3 valid-pixel
   sum S9 >= t*(b^_p), the alpha_w tail of the quasi-Poisson approximation Y = S9 b / g ~ Poisson(9 b^2 / g). Flags
   are dilated by r_dil px (disc). 3. Refit; repeat flagging (2 rounds), final fit.
Pre-registered: alpha_px = alpha_w = 1e-6; r_dil = 2 px at 1k (bin 8), scaled with N / 1024 (4 px at 2k, 1 px at
512), because the spot size is fixed physically.

Truncation bias: removing pure-background pixels with K >= k* removes the mean mass T(b) = sum_{k >= k*} k P(k | b)
from the used set, so the retained mean is (b - T) / (1 - P_tail). At alpha 1e-6 per pixel and per window,
|bias| / b <= (T(b) + b P_tail) / b, evaluated exactly from the count-law PMF (Sec. 2 table) and also MEASURED on
spot-free synthetic frames (rule R2). If it exceeds 0.1 % at any b of the ladder the analytic correction b / (b - T(b))
is applied (pre-registered fallback).

### 1.6 Speed (requirement 5)

All statistics are reduced to polar cells (1-px radial bins x 64 sectors, both checkerboard halves): per iteration two
weighted bincounts of the frame (numba when available, numpy otherwise), the fit solves a (p ~ 400)-dimensional system
built from cell sums with banded B-spline structure, and the per-pixel model is evaluated by bilinear lookup from a polar
grid (0.5 px x 256 sectors) with precomputed pixel indices (a `Plan`, cached per geometry and mask). Flagging is one
fused pass. Variance maps are lazy (computed on a coarse polar grid on request); box variances are exact quadratic
forms per spot. Timing targets: fit + background map <= 50 ms median for 1024^2 with numba; numpy-only and 512^2 /
2048^2 / 4096^2 reported.

### 1.7 Reference methods (generic implementations written here)

Per spot, on the same annulus (r_in, r_out) and box r_box, with the mask, other spots' discs and nothing else excluded:
- FLAT: B = mean of K over the annulus; sigma_I^2 = epc^2 (V_K + n_box^2 / n_ann^2 sum_ann g(K)) with the same V_K form.
- PLANE: least-squares plane b0 + gy dy + gx dx over the annulus pixels, evaluated over the box; Var from the LS
  covariance with g(mean) per pixel.
- ROBUST: Huber M-estimate of a constant with Poisson variance (c = 1.345) and the Poisson Fisher-consistency
  correction (a generic robust-GLM constant background); sigma from its asymptotic variance.
These sizes are the peak-recovery v0 boxes: bin 8 r_box 3, annulus 5-9, spot discs excluded r 9; bin 4 x 2; bin 16
r_box 1.5, annulus 2.5-4.5, disc 4.5.

### 1.8 Validation design

SYNTHETIC (stage 3; truth known exactly). Frames are made with the forward model's electrons
(`ForwardModel(S_family_flyscan_v2).electrons`, t 20 nm; the radial chain of `forward.render(sample=True)`: Poisson at
the dead-time-corrected mean, then `binning.quantise` with tie_up 0.35) on a 2-D mu map that adds what `render` lacks:
- ellipticity e in {0, 0.008} (k_x / k_y = 1.008 measured) at a random axis; centre known to the estimator except a
  random 0.5-px error;
- an m = 1 one-sided pedestal mu_ped = P0 mu(s0) (s / s0)^1 max(0, cos(phi - phi_b)) for s > 0.3, P0 in {0, 0.15}
  (the BAR ADJACENT +15-37 % high-angle pedestal, smooth and ring-free);
- the beam stop and arm: the real session mask of s0047f887 (bin 8; nearest-neighbour resampled for bin 4 / 16) plus
  the seam lines;
- the steep inner halo is the model's own (mu ~ r^-2 to r^-3 inside 150 bin-8 px).
Ladder: bin 4 (2048^2), 8 (1024^2), 16 (512^2) of the same physical frame; dose so that mu at r = 300 bin-8 px is
{0.8, 2.4, 8, 24} e per bin-8 px (stored b there: ~0.03 / 0.3 / 1.6 / 5.6 counts at bin 8). Spots: per frame 7 P/B
{0.03, 0.1, 0.3, 1, 3, 10, 30} x 5 d bins: D0 s 0.13-0.25 (the steep inner halo, added here), D4 0.25-0.50, D3
0.50-0.80, D2 0.80-1.00, D1 1.00-1.163 (bin-8 s = 0.0022811 (r + 0.5), scaled by binning), sigma from
`peak_recovery_v0/spot_shape_v0.csv` (bin-8 px, x 8/b), injected with `podbg.peaks.inject` from the KNOWN host
electrons (the latent posterior is not needed). Spots are 24 bin-8 px apart and off the mask. Replicates: 6 frames per
cell (bin x dose x {plain, ellipse + pedestal}) = 2 x 3 x 4 x 6 = 144 frames, 5,040 spots. Seeds: dev 0-99 (debugging
only), test 10,000+ (run once). Spot-free twins of the test frames (same seeds without spots) measure truncation bias.
Reported per method (O oracle, G, GL, FLAT, PLANE, ROBUST): beta (mean and SE), I / lambda_box and I / lambda_eff
(lambda_eff = the law-expected oracle intensity, as peak recovery v0), pulls, timing.

REAL (stage 4; injected-peak method). The peak-recovery v0 harness is committed (`podbg.peaks`, 8715367; the script
9d7c18f), so its injection code is reused read-only: `peaks.spot_pixels`, `peaks.inject` (latent posterior, dead time,
stored-count law) and its host frames (`~/pod_data/background/peak_recovery_v0/hosts/`: F clean film, BA bar-adjacent
0-5 um, CN crystal-neighbour No frames, 20 held-out fly-scan sessions, bin 8) and the cached classic raw-count bin-4 No
frames of s level bin4 v0 (sessions with >= 8 frames, 6 hosts each, mask `classic_v0.bad_mask_for`, centre from
`classic_profiles_v0.pkl`). Peak placement and P/B as peak recovery v0 Sec. 1.2 (D1-D4, plus D0 where the frame allows),
2 replicates per host, seeds fixed per (sid, host, rep) but different from peak recovery's. Truth: the oracle K0.
Hold-out: DEV sessions s0047f887, s4fead7c7, sc19f33b9 (fly-scan) and the first 4 classic sessions (sorted sid) with >=
8 frames; TEST = all the others, run once after the dev run is frozen.

### 1.9 Metrics and decision rules (pre-registered)

Per method x regime x d bin x P/B cell: n; mean beta and its SE (cluster bootstrap over hosts, 400 resamples); the
background error in units of the box noise, rmse_bg = sqrt(mean((I - I_O)^2)) / sqrt(mean(epc^2 sum_box g(b))); median
and mean of I / lambda; pull = (I - lambda_eff) / sigma_I: mean, SD, robust SD (1.4826 MAD); fraction I <= 0. MEANS are
the bias statistics (a median of skewed low-count sums is itself biased, A134); medians are reported for comparability
with peak recovery v0.

- R1 SYNTHETIC UNBIASEDNESS: G passes iff in every (binning x dose x d bin x {plain, structured}) cell, pooled over P/B,
  |mean beta| <= max(0.005, 2 SE). FLAT / PLANE are expected to fail at D0 with beta close to Sec. 1.2 (a check of the
  derivation: measured / predicted within 0.5-2).
- R2 COUNT STATISTICS / TRUNCATION: on the spot-free twins, per binning x dose, the pixel-pooled bias of G
  |mean(b^ - b)| / mean(b) <= max(0.003, 2 SE) in each of four radial bands, and the masking-on minus masking-off
  difference <= 0.001 of b. (If not, the Sec. 1.5 correction is switched on and the change reported.)
- R3 SIGMA HONESTY: synthetic, per binning, pooled over d bins and doses, separately P/B <= 1 and P/B >= 10: pull robust
  SD in [0.9, 1.1] and |pull mean| <= 0.1. Real: robust SD in [0.85, 1.15], |mean| <= 0.25.
- R4 REAL, BETTER THAN FLAT (test set): per regime (F, BA, CN, classic) x d bin, pooled over P/B: G's |mean beta| <=
  max(0.01, 2 SE) in >= 90 % of cells, AND rmse_bg(G) < rmse_bg(FLAT) in every cell, AND G's |mean beta| <= FLAT's
  |mean beta| + 2 SE in every cell.
- R5 SPEED: median fit + map time for 1024^2 (numba) <= 50 ms on this Mac.
- D1 (dev set only, decides the recommended mode before the test run): GL replaces G as the recommendation in a regime
  iff on the DEV real frames G's |mean beta| > max(0.01, 2 SE) in that regime and GL's is not. The test run then
  reports both.
- VERDICT: bgsub v0 is "valid" iff R1, R2, R3 (synthetic) and R5 pass and R4 passes in F and classic; BA and CN are
  reported (structure near bars and crystals may violate C1; GL is there for them). Any failure is reported with its
  cells; no parameter is changed after the test run.

(Pre-registration ends here; everything below was produced afterwards.)

### 1.10 Amendment 1 (2026-11-04, before any evaluation run; prompted by XDS merge v0, f8cdff3 / 748ea3e, A184-A187)

What changed and why: XDS has no input for an external background or a per-pixel variance (it takes variance = GAIN x
value), feeding it background-subtracted or pedestal-shifted images degrades the data and the cause is the zero level
alone (A186, A187); the archive's Apollo TIFFs are MRC + 1 and XDS reads the pedestal as signal with a wrong gain (A184).
Therefore:
- (a) bgsub ships its OWN integration (`podbg.bgsub.integrate`), never modified images: SUMMATION (Sec. 1.3.2 sigma) and
  PROFILE FITTING (method G-PF): with the background B_p and the count-law variance fixed, the spot amplitude is the
  weighted least-squares fit of K_p - B_p = I f_p / epc over the box with a known normalised profile f_p (pixel-integrated
  Gaussian of a given sigma, as `peaks.spot_pixels`), weights 1 / (phi g(B_p + I^0 f_p / epc)) from the summation
  estimate I^0 (one pass; weights from a pilot keep the estimator linear in K given the weights, as Sec. 1.3.1).
  Its variance: epc^2 / sum(f^2 w) + the background term (s_f^T Cov s_f, s_f = sum_box f_p x_p / sum f^2 w ...
  computed exactly from the same linear form). Profile fitting is evaluated with the TRUE injected sigma (an
  upper bound on what a profile learned from strong spots gives) and reported separately from summation; it does not
  enter the R-rules, which are evaluated on summation.
- (b) Zero level and gain are DECLARED, never inferred: `CountLaw` carries `offset` (stored value of zero electrons,
  e.g. 1 for the lab's MRC -> TIFF conversion, 0 for MRC) and the law itself fixes the gain (epc). The estimator works
  on K = frame - offset; a test checks that a frame + 1 with offset 1 gives identical B - offset, I and sigma, and that
  omitting the declared offset changes them (it is not silently absorbed).
- (c) The baseline to beat is the high-resolution intensity bias at sparse counts (A186): the P/B <= 1 cells at D1-D2
  and the lowest two doses (0.05-2 counts per bin-8 px) are tabulated first, for every method.
No decision rule, threshold or default of Sec. 1.1-1.9 changes.

### 1.11 Amendment 2 (2026-11-04, implementation, before any evaluation run)

Found with the ANALYTIC unit tests only (`tests/test_bgsub.py`; no forward-model or archive frame had been fitted):
- (a) Cells cut by the mask or by a spot's exclusion disc have a mean radius different from the ring's, and in the steep
  halo that biased the fine radial part by up to ~1 % next to an excluded spot (unit test: a spot at r = 22 px in an
  r^-2 halo came back 6 % low at P/B 0.14). Fix: the per-cell mean radius of the USED pixels is accumulated with the
  counts, and the fine radial columns are B_k(r_i) + B_k'(r_i) delta_c (first order in delta_c = cell mean radius -
  ring radius; exact assembly verified against a brute-force design to 1e-15). Remaining within-cell term: b'' / 24 per
  1-px ring (8.6e-4 relative at r = 17 px in r^-2, 1e-4 at r = 50).
- (b) Weights: from the OTHER half's m = 0 (radial) part only, g(max(R0, floor)), instead of the full model; the full
  model can go negative in data-poor frame corners, and the 1e-3 floor then gave huge weights. Weights still never
  depend on the data they weight.
- (c) A ridge RIDGE_HARM = 1e-3 x the block's mean information on the harmonic coefficients, in addition to the
  difference penalty: with only four corner arcs per outer ring, 12 trig functions were ill-determined and the Pearson
  dispersion reached 2-6 on pure Poisson frames. The ridge shrinks harmonics only where local information is << the
  block mean (corners); elsewhere the shrinkage is ~1e-3 of the harmonic amplitude (~1e-5 of b).
- (d) Pearson dispersion: g floored at g(0.5 x the ring's data mean), so that an (unbiased) negative fitted value in a
  corner cell does not dominate phi.
- (e) The loop is exactly as pre-registered (OLS pilot fit, then 2 rounds of flag + weighted cross-fit; 3 fits). The
  ring-wise zinger clip before the pilot fit is kept.
Speed after these changes (analytic 1024^2 frame, numba, machine load ~5): 41 ms median.

### 1.12 Amendment 3 (2026-11-04, before any evaluation run; prompted by peak recovery v0, eb59d13, A173-A177)

Peak recovery v0 found that at bin 8 even the exact (oracle) background recovers 0.00 / 0.55 / 0.74 of the injected box
intensity at P/B 1 for d 0.86-1.0 / 1.0-1.25 / 1.25-2 A, and 0.89 at P/B 30 (A173): the stored-count rounding and dead
time act on the PEAK, so a summation I = epc sum(K - B) has expectation lambda_eff(lambda), not lambda; pulls about the
truth have SD 1.7-2.3 (A176); radial backgrounds miss the local excess near bars / crystals (+15-79 % at d < 1.25 A,
local annulus +3-11.5 %); an annulus on all-zero backgrounds is overconfident (A177). Added before evaluating:

- (a) COUNTING-LAW-AWARE INTEGRATION. The background is converted to electrons per pixel, mu_p = E^-1(B_p) (E = the
  stored mean of the law without dead time, as the host electrons are already post-dead-time), and the spot is a known
  normalised profile f_p (pixel-integrated Gaussian, sigma given; summed over the box = 1 for the box intensity). For a
  box intensity lambda the law-expected stored excess is (peak recovery v0's lam_eff, as a FUNCTION of lambda)
      L(lambda) = epc sum_box [E(mu_p e^{-lambda_p tau} + lambda_p e^{-(mu_p + lambda_p) tau}) - E(mu_p)], lambda_p = lambda f_p,
  tau the DECLARED dead time per output pixel per stored frame (`CountLaw.apollo(..., dead_time_tau=)`; 0 = none).
  * method X-LC (law-corrected summation, for every background X): lambda^ = L^-1(I_X), the method-of-moments
    estimate (L is monotone up to the dead-time peak; solved by bisection on [L^-1 domain]); sigma = sigma_X / L'(lambda^).
    Unbiased to first order whenever I_X is unbiased for L(lambda); where L' -> 0 (signal below the rounding threshold)
    the sigma correctly diverges instead of the intensity silently vanishing.
  * method X-ML (profile likelihood under the stored-count law): lambda^ = argmax sum_box log P(K_p | mu_p e^{-lambda_p
    tau} + lambda_p e^{-(mu_p + lambda_p) tau}), P the recorded PMF of `podbg.binning` (bin 4: Poisson), on a lambda
    grid plus parabolic refinement, lambda allowed negative (mu_p + lambda_p >= 0); sigma^2 = 1 / (EXPECTED Fisher
    information) + (d lambda^ / d B)^2 Var(B_box). The expected information is used, not the observed one, so an
    all-zero box keeps a finite, honest sigma (A177).
  The truth for the -LC and -ML methods is lambda_box (pulls about lambda_box); for raw summation it remains lambda_eff.
- (b) LOCAL TERM WITH A PRINCIPLED SWITCH (method AUTO): per frame, the GL local residual L_j = mean_ann(K - G) is
  computed on probe annuli (a grid of positions at the spot-box scale, at least r_excl away from the mask edge and from
  all excluded / flagged discs) together with its model variance s_j^2 = phi g(G) / n_ann. Under C1 the statistic
  Q = sum_j L_j^2 / s_j^2 is chi^2 with n_probe degrees of freedom. If Q's upper tail p < 0.01, the frame has local
  structure at the annulus scale and AUTO = GL for every spot of that frame; otherwise AUTO = G. The decision uses
  probes away from the integrated spots, so it does not depend on the spots' own boxes. GL stays reported.
- (c) ZERO-COUNT VARIANCE: G's background variance is a model variance (never a plug-in of zeros); GL's annulus term uses
  phi g(G_p), not g(K_p); the reference FLAT is reported with both its plug-in sigma (harness form) and the model form.
- (d) Synthetic pedestal (correction to Sec. 1.8 before the test run): the pre-registered form had a step at s = 0.3
  and the film's rings; it is replaced by 0.15 x (the film profile smoothed with a 60-bin-8-px Gaussian: ring-free) x a
  linear ramp from s 0.3 to 0.6 x max(0, cos(phi - phi_b)). The ellipticity is applied to the film only.
- (e) Real-frame evaluation is against the oracle K0 and against lambda_box / lambda_eff, at bin 8 (F, BA, CN) and bin 4
  (classic), with the harness's host frames.
- Rules: R1, R2 and R4 are unchanged (they are background rules). R3 is applied to X-LC and X-ML about lambda_box with
  the same bounds, and to raw summation about lambda_eff. A new reported rule R6 (not part of the verdict, because it
  was added after the harness result): G-LC recovers lambda_box within 5 % (|mean ratio - 1|) in every synthetic cell
  with P/B >= 1 at bin 4 and bin 8, D2-D4.

### 1.13 Amendment 4 (2026-11-04, from the synthetic DEV seeds only, before the test run)

The dev run (seeds 0-99; 1-2 replicates per cell) was used to debug the pipeline, as pre-registered. Changes, all
fixes of defects rather than tuning to a score:
- (a) Pearson dispersion: only cells with >= 5 expected stored counts enter (DISP_MIN_COUNTS; fallback phi = 1 with
  < 200 such cells). With rare K = 1 events (bin 16 at the low doses) the Pearson mean is heavy-tailed and its sample
  value sat at 0.6-0.8 although g is exact (checked: simulated Var(K) = g(E K) to 0.2 % at bin 8 / 16).
- (b) The Pearson floor is applied in b (0.5 x the radial model), not in g: at bin 16 g(b) is NOT monotone
  (0.25 at b 0.5, 0.06 at b 1.0: the stored value is nearly deterministic near integer means), so g(0.5 b) could
  exceed g(b) and depress phi to 0.92 at high counts.
- (c) Counting-law-aware intensities: at the top of the P/B ladder many injected spots pass the paralyzable dead-time
  maximum (peak pixels of 10^3-10^4 e per frame at bin 4-16; the stored excess then FALLS with intensity), so
  L(lambda) = I has two roots. X-ML now starts from a global log-spaced pass (picks the likelihood's global maximum,
  i.e. the branch, from the profile shape) and X-LC solves on the branch that contains the ML estimate.
- (d) Tables: a cell's beta is the ratio of sums sum(B - b) / sum(b) (cluster-bootstrap SE of the same ratio): at
  bin 16 and the low doses the stored background of a high-angle box is ~0 and a mean of per-peak ratios is undefined.
  Added the derivation check columns beta_flat_truth (FLAT applied to the noise-free truth) and beta_flat_pred
  ((1/4) lap(b) (<rho^2>_ann - <rho^2>_box) from the numerical Laplacian): on dev they agree to 0.0001-0.003.
- Observed on dev (not used to change anything): the synthetic film is smooth at the box scale, so the curvature bias
  of FLAT on the truth is only -0.3 to +0.4 % at bin 4 / 8; FLAT's measured bias is larger because of the spot's own
  wings in the 5-9 px annulus (sigma up to 2 px); ROBUST with the Poisson consistency correction is +4-9 % at bin 8.

## 2. Synthetic validation (stage 3; test seeds 10000+, run once; 144 frames, 5,040 spots, 7 P/B x 5 d bins)

Tables: `~/pod_data/background/bgsub_v0/synth_test_{cells,cells_pb,pulls,trunc}_v0.csv`, rules
`synth_test_rules_v0.json`; the re-run of the SAME seeds after the post-hoc ML-sigma fix (Sec. 2.4) is `synth_testb_*`.
Figures: `notes/figures/bgsub_v0/synth_background_bias.png`, `intensity_recovery_vs_pb.png`.

### 2.1 Background under the box (beta = sum(B - b) / sum(b), %, pooled over dose and structure; 2 SE ~ 0.1 % for G)

| bin | method | D0 4-8 A | D4 2-4 A | D3 1.25-2 A | D2 1-1.25 A | D1 0.86-1 A |
|---|---|---|---|---|---|---|
| 4 | G | +0.04 | -0.00 | -0.05 | -0.02 | +0.01 |
| 4 | FLAT | -0.14 | +0.21 | +0.55 | +0.40 | +0.22 |
| 4 | PLANE | -0.14 | +0.20 | +0.55 | +0.40 | +0.22 |
| 4 | ROBUST | -0.21 | +0.12 | +0.57 | +0.40 | +0.23 |
| 8 | G | -0.03 | -0.07 | +0.11 | -0.03 | -0.05 |
| 8 | FLAT | -0.15 | +0.15 | +0.52 | -0.06 | +0.20 |
| 8 | PLANE | -0.14 | +0.15 | +0.51 | -0.03 | +0.20 |
| 8 | ROBUST | +0.45 | +1.22 | +2.60 | +2.87 | +5.96 |
| 16 | G | -0.03 | +0.03 | +0.01 | -0.00 | -0.01 |
| 16 | FLAT | -0.27 | +0.40 | +0.33 | -0.06 | +0.39 |
| 16 | ROBUST | +0.36 | +1.50 | +2.42 | +2.72 | +5.93 |

G is unbiased to <= 0.11 % in every pooled cell; G without the spot positions (G_blind, flagging only) is within
0.2 % (worst -0.18 % at D0, bin 8). FLAT / PLANE carry +0.2-0.55 %: the curvature term of Sec. 1.2 plus the spot's own
wings in the 5-9 px annulus. ROBUST (Huber with the POISSON consistency correction) is +1-6 % at bin 8 / 16, where the
stored counts are not Poisson: the predicted distribution bias. Derivation check (FLAT on the noise-free truth vs the
second-order prediction, mean over spots, %): bin 4 -0.31/-0.31, +0.22/+0.22, -0.06/-0.06, +0.24/+0.25, +0.10/+0.10
(D0, D1, D2, D3, D4); bin 8 within 0.02; bin 16 within 0.1 (D1 2.90 vs 2.83): the derivation holds.

Background error in units of the box's own count noise (rmse_bg): G 0.07-0.18, FLAT / PLANE 0.40-0.83, ROBUST
0.45-0.90, GL 0.38-0.81. G's background adds 0.5-3 % to the variance of a weak spot; a local annulus adds 16-70 %.

### 2.2 Rules R1, R2

- R1 (per binning x dose x structure x d-bin cell, |beta_G| <= max(0.5 %, 2 SE)): FAIL, 7 of 120 cells; 5 at bin 8
  dose 0.8 (beta -0.59 to -3.8 %, SE 0.1-1.8 %, cluster bootstrap over only 6 frames), one at bin 4 dose 0.8
  (-0.71 +/- 0.18 %), one at bin 16 dose 2.4 (+0.63 +/- 0.23 %). POST HOC (24 new frames at bin 8 dose 0.8, seeds
  20000+): plain D0-D4 -0.19 +/- 0.07, +0.50 +/- 0.89, -0.03 +/- 0.54, -0.30 +/- 0.48, -0.10 +/- 0.14 %; structured
  -0.07 +/- 0.16, +1.19 +/- 1.26, -0.16 +/- 0.47, +0.77 +/- 0.27, -0.30 +/- 0.14 %: no persistent bias (A193).
- R2 (spot-free twins, pixel-pooled bias per radial band <= max(0.3 %, 2 SE); masking on - off <= 0.1 %): PASS,
  48 / 48 cells. Analytic truncation at alpha 1e-6: (T + b P_tail) / b <= 5e-5 at bin 4, <= 2e-5 at bin 8, <= 1.2e-6
  at bin 16 over b 0.01-10 stored counts; the Sec. 1.5 correction was not needed.

### 2.3 Counting-law-aware intensities (synthetic; I / lambda_box as ratio of sums, d 1-4 A, all doses)

| bin | method | P/B 0.3 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|
| 4 | G summation | 0.84 | 0.80 | 0.69 | 0.52 | 0.34 |
| 4 | G-ML | 0.97 | 1.00 | 1.00 | 1.01 | 1.00 |
| 8 | oracle summation | 0.87 | 0.79 | 0.70 | 0.53 | 0.33 |
| 8 | G summation | 0.86 | 0.80 | 0.71 | 0.53 | 0.33 |
| 8 | G-LC | 1.00 | 1.00 | 1.03 | 1.03 | 1.23 |
| 8 | G-ML | 1.02 | 1.00 | 1.02 | 1.00 | 1.00 |
| 16 | G-ML | 0.70 | 0.89 | 1.03 | 1.00 | 1.00 |

Summation with the EXACT background loses 13-67 % to the rounding thresholds and the dead time (A173 reproduced);
G-ML recovers the injected box intensity within 0-3 % for P/B >= 0.3 at bin 4 / 8 and P/B >= 3 at bin 16. At P/B
0.03-0.1 the means are noise-dominated. G-LC fails beyond the paralyzable maximum (R6 FAIL, 39 cells).

### 2.4 Sigma honesty (R3; pull robust SD / mean, P/B <= 1 and >= 10)

| bin | raw G vs lambda_eff, low | high | G-ML, low | high | G-ML after fix, low | high |
|---|---|---|---|---|---|---|
| 4 | 1.01 / -0.06 | 0.99 / +0.04 | 0.99 / -0.18 | 0.72 / +0.01 | 1.01 / -0.20 | 0.90 / -0.01 |
| 8 | 0.97 / -0.07 | 0.92 / -0.03 | 0.97 / -0.09 | 0.97 / -0.01 | 0.99 / -0.09 | 1.12 / -0.00 |
| 16 | 0.96 / -0.18 | 0.77 / -0.06 | 0.85 / -0.01 | 0.82 / -0.75 | 0.86 / -0.05 | 0.87 / -0.75 |

R3 FAIL by the rule (bin 16; bin-4 G-ML at P/B >= 10 and |mean| 0.18-0.20 at low P/B). Raw G summation about
lambda_eff is honest at bin 4 and bin 8. POST HOC fix (after the synthetic test run, before the real test run): the
ML background term used the summation sensitivity 1 / L', which diverges past the dead-time maximum; it now re-solves
the ML with a shifted background (f1de599). Re-running the same seeds: bin-4 high P/B 0.72 -> 0.90, bin 8 0.99 / 1.12.
Bin 16 stays dishonest (a few saturated spots on the wrong dead-time branch; g oscillating, A196).

## 3. Real-frame validation (stage 4; injected peaks, test set run once, after the ML-sigma fix)

Hosts: peak recovery v0's cached frames; TEST = 17 fly-scan sessions (F 193 / BA 198 / CN 177 host frames x 2
replicates, bin 8) and 20 classic bin-4 sessions (120 hosts x 2); DEV (3 + 4 sessions) used only for debugging and
rule D1. Injection: `podbg.peaks.inject` (latent host electrons from the posterior, dead time, stored-count law), 35
spots per host (7 P/B x D0-D4). Truth: the oracle K0 and lambda_box / lambda_eff. 880,740 method-spot rows
(`real_test_peaks_v0.parquet`), tables `real_test_{cells,cells_pb,pulls}_v0.csv`, rules `real_test_rules_v0.json`,
figure `notes/figures/bgsub_v0/real_background_bias.png`.

D1 (dev): G's |beta| <= max(1 %, 2 SE) in every regime (F -0.31, BA +0.11, CN -0.66, classic +0.12 %): GL does NOT
replace G anywhere; G stays the recommendation.

### 3.1 Background vs the oracle: beta % (SE) / background error in box-noise units

| regime | method | D0 4-8 A | D4 2-4 A | D3 1.25-2 A | D2 1-1.25 A | D1 0.86-1 A |
|---|---|---|---|---|---|---|
| F | G | -0.70 (0.38) / 0.88 | +0.02 (0.81) / 0.14 | -1.33 (2.1) / 0.00 | -0.69 (3.8) / 0.22 | -2.95 (8.0) / 0.10 |
| F | FLAT | +0.21 (0.30) / 0.39 | +1.87 (0.77) / 0.38 | +0.30 (2.1) / 0.33 | +2.22 (4.3) / 0.48 | +0.55 (8.9) / 0.41 |
| F | ROBUST | +8.3 / 0.74 | +12.1 / 0.63 | +4.1 / 0.39 | +3.4 / 0.47 | +1.1 / 0.41 |
| BA | G | -0.58 (0.38) / 0.41 | -0.43 (0.97) / 0.33 | +0.03 (2.3) / 0.37 | +3.42 (3.7) / 0.12 | +2.01 (5.8) / 0.27 |
| BA | FLAT | +0.16 / 0.51 | +1.65 / 0.55 | +2.61 / 0.52 | +4.05 / 0.37 | +6.01 / 0.53 |
| BA | GL | +0.11 / 0.50 | +0.95 / 0.56 | +2.15 / 0.52 | +4.54 / 0.37 | +5.44 / 0.51 |
| CN | G | -0.98 (0.44) / 1.18 | +0.41 (0.60) / 0.43 | +0.44 (1.7) / 0.29 | -1.68 (2.0) / 0.00 | +2.60 (4.7) / 0.39 |
| CN | FLAT | +0.43 / 0.59 | +1.81 / 0.55 | +1.74 / 0.58 | -1.32 / 0.31 | +2.81 / 0.63 |
| classic | G | -1.51 (0.63) / 7.5 | -0.13 (0.15) / 0.64 | +0.04 (0.24) / 0.29 | -0.11 (0.30) / 0.36 | -0.41 (0.46) / 0.34 |
| classic | FLAT | +0.48 / 6.0 | +0.51 / 0.86 | +0.70 / 0.67 | +0.06 / 0.65 | +0.23 / 0.51 |

(error = sqrt(mean (B - K0)^2 - mean sum_box g) / sqrt(mean sum_box g); 0.00 = not resolved from the box noise.)
At d < 4 A (D4-D1) G's background error is 1.5-5x smaller than the local annulus' in every regime, bars and crystals
included, and its beta is within 2 SE everywhere (the SEs at high resolution are large: the oracle's own count noise
at 0.05-0.3 counts per pixel). The harness's finding that RADIAL backgrounds miss the bar / crystal excess (+15-79 %)
does not carry over to G, whose harmonics absorb the m = 1 pedestal: G's high-resolution beta in BA is +2.0 / +3.4 %
+/- 4-6 %, FLAT's +4.1 / +6.0 %. ROBUST is +1 to +12 % biased on fly-scan frames.

D0 (s 0.13-0.25, r 57-110 bin-8 px, next to the beam stop) is G's failure: beta -0.6 to -1.5 % and, in the first two
s sub-bins (s 0.13-0.19), an error 1.2-2.4x the box noise (F 1.46 / 1.20 vs FLAT 0.32 / 0.37; CN 2.37 / 1.20 vs 0.67 /
0.65); beyond s 0.19 G and FLAT are equal. Real frames have azimuthal structure at r < ~85 px that six harmonics on a
48-px radial spline do not follow (A194). In classic D0 every method fails (error 6-13x the box noise; A195).

R4: BA PASS; F, CN, classic FAIL, each only through D0 (rmse clause; in classic also D0's beta -1.51 +/- 0.63 %).
Without D0 all four regimes pass every R4 clause.

### 3.2 Intensities on real frames (ratio of sums I / lambda_box)

Sparse high resolution first (Amendment 1 c; d 0.86-1.25 A, bin-8 hosts at 0.05-0.3 counts per pixel):

| regime | method | P/B 0.1 | 0.3 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|---|
| F | oracle summation | 0.35 | 0.43 | 0.58 | 0.78 | 0.92 | 0.89 |
| F | FLAT summation | 0.29 | 0.36 | 0.59 | 0.78 | 0.91 | 0.89 |
| F | G summation | 0.28 | 0.41 | 0.59 | 0.78 | 0.91 | 0.89 |
| F | G-ML | -0.49 | 0.30 | 0.75 | 0.94 | 1.01 | 1.00 |
| F | oracle-LC | 0.71 | 0.75 | 0.93 | 1.03 | 1.05 | 1.01 |
| BA | G-ML | -0.56 | 0.30 | 0.69 | 0.91 | 0.99 | 1.00 |
| CN | G-ML | -0.13 | 0.59 | 0.86 | 0.96 | 0.99 | 1.00 |
| classic | G summation | 1.03 | 1.04 | 0.99 | 1.00 | 1.00 | 1.00 |
| classic | G-ML | 1.04 | 1.05 | 0.99 | 1.00 | 1.00 | 1.00 |

At d 1.25-4 A G-ML gives 0.98-1.01 for P/B >= 1 in F, BA and CN, where summation (any background, the oracle included)
gives 0.58-0.97. At bin 4 summation and ML agree within 1 % for P/B >= 0.3. At bin 8 and d < 1.25 A the background no
longer limits: G, FLAT and the oracle give the same summation intensity (A199). G-ML removes most of the loss for P/B
>= 3 but is biased LOW at P/B <= 1 in the sparse regime (A197). No estimator tested is both unbiased and precise
below P/B 1 at bin 8, d < 1.25 A.

Sigma honesty on real frames (R3: robust SD in [0.85, 1.15], |mean| <= 0.25): raw G summation about lambda_eff honest
at bin 4 (1.03 / 1.06), too wide on fly-scan at P/B >= 10 (0.67-0.70; A196). G-ML honest at P/B >= 10 in every regime
(0.97-1.08, |mean| <= 0.08) and at bin 4 for P/B <= 1 (1.06 / -0.15), conservative at P/B <= 1 on fly-scan (0.59-0.78).
R3 FAIL by the rule. The A177 problem (annulus on all-zero backgrounds overconfident) does not occur with G or G-ML:
no sigma uses a plug-in of zero counts.

## 4. Speed (R5)

`~/pod_data/background/bgsub_v0/speed_v0.json`; fit + background map, plan cached, this Mac under a load average of ~5
from parallel agents:

| frame | numba | numpy only | plan (once per geometry + mask) |
|---|---|---|---|
| 512^2 | 15.5 ms | 31 ms | 22 ms |
| 1024^2 | 42.7 ms | 108 ms | 81 ms |
| 2048^2 | 108 ms | 420 ms | 345 ms |
| 4096^2 | 348 ms | 1,640 ms | 1,510 ms |

R5 PASS (42.7 ms at 1024^2). The variance map is lazy (0.2-1 s on request); box variances are exact quadratic forms.
Integration with ML is extra (two vectorised ML solves per method; ~0.1-0.3 s per frame of 35 spots in the validation
code, which also runs all reference methods). A new centre needs a new plan; a rotation series or a fixed-geometry
raster reuses it.

## 5. Verdict (pre-registered rules)

| rule | result |
|---|---|
| R1 synthetic unbiasedness of G | FAIL, 7 / 120 cells; post-hoc enlargement shows no persistent bias (A193) |
| R2 count statistics / truncation | PASS (48 / 48; analytic truncation <= 5e-5) |
| R3 sigma honesty | FAIL (bin 16; bin-4 ML before the fix; fly-scan summation; low-P/B ML conservative) |
| R4 real, better than FLAT | PASS BA; FAIL F, CN, classic, only through D0 next to the beam stop (A194) |
| R5 speed | PASS (42.7 ms at 1024^2) |
| R6 (reported) G-LC within 5 % | FAIL (dead-time branch) |

**bgsub v0 is NOT "valid" by the pre-registered verdict.** What the evidence supports: (i) at d < 4 A the global
model's background under spots is unbiased within the measurement precision on synthetic and real frames, with a
background error 1.5-5x smaller than any local annulus, plane or robust estimate, and the robust (Huber,
Poisson-consistent) estimate is biased +1-12 % wherever counts are not Poisson; (ii) at bin 8 the background is not
what loses intensity, the stored-count law is, and the count-law likelihood (G-ML) recovers it for P/B >= 3 (0.91-1.01
at d 0.86-1.25 A, 0.98-1.01 at d 1.25-4 A on real frames) with honest sigmas for bright spots; (iii) next to the beam
stop G is worse than a flat annulus; (iv) below P/B 1 at bin 8, d < 1.25 A nothing tested is unbiased. No parameter was
changed after the test runs except the ML-sigma fix of Sec. 2.4, reported with before / after.

## 6. Limits and next steps

- C1 fails next to the beam stop (r < ~85 bin-8 px): add stop-relative basis functions or a finer radial spline with
  more harmonics inside ~2 R_stop, or use GL only there; D0 cells are the decision cells.
- AUTO's chi^2 switch rejects on 60-93 % of real frames (n_probe ~10^3 gives power against tiny misfits): needs an
  effect-size criterion (A198).
- Summation sigma: evaluate g at the profile-model expectation instead of the first-order V_K (A196).
- ML at sparse bin-8 counts: biased at P/B <= 1 (A197); a bias-corrected or Bayesian intensity is the next step.
- Bin 16: g oscillates with b; intensities out of scope.
- Real-data truth is the oracle, so high-resolution beta carries 4-9 % SE: bounded, not measured, below ~5 %.

## 7. Anomalies

- A193: R1 failed in 7 / 120 synthetic cells (5 at bin 8, dose 0.8, beta -0.6 to -3.8 %) but 24 new frames at that
  setting show no bias; a 2-SE rule with a 6-cluster bootstrap over d bins correlated through one global fit has a
  false-fail rate near what was observed. Rules for global estimators should test per frame.
- A194: real frames carry azimuthal structure next to the beam stop (s 0.13-0.19, r 57-83 bin-8 px) that G (M = 6,
  H = 48 px) does not follow: error 1.2-2.4x the box noise vs 0.3-0.7 for the flat annulus, beta -0.7 to -1.9 %; the
  synthetic frames (smooth halo, hard-edged real mask) did not show it.
- A195: classic bin-4 D0 (s 0.19-0.25) has background errors 6-13x the box noise for EVERY method: the classic "No"
  frames carry non-smooth structure there (sample-like scattering, cf. A139?).
- A196: the first-order box variance sum[g(B) + g'(B)(K - B)] overestimates bright-spot variance on sparse bin-8
  backgrounds (pull robust SD 0.67-0.70) because g' at the background (~0.9) is far from g' at the spot pixels (0.25);
  at bin 16 g(b) oscillates between 0.06 and 0.27 with a period of 1 count.
- A197: the stored-count ML is biased low at P/B <= 1 on real sparse bin-8 hosts (0.30 and 0.69-0.86 at P/B 0.3 and 1,
  d 0.86-1.25 A), more than on synthetic hosts (1.00-1.02 at d 1-4 A): weak spots leave few stored counts above the
  3-e thresholds, the likelihood is flat and the background-to-electron inversion is steep below 1 e.
- A198: AUTO switches to GL on 60-93 % of real frames (0-12 % of synthetic) although G's beta is within 2 SE outside D0.
- A199: on real fly-scan hosts the background method is irrelevant for d < 1.25 A summation intensities (oracle, FLAT
  and G within 0.01-0.05 at every P/B): the loss there is the counting law alone, as A173.

A short user README is `src/podbg/bgsub/README.md`.
