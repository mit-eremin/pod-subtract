# bgsub (moved from podbg.bgsub on 2026-11-05): background and intensities for single diffraction frames (v1 default; v2 and v2.1 opt-in 2026-11-05; v0 behind `version="v0"`)

> Moved verbatim from pod/background `src/podbg/bgsub/README.md` at background commit f1e93f5. Report paths below
> (`notes/reports/...`) are copied into this repo's `notes/reports/`. SCOPE (owner, 2026-11-05): the product is the
> background estimate and its variance; the integrators, -ML outputs and intervals described here are kept but are
> not product criteria. See ../../CLAUDE.md.

Design, derivations, validation and verdict: `notes/reports/2026-11-04_bgsub_algorithm_v1.md` (v1, default) and
`notes/reports/2026-11-04_bgsub_algorithm_v0.md` (v0). `version="v0"` (in `make_plan`, `estimate_background`,
`integrate_spots`) reproduces v0 to floating-point order (regression test against outputs frozen at 07a06b9).

## What it does

`estimate_background` fits ONE frame with a smooth global model in stored counts: a fine radial cubic B-spline (rings,
steep inner halo), azimuthal harmonics m <= 6 on a coarse radial spline (ellipticity, the bar-side m = 1 pedestal,
tilts), four centre / ellipticity derivative columns and (v1) PENUMBRA columns next to the beam stop: the shadow
transmission of the stop's disc and of the two sides of its arm on d = 1-4 px from the stop, times the radial profile
of the other half. The fit is linear least squares on polar-cell sufficient statistics, weighted by the detector's
count-law variance taken from the OTHER checkerboard half (cross-fitting), with spots, hot pixels and zingers excluded
by one-sided count-law tests (alpha 1e-6, per pixel and per 3 x 3 window). Because the estimator is linear in the counts
and the weights never depend on the data they weight, the background under a spot is unbiased for ANY count law
whenever the truth is in the model's span; no median, no Poisson assumption. Sandwich variances with the Pearson
dispersion of the frame.

`integrate_spots` integrates given spot positions with that background: summation (G), global + local residual (GL),
AUTO (v1: per radial zone of the frame, GL replaces G only where its estimated expected squared error is smaller,
D = mean Lb^2 - mean (La - Lb)^2 > 0 on spot-free probes; v0: a chi^2 test), and counting-law-aware intensities: X-ML
(profile likelihood of the stored counts under the rounding law and dead time; recommended at bin 8 / 16) and X-LC.
Every -ML output carries `lam_min5`, the information limit (electrons): the box intensity at which the Cramer-Rao
relative SD is 0.45; above it the ML mean is within 5 % (MC: 2.3 %), below it the ML is biased LOW at bin 8 (sparse
stored counts respond convexly to the spot). `spotlaw.pooled_ml` fits one intensity to several equivalent observations
(pooling before inverting removes most of that bias). Reference local methods (FLAT, PLANE, ROBUST) for comparison.

## v2 (opt-in: `version="v2"`; report `notes/reports/2026-11-05_bgsub_algorithm_v2.md`)

v2 = the v1 fit plus: (1) RE-CENTRING when the centre-error columns imply a beam-centre error above 2 px x N / 1024
(Gauss-Newton re-plan; `res.centre_info`; `recentre=False` to disable); (2) a RING REFINEMENT: a shrunken 1-px radial
correction about the refined centre for radial structure narrower than the spline knots (sharp film rings), with the
shrinkage w = E[rho_A rho_B] / E[rhobar^2] from the two checkerboard halves (`res.ring`; `ring_refine=False`);
(3) AUTO = H2, a CONTINUOUS per-spot weight w* = E[La Lb] / E[La^2] on the local annulus residual, estimated from
spot-free probes at the spot's radius (`out["AUTO"]["auto_w"]`); (4) every summation and -ML output carries `ci_lo`,
`ci_hi`: the 68.3 % interval by inverting |estimate - E(lambda)| <= sigma(lambda) with sigma at the HYPOTHESISED
intensity (no data of the spot in sigma). Evaluation hook: `integrate_spots(..., lam_eval=...)` -> `sigma_at`.

Status: NOT VALID by its pre-registered rules (R1: H2 biased in 2 of 60 synthetic cells, up to -3 % at 0.01 counts per
pixel and +0.7 % across the arm shadow; MSE-optimal is not unbiased). G2 (the v2 global background, `out["G"]`)
passes R1, R2 and R4; H2 has the lowest real background error next to the stop (D0: 0.03-0.46 box noise vs FLAT
0.40-0.60). Use v2's G for unbiasedness and v2's AUTO where the stop zone or local structure dominates; read the
intervals, not `sigma`, for weak spots. Limits: sharp rings are only half removed (ellipticity, flagged ring pixels);
sample Bragg spots in the host are overlaps, not background; integration with intervals costs ~2x v1.

## v2.1 (opt-in: `version="v2.1"`; report `notes/reports/2026-11-05_bgsub_algorithm_v2_1.md`)

v2.1 = v2 with three changes and one speed-up:
(1) AUTO = H2.1: G2.1 by default; v2's continuous weight w* = E[La Lb] / E[La^2] is used only where its signal is
significant, T = S_ab / SE_block(S_ab) > 2 (block standard error over 2 r_out x 2 r_out blocks of the overlapping
probes; `integrate.local_weight_v21`, `out["AUTO"]["auto_w"]`, `["auto_T"]`), and the spot's OWN light is removed from
the annulus residual (the profile tail at the G-ML box intensity, dead time included; needs `sigmas`; also in v2.1's
GL). On a frame without local structure the gate is on for ~2.5 % of spots, so H2.1 is G2.1 there, and its bias is
bounded by sqrt(P(T > 2)) x rms(annulus misfit of G).
(2) The ring refinement runs on ELLIPTICAL rings, r' = |p - c'| (1 + ea cos 2phi' + eb sin 2phi') with (ea, eb) from the
fit's own ellipticity columns (`refine.implied_ellipticity`), with skipped ring means re-centred on G + c (up to 3
re-flag rounds). A free per-ring m = 2 term with robust unflagged means was tried and rejected (it broke the background
next to the stop on real frames).
(3) Intervals: `ci_lo`, `ci_hi` are the IMAGE of the confidence set {lambda : |est - E(lambda)| <= sigma(lambda)}
(v2 took E at the set's end points, wrong where E is not monotone, i.e. past the dead-time maximum); the lambda grid
reaches the dead-time branch; summation methods also return the box-intensity interval `ci_lam_lo`, `ci_lam_hi` and
`ci_open` (the set reaches the grid end). Read `ci_lam_*` for the electrons in the box.
(4) The probe sums run in one compiled pass (`integrate.lw_probes_fast`).

Status: NOT VALID by its pre-registered rules (R1: H2.1 fails 1 of 60 synthetic cells, the sparse bin-8 D1 cell at -2.5 %, where G2.1 itself sits at 3.05 SE; R5: 54.2 ms at 1k vs 50). R2, R3 (pulls and nominal 0.66-0.69 interval coverage) and R4 (F, BA, CN, classic without host overlap) pass. H2.1 has the lowest structured background error of all versions and keeps most of v2's real D0 gain; the sharp 4.6-A ring of one classic session is still not removed (its pixels are flagged). Integration 286 ms per 35 spots (v1 233, v2 428).

## Minimal use

```python
from bgsub import CountLaw, Geometry, estimate_background, integrate_spots, disc_mask, make_plan
law = CountLaw.apollo(binning=8, tie_up=0.35, dead_time_tau=tau_b)   # Apollo stored-count law (bin 4: Poisson)
# v2 / v2.1: make_plan(..., version="v2.1"); then out[m]["ci_lo"], out[m]["ci_hi"] (v2.1 summation: also ci_lam_lo / ci_lam_hi)
# law = CountLaw.poisson(gain)            # any counting / integrating detector; .with_offset(1) for MRC -> TIFF + 1
plan = make_plan(frame.shape, mask, Geometry((cy, cx)))             # v1; cache per geometry and mask (~0.2 s at 1k)
# plan = make_plan(frame.shape, mask, Geometry((cy, cx)), stop=stop_mask)   # give the stop explicitly if you have it
ex = disc_mask(frame.shape, spots, r_excl)                          # spots you will integrate: ALWAYS exclude them
res = estimate_background(frame, mask, Geometry((cy, cx)), law, exclude=ex, plan=plan)
bg, var = res                                                        # stored counts; variance of the estimate
out = integrate_spots(frame, res, spots, r_box, r_in, r_out, r_excl, sigmas=spot_sigmas)
I, sigma = out["AUTO-ML"]["I"], out["AUTO-ML"]["sigma"]            # box intensity in electrons
weak = I < out["AUTO-ML"]["lam_min5"]                                # information-limited at this background
```

The stop is derived from the mask (components of the mask that survive a 5 x 5 opening at 1k and reach within 0.2 N of
the centre); pass `stop=` if seams or other masked regions touch it. Inputs otherwise as v0: the frame, a mask
(True = bad), the beam centre and the count law with its DECLARED zero level and dead time.

## When to use which output

- Background under spots: `AUTO` (v1) or `G`; see the report for the synthetic and real tables.
- Intensities at bin 8 / 16: `AUTO-ML` / `G-ML`; summation recovers lambda_eff, not the electrons. Check `lam_min5`.
  At archive bin-8 sparsity (0.003-0.02 stored counts per pixel at d 1-1.25 A) a P/B-1 spot is below the information
  limit; bin 4 (Poisson, linear response) is unbiased at every dose.
- Summation sigmas (v1, with `sigmas`): the stored-count variance at the profile model's expected values (A196 fixed);
  the v0 first-order form is kept as `sigma_v0form`.

## Limits (v1)

- Structure that is neither smooth on the model's scales nor linear at the annulus scale is not modelled; near-zone
  harmonics (`zone_kw={"M_near": 12}`) exist but add variance without reducing error on the dev data (off).
- Penumbra columns cost variance in the first 1-4 px next to the stop.
- `var_map` (the lazy per-pixel variance map) ignores the penumbra columns; `box_sums` is exact.
- Below `lam_min5` no estimator tested (ML, LC, Firth) is mean-unbiased at bin 8: the limit is information.
- Speed: see the report (R5).
