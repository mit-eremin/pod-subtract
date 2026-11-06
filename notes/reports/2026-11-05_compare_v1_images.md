# compare v1: the three methods on real frames from every binning and all three detectors, with the images (2026-11-05)

Owner (2026-11-05, after compare v0): "it is hard to compare without looking at the actual images; find a dataset at
each binning and from all three detectors, check which one produces how many diffraction peaks and how they compare
visually." This report does that: one real dataset per detector x binning, the same three methods on each, one image
panel per dataset (original with every method's peaks; each method's background; each method's subtracted image with
its own peaks), and the peak counts. Descriptive, not pre-registered; no truth for real peaks. Code
`scripts/compare_v1.py` (uses `compare_v0.pyreyes_detect`, `compare_v0_eval.bgsub_detect`, `superxds_py`). Outputs
`~/pod_data/bgsub/compare_v1/` (frames referenced; private copies under `private/`, never in the repo). Figures
`notes/figures/compare_v1/panel_<dataset>.png` (full frame) and `panel_<dataset>_crop.png` (central 44 %).

## 1. Datasets (all real, all with a beam stop)

| key | detector, binning, size | what | frames used | reference |
|---|---|---|---|---|
| apollo_b4 | Arctica-Apollo bin 4, 2048^2 | rotation movie mov11 of xds_merge_v0 (small-molecule crystal, 1 deg / frame, CL 420), 120 frames | 1, 25, 49, 73, 97 of a 120-frame sweep | archive XDS SPOT.XDS (owner's run) |
| apollo_b8 | Arctica-Apollo bin 8, 1024^2 | fly-scan Good snapshots of s73dc36c8 (49 ms, CL 420), the compare-v0 frames | first 5 Good of a 15-frame pseudo-sweep | - |
| apollo_b16 | Arctica-Apollo bin 16, 512^2 | fly-scan "verify" miniscan of 2026-08-31 (grossular, 0.196 s, CL 420): 55 single frames, ~30 with patterns | 5 frames with patterns, pseudo-sweep of all 55 | - |
| ceta_b2 | Arctica-CETA (CETA-D) binned 2, 2048^2 | rotation movie, grossular, 0.9 deg / frame, CL 960, 112 frames (ringcal sid s209222c0) | 1, 28, 56, 84, 112 of the 112-frame sweep | archive XDS auto_process SPOT.XDS (1837 spots; INIT gain 4.12, background 74.7) |
| tvips_b2 | F30-TVIPS binned 2, 2048^2, 300 kV | rotation movie, 0.5 deg / frame, CL 1320, 101 frames (ringcal sid sbdc16178) | 1, 25, 50, 75, 101 of the 101-frame sweep | archive XDS auto_process SPOT.XDS (268 spots; INIT gain 6.93, background 46.8) |

Where the data came from: Apollo bin 4 and bin 8 from the compare-v0 caches; bin 16 from the only archive session at
binning 16 found in the mdoc mirror (`Apollo_data/FLYSCAN/20260902/...verify-verify_13..23`, 263-kB single-frame MRCs);
CETA-D and TVIPS from `pod/ringcal`'s XDS ground-truth inventory (252 Arctica-CETA movies in 24 sessions, 12 F30-TVIPS
movies in one session; Apollo bin-2 4096^2 movies exist as well and are not used here), fetched by `ssh ... tar` from
`Arctica_data/2025-02-12-reyes` and `F30_data/2025-02-25-reyes` (images/ TIFFs + the auto_process XDS tables).

## 2. How each method was run

- bgsub v2.1, global estimate (G2.1), count-law flags, detector peaks = flag components with component significance
  >= 5 sigma and >= 3 counts excess (compare v0, Sec. 1.1). Count laws: Apollo bin 4 `apollo(4).with_offset(1)`, bin 8
  `apollo(8, tie 0.35, tau_b)`, bin 16 `apollo(16, tie 0.35, tau_b)`; CETA-D and TVIPS have NO calibrated law in bgsub
  yet, so `poisson(gain).with_offset(1)` with the gain ESTIMATED as the Pearson dispersion of a gain-1 fit on the first
  picked frame (declared; XDS INIT's gain reported alongside). Centre: Friedel pairs of the archive SPOT.XDS (apollo_b4,
  ceta_b2, tvips_b2), the features table (bin 8), image symmetry (bin 16). Mask: stop from the frame stack (local mean
  below 25 % of the radial median; the component holding the beam centre; dilated 4), fixed pattern (Apollo), 2-px
  border. Peaks of EVERY method within 3 px of the mask are dropped.
- pyREyes 3.6.3: the exact re-implementation (compare v0, Sec. 1.2), parameters by frame size (2048: sigma 1 / 5, k 10,
  5 px; 1024 and 512: sigma 1 / 3, k 6, 3 px).
- superxds / XDS: the validated numpy port (compare v0, Sec. 1.4) with XDS defaults (BACKGROUND_PIXEL 2, SIGNAL_PIXEL 3,
  6 px per spot), INIT over the whole sweep (or pseudo-sweep), COLSPOT per frame, JOINCOLSPOT over the sweep; the
  per-frame background E = BKGINIT x RATIO is shown and subtracted. Where the archive holds the owner's XDS run, its
  SPOT.XDS (BACKGROUND_PIXEL 4, SIGNAL_PIXEL 4-7) is drawn as well.

## 3. Peaks per frame (median over the 5 picked frames [min-max]; peaks within 3 px of the mask dropped for every method)

| dataset | mean counts / px | bgsub law | bgsub peaks | pyREyes peaks | superxds spots (defaults) | owner's XDS run | bgsub dispersion | ms per frame bgsub / pyREyes / superxds-COLSPOT |
|---|---|---|---|---|---|---|---|---|
| Apollo bin 4, 2048^2 | 1.63 | apollo(4)+1 | 156 [132-188] | 50 [36-54] | 34 [0-56] | 13 | 1.31 | 157 / 152 / 359 |
| Apollo bin 8, 1024^2 | 0.08 | apollo(8) | 22 [16-48] | 20 [9-25] | 4 [0-6] | - | 1.04 | 57 / 30 / 77 |
| Apollo bin 16, 512^2 | 0.83 | apollo(16) | 53 [47-75] | 19 [15-21] | 2 [0-7] | - | 1.45 | 182 / 8 / 18 |
| CETA-D bin 2, 2048^2 | 58.66 | poisson(g 3.5, rv 8) x f 2.8; XDS gain 4.12 | 101 [74-136] | 20 [7-27] | 12 [0-35] | 16 | 0.60 | 1428 / 152 / 351 |
| TVIPS bin 2, 2048^2 | 53.69 | poisson(g 10.3, rv 0) x f 3.1; XDS gain 6.929 | 34 [22-40] | 9 [4-9] | 12 [0-16] | 3 | 0.38 | 186 / 144 / 353 |

superxds spots are 3-D objects (joined across frames); a spot is drawn on the frame nearest its z-centroid, so on a
single frame some of its triangles sit on features that are weak on that frame (bin 4, frame 49: 59 % of its spots within
3 px of one of the 125 brightest maxima of the subtracted frame, median 2.3 px; pyREyes 86 %, 0.5 px; the owner's XDS run
100 %, 0.7 px). bgsub's list is the longest everywhere because it keeps every count-law outlier component with >= 5 sigma
net excess, i.e. weak spots too (bin 4: 39 % of them are among the 125 brightest maxima); none of the lists is verified
against truth here. The bgsub times include the first frame's plan and JIT warm-up in the range, not in the median; CETA-D
frames took 1.4-3.8 s each (re-centring / ring-refinement iterations on this detector, not investigated).

## 4. What the images show (`notes/figures/compare_v1/panel_*.png`, `_crop`)

- Apollo bin 4 (rotation movie): bgsub's background is smooth and its subtracted frame is flat to the edge with the spots
  standing alone; the superxds BKGINIT x RATIO map is grainier (box means every 4th pixel, averaged over the sweep) and
  its subtracted frame keeps a faint halo; pyREyes' DoG is a band-pass with ringing round the stop. Counts 156 / 50 / 34
  per frame, the owner's XDS run 13.
- Apollo bin 8 (49-ms snapshots, 0.08 counts / px): bgsub 22, pyREyes 20 (half of them single stray electrons, compare
  v0), superxds 4 after a 15-frame pseudo-sweep (single frames fail).
- Apollo bin 16 (0.196-s snapshots, 0.5 counts / px, 512^2): bgsub 53 peaks on clean grossular patterns; pyREyes 19;
  superxds 2 (its 6-pixel spot minimum is most of a spot at 512^2), and its subtracted frame keeps ring and halo
  residuals because its table is one average over 55 frames of very different brightness rescaled by one number.
- CETA-D (integrating, binned 2, ~75 counts / px, gain ~4): bgsub's subtracted frame is flat with the spots once its law
  carries read noise and the noise correlation (Sec. 5); 101 peaks vs pyREyes 20, superxds 12 and the owner's run 16. Both
  bgsub and superxds leave a bright crescent on the right of the stop: the glow next to the stop edge is wider than the
  penumbra columns (1-4 px) and is not modelled by either.
- TVIPS (integrating, binned 2, 300 kV, ~45 counts / px, gain ~7): 34 / 9 / 12 / 3; same picture, with a coarser-grained
  noise field; the superxds table and bgsub's map agree to the eye away from the stop.

## 5. The integrating detectors needed a variance model (found on the way; B7-B9)

- B7: with a plain Poisson law (gain from the dispersion of a gain-1 fit) bgsub flagged 8 % (CETA-D) and 47 % (TVIPS) of
  the pixels and reported 900-2700 "peaks" per frame; the fit itself was biased low by ~3 counts on TVIPS because the
  flags removed the upper tail. Cause: the gain / read-noise regression used residuals truncated by that fit's own
  flags (variance 2 x too small on TVIPS), and the pixel noise is spatially CORRELATED on both detectors (pixel residual
  variance 1.5-2 x the neighbour-difference estimate), so a 3 x 3 window sum has 2.8 x (CETA-D) / 3.1 x (TVIPS) the
  variance of 9 independent pixels and the window test at alpha 1e-6 fired constantly.
- Fix used here (declared, not a calibration): a first pass with poisson(50) (flags only bright spots), robust MAD
  variance per 32-px band regressed on the model mean (CETA-D: gain 3.5, read variance 8.4; TVIPS: gain 10.3, read
  variance 0; XDS INIT's gains 4.1 / 6.9), times the measured 3 x 3 window factor (2.8 / 3.1). The per-pixel test is then
  conservative (dispersion 0.4-0.6), the window test right. This is ROADMAP item 3 in miniature: CETA-D and TVIPS need a
  calibrated law with gain, read noise AND the spatial correlation (from dark / flat frames), before any validity claim.
- B8: the stop on both detectors is not a dark disc in the pedestal sense (stray counts inside) and its edge glow extends
  ~20-40 px; the mask here is the low-local-mean region dilated 12 px. A proper stop / penumbra model for these
  geometries (wedge on the TVIPS, disc + arm on the CETA) is open.
- B9: at 2048^2 bgsub took 1.4-3.8 s per CETA-D frame against 0.15-0.25 s elsewhere; not investigated.

## 6. Summary

On every detector and binning the three methods differ as their mechanisms predict: bgsub models the frame and leaves a
flat field with the spots (Apollo bin 4 / 8 / 16 with its declared law; CETA-D and TVIPS once a read-noise + correlation
variance is declared); superxds / XDS subtracts a sweep-averaged table rescaled by one number, grainy and with halo /
ring residuals, and finds the bright spots; pyREyes finds the brightest spots on a band-passed image with no background
at all. Peak counts per frame (bgsub / pyREyes / superxds): Apollo bin 4 156 / 50 / 34, bin 8 22 / 20 / 4, bin 16 53 /
19 / 2, CETA-D 101 / 20 / 12, TVIPS 34 / 9 / 12; the owner's XDS runs with stricter thresholds 13, -, -, 16, 3.
The integrating detectors are the first real test outside Apollo and they exposed exactly the missing piece the roadmap
names: a calibrated variance model with noise correlation.
