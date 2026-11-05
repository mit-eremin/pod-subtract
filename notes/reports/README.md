# Reports copied from pod/background (2026-11-05, background commit f1e93f5)

Verbatim copies; do not edit them (they are the record of what was pre-registered and found). New reports go here
with new dates. Path mapping when reading them:

| in the reports | in this repo |
|---|---|
| `src/podbg/bgsub/` | `src/bgsub/` |
| `podbg.binning`, `podbg.detector`, `podbg.peaks`, `podbg.sim` | `bgsub.vendor.binning`, `bgsub.vendor.detector`, `bgsub.validation.peaks`, `bgsub.validation.simbank` (+ `sim_geometry`, `sim_detectors`) |
| `scripts/bgsub_v0_common.py` | `src/bgsub/validation/harness.py` |
| `scripts/bgsub_v1_synth.py` (make_frame, cells) | `src/bgsub/validation/synth.py` |
| `scripts/bgsub_v2_1_synth.py`, `_speed.py` | `scripts/synth_v2_1.py`, `scripts/speed_v2_1.py` |
| other `scripts/bgsub_*.py`, `peak_recovery_v0.py` | `legacy/podbg_scripts/` (import podbg; reference for porting, not runnable here) |
| `tests/test_bgsub*.py` | `tests/` (same tests, imports changed) |
| `~/pod_data/background/bgsub_v*/` | unchanged, READ-ONLY (`bgsub.config.BG_INPUTS`) |
| `notes/figures/bgsub_v*/` | `notes/figures/bgsub_v*/` (copied) |
| anomaly numbers A193-A282 | background's numbering; new anomalies here start at B1 |

- `2026-11-04_bgsub_algorithm_v0.md`: the DERIVATIONS (Sec. 1.1-1.6: observation model, why flat/plane local
  backgrounds are biased, the estimator, GL, exclusion, speed); v0 validation and verdict.
- `2026-11-04_bgsub_algorithm_v1.md`: penumbra columns (stop zone), the expected-error switch, G-ML information limit.
- `2026-11-05_bgsub_algorithm_v2.md`: re-centring, ring refinement, continuous local weight w*.
- `2026-11-05_bgsub_algorithm_v2_1.md`: gated + spill-corrected local weight, elliptical rings; current verdict.
- `2026-11-04_peak_recovery_injection_v0.md`: the injection test bench (oracle background, bin-8 counting-law loss).
- `2026-11-04_xds_merge_background_v0.md`: why an external background does not help inside XDS (context only; XDS is
  out of scope).
