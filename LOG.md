# pod/bgsub log

## 2026-11-05: extraction from pod/background

WHY: the owner's scope correction of 2026-11-05 made the subtraction algorithm a product of its own: background
ESTIMATION AND SUBTRACTION only, no integration, not for XDS, no other packages. In pod/background it was "item 2"
(priority 2, behind the simulator) inside a physics-model repo; split out so it can be developed, validated and shipped
on its own terms, with its own validity spec, without the background physics stack (podbg.model, theory tables).

SOURCE: pod/background at commit f1e93f5 ("Owner scope correction: bgsub product = background estimation/subtraction
only ..."), working tree clean. podbg.bgsub last changed in a3036e4 (v2.1 README status); v2.1 integrated in 2972e5b;
v2.1 pre-registration 4d6a251; v2 5609c14 (swept, A264); v1 bc37be9; v0 36d9d6a.

WHAT WAS DONE
- `src/bgsub/`: podbg.bgsub moved verbatim (v0, v1, v2, v2.1 live in the same files behind `version=`; v0 kept because
  the regression tests and the `version="v0"` path need it). Only change: `from podbg import binning` -> the vendored
  copy. Each moved file has a one-line origin header (commit + sha256).
- `src/bgsub/vendor/`: binning, detector, masks, centre, provenance; `src/bgsub/validation/`: peaks, sim_geometry,
  sim_detectors, harness (= scripts/bgsub_v0_common.py). Each with a VENDORED header (source, commit, sha256, edits,
  KEEP IN STEP).
- Simulator: podbg.sim is NOT vendored (it needs ~9,000 lines of podbg.model and five theory tables). Instead
  `scripts/make_simbank_v1.py` (the one bridge script that imports podbg; run with ../background/.venv) froze the
  simulator-v1 radial tables + geometry of the 704 frames of the v1/v2/v2.1 synthetic design (dev 352, test 352) and
  the 12 R5 speed frames into `~/pod_data/bgsub/simbank_v1/` (26 MB, index.csv + sidecar each). `bgsub.validation.
  simbank` renders them; all 716 entries reproduce podbg's float32 expected frames BIT FOR BIT (sha256). A 46-kB sample
  bundle is in tests/data.
- `bgsub.validation.synth` (make_frame / cells of scripts/bgsub_v1_synth.py on the bank) and `scripts/synth_v2_1.py`
  (port of bgsub_v2_1_synth): 3 published v2.1 synthetic TEST frames (lacey bin 4, lacey bin 8 structured, formvar bin 8)
  re-made and re-fitted here equal the background checkpoints in all 46 numeric columns (max |diff| 0).
  Checkpoints: `~/pod_data/bgsub/synth_v2_1_rerun/checkpoints/synth_test/` (3 files, subset run, no tables).
- `scripts/speed_v2_1.py` ported (frames rendered from the bank with the vendored count law; noise realisation differs
  from the published run). Not run (machine load 13 at the time).
- Tests: the four bgsub test files + test_peak_recovery ported (imports only; the three regression references copied
  unchanged), plus test_simbank, test_vendor_headers, test_synth_reproduction. New venv (uv, Python 3.12, numpy 2.5.4,
  scipy 1.18.1, numba 0.68.1): 47 passed in 22.5 s.
- Real-frame drivers and table scripts (bgsub_v0_real, v1_real, v2_real, v2_1_real, *_tables, v2_1_ring,
  peak_recovery_v0, the reference makers) copied verbatim to `legacy/podbg_scripts/`; they import podbg and are NOT
  runnable here yet (their extra needs: paths -> bgsub.config, podbg.atlas.SEAM_LINES_BIN8 -> vendor.detector.seam_lines).
- Reports v0, v1, v2, v2.1, peak recovery v0, XDS merge v0 copied verbatim to notes/reports (+ README with the path
  mapping); figures bgsub_v0..v2_1 copied.
- Docs: CLAUDE.md, ROADMAP.md, HANDOFF.md, this LOG. `git init`; no remote (the owner creates mit-eremin/pod-bgsub). [Later the same day: the repo was created as
  `mit-eremin/pod-subtract`, see the next entry.]

NOT SELF-CONTAINED (by design or still open): the simulator itself (new synthetic designs need a new bank from
background); the real-frame drivers (legacy, to port); all real data stays in ~/pod_data/background (read-only by path).

## 2026-11-05 (later) — GitHub remote

- The owner created the private repo `mit-eremin/pod-subtract` (name differs from the planned `pod-bgsub`). Added
  `origin`; the first push was rejected because GitHub had seeded the repo with an "Initial commit" (`cf4b6bf`,
  one-line README.md). The owner merged it (`--allow-unrelated-histories`, `6c6941e`) and pushed; `main` tracks
  `origin/main`.
- CLAUDE.md, HANDOFF.md and this LOG now name `pod-subtract`. Nothing else changed; nothing running.
- Two-page PDF `notes/reports/2026-11-05_bgsub_state_2p.pdf` (state of the problem, for the owner), generator
  `scripts/state_pdf_2026_11_05.py` run with the serial3ded micromamba python (reportlab 5.1; neither venv has it).
  Content restates CLAUDE.md status, v2.1 report Sec. 4-8 and ROADMAP 1-6; figure real_d0_v2_v21.png. No new numbers.
