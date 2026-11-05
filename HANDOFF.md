# START HERE (pod/bgsub, a cold session)

## Read, in order

1. `../CLAUDE.md` (umbrella conventions), then `CLAUDE.md` (goal in the owner's words, algorithm, status, detector
   coverage, data rules, git rule, close-out).
2. `ROADMAP.md` (ordered next steps) and `LOG.md` (what was done; the extraction entry of 2026-11-05).
3. `notes/reports/README.md` (path mapping), then `notes/reports/2026-11-05_bgsub_algorithm_v2_1.md` (current state;
   Sec. 2.4 rules, Sec. 7 verdict, Sec. 8 next steps) and `2026-11-04_bgsub_algorithm_v0.md` Sec. 1 (derivations).
4. `src/bgsub/README.md` (API, what each version adds).

## Set up and test (from this folder)

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[numba,harness,dev]"
.venv/bin/python -m pytest -q                      # 47 passed in ~25 s on 2026-11-05 (one test skips without ~/pod_data)
```

Synthetic harness smoke test (reproduces 3 published v2.1 test frames exactly; needs ~/pod_data):

```
.venv/bin/python scripts/synth_v2_1.py --set test --only lacey_utc_b8_d1.0_structured_r0 --out synth_v2_1_check
.venv/bin/python scripts/check_rerun_vs_background.py --set test --out synth_v2_1_check
```

The simulator bank (`~/pod_data/bgsub/simbank_v1/`) is already built. Rebuild or extend it only with background's venv:
`../background/.venv/bin/python scripts/make_simbank_v1.py --design v1synth --set dev` (see the script docstring).

## Open owner questions (ask; do not decide)

- OUTPUT FORMAT: what should bgsub hand over per frame? Default proposal: background map + per-pixel variance (float32,
  stored counts) with a sidecar; alternatives in ROADMAP item 6.
- DATASETS for CETA-D bin 2 and TVIPS F30 validation: which sessions (sids) to use, and whether dark / flat frames exist
  for an integrating-detector variance model.
- (Done 2026-11-05) remote `origin` = `git@github.com:mit-eremin/pod-subtract.git`; `main` tracks `origin/main`.

## First task

ROADMAP item 1: write the BACKGROUND-ONLY validity spec as a dated report in `notes/reports/` (rules on bias everywhere
and under masked spots with stated power, per-pixel variance honesty, stop / bar / crystal zones, count-law regimes,
speed), and commit it BEFORE any test run (`git commit -m "..." -- notes/reports/<file>`). Then item 2 (v2.1 to valid:
speed, a power-adequate sparse-cell re-test, A281 when porting the real drivers from `legacy/podbg_scripts/`).

Rules that bite: no HPC jobs; one worker by default and check completeness; path-limited commits; `/wrap` + memory at
the end of the session.
