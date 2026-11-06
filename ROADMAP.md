# pod/bgsub roadmap

Ordered. Each step is pre-registered before its test run (rules committed first), produces a dated report in
`notes/reports/`, versioned outputs under `~/pod_data/bgsub/` with provenance sidecars, and a LOG.md entry.
Judge everything on the BACKGROUND only (CLAUDE.md, Goal).

## Now

1. **Background-only validity spec (pre-registered).** Replace the integration-flavoured R1-R7 of v0-v2.1 with rules
   on the background estimate alone, written and committed before any test run:
   - bias everywhere (per radial band, per regime) and UNDER MASKED SPOTS (spot discs excluded, background compared with
     the oracle / truth), with power stated per cell (frames per cell chosen so the rule can detect the bias it
     targets; R1's sparse cell had no power);
   - per-pixel VARIANCE HONESTY of the estimate (`res.var` / `var_map` against replicate scatter on synthetic frames
     and split-half / twin frames on real data), including the penumbra columns (`var_map` ignores them today);
   - zones: beam STOP (D0, penumbra 1-4 px, arm shadow), grid BAR edges (bar-adjacent regime), CRYSTAL neighbours;
   - COUNT-LAW regimes: Apollo bin 4 / 8 / 16, sparse (<= 0.02 stored counts / px) to bright, gain flag on / off, dead
     time near the ceiling, the zero level (MRC + 1);
   - SPEED of the background estimate (plan cached), measured on an idle machine with the load recorded;
   - injected peaks allowed only as a test against an ORACLE-background subtraction, never against true intensity.
2. **Bring v2.1 to valid under that spec.**
   - Speed < 50 ms at 1024^2: fuse the elliptical ring accumulation / application into the existing passes (a
     precomputed r' map per plan; ~2.5 ms, A278); measure on an idle machine (`scripts/speed_v2_1.py`).
   - A POWER-ADEQUATE re-test of the sparse bin-8 cell (bin 8, dose x3, plain, D1): enough fresh frames to resolve
     0.5 % (the 168-spot cell had SE 0.7 %); consider the gate options of the v2.1 report Sec. 8 (count-level-aware gate
     threshold, or no local correction where G's error is already < 0.1 box noise). Synthetic frames: new bank entries
     from background (`scripts/make_simbank_v1.py` with new seeds) or the existing 704.
   - Fix A281 in the real-table script (merge on a unique key, not sid:frame) when the real drivers are ported from
     `legacy/podbg_scripts/` (bgsub_v0_real, bgsub_v2_real, bgsub_v2_1_real, *_tables); re-issue the affected tables.
   - Optional: the sharp-ring defect (A277): a high-breakdown ring-zone first pass.

## Follow-ups from compare v0 (2026-11-05)

- Owner clarified: the 50 ms gate is relative (single-CPU Python); judge on correctness and overall efficiency.
  Keep timing as a reported number, not a verdict, in the validity spec (item 1).
- superxds math runs here as a validated numpy port (`scripts/superxds_py.py`); the Linux binary would add nothing (B1 closed).
- Check the xds_merge study's IDXREF/INTEGRATE for the wrong beam origin of mov11 (B2, 75 px).
- A peak finder on the subtracted frame (split merged flag components, B5) only if the owner wants one; not the product.

## Next

3. **Integrating-detector variance model, then CETA-D bin 2 validation.** `CountLaw.poisson(gain, read_var)` exists but
   no CETA calibration does (gain per electron, read noise, offset, correlated noise, no rounding). Derive it from data
   (dark / flat or film frames), pre-register, then validate on CETA bin 2 sessions (datasets: OWNER QUESTION).
4. **TVIPS F30 validation** (TF30, 300 kV). Reuse `pod/ringcal`'s TVIPS readers (vendor them with a header, as here).
   Datasets: OWNER QUESTION.
5. **Honest variance at bin 16** (A196: saturated cores on the wrong dead-time branch, g oscillating), and real bin-16
   data if any exists in the archive.

## Open question for the owner

6. **OUTPUT FORMAT.** What does the owner's downstream use need? Default proposal: per frame, a background map and a
   per-pixel variance map in stored counts (float32), plus the subtracted frame on request, with a sidecar (version,
   count law, zero level, mask, centre). Alternatives: radial-model coefficients + covariance (compact), or electrons
   instead of stored counts. Decide before building any file writer or CLI.

## Done

- 2026-11-05: extraction from pod/background (code v0-v2.1, tests, vendored count law / detector / masks / centre /
  provenance / peaks / sim pieces, simulator-v1 bank with bit-for-bit rendering, docs).
