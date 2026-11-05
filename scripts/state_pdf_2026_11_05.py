"""Two-page state-of-the-problem PDF for pod/bgsub (2026-11-05).

Numbers come from CLAUDE.md (status), notes/reports/2026-11-05_bgsub_algorithm_v2_1.md (Sec. 4-8) and ROADMAP.md
at the commit named in HEAD below; nothing here is a new measurement. Figure: notes/figures/bgsub_v2_1/real_d0_v2_v21.png.

Run (no reportlab in .venv):  ~/micromamba/envs/serial3ded/bin/python scripts/state_pdf_2026_11_05.py
"""
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes/reports/2026-11-05_bgsub_state_2p.pdf"
FIG = ROOT / "notes/figures/bgsub_v2_1/real_d0_v2_v21.png"
HEAD = "1e10212"

INK = colors.HexColor("#1b1f23"); MUTED = colors.HexColor("#5b6470"); RULE = colors.HexColor("#c9ced6")
ACCENT = colors.HexColor("#0b5fa5"); BAND = colors.HexColor("#eef3f8")
GOOD = colors.HexColor("#dff3e3"); BAD = colors.HexColor("#fbe3e0"); WARN = colors.HexColor("#fdf1d6")

W = 7.3 * inch
st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14.5, leading=17.5, textColor=INK, spaceAfter=2)
st_sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, textColor=MUTED, spaceAfter=6)
st_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, leading=13.5, textColor=ACCENT, spaceBefore=6, spaceAfter=2.5)
st_b = ParagraphStyle("b", fontName="Helvetica", fontSize=8.7, leading=11.1, textColor=INK, spaceAfter=3)
st_bl = ParagraphStyle("bl", parent=st_b, leftIndent=10, bulletIndent=1, spaceAfter=1.5)
st_cap = ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=7.6, leading=9.4, textColor=MUTED, spaceBefore=1.5, spaceAfter=4)
st_tb = ParagraphStyle("tb", fontName="Helvetica", fontSize=7.8, leading=9.5, textColor=INK)
st_tbh = ParagraphStyle("tbh", fontName="Helvetica-Bold", fontSize=7.8, leading=9.5, textColor=INK)


def tbl(rows, widths, status_col=None):
    data = [[Paragraph(c, st_tbh if i == 0 else st_tb) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
           ("BACKGROUND", (0, 0), (-1, 0), BAND), ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT)]
    for i in range(1, len(rows)):
        if status_col is not None:
            txt = rows[i][status_col].upper()
            col = GOOD if txt.startswith("PASS") else (BAD if txt.startswith("FAIL") else WARN)
            sty.append(("BACKGROUND", (status_col, i), (status_col, i), col))
        elif i % 2 == 0:
            sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(sty)); return t


def bullets(items):
    return [Paragraph(x, st_bl, bulletText="•") for x in items]


def img(path, width):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(0.6 * inch, 0.42 * inch, f"pod/bgsub  -  state of the problem 2026-11-05  -  D. Eremin, Nelson Lab, Caltech  -  repo mit-eremin/pod-subtract @ {HEAD}")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"page {doc.page} of 2")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5); canvas.line(0.6 * inch, 0.58 * inch, letter[0] - 0.6 * inch, 0.58 * inch); canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.7 * inch,
                        title="bgsub: state of the problem 2026-11-05", author="Dmitry Eremin", subject="pod/bgsub background estimation and subtraction")
S = []

# ------------------------------------------------------------------ page 1
S.append(Paragraph("Background subtraction for diffraction frames: state of the problem", st_title))
S.append(Paragraph("Subproject <b>pod/bgsub</b> (GitHub mit-eremin/pod-subtract, main at " + HEAD + "), Dmitry Eremin, Nelson Lab, Caltech. "
                   "Split out of pod/background on 2026-11-05 at background commit f1e93f5; package <b>bgsub</b>, 47 tests pass, self-contained "
                   "(vendored count law, detector, masks, centre, provenance; a frozen simulator-v1 frame bank, 716 frames reproduced bit for bit). "
                   "This page states the problem and where the algorithm stands; page 2 gives the measured performance, the known defects and the next steps. "
                   "Session ids are anonymous (sids).", st_sub))

S.append(Paragraph("1. The problem, in the owner's words (re-scope 2026-11-04, scope correction 2026-11-05)", st_h))
S.append(Paragraph("A very <b>fast</b>, <b>mathematically correct</b>, <b>universal</b> background estimation and subtraction algorithm for electron-diffraction frames, "
                   "better than the flat local backgrounds of XDS / DIALS (flat boxes are biased on a curved background and lose intensity; their code is not copied). "
                   "Correct means: unbiased background <b>everywhere</b> and <b>under masked spots</b>; honest per-pixel background <b>variance</b>; right behaviour under the detector's "
                   "real <b>count statistics</b>; right <b>near the beam stop, grid bars and crystals</b>; and a fast estimate (target <= 50 ms per 1024<super>2</super> frame).", st_b))
S.extend(bullets([
    "<b>Background only.</b> Peak integration is not the product and the result is not for XDS. The integrators in the code (G-ML, LC, intervals, coverage) were the orchestrator's addition; "
    "they stay but nothing is judged on them. Injected peaks remain a <i>test</i>, scored against an oracle-background subtraction, never against true intensity.",
    "<b>No other packages</b> (no DIALS, nXDS); methods are developed here. The physics forward model / simulator stays in pod/background and is a test bench only.",
    "<b>Universal</b> today means Apollo (bin 4, 8, 16) with its stored-count law; CETA-D and TVIPS need their own variance models before any claim.",
]))

S.append(Paragraph("2. The approach (v1 default; v2, v2.1 opt-in; v0 reproducible)", st_h))
S.append(Paragraph("The frame's stored counts are reduced to sufficient statistics on polar cells (1-px rings x 64 sectors x 2 checkerboard halves) and fitted with a <b>linear smooth model in stored "
                   "counts</b>: a fine radial cubic B-spline (rings, steep halo) + azimuthal harmonics m <= 6 on a coarse radial spline + centre / ellipticity derivative columns + (v1) penumbra columns "
                   "on the 1-4 px next to the beam stop; (v2) re-centring and a shrunken 1-px ring refinement, (v2.1) on elliptical rings. The fit is <b>cross-fitted weighted least squares</b>: the weights "
                   "are the count-law variances taken from the <i>other</i> checkerboard half, so the estimator is linear in the counts and unbiased for <i>any</i> count law whenever the truth lies in the "
                   "model's span (no median, no Poisson assumption). Spots, hot pixels and zingers are excluded by one-sided count-law tests (alpha 10<super>-6</super> per pixel and per 3 x 3 window); "
                   "spots the caller will measure are always excluded by discs. Variance is a sandwich with the frame's Pearson dispersion. The hybrid <b>H2.1</b> adds a local annulus residual "
                   "only where its signal is significant (T > 2 with block standard errors), with the spot's own light removed (spill correction); otherwise it is the global <b>G2.1</b>.", st_b))
S.append(Paragraph("The Apollo count law it carries: stored value = round-half-even of the 3-raw-frame electron sum / m (m = 1, 4, 16 e per count at bin 4, 8, 16), tie rule p = 0.403, "
                   "paralyzable dead time tau = 0.206 per e per physical pixel per raw frame; archive TIFFs are MRC + 1; seams and borders masked.", st_b))

S.append(Paragraph("3. Where it stands: v2.1 against its own pre-registered rules (2026-11-05)", st_h))
S.append(Paragraph("Every version is pre-registered (rules committed before the test run, no parameter changes after it, failures reported with their cells). "
                   "The verdict rule: VALID iff R1, R2, R3 and R5 pass and R4 passes in the F and classic regimes. <b>v2.1 is NOT VALID, by two narrow margins, both about the background.</b>", st_b))
S.append(tbl([
    ["Rule", "What it tests", "v2 (previous)", "v2.1 (current)", "v2.1 status"],
    ["R1", "Synthetic unbiasedness: 60 lacey cells (bin 4 / 8 x dose x1 / 3 / 10 x plain / structured x D0-D4), |beta| <= max(0.5 %, 3.2 SE), for G and H",
     "G pass 0/60; H fail 2/60 (arm cell +0.70 %, sparse cell -3.16 %)", "G2.1 pass 0/60; H2.1 fail 1/60: sparse bin-8 cell (dose x3, plain, D1; 0.2 stored counts per box) -2.46 +- 0.72 %; arm cell fixed (+0.34 %)", "FAIL 1/60 (H); no power at 0.2 counts/box"],
    ["R2", "Count statistics / truncation: spot-free twins, 24 cells, paired bias of G <= max(0.3 %, 3.1 SE); masking on - off <= 0.1 %", "pass (max 0.45 %)", "pass (max 0.45 %)", "PASS"],
    ["R3", "Sigma honesty and interval coverage of the injected-spot estimates (integration; out of scope now)", "pass; coverage 0.57-0.58", "pass; coverage 0.66-0.69 (fix 3)", "PASS (not a product criterion)"],
    ["R4", "Real frames, better than FLAT per regime x d bin, spots without host overlap: F, BA, CN, classic (17 fly-scan + 20 classic test sessions, 48,930 spots; 9 confirmation sessions)", "P / P / P / P", "P / P / P / P; confirmation P; all-spot classic F (host Bragg overlap, not background)", "PASS"],
    ["R5", "Speed of the background estimate: 1024<super>2</super>, numba, plan cached, <= 50 ms on this Mac", "pass 46.8 ms (own run), 50.2 ms in the v2.1 run", "54.2 ms under load (v1 46.3 ms in the same run); elliptical ring kernels +2.7 ms (A278)", "FAIL 54.2 ms"],
], [0.4 * inch, 2.35 * inch, 1.3 * inch, 2.05 * inch, 1.2 * inch], status_col=4))
S.append(Paragraph("R6 and R7 (sharp ring) are reported, not in the verdict. Report of record: notes/reports/2026-11-05_bgsub_algorithm_v2_1.md (Sec. 2.4 rules, Sec. 7 verdict, Sec. 8 limits).", st_cap))

S.append(PageBreak())

# ------------------------------------------------------------------ page 2
S.append(Paragraph("4. Measured performance of the background estimate", st_h))
S.append(Paragraph("Background error next to the beam stop (D0, d 4-8 A) on real Apollo frames with injected peaks, in units of the spot box's own count noise, spots without a host overlap. "
                   "Smaller is better; FLAT is the XDS / DIALS-style local flat background on the same spots. Synthetic plain lacey frames: G2.1 and H2.1 0.07-0.16 vs FLAT 0.38-0.62; "
                   "synthetic structured frames at D4: H2.1 0.39-0.40 vs v2 0.45-0.49 vs G 0.84-0.97.", st_b))
S.append(tbl([
    ["Regime (real frames)", "Spots", "v2.1 H2.1 (hybrid)", "v2.1 G2.1 (global)", "v2 H2", "FLAT"],
    ["F: fly-scan, clean film, bin 8", "13,650", "<b>0.08</b>", "0.32", "0.03", "0.40"],
    ["BA: bar-adjacent, bin 8", "14,280", "<b>0.34</b>", "0.46", "0.33", "0.50"],
    ["CN: crystal neighbour, bin 8", "12,600", "<b>0.37</b>", "0.57", "0.35", "0.60"],
    ["classic, bin 4 (20 test sessions)", "8,400", "<b>1.57</b>", "1.58", "1.76", "2.98"],
    ["classic confirmation (9 sessions)", "3,430", "0.50", "<b>0.48</b>", "0.51", "1.30"],
], [2.3 * inch, 0.7 * inch, 1.15 * inch, 1.15 * inch, 0.9 * inch, 1.1 * inch]))
S.append(Spacer(1, 4))
S.append(img(FIG, 0.9 * W))
S.append(Paragraph("D0 background error (rmse / box noise, log scale), v2 H2 vs v2.1 H2.1 vs FLAT: left all test spots, middle test spots without a host overlap (the verdict form), right the "
                   "classic confirmation set. With all spots, classic H2.1 7.20 vs FLAT 5.96 and F 0.36: sample Bragg spots of the host under the injected spot (an overlap, not background; A266, A279). "
                   "Outside D0, H2.1 = G2.1 within 0.01-0.03. Figure: notes/figures/bgsub_v2_1/real_d0_v2_v21.png.", st_cap))

S.append(Paragraph("5. Known defects and limits", st_h))
S.extend(bullets([
    "<b>Sparse counts (R1).</b> At 0.2 stored counts per box the gate fires at its nominal 2.5 % rate on noise and five firings move a cell by 0.5 %; the failing cell also has G2.1 itself at 3.05 SE on those frames "
    "(on 64 fresh frames H2.1 is -0.47 +- 0.72 %, A276). The test had no power; it needs a re-test with enough frames and possibly a count-level-aware gate.",
    "<b>Sharp rings (A277, R7 reported).</b> A film ring narrower than the spline knots (classic sb7857cd8, 4.6 A) is not removed (ring-zone dispersion 16.2 -> 15.1 against a target of 8.1; no harm elsewhere, max ratio 1.002): 34-81 % of its pixels are count-law flags and the skipped ring mean breaks down. "
    "A full-frame robust ring term was tried and rejected (it breaks the background next to the stop, A275); a ring-zone-only high-breakdown first pass is the open option.",
    "<b>Speed (A278).</b> 54.2 ms at 1024<super>2</super> against 50 ms, measured under load; the elliptical kernels cost +2.7 ms and can be fused into existing passes (~2.5 ms back).",
    "<b>Structure outside the model.</b> Anything neither smooth on the model's scales nor linear at the annulus scale is not modelled; a compact host overlap under a spot is invisible to the probes (A279).",
    "<b>Variance at bin 16</b> is not honest (saturated cores on the wrong dead-time branch, A196); bin 16 is synthetic-only. The penumbra columns are ignored by var_map today.",
    "<b>Detector coverage.</b> Apollo bin 8 (fly-scan) and bin 4 (classic) are tested on real frames; <b>CETA-D bin 2 and TVIPS F30 are untested</b> and need an integrating-detector variance model "
    "(gain per electron, read noise, offset, no rounding) before any run. Evaluation-code defect A281 (duplicated rows in the v2 / v2.1 pull tables; R4 and D0 unaffected) is open.",
]))

S.append(Paragraph("6. Next steps (ROADMAP.md, in order) and what the owner must decide", st_h))
S.extend(bullets([
    "<b>1. Background-only validity spec</b>, pre-registered: bias everywhere and under masked spots with stated power per cell; per-pixel variance honesty (replicates on synthetic frames, split-half / twins on real); "
    "zones (stop, penumbra, arm, bar edges, crystal neighbours); count-law regimes (bin 4 / 8 / 16, sparse to bright, gain on / off, dead-time ceiling, zero level); speed on an idle machine; "
    "injected peaks only against an oracle background. Replaces the integration-flavoured R1-R7.",
    "<b>2. Bring v2.1 to valid under it:</b> fuse the elliptical kernels (< 50 ms); a power-adequate sparse-cell re-test; fix A281 when the real-frame drivers are ported from legacy/podbg_scripts; optionally the ring-zone first pass.",
    "<b>3-5.</b> Integrating-detector variance model, then CETA-D bin 2 validation; TVIPS F30 validation (ringcal's readers, vendored); honest variance at bin 16.",
    "<b>Owner decisions (ask, do not decide):</b> (a) <b>output format</b> per frame: proposed a float32 background map + per-pixel variance map in stored counts, the subtracted frame on request, and a sidecar "
    "(version, count law, zero level, mask, centre); alternatives are radial-model coefficients + covariance, or electrons; decide before any file writer or CLI is built. "
    "(b) <b>datasets</b> for CETA-D bin 2 and TVIPS F30 (which sids; whether dark / flat frames exist).",
]))

doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
