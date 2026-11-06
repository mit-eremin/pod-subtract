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
FIG = ROOT / "notes/figures/compare_v0/zoom_b4_mov11_001.png"
FIG2 = ROOT / "notes/figures/compare_v0/overlays.png"
HEAD = "see footer date"

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
st_tb = ParagraphStyle("tb", fontName="Helvetica", fontSize=7.5, leading=9.0, textColor=INK)
st_tbh = ParagraphStyle("tbh", fontName="Helvetica-Bold", fontSize=7.5, leading=9.0, textColor=INK)


def tbl(rows, widths, status_col=None):
    data = [[Paragraph(c, st_tbh if i == 0 else st_tb) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
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
    canvas.drawString(0.6 * inch, 0.42 * inch, "pod/bgsub  -  state of the problem 2026-11-05 (evening)  -  D. Eremin, Nelson Lab, Caltech  -  repo mit-eremin/pod-subtract")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"page {doc.page} of 2")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5); canvas.line(0.6 * inch, 0.58 * inch, letter[0] - 0.6 * inch, 0.58 * inch); canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.7 * inch,
                        title="bgsub: state of the problem 2026-11-05", author="Dmitry Eremin", subject="pod/bgsub background estimation and subtraction")
S = []

# ------------------------------------------------------------------ page 1
S.append(Paragraph("Background subtraction for diffraction frames: state of the problem", st_title))
S.append(Paragraph("Subproject <b>pod/bgsub</b> (GitHub mit-eremin/pod-subtract), Dmitry Eremin, Nelson Lab, Caltech. "
                   "Split out of pod/background on 2026-11-05 at background commit f1e93f5; package <b>bgsub</b>, 47 tests pass, self-contained "
                   "(vendored count law, detector, masks, centre, provenance; a frozen simulator-v1 frame bank, 716 frames reproduced bit for bit). "
                   "Page 1 states the problem and where the algorithm stands by its own rules; page 2 sets the three background-subtraction methods side by side, measured on real frames (compare v0, same day), with the validated numpy port of the superxds math. "
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
                   "The verdict rule: VALID iff R1, R2, R3 and R5 pass and R4 passes in the F and classic regimes. <b>v2.1 is NOT VALID by that rule, by two narrow margins.</b> "
                   "All thresholds (0.5 %, 3.2 SE, 50 ms, ...) were set by the orchestrator, not the owner; the owner ruled on 2026-11-05 that the <b>speed gate is relative</b> (single-CPU Python), so R5 is reported, not a verdict.", st_b))
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
S.append(Paragraph("4. The three background-subtraction methods, side by side (compare v0, 2026-11-05)", st_h))
S.append(Paragraph("Measured on 40 real Apollo frames: 30 bin-8 fly-scan snapshots (1024<super>2</super>, 20 REyes Good + 10 Bad, ~0.08 counts / px background) and 10 frames of a bin-4 rotation movie "
                   "(2048<super>2</super>, with XDS's own SPOT.XDS as a reference). pyREyes was re-implemented exactly (bit-identical to its code); the superxds / XDS algorithm was run both as the installed XDS and as a "
                   "numpy port of the superxds source, validated against XDS (table below). All peaks were measured on the ORIGINAL frame in the same disc (3 px at bin 8, 5 px at bin 4).", st_b))
S.append(tbl([
    ["", "bgsub v2.1 (this project)", "pyREyes 3.6.3 (dif_map.py)", "superxds = XDS (INIT + COLSPOT)"],
    ["What is estimated", "A smooth global background of THIS frame in stored counts: radial B-spline + azimuthal harmonics + centre / ellipticity + penumbra columns, elliptical 1-px ring refinement; a local correction only where significant", "Nothing. A difference of Gaussians (sigma 1 - 5 at 2048, 1 - 3 at 1024) is thresholded at mean + k std; no background value exists", "A static per-pixel table (BKGINIT) from the first frames: 7 x 7 box means at every 4th pixel, bilinear in between, averaged over frames; rescaled by ONE scalar per frame (RATIO)"],
    ["Statistical model", "Declared count law (Apollo stored-count law, dead time, pedestal); cross-fitted weighted LS, unbiased for ANY law when the truth is in the span; sandwich variance per pixel", "None (global mean and std of the band-passed image)", "Poisson with a per-cell gain = box variance / mean (median + 4.88 MAD cut); candidates where the 7 x 7 variance exceeds Poisson x FAC"],
    ["Spots kept out", "One-sided count-law tests per pixel and per 3 x 3 window (alpha 1e-6), dilated; discs for known spots", "-", "Candidate pixels excluded from RATIO; INIT's gain cut drops bright nodes"],
    ["Stop, seams, bars", "Mask + penumbra columns next to the stop, re-centring, seams and border masked", "3 % border and a central box; fires on the stop edge and arm tip (47 of 639 bin-8 peaks on the mask)", "TRUSTED_REGION and untrusted masks only"],
    ["Output per frame", "Background map, per-pixel variance, flags (and a subtracted frame)", "Binary mask and a peak count; no intensity per peak", "BKGINIT.cbf, GAIN.cbf; spots with 'intensity' = sum of significance labels; the background map is never written"],
    ["Bias on spot-free pixels", "< 1 % in every 32-px band at both binnings; bin 8: mean and count-law variance match the data to 1e-4", "n/a", "bin 4: -9 to -14 % in the halo, +5 to +51 % in the tail; bin 8: 0 to -8 % inner, large outer"],
    ["Background under the same peaks", "1 (reference); flat annulus = 1.15 x bgsub, losing 5 % (bin 8) to 14 % (bin 4) of the net intensity", "n/a", "1.04 (bin 8), 1.01 (bin 4): agrees under peaks, wrong between them"],
    ["Sparse counts (bin 8)", "Works: the fit and the law hold at 0.08 counts / px", "At its noise floor: half the peaks are single stray electrons; Good and Bad frames alike", "0.06 % of pixels pass the 7 x 7 screen as candidates, none strong at SIGNAL_PIXEL 7; needs the defaults and a sweep to run"],
    ["Peaks per frame (median)", "bin 8: 23 Good / 14 Bad; bin 4: 187 (holds 74-87 % of the others' peaks + weak high-angle ones)", "bin 8: 14 / 17; bin 4: 45", "bin 8: 4 / 2 (defaults, pseudo-sweep); bin 4: 46 (defaults), 9 (owner 4/7/5)"],
    ["Time per frame, this Mac", "62 ms (1024<super>2</super>), 155 ms (2048<super>2</super>), Python + numba", "31 / 149 ms", "33 / 125 ms Fortran (4 threads); the same math in numpy 130 / 550 ms"],
], [1.05 * inch, 2.15 * inch, 1.85 * inch, 2.25 * inch]))
S.append(Spacer(1, 3))
S.append(Paragraph("superxds math in Python (scripts/superxds_py.py: INIT, COLSPOT, JOINCOLSPOT ported from the superxds source), validated against the installed XDS on the same inputs:", st_b))
S.append(tbl([
    ["run", "BKGINIT table", "GAIN table", "strong pixels per frame", "spots"],
    ["bin 8, 15-frame pseudo-sweep", "99.98 % exact, rest +-1 in the last digit", "99.97 % exact", "exact on 15 / 15 frames (1027 px)", "58 / 58 matched within 0.007 px, intensities identical"],
    ["bin 4, 120-frame mov11 sweep", "99.985 % exact, rest +-1", "99.98 % exact", "exact on 116 / 120 (509,619 vs 509,623 px)", "5444 / 5444 within 0.008 px, 5442 intensities identical"],
], [1.7 * inch, 1.6 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch]))
S.append(Paragraph("The port also shows what XDS never prints: COLSPOT.LP's 'NBKG' is the count of cleared candidates, not of background pixels; RATIO rests on 99.9 % of the frame at bin 8 and on 97 % (frame 1) to 68 % (frame 120) at bin 4.", st_cap))
S.append(Paragraph("Spot by spot (zoom figure in the report, 8 spots found by all three methods on one bin-4 frame): the flat-annulus net intensity is 8-25 % below bgsub's on every spot. Full tables, overlays of every "
                   "method's peaks on the original frames and radial residuals: notes/reports/2026-11-05_compare_v0.md, notes/figures/compare_v0/.", st_cap))

S.append(Paragraph("5. Known defects, next steps, owner decisions", st_h))
S.extend(bullets([
    "<b>Defects.</b> The one failing synthetic cell (0.2 counts per box) had no statistical power (needs a re-test); a film ring sharper than the spline knots is not removed (A277); bin 16 variance not honest (A196); "
    "<b>CETA-D bin 2 and TVIPS F30 untested</b> (need an integrating-detector variance model). compare v0 is descriptive, not pre-registered, with no truth for real peaks.",
    "<b>Next (ROADMAP).</b> 1. Background-only validity spec, pre-registered, with thresholds the owner endorses (bias everywhere and under masked spots with stated power, variance honesty, zones, count-law regimes; timing reported, not gated). "
    "2. Bring v2.1 to valid under it. 3-5. Integrating-detector variance model, CETA-D and TVIPS validation, bin-16 variance.",
    "<b>Owner decisions.</b> (a) per-frame output format (proposed: float32 background map + variance map in stored counts, subtracted frame on request, sidecar); (b) CETA-D / TVIPS datasets; "
    "(c) the mov11 XDS.INP beam centre is 75 px off and was never refined by XDS (B2): re-check the XDS-merge study.",
]))

doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
