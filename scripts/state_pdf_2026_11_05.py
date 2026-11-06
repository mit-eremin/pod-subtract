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
HEAD = "00f13bf"

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
                   "Page 1 states the problem and where the algorithm stands by its own rules; page 2 measures it against pyREyes and XDS / superxds on real frames (compare v0, same day), then the defects and next steps. "
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
S.append(Paragraph("4. Measured against pyREyes and XDS / superxds on real frames (compare v0, 2026-11-05)", st_h))
S.append(Paragraph("40 real Apollo frames: 30 bin-8 fly-scan snapshots (1024<super>2</super>, 20 REyes Good + 10 Bad, two sessions, ~0.08 counts / px background) and 10 frames of a bin-4 rotation movie "
                   "(2048<super>2</super>, XDS's own SPOT.XDS as a reference). <b>pyREyes</b> 3.6.3 has no background model (difference of Gaussians, global mean + k std threshold, components, a count; no per-peak intensity; "
                   "re-implemented and verified bit for bit). <b>superxds</b> is a line-by-line Fortran port of XDS (static box-mean table x one scalar per frame; Poisson strong-pixel test; 'intensity' = sum of significance labels); "
                   "it does not build on macOS / arm64, so the installed XDS stood in. All peaks were measured on the ORIGINAL frame in the same disc (3 px at bin 8, 5 px at bin 4).", st_b))
S.append(tbl([
    ["Question", "bgsub v2.1", "pyREyes", "XDS / superxds", "flat annulus (XDS / DIALS-style local box)"],
    ["Background on spot-free pixels, by radius", "unbiased: |residual| < 1 % in every 32-px band, both binnings; bin 8: mean and count-law variance match the data to 1e-4", "none (band-pass only)", "bin 4: -9 to -14 % in the halo, high in the tail; bin 8: collapses (0.07 % of pixels pass its background screen)", "fit to the same pixels: not a test"],
    ["Peaks per frame (median)", "bin 8 Good 23 / Bad 14; bin 4 187 (count-law flags, >= 5 sigma per component)", "bin 8 Good 14 / Bad 17 (half are single stray electrons; fires on the stop edge); bin 4 45", "bin 8: 4 (defaults, 15-frame pseudo-sweep; single frames fail); bin 4: 46 (defaults), 9 (owner 4/7/5)", "-"],
    ["Agreement", "holds 74-87 % of XDS and pyREyes peaks at bin 4 (74 % of XDS at bin 8) + 2-4x more weak high-angle candidates", "87 % of its bin-4 peaks are in bgsub; 58 % in XDS", "76 % in bgsub; 56 % in pyREyes", "-"],
    ["Background under the same peaks (median ratio to bgsub) and net intensity", "1 (reference); net I median 23 (bin 8), 101 (bin 4) counts", "no background", "1.04 (bin 8), 1.01 (bin 4): agrees under peaks although wrong between them", "<b>1.15 at both binnings: loses 5 % (bin 8) and 14 % (bin 4) of the net intensity, ~17 % on bright spots</b>"],
    ["Time per frame (this Mac)", "62 ms (1024<super>2</super>), 155 ms (2048<super>2</super>), Python + numba", "31 / 149 ms (two blurs)", "33 / 125 ms (Fortran, 4 threads)", "-"],
], [1.35 * inch, 1.75 * inch, 1.35 * inch, 1.55 * inch, 1.3 * inch]))
S.append(Spacer(1, 3))
S.append(img(FIG, 0.9 * W))
S.append(Paragraph("Eight spots on one bin-4 frame found by all three methods: raw (top) and after bgsub subtraction (bottom), same colour scale; captions give the disc sum, bgsub's background and the net intensity, with the flat-annulus net in brackets "
                   "(8-25 % lower on every spot). Full tables and the overlays of every method's peaks on the original frames: notes/reports/2026-11-05_compare_v0.md, notes/figures/compare_v0/.", st_cap))

S.append(Paragraph("5. Known defects and limits", st_h))
S.extend(bullets([
    "<b>Sparse counts (R1).</b> The one failing synthetic cell (0.2 counts per box) had no statistical power (SE 0.72 % vs a 0.5 % limit); on 64 fresh frames H2.1 is -0.47 +- 0.72 %. Needs a power-adequate re-test, possibly a count-level-aware gate.",
    "<b>Sharp rings (A277).</b> A film ring narrower than the spline knots (4.6 A, one classic session) is not removed: most of its pixels are count-law flags. Open option: a ring-zone high-breakdown first pass.",
    "<b>Coverage.</b> Apollo bin 8 and bin 4 tested on real frames; bin 16 synthetic only (variance not honest, A196); <b>CETA-D bin 2 and TVIPS F30 untested</b>, need an integrating-detector variance model first. Not modelled: structure neither smooth on the model's scales nor linear at the annulus scale (A279). compare v0 is descriptive, not pre-registered, no truth for real peaks.",
]))

S.append(Paragraph("6. Next steps (ROADMAP.md) and what the owner must decide", st_h))
S.extend(bullets([
    "<b>1. Background-only validity spec</b>, pre-registered, with thresholds the owner endorses: bias everywhere and under masked spots with stated power; per-pixel variance honesty; stop / bar / crystal zones; count-law regimes; timing reported, not gated. "
    "<b>2.</b> Bring v2.1 to valid under it (sparse-cell re-test; optional ring-zone pass). <b>3-5.</b> Integrating-detector variance model, then CETA-D bin 2 and TVIPS F30 validation; honest variance at bin 16.",
    "<b>Owner decisions:</b> (a) per-frame <b>output format</b> (proposed: float32 background map + per-pixel variance map in stored counts, subtracted frame on request, sidecar with version / count law / zero level / mask / centre); "
    "(b) <b>datasets</b> for CETA-D bin 2 and TVIPS F30; (c) whether to run superxds itself on a Linux x86-64 box (B1); (d) the mov11 XDS.INP beam centre is 75 px off and was never refined by XDS (B2): re-check the XDS-merge study.",
]))

doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
