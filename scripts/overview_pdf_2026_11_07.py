"""Two-page overview PDF for pod/bgsub (2026-11-07): what the subtraction is, the mathematics, how universal it is, how it
looks on real frames, where the project stands.  Numbers from CLAUDE.md, the v2.1 report and compare v0 / v1 (no new
measurement).  Figure: notes/figures/overview_2026-11-07/how_it_works.png (made from the compare-v1 frames).
Run: ~/micromamba/envs/serial3ded/bin/python scripts/overview_pdf_2026_11_07.py
"""
import io
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes/reports/2026-11-07_bgsub_overview_2p.pdf"
FIG = ROOT / "notes/figures/overview_2026-11-07/how_it_works.png"
INK = colors.HexColor("#1b1f23"); MUTED = colors.HexColor("#5b6470"); RULE = colors.HexColor("#c9ced6"); ACCENT = colors.HexColor("#0b5fa5"); BAND = colors.HexColor("#eef3f8")
W = 7.3 * inch
st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14.5, leading=17.5, textColor=INK, spaceAfter=2)
st_sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, textColor=MUTED, spaceAfter=5)
st_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, leading=13.5, textColor=ACCENT, spaceBefore=5, spaceAfter=2.5)
st_b = ParagraphStyle("b", fontName="Helvetica", fontSize=8.5, leading=10.6, textColor=INK, spaceAfter=2.5)
st_bl = ParagraphStyle("bl", parent=st_b, leftIndent=10, bulletIndent=1, spaceAfter=1.5)
st_cap = ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=7.6, leading=9.4, textColor=MUTED, spaceBefore=1.5, spaceAfter=4)
st_eq = ParagraphStyle("ec", fontName="Helvetica", fontSize=8.0, leading=10.0, textColor=MUTED)
st_tb = ParagraphStyle("tb", fontName="Helvetica", fontSize=7.6, leading=9.2, textColor=INK); st_tbh = ParagraphStyle("tbh", fontName="Helvetica-Bold", fontSize=7.6, leading=9.2, textColor=INK)


def eq(tex, size=9.2, dpi=300, maxw=3.3 * inch):
    fig = plt.figure(figsize=(8, 1), dpi=dpi); fig.text(0.01, 0.5, tex, fontsize=size, color="#1b1f23", va="center")
    buf = io.BytesIO(); fig.savefig(buf, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.03); plt.close(fig); buf.seek(0)
    w, h = PILImage.open(buf).size; W0, H0 = w * 72 / dpi, h * 72 / dpi
    if W0 > maxw: H0 = H0 * maxw / W0; W0 = maxw
    buf.seek(0); return Image(buf, width=W0, height=H0)


def eqrow(tex, caption, maxw=3.3 * inch):
    e = eq(tex, maxw=maxw); cw = max(e.drawWidth + 8, 1.6 * inch)
    t = Table([[e, Paragraph(caption, st_eq)]], colWidths=[cw, W - cw])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    return t


def tbl(rows, widths):
    data = [[Paragraph(c, st_tbh if i == 0 else st_tb) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("BACKGROUND", (0, 0), (-1, 0), BAND), ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT)] +
                          [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fb")) for i in range(2, len(rows), 2)]))
    return t


def img(path, width):
    w, h = PILImage.open(path).size; return Image(str(path), width=width, height=width * h / w)


def bullets(items):
    return [Paragraph(x, st_bl, bulletText="•") for x in items]


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(0.6 * inch, 0.42 * inch, "pod/bgsub  -  overview 2026-11-07  -  D. Eremin, Nelson Lab, Caltech  -  repo mit-eremin/pod-subtract")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"page {doc.page} of 2")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5); canvas.line(0.6 * inch, 0.58 * inch, letter[0] - 0.6 * inch, 0.58 * inch); canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.7 * inch, title="bgsub overview 2026-11-07", author="Dmitry Eremin")
S = []
# ------------------------------------------------------------------ page 1
S.append(Paragraph("Background subtraction for electron diffraction frames: what it is, the mathematics, how far it reaches", st_title))
S.append(Paragraph("Subproject <b>pod/bgsub</b> (GitHub mit-eremin/pod-subtract), Dmitry Eremin, Nelson Lab, Caltech, 2026-11-07. Package <b>bgsub</b> (Python + numba, 47 tests). "
                   "Goal in the owner's words: a fast, mathematically correct, universal background ESTIMATION AND SUBTRACTION for diffraction frames, better than the flat local boxes of XDS / DIALS; background only, not an integrator, not for XDS. "
                   "Page 1: what the algorithm does and why it is correct; page 2: how it looks on real frames from three detectors and where the project stands.", st_sub))

S.append(Paragraph("1. What is going on", st_h))
S.append(Paragraph("A diffraction frame is Bragg spots on a smooth, bright, structured background (beam halo, film rings, stop shadow and penumbra, grid bars). Integration programs estimate it locally and flatly (a box around each spot) and lose intensity on a curved background; "
                   "screening programs threshold a band-passed image and never estimate it. bgsub fits <b>one smooth model of the whole frame's background in the detector's own stored counts</b>, with weights and outlier tests from the detector's declared count statistics, "
                   "and returns the background map, its per-pixel variance and the subtracted frame. Versions v0-v2.1 (v1 default; v2.1 adds re-centring and an elliptical ring refinement), all frozen and reproducible.", st_b))

S.append(Paragraph("2. The mathematics", st_h))
S.append(eqrow(r"$K_p \sim \mathrm{law}\;(\mu_p),\quad \mathrm{E}[K_p]=b_p=B(r_p,\varphi_p),\quad \mathrm{Var}[K_p]=g(b_p)$",
               "<b>Observation model.</b> K<sub>p</sub> is the stored value of pixel p. Its mean is a smooth background B at the pixel's polar position; its variance is a known function g of the mean, the detector's <b>count law</b>: g(b) = b for a Poisson counter; "
               "for the Apollo the law of round-half-even(3-frame electron sum / m) with tie rule and dead time (sub-linear at bin 8, 16); g(b) = gain·b + read variance for an integrating camera. Only g is declared; no distribution shape is assumed."))
S.append(eqrow(r"$B(r,\varphi)=\sum_k \theta_k\,\phi_k(r,\varphi)$",
               "<b>Linear smooth model.</b> A fine cubic B-spline in radius (rings, the steep halo) times azimuthal harmonics m &le; 6 on a coarse radial spline, plus four derivative columns that absorb a centre error and an ellipticity, plus "
               "penumbra columns on the 1-4 pixels next to the beam stop. Everything is linear in the coefficients &theta;; the frame is first reduced to sufficient statistics on polar cells (1-px rings x 64 sectors x 2 checkerboard halves): cell sums S<sub>c</sub> and counts n<sub>c</sub>."))
S.append(eqrow(r"$\hat\theta_A=\arg\min_\theta\sum_{c\in A} w_c\,(S_c-n_c\,X_c\theta)^2,\qquad w_c=\frac{1}{n_c\,g(\hat b_c^{(B)})}$",
               "<b>Cross-fitted weighted least squares.</b> Half A of the checkerboard is fitted with weights computed from the OTHER half's estimate of the mean, and vice versa; the two fits are averaged. Because the weights do not depend on the "
               "data being fitted, the estimator stays <b>linear in the counts</b>: E[the estimate of &theta;] = &theta; exactly whenever the true background lies in the model's span, <b>for any count law</b>: the law affects only the efficiency (weights, thresholds, variances), never the bias. No median, no Poisson assumption, no iteration on the same data."))
S.append(eqrow(r"$P(K_p > k^\ast(\hat b_p))=10^{-6},\qquad P(\sum_{3\times3}K > k^\ast_{9})=10^{-6}$",
               "<b>What is kept out.</b> Spots, hot pixels and zingers are excluded by one-sided tail tests of the declared law, per pixel and per 3 x 3 window, dilated by one pixel; spots the caller will measure are always excluded by discs. "
               "Excluded pixels never enter the fit, so the background under a spot is what the surrounding smooth model says, not the spot's own tails (the flat-box failure)."))
S.append(eqrow(r"$\mathrm{Var}[\hat\theta]=\hat\phi\,(X^{\!\top}WX)^{-1}X^{\!\top}W\,\Sigma\,WX(X^{\!\top}WX)^{-1},\quad \hat B_p=X_p\hat\theta,\quad \mathrm{Var}[\hat B_p]=X_p\mathrm{Var}[\hat\theta]X_p^{\!\top}$",
               "<b>Honest variance.</b> A sandwich covariance with the frame's own Pearson dispersion &phi; (the ratio of observed to law variance on the used pixels) gives a per-pixel variance of the background map; the subtracted frame K - B carries "
               "g(B) + Var[B]. The dispersion is itself a diagnostic: 1.02 on Apollo bin 8 real frames, i.e. the law and the fit agree with the data.", maxw=4.4 * inch))
S.append(Paragraph("Measured on real Apollo frames (compare v0, 2026-11-05): the fitted background matches the spot-free data to within 1 % in every 32-px radial band at bin 4 and bin 8 (bin 8: mean and variance agree with the law to 10<super>-4</super> counts / px); "
                   "a flat local annulus on the same spots removes 15 % more 'background' and loses 5-14 % of the net intensity (17 % on bright spots); the XDS table x one scalar is 9-14 % low in the halo. Fit time 60-160 ms per 1024<super>2</super>-2048<super>2</super> frame.", st_b))

S.append(Paragraph("3. How universal it is", st_h))
S.extend(bullets([
    "<b>Universal by construction:</b> the unbiasedness argument above holds for any detector whose stored value has a declared mean-variance law, any geometry the centre / ellipticity columns and the mask can express, any stop (penumbra columns from the mask), any binning. Nothing is tuned per sample.",
    "<b>Declared per detector, not inferred:</b> the count law g(b) and the zero level. Apollo: the stored-count law per binning (m = 1, 4, 16 e per count, tie rule, dead time, TIFF pedestal) is measured and built in; tested on real bin 4, 8 and 16 frames. CETA-D and TVIPS: a Poisson(gain, read noise) law exists but is NOT calibrated, "
    "and compare v1 showed their noise is spatially correlated (a 3 x 3 window carries ~3 x the variance of nine independent pixels), so the law must carry a correlation term; with a declared estimate both already run (real binned-2 movies). Not yet: Apollo bin 2 (4096<super>2</super>).",
    "<b>Not universal:</b> structure outside the model's span is not modelled: a film ring sharper than the spline knots (one classic session), the 20-40 px glow at the edge of a CETA-D / TVIPS stop, a sample's own diffuse scattering at the annulus scale. These are the known limits, each with a named anomaly.",
]))

S.append(PageBreak())
# ------------------------------------------------------------------ page 2
S.append(Paragraph("4. What it looks like on real frames", st_h))
S.append(img(FIG, 0.88 * W))
S.append(Paragraph("Three detectors / binnings from compare v1 (2026-11-05): Apollo bin 8 fly-scan snapshot at 0.08 counts / px (top), Apollo bin 16 grossular at 0.5 counts / px (middle), CETA-D grossular rotation frame at 59 counts / px (bottom). "
                   "Columns: the original frame (log), bgsub's fitted background B on the same scale, the subtracted frame K - B (linear: a flat field with the Bragg spots standing alone), and the radial profile of spot-free pixels against the model with the residual in percent (green). "
                   "The CETA-D row uses the declared read-noise + correlation law estimated from the frame; the glow crescent right of its stop is the known unmodelled edge effect. Side-by-side panels against pyREyes and XDS / superxds for all five datasets: notes/reports/2026-11-05_compare_v1_panels.pdf.", st_cap))

S.append(Paragraph("5. Where we are", st_h))
S.append(tbl([
    ["", "status (2026-11-07)"],
    ["Algorithm", "v0-v2.1 implemented, frozen, 47 tests; v1 default, v2.1 opt-in (re-centring, elliptical ring refinement, gated local layer). Fit 60-160 ms per frame, plan cached."],
    ["Correctness, synthetic (truth known)", "60 lacey cells x bin 4 / 8 x dose x structure: global estimator unbiased in 60 / 60 cells; the hybrid fails 1 cell at 0.2 counts per box where the test had no statistical power. Count statistics / truncation: pass (24 cells)."],
    ["Correctness, real Apollo", "Background matches spot-free data to < 1 % in every radial band (bin 4, 8); better than a flat local background in every injected-peak regime (flat loses 5-14 % intensity). Peaks per frame bgsub / pyREyes / XDS: bin 4 156 / 50 / 34, bin 8 22 / 20 / 4, bin 16 53 / 19 / 2."],
    ["Yardsticks", "pyREyes 3.6.3 re-implemented exactly (no background model); superxds = XDS ported to numpy and validated against XDS (tables 99.98 % exact, spot lists identical): a sweep-averaged table x one scalar, 9-14 % off in the halo, fails on single sparse bin-8 frames."],
    ["Other detectors", "CETA-D and TVIPS run with a declared law estimated from the frame (gain 3.5 / 10, read variance 8 / 0, correlation factor 2.8 / 3.1); peaks 101 / 20 / 12 and 34 / 9 / 12. A calibrated law from dark / flat frames is the next real-data task."],
    ["Verdict by its own pre-registered rules", "v2.1 NOT VALID by two narrow margins (one sparse synthetic cell; 54 vs 50 ms); the thresholds were the orchestrator's, and the owner ruled timing relative. A background-only validity spec with owner-endorsed thresholds is ROADMAP item 1."],
], [1.7 * inch, 5.6 * inch]))
S.append(Spacer(1, 3))
S.append(Paragraph("6. Next steps and the owner's decisions", st_h))
S.append(Paragraph("<b>Next:</b> 1. a background-only validity spec, pre-registered (bias everywhere and under masked spots with stated power, variance honesty, stop / bar / crystal zones, count-law regimes; timing reported); 2. v2.1 to valid under it; "
                   "3. a calibrated integrating-detector law with noise correlation, then CETA-D / TVIPS validation on the movies held locally; 4-5. Apollo bin 2, bin-16 variance. "
                   "<b>Owner decisions:</b> (a) thresholds to endorse; (b) per-frame output format (proposed: float32 background + variance maps in stored counts, subtracted frame on request, sidecar); (c) CETA-D / TVIPS sessions and dark / flat frames to calibrate on; (d) the XDS-merge mov11 beam origin is 75 px off, never refined by XDS: re-check.", st_b))
doc.build(S, onFirstPage=footer, onLaterPages=footer); print("wrote", OUT)
