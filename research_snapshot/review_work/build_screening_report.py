"""A compact private scorecard; uses verified aggregate experiment summaries only."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

WORKSPACE = Path(__file__).resolve().parents[1]
PRIVATE = WORKSPACE / "private_runs/screening_230"
OUT = WORKSPACE / "output/PRC_2026_Screening_Results.pdf"
S = json.loads((PRIVATE / "screening_summary.json").read_text())
R = json.loads((PRIVATE / "followup_summary.json").read_text())
N = json.loads((PRIVATE / "next_directions.json").read_text())
for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
INK = colors.HexColor("#202629")
TEAL = colors.HexColor("#087E78")
RED = colors.HexColor("#B64D36")
GRAY = colors.HexColor("#657278")
LIGHT = colors.HexColor("#EEF3F4")
GRID = colors.HexColor("#D3DCDE")
W, H, M = 612, 792, 42
C = canvas.Canvas(str(OUT), pagesize=(W, H))
C.setTitle("PRC 2026 - Eight Screens and the Path to 230 Seconds")
C.setAuthor("Local Experiment Review")
LAYOUT = []


def text(value, x, top, size=10, bold=False, color=INK, align="left"):
    C.setFont("Arial-Bold" if bold else "Arial", size)
    C.setFillColor(color)
    draw = C.drawRightString if align == "right" else C.drawString
    draw(x, H - top - size, str(value))


def para(value, top, x=M, width=528, size=10.5, leading=15, max_h=100):
    p = Paragraph(value, ParagraphStyle("p", fontName="Arial", fontSize=size, leading=leading, textColor=INK))
    _, h = p.wrap(width, H)
    assert h <= max_h and top + h < 738, (C.getPageNumber(), value[:50], h, top)
    p.drawOn(C, x, H - top - h)
    LAYOUT.append({"page": C.getPageNumber(), "top": top, "x": x, "width": width, "height": h, "text": value})
    return top + h


def line(top, x=M, width=528):
    C.setStrokeColor(GRID)
    C.setLineWidth(.6)
    C.line(x, H - top, x + width, H - top)


def page(n, title, subtitle):
    if n > 1:
        C.showPage()
    text("PRC 2026  /  PRIVATE EXPERIMENT UPDATE", M, 24, 8.5, True, TEAL)
    text(f"{n:02d} / RESULTS", 570, 24, 8.5, color=GRAY, align="right")
    para(title, 65, size=25, leading=29, max_h=60)
    para(subtitle, 108, size=10.5, leading=15, max_h=45)
    line(746)
    text("Verified local experiments. Raw records and models remain outside Git.", M, 758, 7.5, color=GRAY)
    text(f"15 SEP 2026   |   {n} / 4", 570, 773, 7.5, color=GRAY, align="right")


LABELS = {"baseline": "Baseline", "A_rows": "A  More rows", "B_trees": "B  More trees",
          "C_prefix": "C  Flight prefix", "D_rome": "D  Rome correction",
          "E_rome_local": "Rome-only refinement", "F_stronger_residual": "Stronger residual",
          "B_plus_D": "400 trees + Rome", "F_plus_D": "Stronger residual + Rome",
          "G_shared_missing": "Movement-only transfer", "B_plus_D3": "400 trees + 3 Rome heads",
          "F_plus_D3": "Stronger + 3 Rome heads", "FB_plus_D": "Stronger + B/D fallbacks",
          "H_gpu_full": "Full-data GPU residual"}
best_name = N["best_candidate"]
best = R["candidates"][best_name]
base = R["candidates"]["baseline"]
score = best["seasonal_rmse_sec"]

page(1, "Eight screens. Measured gains.", "The target is now 230 seconds. Complete July and November cohorts, original folds and raw labels.")
for x, value, label in [(42, "8 / 8", "SCREENS VERIFIED"), (228, f"{score:.2f}s", "LEADING RECIPE"), (432, "230s", "TARGET")]:
    text(value, x, 170, 28, True, TEAL)
    text(label, x, 207, 7.8, True, GRAY)
line(239)
text("Original screening batch", M, 256, 15, True)
text("Direction", M, 292, 9, True, GRAY)
for title, x in [("F1 / July", 346), ("F3 / Nov", 443), ("Seasonal", 560)]:
    text(title, x, 292, 9, True, GRAY, "right")
for i, name in enumerate(["baseline", "A_rows", "B_trees", "C_prefix", "D_rome"]):
    top = 319 + i * 35
    if name in ["B_trees", "D_rome"]:
        C.setFillColor(LIGHT)
        C.rect(M, H - top - 26, 528, 32, stroke=0, fill=1)
    value = S["candidates"][name]
    text(LABELS[name], M + 8, top, 10.5, name == "baseline")
    for f, x in [("F1", 346), ("F3", 443)]:
        text(f"{value['folds'][f]['overall']['rmse_sec']:.2f}", x, top, 10.5, align="right")
    text(f"{value['seasonal_rmse_sec']:.2f}", 560, top, 10.5, True, TEAL if name in ["B_trees", "D_rome"] else INK, "right")
para("<b>Advance:</b> more boosting capacity and the Rome schedule route. Doubling the shared sample at 180 trees brought only 0.28 seconds; the prefix feature did not improve the combined score.", 520, max_h=65)
para("Seasonal RMSE weights July and November mean squared errors using the January/July ranking proportions. Lower is better. This local estimate is not a measured leaderboard result.", 592, size=9.5, leading=14, max_h=65)
para(f"The original baseline reproduced exactly on F1, F2, F3 and G1. The best tested configuration reduces mean squared error by <b>{best['mse_reduction_pct']:.1f}%</b> relative to that baseline.", 665, size=10, max_h=55)

page(2, "Build on the successful routes", "The follow-up models preserve the baseline where their new component does not apply.")
text("Local seasonal RMSE, seconds", M, 172, 12, True)
names = [n for n in ["baseline", "B_trees", "D_rome", "E_rome_local", "B_plus_D", "F_stronger_residual", "F_plus_D", "FB_plus_D", "H_gpu_full", "G_shared_missing"] if "seasonal_rmse_sec" in R["candidates"].get(n, {})]
x0, width, top0, step = 225, 305, 226, 26
for tick in [230, 280, 330, 380, 430]:
    x = x0 + (tick - 220) / 230 * width
    C.setStrokeColor(GRID)
    C.line(x, H - 213, x, H - top0 - (len(names) - 1) * step - 20)
    text(str(tick), x - 8, 197, 8, color=GRAY)
for i, name in enumerate(names):
    top = top0 + i * step
    v = R["candidates"][name]["seasonal_rmse_sec"]
    text(LABELS[name], M, top, 9.6, name == best_name)
    x = x0 + (v - 220) / 230 * width
    C.setFillColor(TEAL if name == best_name else GRAY)
    C.circle(x, H - top - 6, 4, stroke=0, fill=1)
    text(f"{v:.1f}", x + 8, top - 1, 9.5, name == best_name)
para("Point chart uses a 220-450 second axis. The failed transfer experiment is retained. The 230 target is supplied by the user, not an observed result on these folds.", 516, size=9, leading=13, max_h=42)
text("How the best candidate works", M, 578, 14, True)
para(escape(N["best_architecture"]), 607, max_h=95)
para("The GPU candidate changes device, sample size and tree ceiling together. Later tests are adaptive development evidence. No blend weight or routing threshold was fitted to scoring labels.", 700, size=8.8, leading=12, max_h=36)

page(3, "Check the strength of the result", "A lower headline score must survive different dates, route checks and saved-model verification.")
text("Broader fold checks", M, 176, 14, True)
for title, x in [("Model", 42), ("F1", 290), ("F2", 378), ("F3", 466), ("G1", 560)]:
    text(title, x, 212, 9, True, GRAY, "right" if x > 42 else "left")
display = list(dict.fromkeys(["baseline", "B_trees", "D_rome", best_name]))
for i, name in enumerate(display):
    top = 243 + i * 33
    text(LABELS.get(name, name), M, top, 10, name == best_name)
    for f, x in [("F1", 290), ("F2", 378), ("F3", 466), ("G1", 560)]:
        metric = R["candidates"][name]["folds"].get(f)
        text(f"{metric['overall']['rmse_sec']:.1f}" if metric else "Pending", x, top, 10, align="right")
line(391)
text("What the checks establish", M, 412, 14, True)
para(escape(N["validation_summary"]), 442, max_h=90)
text("What still needs care", M, 549, 14, True)
para("The Rome head improves rare extreme durations but increases error on ordinary Rome missing-clock flights: about 8.7% on F1 and 13.1% on F3. Its single-fold bootstrap intervals cross zero even though every leave-one-day-out comparison improves. Keep this tradeoff visible.", 579, max_h=85)
para(escape(N["seed_summary"]), 673, size=9.5, leading=14, max_h=56)

page(4, "The next moves toward 230", f"Perfect missing-clock predictions alone would still leave {best['perfect_missing_floor_sec']:.1f}s, with other predictions fixed.")
text(f"{score - 230:.1f}s", M, 168, 30, True, RED)
text("REMAINING LOCAL RMSE GAP", M, 207, 8.5, True, GRAY)
text(f"{100*(1-(230/score)**2):.1f}%", 328, 168, 30, True, RED)
text("FURTHER MSE REDUCTION NEEDED", 328, 207, 8.5, True, GRAY)
if score > 280:
    text(f"280 milestone: {100*(1-(280/score)**2):.1f}% less total MSE, with scoring cohorts unchanged.", M, 226, 8.5, color=GRAY)
line(242)
for i, item in enumerate(N["directions"]):
    top = 265 + i * 101
    text(f"0{i+1}", M, top, 17, True, TEAL)
    text(item["title"], 84, top, 12, True)
    para(escape(item["detail"]), top + 26, x=84, width=486, size=10, leading=14, max_h=68)
line(677)
para("<b>Privacy verified:</b> all 14 raw hashes unchanged; no private Parquet files or trained models in either checkout. Data, caches, models, predictions and this report remain in external sibling folders. Nothing was uploaded or published.", 695, size=9, leading=12.5, max_h=40)
C.save()
(PRIVATE / "report_layout.json").write_text(json.dumps(LAYOUT, indent=2), encoding="utf-8")
print(OUT)
