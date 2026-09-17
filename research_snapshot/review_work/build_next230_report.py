"""Short private experiment report from independently verified aggregate receipts."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'private_runs/next_230'
S = json.loads((PRIVATE / 'summary.json').read_text())
D = json.loads((PRIVATE / 'decision.json').read_text())
V = json.loads((PRIVATE / 'inference_verification.json').read_text())
FINAL = ROOT / 'output/PRC_2026_Next_Batch_Results.pdf'
for name, file in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold')
INK, TEAL, RED, GRAY, GRID = [colors.HexColor(v) for v in ['#202629', '#087E78', '#B64D36', '#657278', '#D3DCDE']]
W, H, M = 612, 792, 42
C = canvas.Canvas(str(FINAL), pagesize=(W, H))
C.setTitle('PRC 2026 - Next Batch Toward 230 Seconds')
C.setAuthor('Private Local Experiment Review')
LAYOUT = []


def text(value, x, top, size=10, bold=False, color=INK, right=False):
    C.setFont('Arial-Bold' if bold else 'Arial', size)
    C.setFillColor(color)
    draw = C.drawRightString if right else C.drawString
    draw(x, H - top - size, str(value))


def para(value, top, x=M, width=528, size=10.3, leading=14.5, max_h=100):
    p = Paragraph(value, ParagraphStyle('p', fontName='Arial', fontSize=size, leading=leading, textColor=INK))
    _, h = p.wrap(width, H)
    assert h <= max_h and top + h <= 731, (C.getPageNumber(), value[:50], h, top)
    p.drawOn(C, x, H - top - h)
    LAYOUT.append({'page': C.getPageNumber(), 'top': top, 'height': h, 'text': value})


def line(top):
    C.setStrokeColor(GRID)
    C.setLineWidth(.6)
    C.line(M, H-top, W-M, H-top)


def page(n, title, subtitle):
    if n > 1:
        C.showPage()
    text('PRC 2026  /  PRIVATE TRAINING UPDATE', M, 25, 8.5, True, TEAL)
    text(f'{n:02d} / NEXT BATCH', W-M, 25, 8.5, color=GRAY, right=True)
    para(title, 65, size=25, leading=29, max_h=60)
    para(subtitle, 110, size=10, leading=14, max_h=42)
    line(746)
    text('Local development folds. Confidential artifacts stay outside Git.', M, 757, 7.5, color=GRAY)
    text(f'SEPTEMBER 2026  |  {n} / 4', W-M, 773, 7.5, color=GRAY, right=True)


LABELS = {
    'capacity_d8_5000': 'Depth 8 / 5000 trees', 'capacity_d6_5000': 'Depth 6 / 5000 trees',
    'capacity_d10_2500': 'Depth 10 / 2500 trees', 'runway_d8_2500': 'Prior runway context',
    'rome_base_1200': 'Rome: more capacity', 'rome_context_1200': 'Rome: flight + timing',
    'clock_global': 'Global clock blend', 'clock_airport': 'Airport clock blend',
    'clock_cpu_global': 'Global clock blend', 'clock_cpu_airport': 'Airport clock blend',
    'rome_schedule_mixture': 'Rome: timing reliability',
    'rome_mixture_calibrated': 'Rome: calibrated reliability',
    'rome_mixture_ensemble': 'Rome: three-seed average',
    'capacity_and_rome_ensemble': 'Capacity + Rome ensemble',
    'clock_and_rome_ensemble': 'Clock blend + Rome ensemble',
    'runway_and_rome_ensemble': 'Runway + Rome ensemble',
    'clock_learned_gate': 'Feature-based clock gate',
}


def label(name):
    if name in LABELS:
        return LABELS[name]
    if '_seed' in name:
        base, seed = name.split('_seed')
        return f'{LABELS.get(base, base)} / seed {seed[-2:]}'
    return name.replace('_', ' ')


score = D['seasonal_rmse_sec']
page(1, 'Moving toward 230 seconds', 'Measured progress from the 299.05-second reference. Lower RMSE is better; this is not a leaderboard score.')
for x, value, title in [(42, f'{score:.2f}s', 'BEST VALIDATED RECIPE'), (237, f'{D["gain_sec"]:.2f}s', 'GAIN THIS BATCH'), (445, '230s', 'TARGET')]:
    text(value, x, 171, 29, True, TEAL)
    text(title, x, 209, 7.6, True, GRAY)
line(241)
text('Progress against the objective', M, 262, 14, True)
plot_left, plot_right = 205, 535
for tick in [230, 250, 280, 300, 330]:
    x = plot_left + (tick-220)/120*(plot_right-plot_left)
    C.setStrokeColor(GRID)
    C.line(x, H-316, x, H-468)
    text(str(tick), x-9, 295, 8, color=GRAY)
for i, (name, value, color) in enumerate([('Original', 331.175989, GRAY), ('Previous reference', 299.053253, GRAY), ('This batch', score, TEAL), ('Goal', 230, RED)]):
    top = 329+i*34
    x = plot_left + (value-220)/120*(plot_right-plot_left)
    C.setFillColor(color)
    C.circle(x, H-top-5, 4, fill=1, stroke=0)
    text(name, M, top-1, 9.5, color=color)
    text(f'{value:.2f}', x+9, top-1, 9, True, color)
para(f'<b>Decision:</b> continue from {escape(label(D["recommended_development_reference"]))}. '
     f'This batch reduces seasonal mean squared error by {D["mse_reduction_pct"]:.2f}% versus the previous reference. '
     f'The 280-second milestone is {"reached" if D["intermediate_280_reached"] else "still ahead"}.', 507, max_h=65)
para(f'<b>The remaining gap:</b> {score-230:.2f} seconds of RMSE, equivalent to '
     f'{D["additional_mse_reduction_to_230_pct"]:.1f}% further MSE reduction. More compute alone is not evidence that this gap is attainable.', 597, max_h=60)
para('Seasonal RMSE weights July and November squared errors using ranking-season proportions: 55.71% July and 44.29% winter. The full cohorts and original raw labels are preserved.', 681, size=9, leading=12.5, max_h=40)

page(2, 'What the experiments found', 'Independent changes first. All scored attempts stay in the results, including regressions.')
rows = [(n, c) for n, c in S['candidates'].items() if 'seasonal_rmse_sec' in c and '_seed' not in n]
rows.sort(key=lambda pair: pair[1]['seasonal_rmse_sec'])
assert len(rows) <= 18
row_step=24 if len(rows)<=14 else 21.5
text('Recipe', M, 174, 9, True, GRAY)
for title, x in [('July', 342), ('November', 443), ('Seasonal', 566)]:
    text(title, x, 174, 9, True, GRAY, True)
for i, (name, item) in enumerate(rows):
    top = 202 + i*row_step
    winner = name == D['recommended_development_reference']
    text(label(name), M, top, 9.4, winner, TEAL if winner else INK)
    for key, x in [('F1', 342), ('F3', 443)]:
        text(f'{item["folds"][key]["overall"]["rmse_sec"]:.2f}', x, top, 9.5, right=True)
    text(f'{item["seasonal_rmse_sec"]:.2f}', 566, top, 9.5, winner, TEAL if winner else INK, True)
bottom = 202+len(rows)*row_step+14
line(bottom)
para('Advance only when the gain clears 2 seconds, improves both screening months, survives every day-removal comparison, and keeps missing-clock regression below 5%. Then check F2 and G1.', bottom+20, size=9.7, leading=14, max_h=60)
para('Clock blends use fit-only expert predictions on the tuning month. In the flight-code screen, unseen exact codes used the paired base head. The selected Rome mixture instead learns soft reliability from observed features. Runway features include only strictly earlier movements, with month-isolated coverage.', bottom+87, size=9.7, leading=14, max_h=70)

page(3, 'How strong is the evidence?', 'Prediction replay and temporal checks support the result; reused development folds still limit generalization claims.')
text('Fold', M, 176, 9, True, GRAY)
text('Previous', 296, 176, 9, True, GRAY, True)
text('Selected', 419, 176, 9, True, GRAY, True)
text('Change', 566, 176, 9, True, GRAY, True)
best = S['candidates'].get(D['recommended_development_reference'])
for i, fold in enumerate(['F1', 'F2', 'F3', 'G1']):
    top = 208+i*34
    a = S['reference']['folds'][fold]['rmse_sec']
    b = best['folds'][fold]['overall']['rmse_sec'] if best else a
    text({'F1':'F1 / July', 'F2':'F2 / October', 'F3':'F3 / November', 'G1':'G1 / distant October'}[fold], M, top, 10)
    text(f'{a:.2f}', 296, top, 10, right=True)
    text(f'{b:.2f}', 419, top, 10, True, TEAL, True)
    text(f'{b-a:+.2f}', 566, top, 10, right=True)
line(367)
text('How prediction works', M, 389, 14, True)
def node(title,x,top,width,height=27):
    C.setFillColor(colors.HexColor('#EEF3F4'))
    C.setStrokeColor(GRID)
    C.roundRect(x,H-top-height,width,height,3,fill=1,stroke=1)
    text(title,x+7,top+8,8.1,True)

def arrow(x1,t1,x2,t2):
    C.setStrokeColor(TEAL)
    C.setLineWidth(1)
    C.line(x1,H-t1,x2,H-t2)
    C.line(x2,H-t2,x2-4,H-t2+3)
    C.line(x2,H-t2,x2-4,H-t2-3)

node('Observed features',42,440,117)
node('Direct estimate',193,420,109)
node('NM + correction',193,459,109)
node('Learned gate',352,440,98)
node('Prediction',487,440,83)
arrow(159,453,193,434)
arrow(159,453,193,472)
arrow(302,434,352,453)
arrow(302,472,352,453)
arrow(450,453,487,453)
text('NM: Network Manager off-block clock. Its difference from takeoff is the proxy.',42,490,8.5,color=GRAY)
text('Missing clocks use a specialist; the Rome schedule route uses a reliability mixture.',42,503,8.5,color=GRAY)
text('Verification and tradeoffs', M, 529, 14, True)
notes = json.loads((PRIVATE / 'report_notes.json').read_text())
para(f'{V["count"]} saved prediction pipelines replayed exactly from checked model files. '+escape(notes['stability']), 559, max_h=108)
para('F2 and G1 share October outcomes, so they are not independent replication. All these folds were exposed during development; December is also already exposed. A genuinely new labeled period remains the strongest external check.', 672, size=9, leading=12.5, max_h=50)

page(4, 'Where the next gains must come from', 'Follow the remaining squared-error budget. Keep the scientific and privacy checks intact.')
text('Remaining seasonal squared error', M, 174, 14, True)
shares = D['seasonal_error_share_pct']
palette = [TEAL, colors.HexColor('#327CA0'), RED, colors.HexColor('#8C9498')]
keys = ['LIRF_missing', 'LIRF_present_or_invalid', 'other_missing', 'other_present_or_invalid']
labels = ['Rome / missing clock', 'Rome / observed or invalid', 'Other airports / missing', 'Other / observed or invalid']
x = M
for key, color in zip(keys, palette):
    width = shares[key]/100*528
    C.setFillColor(color)
    C.rect(x, H-240, width, 29, fill=1, stroke=0)
    x += width
for i, (key, title, color) in enumerate(zip(keys, labels, palette)):
    xx, top = M+(i%2)*274, 259+(i//2)*30
    C.setFillColor(color)
    C.rect(xx, H-top-9, 8, 8, fill=1, stroke=0)
    text(f'{title}: {shares[key]:.1f}%', xx+15, top-2, 8.6)
para(f'Even perfect missing-clock predictions, with other predictions fixed, would leave '
     f'{D["perfect_missing_floor_sec"]:.1f}s. This is a component error budget, not a noise floor.', 334, size=9.7, leading=14, max_h=50)
for i, direction in enumerate(notes['directions']):
    top = 408+i*84
    text(f'0{i+1}', M, top, 16, True, TEAL)
    text(direction['title'], 82, top, 11, True)
    para(escape(direction['detail']), top+24, x=82, width=488, size=9.7, leading=13.5, max_h=52)
line(673)
para('<b>Private by construction:</b> raw files are read in place. Models, caches, predictions, reports and source snapshots are stored outside both Git checkouts. All 14 input hashes remain unchanged. Nothing was uploaded or submitted.', 693, size=9, leading=12.5, max_h=38)
C.save()
(PRIVATE / 'report_layout.json').write_text(json.dumps(LAYOUT, indent=2), encoding='utf-8')
print(FINAL)
