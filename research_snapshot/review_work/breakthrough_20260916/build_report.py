"""Render an evidence-selected private report; no automatic model promotion."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import csv
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import report_builder as layout
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Spacer, PageBreak, Flowable
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
import pypdfium2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ErrorBand(Flowable):
    def __init__(self, parts):
        super().__init__()
        self.parts = parts
        self.width, self.height = layout.WIDTH, 103

    def draw(self):
        x = 0.
        for name, fraction, color in self.parts:
            self.canv.setFillColor(color)
            self.canv.rect(x, 70, self.width * fraction, 23, fill=1, stroke=0)
            x += self.width * fraction
        for i, (name, fraction, color) in enumerate(self.parts):
            y = 50 - i * 17
            self.canv.setFillColor(color)
            self.canv.rect(0, y, 7, 7, fill=1, stroke=0)
            self.canv.setFillColor(layout.INK)
            self.canv.setFont('Helvetica', 8.5)
            self.canv.drawString(15, y, name)
            self.canv.drawRightString(self.width, y, f'{fraction:.1%} of squared error')


def build(args):
    args.output = layout.external_path(args.output)
    args.inventory = layout.external_path(args.inventory)
    args.decision = layout.external_path(args.decision)
    decision = read(args.decision)
    inventory = read(args.inventory / 'inventory.json')
    hashes = read(args.inventory / 'outputs.json')['hashes']
    for filename, digest in hashes.items():
        assert sha(args.inventory / filename) == digest
    with (args.inventory / 'seasonal.csv').open(newline='', encoding='utf-8') as file:
        records = list(csv.DictReader(file))
    with (args.inventory / 'folds.csv').open(newline='', encoding='utf-8') as file:
        folds = list(csv.DictReader(file))
    selected = decision['selected']
    chosen = [r for r in records if r['recipe'] == selected['recipe'] and r['variant'] == selected['variant']]
    if len(chosen) != 1 or chosen[0]['eligible_for_predictive_comparison'] != 'True':
        raise ValueError('Decision must select one classified predictive result, never a support diagnostic')
    chosen = chosen[0]
    rows = [r for r in folds if r['recipe'] == selected['recipe'] and r['variant'] == selected['variant'] and r['fold'] in ['F1', 'F3']]
    weights = {'F1': 192122 / 344841, 'F3': 152719 / 344841}
    mse = sum(weights[r['fold']] * float(r['rmse']) ** 2 for r in rows)
    shares = {name: sum(weights[r['fold']] * float(r['rmse']) ** 2 * float(r[name]) for r in rows) / mse
              for name in ['missing_sse_share', 'source_gap_sse_share']}
    args.output.mkdir(parents=True, exist_ok=False)
    pdf = args.output / 'PRC_2026_Breakthrough_Search.pdf'
    story = []
    p = layout.p
    def head(number, title, subtitle):
        if number > 1:
            story.append(PageBreak())
        story.extend([p(f'PRC 2026 / RESEARCH EVIDENCE / {number:02d}', 'SmallPRC'),
                      p(escape(title), 'TitlePRC'), p(escape(subtitle), 'SubtitlePRC')])
    def para(text):
        story.append(p(escape(text)))
    def sub(title, text):
        story.append(p(escape(title), 'HeadPRC'))
        para(text)

    stage = 'Completed five-hour search' if decision['completed_window'] else 'Interim evidence snapshot; search continues'
    head(1, decision['headline'], stage + '. ' + decision['window'])
    para(decision['outcome'])
    story.append(layout.table([
        ['Evidence', 'RMSE, seconds', 'Meaning'],
        ['Submitted V2', '294.626', 'Official organizer score; unchanged'],
        ['V2 local reference', '292.813', 'Same July/November scoring cohorts'],
        [selected['label'], f"{float(chosen['rmse']):.3f}", 'Local development result; not submitted'],
        ['Historical leader snapshot', '243.5805', 'Different official ranking data; not a current rank'],
        ['Research objective', '230', 'Ambitious target; attainability unproven'],
    ], [169, 94, layout.WIDTH - 263]))
    story.append(Spacer(1, 14))
    for finding in decision['key_findings'][:3]:
        para(finding)
    para('All new work remains outside both Git checkouts. Original labels, full scoring cohorts and flight-ID purges are retained. No competition submission, paid compute or private-data transfer was performed.')

    head(2, 'The information gap', 'What the model is estimating, and why the representation matters.')
    story.append(p('Taxi time Y = T - B<br/>NM proxy P = T - N<br/><b>Residual Y - P = N - B</b>', 'HeadPRC'))
    para('T is supplied takeoff, B is the hidden airport off-block time, and N is the supplied NM off-block time. The residual learns disagreement between sources. It is not an additional physical taxi delay.')
    para('Official EUROCONTROL metadata allows an NM actual off-block time to be reconstructed from takeoff minus a standard taxi allowance when updates are absent. The challenge has no per-row provenance flag, so repeated durations are suggestive rather than proof of that fallback.')
    comparison = read(ROOT / 'private_runs/breakthrough_20260916/information_capacity_1100/audit.json')
    story.append(layout.table([['Matched comparison', 'Original30 fields', 'Combined225 fields', 'Information gain']] + [
        [capacity, f"{next(r['seasonal_rmse'] for r in comparison['rows'] if r['capacity'] == capacity and r['information'] == 'base30'):.2f}",
         f"{next(r['seasonal_rmse'] for r in comparison['rows'] if r['capacity'] == capacity and r['information'] == 'combined225'):.2f}",
         f"{gain:.2f} seconds"] for capacity, gain in comparison['contrasts']['information_gain_seconds'].items()],
        [165, 99, 112, layout.WIDTH - 376]))
    story.append(Spacer(1, 12))
    para('The combined information includes explicit clock differences and order, source-peer context, supplied trajectory summaries, surface counts, prior weather observations and approximate static geometry. Matched information gains are about11 seconds; the capacity change adds about2-3 seconds. This does not isolate every feature block or establish causal real-time availability.')
    para('LightGBM, CatBoost and the neural models are trained predictors. The useful structure comes from representing the observations and optimizing prediction error; arbitrary hand-written corrections are not a substitute for measurable signal.')

    head(3, 'What was tested', 'Complete local scoring cohorts; lower seasonal RMSE is better.')
    table = [['Experiment', 'RMSE', 'Interpretation']]
    for item in decision['comparison_rows']:
        matches = [r for r in records if r['recipe'] == item['recipe'] and r['variant'] == item['variant']]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous comparison row: ' + item['label'])
        table.append([item['label'], f"{float(matches[0]['rmse']):.2f}", item['note']])
    story.append(layout.table(table, [155, 58, layout.WIDTH - 213], small=True))
    story.append(Spacer(1, 12))
    para(decision['breadth_note'])
    para('The complete CSV includes unsuccessful variants, controls and diagnostics with explicit roles. Counts represent prediction artifacts and variants, not independent discoveries. Small sampled fits are labeled separately from full-data fits.')

    head(4, 'Where error remains', 'Squared-error concentration in the selected complete-cohort predictor.')
    story.append(ErrorBand([
        ('Missing NM record', shares['missing_sse_share'], colors.HexColor('#B55249')),
        ('Present NM, source gap over30 minutes', shares['source_gap_sse_share'], colors.HexColor('#B28B26')),
        ('All other departures', 1 - sum(shares.values()), layout.GREEN)]))
    para('The source-gap group uses the true target and is diagnostic only. It cannot be used to route an unseen flight. Missing records also lack the rest of the NM feature block, which makes this a different prediction problem.')
    para(decision['error_interpretation'])
    sub('An oracle is not a model', 'Choosing the best of eight saved predictions after seeing each true label scores219.47 locally. This establishes complementarity, not a learnable gate, a noise floor or a path proven to230. Earlier-month tests of simple contextual gates did not recover that advantage.')
    sub('Support-floor caveat', 'A fit-derived lower bound removes two extreme negative V2 local predictions, but changes zero existing V2 ranking predictions. Projected scores are listed as diagnostics and excluded from candidate rankings. The effect on future newly trained ranking predictions is unknown.')
    para(decision['validation_note'])

    head(5, 'Decision and limits', 'What the evidence supports, what remains open, and how to reproduce it.')
    para(selected['decision'])
    for text in decision['limitations'][:4]:
        para(text)
    for text in decision['next_steps'][:3]:
        para(text)
    story.append(p('Evidence and reproduction', 'HeadPRC'))
    para('External runners: review_work/breakthrough_20260916. Frozen run manifests, models and row-level predictions: private_runs/breakthrough_20260916. This report is bound to the inventory and decision hashes in report_receipt.json. Read the accompanying model shortlist for versions, licenses, sample limits and untested families.')
    sources = [
        ('Competition data and availability', 'https://prc-data-challenge-2026.netlify.app/data.html'),
        ('EUROCONTROL ADRR metadata; AOBT fallback', 'https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf'),
        ('TabM implementation and paper', 'https://github.com/yandex-research/tabm'),
        ('TabICL implementation', 'https://github.com/soda-inria/tabicl'),
        ('Full source and feasibility review', 'model_shortlist.md and sources.json in output/breakthrough_20260916/research_gap'),
    ]
    for label, url in sources:
        story.append(p(escape(label + ': ' + url), 'SmallPRC'))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D7E0DC'))
        canvas.line(42, 39, A4[0] - 42, 39)
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(layout.GRAY)
        canvas.drawString(42, 27, 'PRIVATE LOCAL RESEARCH | PRC 2026 | 16 SEP 2026')
        canvas.drawRightString(A4[0] - 42, 27, str(doc.page) + ' / 5')
        canvas.restoreState()
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=42, bottomMargin=55, title='PRC2026 Breakthrough Search', author='Local PRC Research')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(pdf))
    if len(reader.pages) != 5:
        raise ValueError(f'Expected5pages, got{len(reader.pages)}; inspect layout and revise before delivery')
    rendered = pypdfium2.PdfDocument(str(pdf))
    previews = []
    for i in range(len(rendered)):
        path = args.output / f'page_{i + 1}.png'
        rendered[i].render(scale=1.5).to_pil().save(path)
        previews.append(str(path))
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'pdf': str(pdf), 'pdf_sha256': sha(pdf),
        'decision_sha256': sha(args.decision), 'inventory_sha256': sha(args.inventory / 'inventory.json'),
        'source_sha256': sha(__file__), 'layout_source_sha256': sha(layout.__file__), 'pages': len(reader.pages),
        'page_text_lengths': [len(page.extract_text()) for page in reader.pages], 'previews': previews,
        'visual_review_required': True, 'final': decision['completed_window']}
    (args.output / 'report_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
