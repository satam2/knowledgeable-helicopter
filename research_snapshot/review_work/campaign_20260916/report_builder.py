"""Build the external six-page decision report from recorded campaign evidence."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
RUN = external_path(ROOT / "private_runs/campaign_20260916")
OUT = external_path(ROOT / "output/campaign_20260916")
RESEARCH = ROOT / "output/research_20260916/research.md"
INK = colors.HexColor("#20272A")
GREEN = colors.HexColor("#176B59")
RED = colors.HexColor("#B55249")
GRAY = colors.HexColor("#687277")
LIGHT = colors.HexColor("#EFF3F2")
WIDTH = A4[0] - 84
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitlePRC", fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="SubtitlePRC", fontName="Helvetica", fontSize=12, leading=17, textColor=GRAY, spaceAfter=16))
styles.add(ParagraphStyle(name="HeadPRC", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="BodyPRC", fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK, spaceAfter=10))
styles.add(ParagraphStyle(name="SmallPRC", fontName="Helvetica", fontSize=8, leading=11.5, textColor=GRAY, spaceAfter=8))
styles.add(ParagraphStyle(name="CellPRC", fontName="Helvetica", fontSize=7.5, leading=10, textColor=INK))
styles.add(ParagraphStyle(name="CellSmallPRC", fontName="Helvetica", fontSize=6.7, leading=8.6, textColor=INK))


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else ({} if default is None else default)


def fmt(value, digits=2):
    return "--" if value is None else f"{value:,.{digits}f}"


def p(text, style="BodyPRC"):
    return Paragraph(text, styles[style])


def safe(value):
    return escape(str(value))


def table(rows, widths, small=False):
    style = "CellSmallPRC" if small else "CellPRC"
    cells = [[p(safe(c).replace("\n", "<br/>"), style) for c in row] for row in rows]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), .8, GREEN),
        ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#D8E0DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


class BarChart(Flowable):
    def __init__(self, rows, label_width=200, reference=None, suffix=" s", maximum=None):
        super().__init__()
        self.rows, self.label_width, self.reference = rows, label_width, reference
        self.suffix, self.maximum = suffix, maximum
        self.width, self.height = WIDTH, len(rows) * 30 + 28

    def draw(self):
        c = self.canv
        area = WIDTH - self.label_width - 64
        maximum = self.maximum or max(v for _, v, _ in self.rows) * 1.08
        for i, (label, value, color) in enumerate(self.rows):
            y = self.height - 25 - i * 30
            c.setFillColor(INK)
            c.setFont("Helvetica", 8.2)
            c.drawString(0, y + 4, label)
            c.setFillColor(LIGHT)
            c.rect(self.label_width, y, area, 16, fill=1, stroke=0)
            c.setFillColor(color)
            c.rect(self.label_width, y, area * value / maximum, 16, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawRightString(WIDTH, y + 4, f"{value:.2f}{self.suffix}")
        if self.reference is not None:
            x = self.label_width + area * self.reference / maximum
            c.setStrokeColor(RED)
            c.setDash(2, 2)
            c.line(x, 10, x, self.height - 5)
            c.setDash()


def label(row):
    features = row.get("features") or "base"
    variant = row.get("variant") or "candidate"
    return f"{family_name(row.get('family'))} / {target_name(row.get('target'))} / {features} / {variant}"


def family_name(value):
    return {"lightgbm_soft_missing": "Soft missing-clock expert"}.get(value, str(value))


def target_name(value):
    return {"conditional_mean_mixture": "conditional mean mixture"}.get(value, str(value))


def build(final=False):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read(RUN / "summary.json")
    protocol = read(RUN / "protocol.json")
    decision = read(RUN / "decision.json")
    aviation = read(RUN / "aviation/manifest.json")
    review_paths = sorted((RUN / "validation").glob("completed_runs_*.json"))
    review = read(review_paths[-1]) if review_paths else {}
    final_validation = read(RUN / "validation/final_validation.json")
    manifests = [read(path) for path in sorted((RUN / "models").glob("*/manifest.json"))]
    research_text = RESEARCH.read_text(encoding="utf-8")
    if final and (not decision or not final_validation):
        raise ValueError("Final rendering requires decision.json and final_validation.json")
    rows = summary.get("rows", [])
    candidates = [row for row in rows if row.get("family")]
    reference = summary.get("reference_seasonal_rmse_sec", 292.8133464058)
    best = min(candidates, key=lambda r: r["seasonal_rmse_sec"]) if candidates else None
    gate_passers = [row for row in candidates if row.get("meets_screen_gates")]
    stage = "DECISION CHECKPOINT" if final else "WORKING DRAFT - EXPERIMENTS IN PROGRESS"
    filename = "PRC_2026_Diverse_Campaign.pdf" if final else "PRC_2026_Diverse_Campaign_DRAFT.pdf"
    pdf = OUT / filename
    story = []

    def head(kicker, title, subtitle=None):
        story.append(p(kicker.upper(), "SmallPRC"))
        story.append(p(title, "TitlePRC"))
        if subtitle:
            story.append(p(subtitle, "SubtitlePRC"))

    def section(title, text):
        story.append(p(title, "HeadPRC"))
        story.append(p(text))

    # Page 1: outcome before process.
    head(stage, "PRC Data Challenge 2026", "Diverse models, distinct taxi-time formulations, and observed surface conditions")
    if best:
        gain = reference - best["seasonal_rmse_sec"]
        if gate_passers:
            outcome = f"{len(gate_passers)} recorded variants clear the initial screen gates."
        else:
            outcome = "No recorded variant clears all initial screen gates."
        section(outcome, f"The lowest measured candidate is <b>{safe(label(best))}</b> at <b>{fmt(best['seasonal_rmse_sec'])} seconds</b> seasonal RMSE. Its gain relative to the frozen local reference is <b>{gain:+.2f} seconds</b> (positive is better). This is development evidence, not an official competition score.")
        uncertainty = review.get("seasonal_paired_uncertainty", {}).get(best["candidate"], {}).get(best["variant"], {})
        interval = uncertainty.get("bootstrap", uncertainty).get("candidate_minus_reference_rmse_95pct")
        detail = f"Selected variant: July {fmt(best.get('F1_rmse_sec'))} s; November {fmt(best.get('F3_rmse_sec'))} s; missing-clock RMSE ratio {fmt(best.get('max_missing_rmse_ratio'), 3)}."
        if interval:
            detail += f" Paired airport-day bootstrap 95% interval for candidate minus reference: [{interval[0]:+.2f}, {interval[1]:+.2f}] s (negative is better)."
        story.append(p(detail, "SmallPRC"))
        seed_repeats = [r for r in candidates if r.get("family") == best.get("family") and r.get("target") == best.get("target") and r.get("features") == best.get("features") and r.get("training") == best.get("training") and r.get("variant") == best.get("variant") and r.get("seed") != best.get("seed")]
        if seed_repeats:
            story.append(p("Component-seed repeat: " + "; ".join(f"seed {int(r['seed'])}: {r['seasonal_rmse_sec']:.3f} s, gain {r['gain_sec']:+.3f} s" for r in seed_repeats) + ". The frozen reference components are unchanged; this is not full-pipeline seed replication.", "SmallPRC"))
    else:
        section("No completed seasonal comparison yet", "The report will populate when both July and November results are recorded. No gain is inferred from training progress or partial cohorts.")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["family"]].append(row)
    bars = [("Submitted recipe: local reference", reference, GREEN)]
    for family, values in sorted(grouped.items()):
        b = min(values, key=lambda r: r["seasonal_rmse_sec"])
        chart_label = "Soft missing-clock: best variant" if family == "lightgbm_soft_missing" else f"{family}: best declared variant"
        bars.append((chart_label, b["seasonal_rmse_sec"], GREEN if b["seasonal_rmse_sec"] < reference else GRAY))
    story.append(BarChart(bars[:7], reference=reference))
    story.append(p("Chart shows the best recorded declared variant within each family, including fixed blends and routes. It is a descriptive selection across correlated results, not an unbiased family ranking. Lower RMSE is better. Bar axis starts at zero.", "SmallPRC"))
    story.append(table([["Local frozen reference", "Official submitted V2", "Ambitious objective"], [f"{reference:.3f} seconds", "294.626 seconds", "230 seconds; not guaranteed"]], [WIDTH/3]*3))
    story.append(Spacer(1, 10))
    story.append(p("Historical leaderboard leader: 243.5805 seconds. That snapshot is external context, not a current ranking or the same validation population. The original submitted model and completed evidence remain preserved.", "SmallPRC"))
    story.append(PageBreak())

    # Page 2: concise consolidated model recipe table.
    head("Measured comparisons", "Families and formulations", "Complete F1 July and F3 November score cohorts; training scope shown per recipe")
    by_recipe = defaultdict(list)
    for row in candidates:
        by_recipe[row["candidate"]].append(row)
    table_rows = [["Family / target / features", "Train", "Alone", "Best variant", "Seasonal", "Gain", "Gate"]]
    for key, values in sorted(by_recipe.items(), key=lambda kv: min(r["seasonal_rmse_sec"] for r in kv[1])):
        b = min(values, key=lambda r: r["seasonal_rmse_sec"])
        standalone = next((r for r in values if r["variant"] == "candidate"), None)
        seed_note = f"; seed {int(b['seed'])}" if b.get("training") == "full" and b.get("seed") else ""
        table_rows.append([f"{family_name(b['family'])} / {target_name(b['target'])}\n{b.get('features', 'base')}{seed_note}",
            "full" if b.get("training") == "full" else "200k", fmt(standalone["seasonal_rmse_sec"] if standalone else None),
            b["variant"], fmt(b["seasonal_rmse_sec"]), f"{b['gain_sec']:+.2f}", "pass" if b.get("meets_screen_gates") else "fail"])
    if len(table_rows) == 1:
        table_rows.append(["Awaiting paired folds", "--", "--", "--", "--", "--", "--"])
    story.append(table(table_rows, [140, 35, 50, 108, 60, 55, WIDTH-448], small=True))
    story.append(Spacer(1, 12))
    story.append(p("All numbers are seconds. Alone means the candidate's declared routing, which may retain the reference on unsupported clock regimes. Best variant is the lowest recorded member of the predeclared comparison set; it is not a newly fitted score-optimal blend. The complete per-variant results, including MAE, bias, slices, timing and stability, are in experiment_table.csv.", "SmallPRC"))
    story.append(p("A screen passes only with at least 2 seconds seasonal gain, both months improving, negative worst leave-one-day delta in each month, and missing-clock RMSE at most 1.05 times the reference. Passing these gates does not remove selection bias; a promising sampled model must advance to full eligible training.", "SmallPRC"))
    story.append(PageBreak())

    # Page 3: remaining error budget and physical interpretation.
    head("Where errors remain", "Concentrated tail, distinct processes", "Reference and best new blend recomputed from saved complete-cohort predictions")
    budget = summary.get("error_budget_reference", {})
    error_bars = [("Missing clocks: error share", budget.get("missing_clock_error_share_pct", 0), RED),
                  ("Largest 1% of errors: share", budget.get("largest_1pct_error_share_pct", 0), GRAY)]
    story.append(BarChart(error_bars, label_width=220, suffix="%", maximum=100))
    story.append(p(f"Missing clocks represent <b>{fmt(budget.get('missing_clock_row_share_pct'))}% of weighted rows</b>, yet account for {fmt(budget.get('missing_clock_error_share_pct'))}% of squared error. Tail membership uses labels for diagnosis only and cannot become an inference-time route."))
    new_budget = summary.get("error_budget_best_new", {})
    if new_budget:
        story.append(table([["Error share", "Reference", "Best new blend"],
            ["Missing clocks", f"{budget['missing_clock_error_share_pct']:.2f}%", f"{new_budget['missing_clock_error_share_pct']:.2f}%"],
            ["Largest 1% of errors", f"{budget['largest_1pct_error_share_pct']:.2f}%", f"{new_budget['largest_1pct_error_share_pct']:.2f}%"],
            ["Missing or clock gap >30 minutes", fmt(budget.get('missing_or_disagreement_error_share_pct')) + "%", fmt(new_budget.get('missing_or_disagreement_error_share_pct')) + "%"]], [WIDTH*.5, WIDTH*.25, WIDTH*.25]))
        story.append(Spacer(1, 8))
        story.append(p("Missing-clock predictions are unchanged in this correction blend. Their fraction of remaining error increases because ordinary-clock error falls, not because missing-clock predictions worsen.", "SmallPRC"))
    airports = sorted(budget.get("airport_error_share_pct", {}).items(), key=lambda kv: -kv[1])[:5]
    story.append(table([["Airport", "Share of remaining squared error"]] + [[a, f"{v:.2f}%"] for a, v in airports], [WIDTH*.45, WIDTH*.55]))
    story.append(Spacer(1, 18))
    section("Two different learning problems", "With takeoff T, hidden airport off-block B, and observed NM off-block N:<br/><b>Taxi time Y = T - B. Clock proxy P = T - N. Correction Y - P = N - B.</b><br/>The correction models disagreement between off-block sources. A physical expert instead predicts taxi time from stand/runway reference, demand and operating conditions. Arrival taxi-in describes related surface conditions, not the same route or waiting process.")
    story.append(p("The airport table shows reference shares. Airport, missing-clock and extreme-tail shares overlap; do not add them. A label-aware blend oracle from prior work is diagnostic only, not an achievable score or a noise floor.", "SmallPRC"))
    story.append(PageBreak())

    # Page 4: information and validation boundaries.
    head("Aviation and evaluation", "New information, explicit cutoffs", "Availability evidence and prediction evidence are kept separate")
    arrivals = aviation.get("ranking_availability", {})
    section("Arrival-derived surface conditions", f"The current arrival audit retains {arrivals.get('arrivals', 0):,} ranking arrivals with observed in-block/taxi-in fields. Features use completed arrivals strictly before the departure query and isolate month boundaries. The prepared training feature artifact contains {aviation.get('rows', 0):,} departures. Availability alone is not an accuracy result.")
    story.append(table([["Block", "Interpretation / boundary"],
        ["Recent completed-arrival durations", "Airport/runway/stand counts, mean, dispersion and long-arrival share; completion must precede query."],
        ["Arrival inventory and sequence", "Observed open arrivals are distinct from completed throughput; no claim of total departure queue length."],
        ["Timestamp precision", "Minute/second recording conventions and missingness can signal source reliability."],
        ["Stand/runway q10 reference", "Label-derived predictive prior requires chronological training construction, sparse fallbacks and unchanged score labels."],
        ["Physical expert", "Can differ in both target origin and clock information; report each difference, not just the model name."]], [158, WIDTH-158]))
    story.append(Spacer(1, 15))
    section("Validation contract", "F1: Jan-May fit, June tune, July score. F3: Jan-Sep fit, October tune, November score. Refit follows tuning. Preserve flight-ID purges, every score row and original labels. Seasonal RMSE is the square root of weighted MSE, with July weight 192122/344841 and November 152719/344841; it is not an average of RMSEs.")
    story.append(p("All these score folds, October F2/G1 and December have already been exposed during development. November is a winter proxy, not a January holdout. Sampled 200k fits are feasibility evidence; full scoring does not make them full-data fits. A single candidate seed is not full-pipeline seed replication. Bootstrap/day-removal stability is useful but cannot restore an untouched validation set.", "SmallPRC"))
    if final_validation:
        story.append(p(f"Final independent audit ({safe(final_validation.get('created_utc', ''))}): {final_validation.get('complete_containers', 0)} complete model/fold containers checked for output hashes, score/label parity, metrics, slices and declared routes; all {final_validation.get('independently_replayed_models', 0)} saved models independently replayed. Source/protocol corrections remain documented in the audit; passing replay does not create fresh validation.", "SmallPRC"))
    elif review:
        story.append(p(f"Latest independent audit ({safe(review.get('created_utc', ''))}): {len(review.get('verified', []))} model/fold records checked for output hashes, score/label parity, metrics, slices and declared routes; {len(review.get('independent_model_replay', []))} named models independently replayed. This is the audit's recorded scope, not a claim of every-model replay.", "SmallPRC"))
    story.append(PageBreak())

    # Page 5: current research, not benchmark marketing.
    head("Public research verified 16 September 2026", "A practical model shortlist", "Primary papers, official repositories and release receipts; no generic benchmark winner assumed")
    story.append(table([["Family", "Live version / evidence", "Scale / license / decision"],
        ["LightGBM", "4.7.0; NeurIPS 2017 method", "MIT. Native missing/categories. Mature full-data histogram candidate."],
        ["XGBoost", "3.4.2; KDD 2016 method", "Apache-2.0. Native categorical hist; JSON/UBJSON persistence. Full-data candidate."],
        ["CatBoost", "1.2.10; mature implementation", "Apache-2.0. Preserve categorical/missing conventions; matched control and ensemble member."],
        ["TabM", "0.0.3; ICLR 2025", "Apache-2.0. Minibatch MLP ensemble; paper includes 13M rows. Compact CPU screen here."],
        ["MLP / FT-Transformer", "rtdl 0.0.2; NeurIPS 2021", "Apache-2.0. Train-fit preprocessing. Attention costs more; kept as future comparator."],
        ["TabPFN-3.5 / Fast", "package 9.0.0; Sep 2026 technical report", "Code Apache-2.0; gated non-commercial weights. Advertised 1M rows below full 2.085M. Local feasibility and prize compatibility unresolved."]], [92, 153, WIDTH-245]))
    story.append(Spacer(1, 14))
    story.append(p("TabPFNv2's Nature paper (8 January 2025) establishes benchmark evidence up to 10k rows/500 features. It does not validate the latest model's million-row claims. TabPFN-3.5 weights expressly allow qualifying data science competitions but retain separate restrictions; no license acceptance, weights download or hosted inference was performed.", "SmallPRC"))
    story.append(p("The version column lists current public releases. The measured XGBoost runtime was 3.4.1, not latest 3.4.2. TabM used torch 2.14.0+cpu; its CPU screen is not evidence of GPU throughput.", "SmallPRC"))
    section("Small external-data pilots", "Public IEM weather requests succeeded for Rome/Paris on January 1 and July 1 in 2025 and 2026: 192 observations, 24 per station-day, maximum internal gap 60 minutes. IEM states public-domain reuse. OSM parking-position/taxiway tags were retrieved under ODbL: 194/190 elements in the Rome bbox and 740/773 in Paris. These are element counts, not validated unique stands or routes.")
    story.append(p("Weather full-period coverage, reporting latency and benefit remain untested. Geometry needs local stand matching, historical snapshots and a connected route network. Numeric stand codes are not distances; straight-line distances are not taxi paths. No confidential rows were sent in public-source requests.", "SmallPRC"))
    story.append(PageBreak())

    # Page 6: bounded checkpoint and audit trail.
    head("Decision and audit trail", "What deserves the next investment", "The current submission stays preserved; no competition upload was authorized or performed")
    if decision:
        text = decision.get("recommendation") or decision.get("decision") or decision.get("status")
        if text:
            story.append(p("<b>" + safe(text) + "</b>"))
    section("Concentrate only after the gate", "The best fixed blend improves both months, survives day removal and preserves missing-clock predictions, but its roughly one-second gain falls below the declared two-second materiality threshold. The full-data LightGBM runs were bounded diagnostic deepening, not promotions of sampled winners. Retain the reference; prioritize source reliability and new information over another broad capacity search.")
    incomplete = summary.get("incomplete_or_failed", [])
    failed_count = sum(item.get("status") == "failed" for item in incomplete)
    running_count = sum(item.get("status") != "failed" for item in incomplete)
    story.append(p(f"Recorded model/fold fits completed: <b>{summary.get('completed_fit_records', 0)}</b>. Saved unfinished/failed records: <b>{len(incomplete)}</b> ({failed_count} failed, {running_count} other). These counts are execution records, not independent discoveries. Complete failures and warnings are retained in their manifests."))
    story.append(p("Two resolved provenance failures remain visible: XGBoost F1 metrics/model replay were recovered after JSON serialization failed; original timings remain unknown. The missing-mixture runner/protocol was edited after execution; exact original bytes and protocol were recovered and linked to the saved predictions. The current arithmetic revision was not measured. See REPRODUCE.md and missing_mixture/provenance_correction.md.", "SmallPRC"))
    resources = [r for r in candidates if r.get("fit_refit_seconds") is not None]
    if resources:
        story.append(p(f"Recorded two-fold fit/refit time spans {min(r['fit_refit_seconds'] for r in resources):.1f}-{max(r['fit_refit_seconds'] for r in resources):.1f} seconds; reported process peak memory spans {min(r['peak_rss_gb'] for r in resources):.2f}-{max(r['peak_rss_gb'] for r in resources):.2f} GiB. Shared caches and preprocessing are outside these fit timings; process peaks are not exclusive model allocations.", "SmallPRC"))
    section("Still untested or unresolved", "F2/G1 were not run for new candidates. Prioritize source/clock reliability, isolate the completed-arrival signal, and audit full-period weather before joining it. The no-q10 physical ablation, physical_base run, FT-Transformer, TabPFN subgroups and real taxi-route geometry remain untested. Weak physical screens change several factors and do not identify which caused the loss.")
    story.append(p("<b>Reproduction:</b> review_work/campaign_20260916/ runners; private_runs/campaign_20260916/ protocol.json, contract.json, summary.json, experiment_table.csv and per-model manifests; output/research_20260916/ research.md, sources.json and public_pilot_summary.json. All are external to frozen Git checkouts. Raw inputs are read in place.", "SmallPRC"))
    sources = [
        ("Competition data", "https://prc-data-challenge-2026.netlify.app/data.html"),
        ("Competition eligibility / open-data rules", "https://prc-data-challenge-2026.netlify.app/eligibility.html"),
        ("TabM paper, ICLR 2025", "https://arxiv.org/abs/2410.24210"),
        ("FT-Transformer / MLP, NeurIPS 2021", "https://arxiv.org/abs/2106.11959"),
        ("TabPFNv2, Nature 2025", "https://www.nature.com/articles/s41586-024-08328-6"),
        ("Current TabPFN limits and licenses", "https://docs.priorlabs.ai/models"),
        ("IEM public-domain terms", "https://mesonet.agron.iastate.edu/disclaimer.php"),
        ("OpenStreetMap ODbL terms", "https://www.openstreetmap.org/copyright"),
    ]
    for name, url in sources:
        story.append(p(f'<link href="{url}" color="#176B59">{name}</link> | {url}', "SmallPRC"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E0DC"))
        canvas.line(42, 40, A4[0]-42, 40)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(42, 27, "PRIVATE LOCAL RESEARCH | PRC 2026 | 16 SEP 2026")
        canvas.drawRightString(A4[0]-42, 27, f"{doc.page} / 6")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=42, bottomMargin=55, title="PRC 2026 Diverse Research Campaign", author="Local PRC Research")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(pdf))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), draft=not final,
        pdf=str(pdf), pages=len(reader.pages), summary_sha256=hashlib.sha256((RUN/"summary.json").read_bytes()).hexdigest(),
        final_validation_sha256=hashlib.sha256((RUN/"validation/final_validation.json").read_bytes()).hexdigest() if final_validation else None,
        decision_sha256=hashlib.sha256((RUN/"decision.json").read_bytes()).hexdigest() if decision else None,
        research_sha256=hashlib.sha256(research_text.encode()).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        page_text_characters=[len(page.extract_text()) for page in reader.pages],
        source_manifests=len(manifests), final_visual_review_required=True)
    (OUT / ("pdf_build_receipt.json" if final else "pdf_draft_receipt.json")).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if len(reader.pages) != 6:
        raise RuntimeError(f"Expected exactly six pages; got {len(reader.pages)}")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    build(parser.parse_args().final)
