"""Produce aggregate information-gap findings from audited private receipts."""
import json
from pathlib import Path
import sys
import hashlib
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_information import ROOT, OUT, external_path


def pooled_stats(packs, path):
    records = []
    for pack in packs:
        value = pack["diagnostics"]
        for key in path:
            value = value[key]
        if value.get("n", 0):
            records.append(value)
    total = sum(v["n"] for v in records)
    return {"n": total, "mae": sum(v["n"]*v["mae"] for v in records)/total,
            "rmse": float(np.sqrt(sum(v["n"]*v["rmse"]**2 for v in records)/total)),
            "within60_pct": sum(v["n"]*v["within60_pct"] for v in records)/total}


def main():
    audit = json.loads((OUT/"information_audit.json").read_text())
    summary = json.loads((OUT/"information_summary.json").read_text())
    packs = [p for p in audit["packs"] if p["file"].startswith("training")]
    train, rank = summary["coverage"]["training"], summary["coverage"]["ranking"]
    landing = pooled_stats(packs, ["same_nm_flight_arrival", "arvt3_minus_landing_sec"])
    inblock = pooled_stats(packs, ["same_nm_flight_arrival", "arvt3_minus_inblock_sec"])
    offsets = pooled_stats(packs, ["arvt3_minus_arvt1_sec"])
    lower = sum(p["phase_id_ranges"]["DEP"]["max"] < p["phase_id_ranges"]["ARR"]["min"] for p in summary["packs"])
    reverse = sum(p["phase_id_ranges"]["ARR"]["max"] < p["phase_id_ranges"]["DEP"]["min"] for p in summary["packs"])
    order_july = [v for v in summary["ordering"] if v["file"].startswith("training_2025-07")]
    order_nov = [v for v in summary["ordering"] if v["file"].startswith("training_2025-11")]
    distilled = {"created_utc": datetime.now(timezone.utc).isoformat(), "same_flight_arvt3_minus_landing": landing,
        "same_flight_arvt3_minus_inblock": inblock, "arvt3_minus_arvt1": offsets,
        "phase_ranges_departures_before_arrivals_packs": lower, "phase_ranges_arrivals_before_departures_packs": reverse,
        "all_audit_pack_count": len(summary["packs"]), "diagnostics_are_not_model_scores": True}
    (OUT/"distilled_findings.json").write_text(json.dumps(distilled,indent=2),encoding="utf-8")
    coverage_rows = []
    descriptions = {
        "ARVT_3_flt": "Unused flown-trajectory arrival clock; retrospective flight consistency",
        "ARVT_1_flt": "Planned duration implemented only in quality=true, absent current base",
        "ADES_FILED_flt": "Unused filed-versus-actual diversion indicator",
        "CALLSIGN_flt": "Unused NM callsign; no rescue when NM record absent",
        "FLIGHT_RULE_mvt": "Unused airport flight-rule category; nearly complete",
        "FLIGHT_RULE_flt": "Unused NM rule; may identify join/flight-type convention",
        "FLIGHT_mvt": "Not ordinary base; already used in schedule/Rome features and previous probes",
        "FLIGHT_ID_mvt": "Purges/context joins; not ordinary predictor; exact flight, not aircraft",
        "MVT_ID_mvt": "Alignment/context identity; weak source-order hypothesis, not random assumed",
        "ADEP_flt": "Quality mismatch flag only, absent current base",
        "ADES_flt": "Quality mismatch flag only, absent current base",
        "AIRCRAFT_TYPE_flt": "Quality mismatch flag only, absent current base",
    }
    for col, use in descriptions.items():
        coverage_rows.append(f"| `{col}` | {train['columns'][col]['coverage_pct']:.3f}% | {rank['columns'][col]['coverage_pct']:.3f}% | {use} |")
    order_rows=[]
    for month, records in [("July",order_july),("November",order_nov)]:
        for record in records:
            order_rows.append(f"| {month} | {record['airport']} | {record['within_day_spearman_block_mean']:.3f} | {record['within_day_spearman_takeoff_mean']:.3f} | {record['id_adjacent_block_reverse_pct']:.2f}% |")
    report = f"""# Information audit: what the current model does not observe

Verified locally 16 September 2026. All 12 training packs plus ranking were read in place, with hashes checked against the submitted protocol. No private rows were sent to any service. No training, submission, new ranking predictions or Git-checkout edits occurred. The existing 49 successful public research receipts were reused; this subtask made zero network requests.

## Findings that change priorities

1. **Missing NM clocks mean missing NM records.** Every one of the {train['missing_aobt']:,} training departures and {rank['missing_aobt']:,} ranking departures missing AOBT also has every `_flt` field missing. The unused NM arrival clock, callsign and operator cannot directly rescue those rows. Airport schedule remains present for all of them; flight number remains present for {train['fields_present_among_missing_aobt']['FLIGHT_mvt']:,} training and {rank['fields_present_among_missing_aobt']['FLIGHT_mvt']:,} ranking missing-clock rows. A missing-record mechanism must rely on airport-side identity, schedule, stand/runway conventions or cohort/source state. It is not an isolated timestamp-imputation problem.
2. **The unused actual-trajectory arrival clock is real operational information.** `ARVT_3_flt` is present on {train['columns']['ARVT_3_flt']['coverage_pct']:.3f}% of training and {rank['columns']['ARVT_3_flt']['coverage_pct']:.3f}% of ranking departures. On {landing['n']:,} same-NM-flight destination arrival links, its MAE relative to observed landing is {landing['mae']:.2f}s, versus {inblock['mae']:.2f}s relative to in-block. {landing['within60_pct']:.2f}% are within one minute of landing. It is an arrival-time/flight-consistency signal, not another airport off-block column. The official field describes the arrival time for the flown M3 trajectory; it does not prove an estimate was published before takeoff.
3. **A literal movement-ID arrival bracket is unavailable.** Same-airport/month arrival-ID brackets cover zero departures in both training and ranking across all {len(summary['packs'])} packs. This is a local phase-ordering result, not globally disjoint ARR/DEP ranges: the global ranges overlap in every pack. ID ordering is not assumed random, but within-day block ordering is noisy and nearly the same as takeoff ordering. A simple adjacent-arrival in-block interpolation mechanism is contradicted by the actual local ID layout.
4. **Exact same-flight arrival linkage mostly duplicates NM information.** It covers {train['same_flight_destination_arrival_coverage_pct']:.2f}% of training and {rank['same_flight_destination_arrival_coverage_pct']:.2f}% of ranking departures in within-file unambiguous joins. AOBT is repeated identically in matched rows; no missing AOBT was recovered. These are the same journey observed at destination, normally after the departure, not a preceding aircraft rotation. No registration is supplied.
5. **Important source features are absent from the ordinary model, but some were tried elsewhere.** Raw schema has 30 columns. The current ordinary feature cache has 30 engineered columns, using only 16 raw operational/context columns. `FLIGHT_mvt`/prefix and schedule conventions already appear in the Rome schedule branch and previous exact-designator probes. Reusing them without a substantive change is not a new discovery.

## Actual unused-field coverage

Percentages below are nonnull departure coverage over all 2,085,047 training and 344,841 ranking rows; they do not establish predictive benefit. Hidden departure labels/block times are excluded from the candidate list. The audit also records complete ARR coverage and unique counts per pack.

| Field | Training | Ranking | Current role / gap |
|---|---:|---:|---|
{chr(10).join(coverage_rows)}

All departures have airport schedule; nearly all have flight number, flight rule and aircraft type. Among missing AOBT cases, just {train['fields_present_among_missing_aobt']['FLIGHT_ID_mvt']} training and {rank['fields_present_among_missing_aobt']['FLIGHT_ID_mvt']} ranking rows retain an NM flight ID, but every NM `_flt` field is still null. The small ID-only subset is not a bulk recovery path.

Filed-versus-actual NM destination differs on {train['diverted_filed_destination_rows']:,} training departures ({100*train['diverted_filed_destination_rows']/train['departure_rows']:.3f}%) and {rank['diverted_filed_destination_rows']:,} ranking departures ({100*rank['diverted_filed_destination_rows']/rank['departure_rows']:.3f}%). This is a sparse source/trajectory-consistency flag, not a sufficient explanation for the whole error tail.

## Ranked mechanisms worth testing

### 1. Missing-record source convention and retrospective peer state

The all-or-nothing NM absence is the clearest structural fact. Candidate explanations include failure to match the airport movement to an NM record, flight-number/airport/type conventions, or a distinct airport recording source. Neither the data page nor schema identifies an explicit source-system ID. Do not invent one or assume missingness is random. Airport flight rule, normalized movement flight number, schedule reliability, stand/runway and source/cohort metadata are available even when all NM fields vanish.

The substantive change from the existing Rome schedule branch would be **modeling the observation/source regime explicitly and using label-free peer cohort state**, with chronological learned calibration. For example, characterize whether nearby departures in the same airport/stand/flight-number cohort also lack NM records, how final NM-versus-airport discrepancies cluster in permitted earlier labels, and how schedule/takeoff patterns change within a source batch. The distinction is mechanism and cohort calibration, not a larger generic predictor. This audit establishes inputs, not a beneficial estimator.

### 2. Flown-versus-planned trajectory consistency for present clocks

Use `ARVT_3 - ARVT_1`, `ARVT_3 - takeoff`, planned duration `ARVT_1 - EOBT_1`, and `(ARVT_3 - AOBT_3) - (ARVT_1 - EOBT_1)` as separate observation/reliability features. The last quantity combines both departure and arrival source revisions, not physical taxi time. `ARVT_3 - ARVT_1` has pooled MAE {offsets['mae']:.1f}s and RMSE {offsets['rmse']:.1f}s against zero; that is feature variation, **not taxi prediction error**. Destination consistency and diversion flags can distinguish a late/changed journey from a stand/runway taxi delay. A learned head must preserve raw-second conditional means and all score rows.

This information is available in the retrospective ranking input, but nearly all final ARVT3 values occur after takeoff. Causal departure-time deployment cannot assume this final observation already existed. Test it only under a declared retrospective variant, alongside the unchanged causal-policy control. It cannot help rows with the entire NM record absent.

### 3. Arrival/source matching consistency, not rotations

Within-file unique `FLIGHT_ID_mvt` joins link {train['same_flight_destination_arrival_rows']:,} training and {rank['same_flight_destination_arrival_rows']:,} ranking departures to their destination arrival. Those arrivals can audit final NM trajectory endpoint consistency. They do not supply previous-aircraft turnaround, exact tail identity or a missing off-block time. Destination arrival duration could characterize that journey's endpoint record reliability under a retrospective contract, but geographic/end-time availability and sparse same-challenge-airport coverage create selection effects. No outcome gain was tested.

### 4. ID-derived recording order: downgraded after direct test

Spearman correlations within each airport/day are averaged over days with at least ten departures. `block` is hidden-label diagnostic only; it is never a feature. Adjacent-ID block reversals are measured within those same days. High correlation shared with takeoff is not evidence that ID encodes hidden off-block precisely.

| Month | Airport | Mean ID/block rho | Mean ID/takeoff rho | Adjacent block reversals |
|---|---|---:|---:|---:|
{chr(10).join(order_rows)}

Same-airport/month departure IDs never lie inside the range spanned by the observed arrival IDs, so nearby ARR anchors cannot interpolate DEP block times directly. Globally the phase ranges overlap because airport-specific ranges interleave; a uniform global phase offset is not established. Normalizing local phase ranks would introduce a new unverified assumption: separate extract orders need not preserve the same clock ordering. The evidence does not justify pursuing that assumption ahead of the measured NM/source gaps. Absolute IDs also risk encoding extraction chronology that will not transport to 2026; historical temporal cohorts and rank-normalized label-free representations would be mandatory if tested.

## Availability contracts and official evidence

The official data page states that ranking contains the same columns as training except departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, which are blanked, and gives January/July 2026 as ranking periods. It defines `_mvt` as airport movement information and `_flt` as NM flight-list information. It supplies takeoff/landing times and final flown M3 fields. The page does **not** establish a real-time prediction deadline or publication-latency guarantee. It therefore supports treating supplied batch fields as observed retrospective inputs, but absence of a stated cutoff is not proof that every conceivable enrichment or external truth source is permitted. Organizer hidden truth, hidden departure fields and validation outcome fitting remain excluded.

Keep these experiments distinct:

| Contract | Permitted inputs in that declared experiment | Important boundary |
|---|---|---|
| Existing causal-policy control | Earlier movement context, month isolated; taxi-in only after observed in-block completion | Event timestamp is not proof of real-time publication; ties/self/same-flight duplicates excluded |
| Supplied retrospective batch context | Provided ranking rows and unhidden fields; potential later final ARVT3, same-flight destination arrival, two-sided traffic/source cohort state | Declare availability relaxation explicitly; no departure hidden labels/block or organizer truth; purges and complete cohorts still apply |
| Label-derived training priors | Only permitted earlier fit/calibration labels, chronologically cross-fit | Never same-month score labels or ranking hidden outcomes; G1 cannot reuse later-year priors |

The official eligibility page requires documented openly usable additional datasets, open-source additional-data licensing, reproducible original methods, and GPLv3 produced source for prize eligibility. No new outside data is required for the first two ranked information gaps. No evidence here authorizes publication of private inputs or a competition upload.

Primary sources already fetched 2026-09-16, immutable receipt references:

- https://prc-data-challenge-2026.netlify.app/data.html (`competition_data` in `output/research_20260916/sources.json`)
- https://prc-data-challenge-2026.netlify.app/eligibility.html (`competition_eligibility` in the same receipt)
- Source implementation: `knowledgeable-helicopter-screening/src/taxiout/schema.py`, `features/pipeline.py`, `features/clocks.py`, `availability.py`; previous schedule features: `review_work/next230_features.py`. These were read, not modified.

## Reproduction and limitations

```powershell
.\\.review-venv\\Scripts\\python.exe -B -u review_work\\mechanism_20260916\\information\\audit_information.py
.\\.review-venv\\Scripts\\python.exe -B -u review_work\\mechanism_20260916\\information\\summarize_information.py
.\\.review-venv\\Scripts\\python.exe -B review_work\\mechanism_20260916\\information\\write_report.py
```

Receipts: `private_runs/mechanism_20260916/information/information_audit.json` (all 13 raw packs, phase/column nonnull/unique counts, raw hashes, NM-clock and same-flight tests), `information_summary.json` (global coverage, phase ranges, within-day ID-order evidence), and `distilled_findings.json` (pooled diagnostic statistics). `initial_failure.json` preserves the first empty-bracket JSON-NaN failure and its correction. The summary script had a one-character syntax typo before its first execution; corrected before any summary artifacts existed.

Same-flight linkage is within supplied file, excluding ambiguous arrival flight IDs; boundary-crossing flights can be missed. Hidden training block time is used only to assess the ID-order hypothesis, not fitted or exposed to feature code. Findings use all training months diagnostically, so they are research hypotheses for exposed development folds, not fresh confirmation. No estimator was trained, no 230-second feasibility bound established, and no ranking outcomes were accessed. Coverage and clock identities do not establish useful predictive gains.
"""
    extra = []
    rule_path = OUT / "flight_rule_slices.json"
    if rule_path.exists():
        rules = json.loads(rule_path.read_text())
        extra.append("## Follow-up: omitted flight-rule slices\n\nThe omitted airport flight rule identifies a different VFR population, but not the dominant missing-clock error mechanism. These are label-aware descriptive score slices, not calibrated routing rules.\n\n| Fold | Missing rule | Rows | Reference RMSE | Share of missing SSE | Schedule within60s |\n|---|---|---:|---:|---:|---:|")
        for fold, values in rules["folds"].items():
            for name, row in values["missing_by_flight_rule"].items():
                extra.append(f"| {fold} | {name} | {row['rows']:,} | {row['reference_rmse_sec']:.2f}s | {row['share_of_missing_sse_pct']:.2f}% | {row['schedule_within60_pct']:.2f}% |")
        extra.append("\nIFR accounts for more than98% of missing-clock SSE in both months. A VFR flag can improve population characterization but cannot plausibly remove the principal error budget alone. All airport/rule slices are retained in flight_rule_slices.json; no score-fitted hard rule was proposed.\n")
    links_path = OUT / "record_linkage_results.json"
    if links_path.exists():
        links = json.loads(links_path.read_text())
        extra.append("## Follow-up: a nonzero record-linkage mechanism\n\n**The old zero-recovery result applied only to NM FLIGHT_ID matching.** That identifier is absent on almost all missing-NM rows. A different test uses exact airport FLIGHT_mvt text plus ADEP/ADES route to match a destination ARR record that retained NM fields. Its declared, label-free filter is arrival UTC date within one day of departure takeoff and candidate NM AOBT no later than takeoff with proxy duration0..7200 seconds. Exactly one distinct candidate NM journey is required. No departure TAXITIME/BLOCK column is loaded.\n\n| Cohort | Missing queries | Unique matches | Missing coverage | Ambiguous after filters | Known-NM masked ID/AOBT precision |\n|---|---:|---:|---:|---:|---:|")
        for cohort, rec in links["cohorts"].items():
            m=rec["counts"]["missing_nm"]
            k=rec["counts"]["known_nm_masked_for_matching"]
            extra.append(f"| {cohort} | {m['query_rows']:,} | {m['unique_journey_candidate']:,} | {m['unique_coverage_pct']:.2f}% | {m['ambiguous_journey_candidates']} | {k['original_aobt_equal_pct']:.3f}% ({k['original_aobt_equal_rows']:,}/{k['unique_journey_candidate']:,}) |")
        extra.append("\nEvery F1/F3 recovered row's destination arrival occurs after the query. This is **retrospective supplied-batch information**, not a causal-pipeline feature. The same-journey linkage is not an aircraft rotation. High known-record precision does not prove precision on the missing population, and even rare mismatches can hurt RMSE. Three ranking matches have destination landing not after query and deserve anomaly review; no ranking labels are available or accessed.\n\nThis establishes new available candidate clocks, not a prediction gain. Main-agent attribution must evaluate a predeclared fixed replacement/blend on unchanged complete score cohorts and retain every unsupported row. Do not learn match thresholds from F1/F3 outcomes. Exact raw text is the only matching version tested here; unsupervised name/prefix translation could be a later separately declared coverage expansion.\n\nProtocol/results: record_linkage_protocol.json and record_linkage_results.json. Private matched records stay in information/F1_exact_missing_links.parquet, F3_exact_missing_links.parquet and ranking_exact_missing_links.parquet. Reproduction: `.review-venv/Scripts/python.exe -B -u review_work/mechanism_20260916/information/record_linkage.py`. This is a meaningful distinction from the previous zero same-FLIGHT_ID recovery, not evidence that a large enough fraction of the error tail is solvable.\n")
    report += "\n" + "\n".join(extra)
    target = external_path(ROOT/"output/mechanism_20260916/information.md")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(report,encoding="utf-8")
    print(target)
    print(json.dumps(distilled,indent=2))


if __name__=="__main__":
    main()
