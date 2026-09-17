"""Bind the final narrative to completed evidence, without rewriting snapshots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--completed-window', action='store_true')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    deadline = datetime(2026, 9, 16, 14, 13, tzinfo=timezone.utc)
    if args.completed_window and now < deadline:
        raise ValueError('Do not claim a completed five-hour window prematurely')
    path = external_path(args.output)
    assert not path.exists()
    base = ROOT / 'private_runs/breakthrough_20260916'
    decision = read(base / 'decision_draft_1233.json')
    budget = read(base / 'missing/route_composition_v4/error_budget/summary.json')['variants']['global9']
    inventory = read(args.inventory / 'inventory.json')
    score = budget['seasonal_rmse']
    assert abs(score - 272.2639669175017) < 1e-9
    cutoff = inventory['created_utc']
    decision.update(completed_window=args.completed_window, created_utc=now.isoformat(),
        evidence_cutoff_utc=cutoff,
        window='Authorized exploration: 16 September 2026, 09:13-14:13 UTC. Model cutoff 14:02 UTC; evidence snapshot ' + cutoff,
        headline='Better information, still no sub-250 breakthrough',
        outcome=f'The strongest verified fixed composition scores {score:.2f} seconds locally, versus V2\'s 292.81 on the same full July/November cohorts. This extends the measured local gain to 20.55 seconds; 250 and 230 remain unachieved. Official V2 remains 294.626; no new submission was made.',
        shortlist_filename='model_shortlist_final_1402.md', sources_filename='sources_final_1402.json')
    decision['selected'] = {
        'recipe': 'missing/route_composition_v4/global9/{fold}', 'variant': 'candidate',
        'label': 'Nine-expert route composition',
        'decision': 'Retain both predeclared research compositions: global9 272.26 and airport9 272.35. The global result is the numerical headline, not an independently selected winner. New ordinary-route gains are stable; missing-route gains remain tail-heavy. No full-year ranking fit, deployment or submission was selected.'}
    decision['key_findings'] = [
        'Matched richer information improves LightGBM about 11 seconds; the tested capacity package adds about 2-3. The gains are not explained by a model-family name alone.',
        'Neighboring departure clocks are valuable: removing their 24 added clock channels loses 98.7% of the matched 600-tree ordered-context gain. This is predictive attribution, not proof of a physical queue mechanism.',
        'Nine fixed experts, with weights learned on preceding tune months, complement each other. New public-source leads were implemented and tested; OPDI aircraft context and monthly-arrival missing specialists did not deliver stable gains.'
    ]
    decision['comparison_rows'] = [
        {'label': 'Fixed complete composition', 'recipe': 'missing/route_composition_v4/global9/{fold}', 'variant': 'candidate', 'note': 'Full rows; unfloored; exact composition replay'},
        {'label': 'Nine-expert ordinary blend', 'recipe': 'models/context_gate/final_simplex9_v1/{fold}_score', 'variant': 'global9', 'note': 'Other routes V2; tune-only weights; KKT verified'},
        {'label': 'Combined context, leaf63', 'recipe': 'deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': '387 fields; shorter 337 control is 281.54'},
        {'label': 'Combined225 CatBoost', 'recipe': 'combined_catboost/catboost_combined_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': 'Matched inputs; full eligible training'},
        {'label': 'Combined225 TabM + PLE', 'recipe': 'full_neural/tabm_combined_ple8_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': 'Same-seed standard control 285.17; full fit'},
        {'label': 'Monthly ARR context, LGB600', 'recipe': 'retrospective_models/monthly_arrival_v3/lightgbm_monthly_arrival_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': '15 fields; matched base 287.16; small gain'},
        {'label': 'OPDI missing, 25% V2 blend', 'recipe': 'models/opdi_missing_v3/models/historical_template_{fold}_s20260916', 'variant': 'blend25', 'note': 'Strict local links; fails both-month gain criterion'},
        {'label': 'Combined225 XGBoost', 'recipe': 'combined_xgb/xgb_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': 'Loses; strict CPU/GPU parity caveat retained'},
        {'label': 'Pretrained TabDPT residual', 'recipe': 'models_retrieval/tabdpt_batch_v3_1/tabdpt_aobt_allfinite_{fold}_s20260916', 'variant': 'candidate', 'note': '32K references, 256 context; full score; loses'},
        {'label': 'GRU sequence, 25% V2 blend', 'recipe': 'sequence_context/models/context_{fold}_s20260919', 'variant': 'blend25', 'note': '200K fit sample; standalone 308.80; loses'},
        {'label': 'Physical, 25% V2 blend', 'recipe': 'physical_v2/models/weather_{fold}_s20260916', 'variant': 'blend25', 'note': 'Independent physical/weather formulation loses'}
    ]
    decision['breadth_note'] = 'Completed tracks include three boosting families, forests, tabular and sequence networks, pretrained inference, source-mixture targets, physical/weather/geometry, ordered and retrospective context. Six public 2026 solution repositories were inspected; public claims were not treated as verified competitor scores. The CSV retains all completed variants and failures separately.'
    decision['error_interpretation'] = (f'The worst 1% still account for {100*budget["groups"]["top1pct"]["sse_share"]:.1f}% of squared error; Rome contributes {100*budget["airports"][0]["sse_share"]:.1f}%. Reaching 250 from {score:.2f} requires another {budget["goals"]["250"]["required_mse_reduction_pct"]:.1f}% MSE reduction; 230 requires {budget["goals"]["230"]["required_mse_reduction_pct"]:.1f}%. The new ordinary-route gain survives every day and top-ten-gain removal. Earlier missing-route gains remain concentrated in two rows per fold.')
    decision['validation_note'] = 'Both full compositions improve their matched four-expert predecessors by over two seasonal seconds and retain every-day-removal gains. Earlier 225-field leaf63 F2/G1 checks improve, but share October; broader 337-field checks were not run. All score months are exposed development. November differs from January, and the new OPDI linkage coverage falls sharply in January 2026.'
    decision['limitations'] = [
        'No sub-250 result or verified leader method was found. One local/official V2 pair cannot calibrate a new recipe; repeated development comparisons create selection bias. Component seed checks do not replicate the whole pipeline.',
        'Final NM publication times are unknown. Future/day/month-arrival features are explicitly retrospective. OPDI links are observed-flight proxies, not true off-block or proven aircraft identity; prize/commercial reuse eligibility remains unresolved.',
        'XGBoost CPU replay differs by less than 1 ms but fails its unchanged 0.1 ms tolerance; its global9 weight is numerically zero, with small airport weights. GRU full CPU replay also fails its tolerance; a bounded same-tensor GPU probe supports a device-arithmetic explanation. Neither failure is waived.',
        'Implementation failures are preserved; raw inputs and prior artifacts remain unchanged. No final full-year ranking predictions were generated.'
    ]
    decision['next_steps'] = [
        'Prioritize demonstrably new information about source reliability and missing-record gate activity. The tested OPDI/monthly-arrival additions do not yet justify replacing the existing missing route.',
        'Untested: chronologically cross-fitted airport-hour label-history forecasts with matched covariates. July 2026 has no 2026 departure-label history; test long-gap or prior-year seasonal transfer against simple seasonal baselines.',
        'Before a future submission, establish transfer across time and airport coverage, reproduce the complete recipe, and resolve required source/provenance and external-data terms. Organizer questions remain a local unsent draft.'
    ]
    decision['supplemental_evidence'] = [
        'private_runs/breakthrough_20260916/missing/route_composition_v4/verification.json',
        'private_runs/breakthrough_20260916/missing/route_composition_v4/error_budget/summary.json',
        'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1/independent_verification.json',
        'private_runs/breakthrough_20260916/missing/clock_ablation_audit/attempt_v3/summary.json',
        'private_runs/breakthrough_20260916/models/sequence_result_audit/missing_information/audit.json',
        'private_runs/breakthrough_20260916/retrospective_research/monthly_arrival/independent_oracle.json',
        'private_runs/breakthrough_20260916/missing/opdi_rotation/independent_oracle.json',
        'private_runs/breakthrough_20260916/closing_implementation_attempts.json',
        'private_runs/breakthrough_20260916/preservation_final_1402/receipt.json',
        'private_runs/breakthrough_20260916/models/sequence_result_audit/innovations_matched_audit.json',
        'private_runs/breakthrough_20260916/models/sequence_result_audit/innovations_native_replay.json',
        'private_runs/breakthrough_20260916/retrospective_research/factorial_analysis/analysis.json',
        'review_work/breakthrough_20260916/missing/sequence_deep_audit/airport_hour_forecast_critique.md',
        'output/breakthrough_20260916/research_gap/validation_transfer.md',
        'output/breakthrough_20260916/research_gap/public_solution_findings_1316.md',
        'output/breakthrough_20260916/research_gap/public_methods_main/review.md',
        'output/breakthrough_20260916/research_gap/model_shortlist_final_1402.md',
        'output/breakthrough_20260916/research_gap/sources_final_1402.json'
    ]
    for source in decision['supplemental_evidence']:
        assert (ROOT / source).is_file(), source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    print('DECISION', path, 'completed_window', args.completed_window, flush=True)


if __name__ == '__main__':
    main()
