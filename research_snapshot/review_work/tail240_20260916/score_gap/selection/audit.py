"""Bounded aggregate selection audit; no model, row labels, or network access."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path

OUT = external_path(ROOT / 'private_runs/tail240_20260916/score_gap/selection/v1')
SOURCES = [
    'private_runs/tail240_20260916/validation/neural_missing_integration_replacement_v1/evaluation.json',
    'private_runs/tail240_20260916/final_submission_v3_v2/organizer_result.json',
    'private_runs/tail240_20260916/models/current_error_budget_v1/receipt.json',
    'private_runs/campaign_20260916/experiment_table.csv',
    'private_runs/campaign_20260916/experiment_registry.json',
    'private_runs/campaign_20260916/summary.json',
    'output/tail240_20260916/validation/NORMALIZED_SCORE_REVIEW.md',
    'output/tail240_20260916/validation/NEURAL_INTEGRATION_REVIEW.md',
    'output/tail240_20260916/state/neural_context/RESULTS.md',
    'output/tail240_20260916/state/neural_context/REFIT_RESULTS.md',
    'output/tail240_20260916/state/learning_curve_design/RESULTS.md',
    'output/tail240_20260916/state/validation_alignment_audit/FINDINGS.md',
    'output/tail240_20260916/forensics/current_ensemble_gating_gap.md',
    'private_runs/breakthrough_20260916/missing/broader_information_audit/analysis.md',
    'private_runs/breakthrough_20260916/missing/forest_audit/extratrees/attempt_v2/analysis.md',
    'output/breakthrough_20260916/research_gap/model_shortlist_final_1402.md',
    'output/PRC_2026_Deep_Review.md',
    'private_runs/tail240_20260916/state/final_ple387/model_v2/protocol.json',
    'private_runs/tail240_20260916/forensics/final_missing/v1/protocol.json',
]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    evals = read(SOURCES[0])['seasonal']
    official = read(SOURCES[1])
    assert official['status'] == 'Succeeded' and official['used_pairs'] == 344841
    assert official['file'] == 'knowledgeable-helicopter_v3.parquet'
    local = evals['baseline272']['candidate_rmse']
    v4 = evals['baseline272']['control_rmse']
    normalized = evals['matched_control']['control_rmse']
    old = evals['originalV2']['control_rmse']
    table = list(csv.DictReader((ROOT / SOURCES[3]).open(encoding='utf-8', newline='')))
    registry = read(SOURCES[4])
    completed = sum(item['status'] == 'complete' for item in registry['experiments'])
    assert completed == read(SOURCES[5])['completed_fit_records'] == 30
    candidates = [r for r in table if r['candidate'] != 'submitted_reference']
    assert len(table) == 39 and len(candidates) == 38
    steps = {}
    for name in ('cat225', 'lgb225', 'lgb387', 'lgb449', 'tabm225'):
        path = f'private_runs/tail240_20260916/final_ordinary/v2/{name}/protocol.json'
        SOURCES.append(path)
        value = read(path)
        steps[name] = {k: value[k] for k in ('fold_steps', 'final_steps')}
        assert value['final_steps'] == sum(value['fold_steps']) // 2
    ple = read(SOURCES[17])
    missing = read(SOURCES[18])
    steps['ple387'] = {k: ple[k] for k in ('fold_epochs', 'final_epochs')}
    steps['missing'] = {k: missing[k] for k in ('forest_fold_leaves', 'forest_trees', 'normalized_fold_steps', 'normalized_steps')}
    mse_gain = v4 * v4 - local * local
    result = dict(
        status='passed', scope='Saved aggregate evidence only; no new fits, prediction search, labels, or network.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        scores=dict(local_v2=old, local_v4=v4, local_normalized=normalized, local_v3=local, official_v3=official['score']),
        arithmetic=dict(
            official_minus_local=official['score']-local,
            official_mse_excess_fraction=(official['score']/local)**2-1,
            local_v2_to_v3_gain=old-local,
            official_v2_to_v3_gain=294.626-official['score'],
            local_vs_official_gain_shortfall=(old-local)-(294.626-official['score']),
            normalized_rmse_gain=v4-normalized, neural_incremental_rmse_gain=normalized-local,
            normalized_share_latest_rmse_gain=(v4-normalized)/(v4-local),
            normalized_share_latest_mse_gain=(v4*v4-normalized*normalized)/mse_gain,
            neural_share_latest_mse_gain=(normalized*normalized-local*local)/mse_gain),
        bounded_campaign_counts=dict(completed_model_fold_records=completed, table_rows=len(table),
            reference_rows=len(table)-len(candidates), correlated_candidate_variant_comparisons=len(candidates),
            warning='Counts for campaign_20260916 only, not the full search or independent experiments.'),
        stopping=steps,
        limitations='Aggregate official result cannot assign a score gap to selection, drift, refit, or any component. Existing exposed-fold bootstrap intervals do not correct adaptive search.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'receipt.json'
    assert not path.exists(), 'Preserve completed audit'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
