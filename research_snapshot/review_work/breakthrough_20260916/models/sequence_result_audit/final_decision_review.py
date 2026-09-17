"""Bind final decision to the audited draft and check diagnostic arithmetic."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', type=Path, required=True)
    args = parser.parse_args()
    output = audit.OUT / 'final_decision_consistency_review.json'
    assert not output.exists(), 'Preserve prior final receipt'
    baseline_path = audit.BASE / 'decision_draft_1403.json'
    original_review = audit.read_json(audit.OUT / 'final_consistency_review.json')
    assert original_review['status'] == 'passed'
    assert original_review['decision_sha256'] == audit.sha256(baseline_path)
    baseline = audit.read_json(baseline_path)
    final = audit.read_json(args.final)
    final_hash = audit.sha256(args.final)
    deadline = datetime(2026, 9, 16, 14, 13, tzinfo=timezone.utc)
    assert datetime.now(timezone.utc) >= deadline
    assert final['completed_window'] is True
    assert datetime.fromisoformat(final['created_utc']) >= deadline
    assert datetime.fromisoformat(final['created_utc']) <= datetime.now(timezone.utc)
    assert set(final) == set(baseline) | {'clarification'}
    expected = deepcopy(baseline)
    expected['completed_window'] = True
    expected['created_utc'] = final['created_utc']
    rows = [row for row in expected['comparison_rows'] if row['label'] == 'GRU sequence, 25% V2 blend']
    assert len(rows) == 1 and rows[0]['note'] == '200K fit sample; standalone 308.80; loses'
    rows[0]['note'] = '200K fit sample; standalone 308.80; replay fails'
    expected['clarification'] = final['clarification']
    assert expected == final, 'Final decision changed beyond approved timestamp/completion/note/provenance'
    clarification = final['clarification']
    intermediate = Path(clarification['prior_decision'])
    intermediate.resolve().relative_to(audit.ROOT.resolve())
    assert audit.sha256(intermediate) == clarification['prior_decision_sha256']
    preceding = audit.read_json(intermediate)
    old_expected = deepcopy(baseline)
    old_expected['created_utc'] = final['created_utc']
    old_expected['completed_window'] = True
    assert preceding == old_expected
    wrapper = audit.ROOT / 'review_work/breakthrough_20260916/prepare_closing_decision_v2.py'
    assert audit.sha256(wrapper) == clarification['wrapper_sha256']
    assert original_review['inventory_sha256'] == audit.sha256(audit.BASE / 'inventory_final_1402/inventory.json')
    assert original_review['inventory_outputs_sha256'] == audit.sha256(audit.BASE / 'inventory_final_1402/outputs.json')
    budget_path = audit.BASE / 'missing/route_composition_v4/error_budget/summary.json'
    budget = audit.read_json(budget_path)['variants']['global9']
    total = budget['seasonal_mse']
    missing = budget['groups']['missing']['weighted_mse']
    fixed = total - missing
    results = {}
    for target in (250, 230):
        remaining = target**2 - fixed
        assert 0 < remaining < missing
        ratio = math.sqrt(remaining / missing)
        results[str(target)] = {'missing_mse_reduction_pct': 100 * (1 - remaining / missing),
            'missing_rmse_multiplier': ratio, 'recomputed_overall_rmse': math.sqrt(fixed + missing * ratio**2)}
        assert abs(results[str(target)]['recomputed_overall_rmse'] - target) < 1e-10
    half = math.sqrt(fixed + .25 * missing)
    assert abs(results['250']['missing_mse_reduction_pct'] - 48.48987619548599) < 1e-9
    assert abs(results['230']['missing_mse_reduction_pct'] - 88.52394185863801) < 1e-9
    assert abs(half - 236.9451079594196) < 1e-9
    assert audit.sha256(args.final) == final_hash
    audit.write_json(output, {'status': 'passed', 'created_utc': audit.utc_now(),
        'source_sha256': audit.sha256(__file__), 'audited_draft_sha256': audit.sha256(baseline_path),
        'prior_consistency_review_sha256': audit.sha256(audit.OUT / 'final_consistency_review.json'),
        'final_decision_path': str(args.final.resolve()), 'final_decision_sha256': final_hash,
        'permitted_changes_only': ['created_utc', 'completed_window=True', 'GRU table note: loses -> replay fails', 'clarification provenance'],
        'created_after_authorized_window': True, 'inventory_unchanged': True,
        'counterfactuals': {'budget_sha256': audit.sha256(budget_path), 'total_mse': total,
            'missing_mse': missing, 'fixed_other_route_mse': fixed, 'targets': results,
            'overall_rmse_if_missing_rmse_halved': half,
            'scope': 'Arithmetic with all other-route squared error unchanged; no attainable-model or noise-floor claim.'},
        'scope': 'Final decision matches audited draft except enumerated clarification/completion fields; all earlier recipe and score checks transfer through exact equality. PDF visual/report verification is separately owned.'})
    print('FINAL_DECISION_DIFF_PASSED', final_hash, 'counterfactuals verified', flush=True)


if __name__ == '__main__':
    main()
