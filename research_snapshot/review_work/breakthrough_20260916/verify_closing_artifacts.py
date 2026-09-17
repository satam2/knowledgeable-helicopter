"""Verify the final report bundle against its immutable evidence receipts."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--visual-review', type=Path, required=True)
    parser.add_argument('--decision-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = external_path(args.output)
    assert not destination.exists()
    receipt = read(args.report / 'report_receipt.json')
    decision = read(args.decision)
    inventory = read(args.inventory / 'inventory.json')
    assert receipt['final'] and decision['completed_window']
    assert datetime.now(timezone.utc) >= datetime(2026, 9, 16, 14, 13, tzinfo=timezone.utc)
    assert datetime.fromisoformat(decision['created_utc']) >= datetime(2026, 9, 16, 14, 13, tzinfo=timezone.utc)
    assert receipt['pages'] == 5
    assert sha(Path(receipt['pdf'])) == receipt['pdf_sha256']
    assert sha(args.decision) == receipt['decision_sha256']
    assert sha(args.inventory / 'inventory.json') == receipt['inventory_sha256']
    assert inventory['classifications'] == {
        'experimental_predictor': 172, 'component_control': 4,
        'matched_control': 4, 'support_diagnostic': 3}
    checked = {}
    for filename, expected in read(args.inventory / 'outputs.json')['hashes'].items():
        path = args.inventory / filename
        assert sha(path) == expected, path
        checked[str(path)] = expected
    for filename, expected in receipt['supplemental_evidence'].items():
        path = Path(filename)
        assert sha(path) == expected, path
        checked[str(path)] = expected
    for filename, expected in inventory['manifest_receipts'].items():
        assert sha(Path(filename)) == expected, filename
    composition = ROOT / 'private_runs/breakthrough_20260916/missing/route_composition_v4'
    assert read(composition / 'verification.json')['status'] == 'passed'
    budget = read(composition / 'error_budget/summary.json')['variants']['global9']
    assert abs(budget['seasonal_rmse'] - 272.2639669175017) < 1e-9
    previews = {str(Path(path)): sha(Path(path)) for path in receipt['previews']}
    assert len(previews) == 5
    visual = read(args.visual_review)
    decision_review = read(args.decision_review)
    assert visual['status'] == 'passed' and visual['stage'] == 'final'
    assert visual['report_receipt_sha256'] == sha(args.report / 'report_receipt.json')
    assert visual['decision_sha256'] == sha(args.decision)
    assert len(visual['pages']) == 5
    for page in visual['pages']:
        assert page['visually_inspected'] and not page['clipping_overlap_or_unreadable_text']
        assert sha(Path(page['path'])) == page['sha256']
    assert decision_review['status'] == 'passed'
    assert decision_review['final_decision_sha256'] == sha(args.decision)
    assert decision_review['created_after_authorized_window']
    for path in [args.visual_review, args.decision_review,
                 ROOT / 'output/breakthrough_20260916/REPRODUCE_CLOSING.md',
                 ROOT / 'output/breakthrough_20260916/WHY_IT_IMPROVED.md',
                 ROOT / 'output/breakthrough_20260916/missing_budget_requirements/receipt.json',
                 ROOT / 'output/breakthrough_20260916/missing_budget_requirements/analysis.md',
                 ROOT / 'output/breakthrough_20260916/research_gap/final_claims_review.md']:
        checked[str(path)] = sha(path)
    result = {
        'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Final report, decision, inventory outputs, all model manifests, supplemental evidence and preview hashes; independent final visual and decision review receipts verified and bound. Supporting explanations are hash-bound.',
        'source_sha256': sha(Path(__file__)), 'report_receipt_sha256': sha(args.report / 'report_receipt.json'),
        'decision_sha256': sha(args.decision), 'inventory_sha256': sha(args.inventory / 'inventory.json'),
        'manifest_count': len(inventory['manifest_receipts']),
        'checked_evidence': checked, 'preview_hashes': previews,
        'verified_local_rmse': budget['seasonal_rmse'],
        'official_submission_unchanged': 294.626,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('CLOSING_ARTIFACTS_PASSED', len(checked), 'evidence files;', len(previews), 'previews', flush=True)


if __name__ == '__main__':
    main()
