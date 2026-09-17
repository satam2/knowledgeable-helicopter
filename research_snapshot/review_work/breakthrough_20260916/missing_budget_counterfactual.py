"""Translate the final error budget into requirements, not model predictions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def main():
    source = ROOT / 'private_runs/breakthrough_20260916/missing/route_composition_v4/error_budget/summary.json'
    budget = json.loads(source.read_text(encoding='utf-8'))['variants']['global9']
    rmse = budget['seasonal_rmse']
    missing = budget['groups']['missing']
    share = missing['sse_share']
    missing_rmse = (missing['weighted_mse'] / missing['weighted_row_share']) ** 0.5
    goals = {}
    for target in [250, 230]:
        fraction = (1 - (target / rmse) ** 2) / share
        factor = (1 - fraction) ** 0.5
        reconstructed = rmse * (1 - share + share * factor ** 2) ** 0.5
        assert abs(reconstructed - target) < 1e-10
        goals[str(target)] = {
            'required_missing_mse_reduction_pct': 100 * fraction,
            'required_missing_rmse_reduction_pct': 100 * (1 - factor),
            'required_weighted_missing_rmse': missing_rmse * factor,
        }
    scenarios = []
    for reduction in [0, 0.25, 0.5, goals['230']['required_missing_rmse_reduction_pct'] / 100, 1]:
        overall = rmse * (1 - share + share * (1 - reduction) ** 2) ** 0.5
        scenarios.append({'missing_rmse_reduction_pct': 100 * reduction, 'hypothetical_overall_rmse': overall})
    assert abs(scenarios[-1]['hypothetical_overall_rmse'] - missing['hypothetical_rmse_if_zero']) < 1e-9
    destination = external_path(ROOT / 'output/breakthrough_20260916/missing_budget_requirements')
    destination.mkdir(exist_ok=False)
    receipt = {
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status': 'arithmetic_verified', 'baseline_rmse': rmse,
        'weighted_missing_rmse': missing_rmse, 'goals': goals, 'scenarios': scenarios,
        'scope': 'Counterfactual only: all nonmissing predictions and route membership fixed. No new model, score-selected routing, attainable floor or forecast.',
    }
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# What a missing-record breakthrough would require', '',
        'This is arithmetic on the independently verified final global9 error budget, not a trained model or an achievable-score claim.', '',
        f'The {100 * missing["weighted_row_share"]:.2f}% weighted missing-NM rows contribute {100 * share:.2f}% of squared error. Their seasonal-weighted RMSE is {missing_rmse:.2f} seconds. All nonmissing errors are held fixed below.', '',
        '| Overall target | Required missing MSE reduction | Required missing RMSE reduction | Resulting missing RMSE |',
        '|---|---:|---:|---:|',
    ]
    for target, values in goals.items():
        lines.append(f'| {target} | {values["required_missing_mse_reduction_pct"]:.2f}% | {values["required_missing_rmse_reduction_pct"]:.2f}% | {values["required_weighted_missing_rmse"]:.2f} s |')
    lines.extend(['', '| Hypothetical missing RMSE reduction | Overall RMSE |', '|---:|---:|'])
    for row in scenarios:
        lines.append(f'| {row["missing_rmse_reduction_pct"]:.2f}% | {row["hypothetical_overall_rmse"]:.2f} s |')
    lines.extend([
        '',
        'Thus, halving missing-route RMSE would still leave about 236.95 seconds overall. Reaching 230 through this route alone needs about a two-thirds reduction in its RMSE. None of the tested missing-route additions demonstrates that scale of gain.',
        '',
        'Missingness is observable and can define an inference route. The true-source-gap and worst-error groups are label-aware diagnostics and cannot define such a route. These groups overlap other diagnostics; their shares must not be added indiscriminately.',
        '',
        'The perfect-missing value is a zero-error counterfactual, not a data noise floor. Source provenance, gate activity and longer-term airport state deserve tests only if they supply information unavailable to the current predictors. The tested public OPDI links and monthly-arrival features did not meet that requirement in stable matched results.',
        '',
        'This calculation uses the local exposed development folds. It does not forecast an official competition score. See receipt.json for the exact input hash and unrounded arithmetic.',
    ])
    (destination / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
