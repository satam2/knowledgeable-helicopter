"""Aggregate-only error budget from the independently verified composition."""
import normalized_missing_tune as prior
import math

common, ROOT = prior.common, prior.ROOT
EVAL = ROOT / 'private_runs/tail240_20260916/validation/neural_missing_integration_replacement_v1/evaluation.json'
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/current_error_budget_v1')
REPORT = common.external_path(ROOT / 'output/tail240_20260916/CURRENT_ERROR_BUDGET.md')


def main():
    data = common.read_json(EVAL)
    assert data['status'] == 'passed'
    weights = {'F1':192122/344841, 'F3':152719/344841}
    rmse = data['seasonal']['baseline272']['candidate_rmse']
    total = rmse**2
    def contribution(section, key):
        return sum(weights[f]*data['folds'][f][section][key]['candidate']['sse']/data['folds'][f]['score_rows'] for f in weights)
    slices = {k:contribution('slices',k) for k in data['folds']['F1']['slices']}
    airports = {k:contribution('airports',k) for k in data['folds']['F1']['airports']}
    assert abs(sum(slices[k] for k in ['ordinary_nm','missing_nm','finite_nonordinary_nm'])-total) < 1e-8
    assert abs(sum(airports.values())-total) < 1e-8
    scenarios = {}
    for key in ['missing_nm','ordinary_nm','baseline_top1pct_error']:
        scenarios[key] = {str(reduction):math.sqrt(total-slices[key]*(1-(1-reduction)**2)) for reduction in [.1,.25,.5,1.]}
    required = {}
    for target in [250.,240.,230.]:
        delta = total-target**2
        required[str(target)] = dict(total_mse_reduction_fraction=delta/total,
            missing_rmse_reduction_fraction=1-math.sqrt(1-delta/slices['missing_nm']) if delta <= slices['missing_nm'] else None,
            ordinary_rmse_reduction_fraction=1-math.sqrt(1-delta/slices['ordinary_nm']) if delta <= slices['ordinary_nm'] else None)
    report = dict(status='complete', source_sha256=common.sha256(__file__), evaluation_sha256=common.sha256(EVAL),
        rmse=rmse, mse=total, slice_mse=slices, airport_mse=airports, scenarios=scenarios, required=required,
        caveat='Counterfactual arithmetic only. All other row errors held fixed. Error-defined baseline top1% is retrospective diagnostic, not inference routing or current candidate top1%. Groups overlap except the three explicit availability routes. No attainability or irreducibility conclusion.')
    OUT.mkdir(parents=True,exist_ok=False)
    common.write_json(OUT/'receipt.json', report)
    lines = ['# Current error budget', '', f'The independently verified local composition scores **{rmse:.6f} seconds**. Reaching 240 requires **{100*required["240.0"]["total_mse_reduction_fraction"]:.2f}% less total squared error**. These are exposed development results, not an official score forecast.', '',
        '| Observable route | Remaining weighted squared error |', '| --- | ---: |']
    for key in ['ordinary_nm','missing_nm','finite_nonordinary_nm']:
        lines.append(f'| {key} | {100*slices[key]/total:.2f}% |')
    lines += ['', 'These three routes partition every row. Missing-clock cases remain valuable, but most error still lies in ordinary-clock cases.', '',
        '| Airport | Remaining weighted squared error |', '| --- | ---: |']
    for key,value in sorted(airports.items(),key=lambda item:-item[1]):
        lines.append(f'| {key} | {100*value/total:.2f}% |')
    lines += ['', '## Counterfactual requirements', '', '| Target overall RMSE | Total MSE reduction | Required missing-route RMSE reduction if every other error stays fixed |', '| --- | ---: | ---: |']
    for target,row in required.items():
        amount = row['missing_rmse_reduction_fraction']
        lines.append(f'| {target} | {100*row["total_mse_reduction_fraction"]:.2f}% | {"Impossible through this route alone" if amount is None else f"{100*amount:.2f}%"} |')
    lines += ['', 'This does not show that those reductions are achievable. An error-concentration plot identifies valuable outcomes to explain; it does not identify those flights before their hidden labels are known. A useful predictor must both recognize a problem from observables and estimate the signed correction without damaging other flights.', '',
        'The previous baseline top-1% error group, now evaluated with current predictions, is only a retrospective diagnostic. It overlaps route and airport groups and must not be added to their shares or used for inference routing.', '',
        'Source: `private_runs/tail240_20260916/validation/neural_missing_integration_replacement_v1/evaluation.json`. Receipt and all counterfactual formulas: `private_runs/tail240_20260916/models/current_error_budget_v1/receipt.json`.', '']
    assert not REPORT.exists()
    REPORT.write_text('\n'.join(lines),encoding='ascii')
    print('ERROR_BUDGET', required, flush=True)


if __name__ == '__main__':
    main()
