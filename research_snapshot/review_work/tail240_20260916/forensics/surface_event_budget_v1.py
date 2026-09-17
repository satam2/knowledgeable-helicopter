"""Aggregate-only upper bounds for an airport-limited public event source."""
import math
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
OUT = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_budget_v1'
EVALUATION = ROOT / 'private_runs/tail240_20260916/validation/normalized_refit_v2/observed_schedule_scale_blend25_evaluation/evaluation.json'
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    evaluation = common.read_json(EVALUATION)
    binding = common.read_json(BINDING)
    assert evaluation['status'] == 'passed'
    assert common.sha256(BINDING) == evaluation['baseline_binding_sha256']
    assert common.sha256(ROOT / 'review_work/tail240_20260916/validation/validate_candidate.py') == evaluation['source_sha256']
    results = {}
    for model in ['control', 'candidate']:
        total_mse = sum(binding['weights'][fold] * record['slices']['all'][model]['mse'] for fold, record in evaluation['folds'].items())
        rmse = math.sqrt(total_mse)
        expected = evaluation['seasonal']['baseline272']['control_rmse' if model == 'control' else 'candidate_rmse']
        assert abs(rmse - expected) < 1e-10
        airport_results = {}
        for name, airports in [('EDDF', ['EDDF']), ('LSZH', ['LSZH']), ('EDDF_and_LSZH', ['EDDF', 'LSZH'])]:
            per_fold = {}
            contribution = 0.
            for fold, record in evaluation['folds'].items():
                total = record['slices']['all'][model]
                sse = sum(record['airports'][airport][model]['sse'] for airport in airports)
                n = sum(record['airports'][airport][model]['n'] for airport in airports)
                contribution += binding['weights'][fold] * sse / total['n']
                per_fold[fold] = dict(rows=n, all_rows=total['n'], row_fraction=n/total['n'], sse=sse,
                    sse_fraction=sse/total['sse'], airport_group_rmse=math.sqrt(sse/n),
                    all_rmse=total['rmse'], perfect_correction_all_rmse=math.sqrt(max(0., (total['sse'] - sse)/total['n'])))
            upper_rmse = math.sqrt(max(0., total_mse-contribution))
            airport_results[name] = dict(per_fold=per_fold, seasonal_mse_contribution=contribution,
                seasonal_mse_fraction=contribution/total_mse, perfect_correction_seasonal_rmse=upper_rmse,
                perfect_correction_rmse_gain=rmse-upper_rmse,
                fraction_of_mse_gap_to240=contribution/(total_mse-240**2),
                fraction_of_this_airport_sse_needed_for2sec_gain=(total_mse-(rmse-2)**2)/contribution,
                scenarios={str(fraction):dict(seasonal_rmse=math.sqrt(total_mse-fraction*contribution),
                    rmse_gain=rmse-math.sqrt(total_mse-fraction*contribution)) for fraction in [.01, .05, .1, .25, .5, 1.]})
        results[model] = dict(seasonal_mse=total_mse, seasonal_rmse=rmse, airports=airport_results)
    source_dir = ROOT / 'output/breakthrough_20260916/research_gap/public_methods_aviation/primary_docs_v2'
    receipts = common.read_json(source_dir / 'receipts.json')
    for name in ['opdi_data', 'opdi_methodology']:
        expected = next(row['sha256'] for row in receipts if row['name'] == name)
        assert common.sha256(source_dir / (name + '.html')) == expected
    peak = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
    assert peak < 1024**3
    common.write_json(OUT / 'receipt.json', dict(status='complete', source_sha256=common.sha256(__file__),
        evaluation_sha256=common.sha256(EVALUATION), baseline_binding_sha256=common.sha256(BINDING), weights=binding['weights'],
        sources={str((source_dir / (name + '.html')).relative_to(ROOT)):common.sha256(source_dir / (name + '.html')) for name in ['opdi_data', 'opdi_methodology']},
        models={'control':'Frozen V4 global9 baseline272','candidate':'Verified normalized missing25 composition269'},
        results=results, peak_bytes=peak, private_rows_read=False, new_prediction=False, external_request=False,
        limitation='Previously exposed score airport aggregates only. Perfect correction is an oracle error-budget ceiling, not attainable performance. Fractional scenarios refer to residual SSE removed, not flight/event coverage.'))
    print('SURFACE_BUDGET', results, flush=True)


if __name__ == '__main__':
    main()
