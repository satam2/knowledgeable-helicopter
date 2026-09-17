"""Frozen diagnostic component replacement and two-fold scaling gate."""
import scaled_finite_tune as fit
import numpy as np
import pandas as pd

common, ROOT, ID, TARGET = fit.common, fit.ROOT, fit.ID, fit.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/scaled_finite_assessment_v1')


def run():
    protocol = fit.declare()
    OUT.mkdir(parents=True, exist_ok=False)
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, TARGET, 'proxy_sec'])
    reports = {}
    for fold in ('F1','F3'):
        folder = fit.OUT / fold
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete' and marker['protocol_sha256'] == common.sha256(fit.OUT / 'protocol.json')
        assert marker['source_sha256'] == common.sha256(fit.__file__) == protocol['source_sha256']
        assert common.sha256(folder / 'tune.parquet') == marker['outputs']['tune.parquet']
        pred = pd.read_parquet(folder / 'tune.parquet')
        idx, split, _ = common.fold_data(meta, fold, full=True)
        assert common.object_hash(split) == common.object_hash(marker['split'])
        truth = meta.iloc[idx['tune']]
        truth = truth.loc[np.isfinite(truth.proxy_sec)]
        np.testing.assert_array_equal(pred[ID], truth[ID])
        ordinary = truth.proxy_sec.between(0,7200).to_numpy()
        base = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
        prep = common.read_json(base / 'preparation.json')['folds'][fold]
        path = base / f'{fold}_aligned_tune.parquet'
        assert common.sha256(path) == prep['aligned_tune_sha256']
        aligned = pd.read_parquet(path)
        weights = common.read_json(base / f'{fold}_weights.json')
        assert weights == prep['weights']
        np.testing.assert_array_equal(aligned[ID], truth.loc[ordinary, ID])
        np.testing.assert_array_equal(aligned[TARGET], truth.loc[ordinary, TARGET])
        y = aligned[TARGET].to_numpy(float)
        dates = truth.loc[ordinary, common.MOVEMENT].dt.floor('D').to_numpy()
        reference = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
        weight = weights['global'][weights['experts'].index('lgb63_union')]
        candidate = reference + weight * (pred.prediction_sec.to_numpy()[ordinary] - aligned['lgb63_union'].to_numpy())
        replacement = fit.shared.metric.comparison(y, candidate, reference, dates)
        replacement.update(reference_rmse=float(np.sqrt(np.mean((reference-y)**2))), rmse=float(np.sqrt(np.mean((candidate-y)**2))))
        reports[fold] = dict(manifest_sha256=common.sha256(folder / 'manifest.json'),
            matched=marker['matched'], fixed25=marker['ordinary_fixed25'], replacement_diagnostic=replacement)
    summer = 192122/344841
    seasonal = {}
    for name in ('fixed25','replacement_diagnostic'):
        candidate = np.sqrt(summer*reports['F1'][name]['rmse']**2 + (1-summer)*reports['F3'][name]['rmse']**2)
        reference = np.sqrt(summer*reports['F1'][name]['reference_rmse']**2 + (1-summer)*reports['F3'][name]['reference_rmse']**2)
        seasonal[name] = dict(rmse=float(candidate), reference_rmse=float(reference), gain=float(reference-candidate))
    gate = seasonal['fixed25']['gain'] >= 2 and all(reports[f][n]['mse_gain'] > 0 and reports[f][n]['all_day_removals_improve'] for f in reports for n in ('matched','fixed25'))
    common.write_json(OUT / 'summary.json', dict(status='complete', source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(fit.OUT / 'protocol.json'), folds=reports, seasonal=seasonal,
        declared_gate_passed=gate, score_predictions_read=False,
        limitation='Component replacement is diagnostic only, not alternate advancement. All tune data exposed. Independent native replay still required.'))
    print('SCALE_GATE', gate, seasonal, flush=True)


if __name__ == '__main__':
    run()
