"""Posthoc F1 diagnostic; does not alter the frozen fixed25 qualification."""
import linear_finite_tune as fit
import numpy as np
import pandas as pd

common, ROOT = fit.common, fit.ROOT


def main():
    dest = common.external_path(ROOT / 'private_runs/tail240_20260916/models/linear_replacement_diagnostic_v1')
    dest.mkdir(parents=True, exist_ok=False)
    folder = fit.OUT / 'F1'
    protocol = fit.declaration()
    marker = common.read_json(folder / 'manifest.json')
    assert marker['status'] == 'complete'
    sources = {}
    for name, digest in marker['outputs'].items():
        assert common.sha256(folder / name) == digest
        sources[str(folder / name)] = digest
    arm = common.read_json(folder / 'linear/manifest.json')
    for name, digest in arm['outputs'].items():
        assert common.sha256(folder / 'linear' / name) == digest
        sources[str(folder / 'linear' / name)] = digest
    ensemble = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    preparation = common.read_json(ensemble / 'preparation.json')['folds']['F1']
    path = ensemble / 'F1_aligned_tune.parquet'
    assert common.sha256(path) == preparation['aligned_tune_sha256']
    aligned = pd.read_parquet(path)
    weights = common.read_json(ensemble / 'F1_weights.json')
    assert weights == preparation['weights']
    neural_folder = fit.NEURAL['F1']
    assert common.sha256(neural_folder / 'manifest.json') == protocol['neural_controls']['F1']
    neural_marker = common.read_json(neural_folder / 'manifest.json')
    neural_path = neural_folder / 'tune_predictions.parquet'
    assert common.sha256(neural_path) == neural_marker['outputs'][neural_path.name]
    neural = pd.read_parquet(neural_path).set_index(fit.ID).loc[aligned[fit.ID]]
    np.testing.assert_array_equal(neural[fit.TARGET], aligned[fit.TARGET])
    linear = pd.read_parquet(folder / 'linear/tune.parquet').set_index(fit.ID).loc[aligned[fit.ID]]
    baseline = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    baseline += weights['global'][weights['experts'].index('tabm_ple8')] * (neural.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
    weight = weights['global'][weights['experts'].index('lgb63_union')]
    candidate = baseline + weight * (linear.prediction_sec.to_numpy() - aligned.lgb63_union.to_numpy())
    y = aligned[fit.TARGET].to_numpy(float)
    dates = aligned[common.MOVEMENT].dt.floor('D').to_numpy()
    result = fit.compare(y, candidate, baseline, dates)
    gains = (y-baseline)**2 - (y-candidate)**2
    order = np.argsort(gains)[::-1]
    result['remove_best_rows_gain_sec'] = {}
    for count in [1, 5, 10]:
        keep = np.ones(len(y), dtype=bool)
        keep[order[:count]] = False
        result['remove_best_rows_gain_sec'][str(count)] = float(np.sqrt(np.mean((y[keep]-baseline[keep])**2))-np.sqrt(np.mean((y[keep]-candidate[keep])**2)))
    common.write_json(dest / 'receipt.json', dict(status='complete', result=result,
        source_sha256=common.sha256(__file__), producer_sources=sources,
        rows=len(y), id_hash=common.object_hash(aligned[fit.ID].tolist()),
        coefficient=weight, operation='Current387 ensemble + original lgb63_union weight * (linear387 - original lgb63_union)',
        status_of_original_gate='Fixed25 gate failed and is unchanged',
        limitation='Posthoc diagnostic after F1 results; no fitted weight, no score or ranking access, no qualification or promotion. Same exposed tune labels selected model stopping and original weights.'))
    print(result, flush=True)


if __name__ == '__main__':
    main()
