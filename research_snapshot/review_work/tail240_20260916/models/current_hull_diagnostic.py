"""Label-aware error bound for current experts; never a prediction candidate."""
import linear_finite_tune as shared
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT, common, ID, TARGET = shared.ROOT, shared.common, shared.ID, shared.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/current_hull_diagnostic_v1')
ENSEMBLE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
CURRENT = ROOT / 'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
NEURAL = ROOT / 'private_runs/tail240_20260916/state/neural_context/refit_score_v3'
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'


def declare():
    record = dict(source_sha256=common.sha256(__file__),
        preparation_sha256=common.sha256(ENSEMBLE / 'preparation.json'),
        current_manifest_sha256=common.sha256(CURRENT / 'manifest.json'),
        binding_sha256=common.sha256(BINDING),
        neural_manifests={f: common.sha256(NEURAL / f / 'manifest.json') for f in ['F1', 'F3']},
        definition='Ordinary route only: replace PLE225 expert by PLE387; per-row oracle clip(rawY, min(experts), max(experts)) is best scalar convex combination using the hidden label. Also best single expert using hidden label. Preserve every missing/nonordinary prediction exactly.',
        expert_sets='All original9; additionally original fold-global weights>1e-8, determined before this diagnostic. No candidate selection or coefficient estimation.',
        role='DIAGNOSTIC ONLY. Exposed score labels deliberately used in oracle, never available at inference. Do not export oracle predictions as a candidate. Does not show learnability or achievable noise floor.',
        scope='Fixed current learned experts only; unchanged all353045score rows/labels and weightedMSE192122/344841,152719/344841. No training, refit, submission or ranking read.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def run():
    protocol = declare()
    assert not (OUT / 'receipt.json').exists()
    preparation = common.read_json(ENSEMBLE / 'preparation.json')
    current_manifest = common.read_json(CURRENT / 'manifest.json')
    binding = common.read_json(BINDING)
    results = {}
    for fold in ['F1', 'F3']:
        bound = binding['folds'][fold]
        assert common.sha256(bound['prediction_path']) == bound['prediction_sha256']
        reference = pd.read_parquet(bound['prediction_path'], columns=[ID, TARGET, 'proxy_sec', common.MOVEMENT])
        assert reference[ID].is_unique
        current_path = CURRENT / current_manifest['folds'][fold]['prediction']['path']
        assert common.sha256(current_path) == current_manifest['folds'][fold]['prediction']['sha256']
        current = pd.read_parquet(current_path)
        np.testing.assert_array_equal(current[ID], reference[ID])
        ordinary = reference.proxy_sec.between(0, 7200).to_numpy()
        y = reference[TARGET].to_numpy(float)
        values = current.prediction_sec.to_numpy(float)
        weights = preparation['folds'][fold]['weights']
        experts = weights['experts']
        vectors = []
        for expert in experts:
            receipt = preparation['sources'][fold][expert]
            directory = Path(receipt['directory'])
            assert common.sha256(directory / 'manifest.json') == receipt['manifest_sha256']
            path = directory / 'candidate.parquet'
            assert common.sha256(path) == receipt['outputs']['candidate.parquet']
            frame = pd.read_parquet(path, columns=[ID, TARGET, 'prediction_sec'])
            np.testing.assert_array_equal(frame[ID], reference[ID])
            np.testing.assert_array_equal(frame[TARGET], y)
            vectors.append(frame.prediction_sec.to_numpy(float)[ordinary])
        neural_marker = common.read_json(NEURAL / fold / 'manifest.json')
        neural_path = NEURAL / fold / 'finite_score_predictions.parquet'
        assert common.sha256(neural_path) == neural_marker['outputs'][neural_path.name]
        neural = pd.read_parquet(neural_path).set_index(ID).loc[reference.loc[ordinary, ID]]
        np.testing.assert_array_equal(neural[TARGET], y[ordinary])
        vectors[experts.index('tabm_ple8')] = neural.prediction_sec.to_numpy(float)
        matrix = np.column_stack(vectors)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix @ np.asarray(weights['global']), values[ordinary], rtol=0, atol=1e-7)
        entry = dict(rows=len(y), ordinary_rows=int(ordinary.sum()),
            id_hash=common.object_hash(reference[ID].tolist()), label_hash=common.object_hash(y.tolist()),
            current_mse=float(np.mean((y-values)**2)), cases={})
        for name, selected in [('all9', np.ones(len(experts), bool)), ('positive_weight', np.asarray(weights['global']) > 1e-8)]:
            p = matrix[:, selected]
            lower, upper = p.min(axis=1), p.max(axis=1)
            oracle = np.minimum(np.maximum(y[ordinary], lower), upper)
            best_single = np.min((p-y[ordinary, None])**2, axis=1)
            protected_sse = float(np.sum((y[~ordinary]-values[~ordinary])**2))
            hull_sse = float(np.sum((y[ordinary]-oracle)**2))
            outside = (y[ordinary] < lower) | (y[ordinary] > upper)
            base_ordinary_loss = (y[ordinary]-values[ordinary])**2
            entry['cases'][name] = dict(experts=np.asarray(experts)[selected].tolist(),
                oracle_mse=(protected_sse+hull_sse)/len(y),
                best_single_mse=(protected_sse+float(best_single.sum()))/len(y),
                all_experts_wrong_side_rows=int(outside.sum()),
                all_experts_wrong_side_fraction=float(outside.mean()),
                current_ordinary_sse_on_wrong_side_fraction=float(base_ordinary_loss[outside].sum()/base_ordinary_loss.sum()),
                protected_sse=protected_sse, oracle_ordinary_sse=hull_sse)
        results[fold] = entry
    weights = [192122/344841, 152719/344841]
    seasonal = dict(current=float(np.sqrt(sum(w*results[f]['current_mse'] for w, f in zip(weights, results)))))
    assert abs(seasonal['current']-268.662991126314) < 1e-8
    for name in ['all9', 'positive_weight']:
        for kind in ['oracle_mse', 'best_single_mse']:
            seasonal[name+'_'+kind.replace('_mse', '_rmse')] = float(np.sqrt(sum(w*results[f]['cases'][name][kind] for w, f in zip(weights, results))))
    common.write_json(OUT / 'receipt.json', dict(status='diagnostic_complete',
        protocol_sha256=common.sha256(OUT / 'protocol.json'), folds=results, seasonal=seasonal,
        model_trained=False, inference_usable=False, row_level_oracle_predictions_exported=False))
    print('LABEL_AWARE_DIAGNOSTIC_NOT_MODEL', seasonal, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare() if args.declare_only else run()
