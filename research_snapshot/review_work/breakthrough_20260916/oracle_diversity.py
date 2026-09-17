"""Label-aware diagnostic of model complementarity; never an inference recipe."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import common

REGISTRY = {
    'lgb63': 'deeper_lgb/combined/lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
    'lgb600': 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T/lightgbm_aobt_allfinite_{fold}_s20260916',
    'tabm_source': 'models/augmented/source_past__source_twosided/tabm_aobt_allfinite_{fold}_s20260916',
    'tabm_ple8': 'models/tabm_ple/base/tabm_ple8_aobt_allfinite_{fold}_s20260916',
    'catboost_conventions': 'missing/conventions_models/models/global_conventions_{fold}_s20260916',
    'missing_template': 'missing/id_context_v1/models/historical_template_{fold}_s20260916',
    'missing_rf': 'models/missing_forest/randomforest_missing_template_idcontext_{fold}_s20260916',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = common.external_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    root = common.WORKSPACE / 'private_runs/breakthrough_20260916'
    receipts, frames = {}, {}
    for fold in ['F1', 'F3']:
        receipts[fold], frames[fold] = {}, {}
        for expert, pattern in REGISTRY.items():
            directory = root / pattern.format(fold=fold)
            manifest = common.read_json(directory / 'manifest.json')
            assert manifest['status'] == 'complete'
            path = directory / 'candidate.parquet'
            assert common.sha256(path) == manifest['outputs'][path.name]
            receipts[fold][expert] = {'manifest_sha256': common.sha256(directory / 'manifest.json'),
                                     'prediction_sha256': manifest['outputs'][path.name], 'path': str(path)}
            frames[fold][expert] = path
    common.write_json(output / 'protocol.json', {
        'created_utc': common.utc_now(), 'script_sha256': common.sha256(__file__), 'inputs': receipts,
        'purpose': 'Optimistic label-aware complementarity diagnostic. Neither oracle is deployable, a noise floor, or proof a gate can learn the selection.',
        'restriction': 'No fitted model, no exported per-row oracle predictions, no use in training or stacking.',
        'variants': 'Best discrete expert per labeled row; perfect convex interpolation inside expert prediction range.'})
    rows = []
    for fold in frames:
        reference, _ = common.reference(fold)
        y = reference[common.TARGET].to_numpy(float)
        predictions = {'reference': reference.prediction_sec.to_numpy(float)}
        for expert, path in frames[fold].items():
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame[common.ID], reference[common.ID])
            np.testing.assert_array_equal(frame[common.TARGET], y)
            predictions[expert] = frame.prediction_sec.to_numpy(float)
        matrix = np.column_stack(list(predictions.values()))
        assert np.isfinite(matrix).all()
        squared = (matrix - y[:, None]) ** 2
        best = squared.min(axis=1)
        envelope = np.clip(y, matrix.min(axis=1), matrix.max(axis=1))
        proxy = reference.proxy_sec.to_numpy(float)
        missing = ~np.isfinite(proxy)
        gap = ~missing & (np.abs(y - proxy) > 1800)
        for name, mask in {'all': np.ones(len(y), bool), 'missing': missing,
                           'large_source_gap': gap, 'ordinary': ~(missing | gap)}.items():
            row = {'fold': fold, 'group': name, 'rows': int(mask.sum()),
                   'reference_mse_contribution': float(squared[mask, 0].sum() / len(y)),
                   'lgb63_mse_contribution': float(squared[mask, 1].sum() / len(y)),
                   'discrete_oracle_mse_contribution': float(best[mask].sum() / len(y)),
                   'convex_oracle_mse_contribution': float(((envelope - y)[mask] ** 2).sum() / len(y))}
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'fold_groups.csv', index=False)
    weights = {'F1': 192122 / 344841, 'F3': 152719 / 344841}
    summary = {}
    for group, values in table.groupby('group'):
        values = values.set_index('fold')
        summary[group] = {col: sum(weights[f] * values.loc[f, col] for f in weights)
                          for col in table if col.endswith('_contribution')}
    summary['seasonal_rmse'] = {key.removesuffix('_mse_contribution'): float(np.sqrt(value))
                                for key, value in summary['all'].items()}
    common.write_json(output / 'summary.json', summary)
    print(summary['seasonal_rmse'], flush=True)


if __name__ == '__main__':
    main()
