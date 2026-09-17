"""Recorded two-by-two information/capacity comparison, without new fitting."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = common.WORKSPACE / 'private_runs/breakthrough_20260916'
    registry = {
        ('600trees31leaves', 'base30'): 'cpu/base/lightgbm_aobt_allfinite_{fold}_s20260916',
        ('600trees31leaves', 'combined225'): 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T/lightgbm_aobt_allfinite_{fold}_s20260916',
        ('2500trees63leaves', 'base30'): 'deeper_lgb/base/lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
        ('2500trees63leaves', 'combined225'): 'deeper_lgb/combined/lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
    }
    rows, receipts = [], {}
    weights = {'F1': 192122 / 344841, 'F3': 152719 / 344841}
    references = {fold: common.reference(fold)[0] for fold in weights}
    for (capacity, information), pattern in registry.items():
        errors, columns = {}, None
        for fold in weights:
            directory = root / pattern.format(fold=fold)
            manifest = common.read_json(directory / 'manifest.json')
            assert manifest['status'] == 'complete'
            path = directory / 'candidate.parquet'
            assert common.sha256(path) == manifest['outputs'][path.name]
            frame = pd.read_parquet(path)
            reference = references[fold]
            np.testing.assert_array_equal(frame[common.ID], reference[common.ID])
            np.testing.assert_array_equal(frame[common.TARGET], reference[common.TARGET])
            missing = ~np.isfinite(reference.proxy_sec.to_numpy(float))
            np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing], reference.prediction_sec.to_numpy()[missing])
            if columns is not None:
                assert columns == manifest['feature_columns']
            columns = manifest['feature_columns']
            errors[fold] = float(np.mean((frame.prediction_sec - frame[common.TARGET]) ** 2))
            receipts[str(path)] = {'manifest_sha256': common.sha256(directory / 'manifest.json'),
                                   'prediction_sha256': manifest['outputs'][path.name],
                                   'fit_ids': manifest['fit_ids'], 'columns': columns}
        rows.append({'capacity': capacity, 'information': information, 'features': len(columns),
                     'F1_rmse': np.sqrt(errors['F1']), 'F3_rmse': np.sqrt(errors['F3']),
                     'seasonal_rmse': np.sqrt(sum(weights[f] * errors[f] for f in weights))})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index='capacity', columns='information', values='seasonal_rmse')
    changes = {'information_gain_seconds': {capacity: float(row.base30 - row.combined225) for capacity, row in pivot.iterrows()},
               'capacity_gain_seconds': {information: float(pivot.loc['600trees31leaves', information] - pivot.loc['2500trees63leaves', information])
                                         for information in ['base30', 'combined225']}}
    output = common.external_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / 'two_by_two.csv', index=False)
    common.write_json(output / 'audit.json', {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'inputs': receipts, 'contrasts': changes, 'rows': table.to_dict('records'),
        'interpretation': 'Matched model settings within each information contrast; capacity changes several hyperparameters together, not leaf count alone. Each fit independently early-stops on original tune. Adaptive exposed development folds, one seed.',
        'availability': 'Combined includes separately declared retrospective batch information, not a strict real-time claim.',
        'provenance_limit': 'Original deeper runner omitted imported encoder module in historical source-hash map; independent audit records current reviewed source, not retroactive historical proof.'})
    print(table.to_string(index=False), flush=True)
    print(changes, flush=True)


if __name__ == '__main__':
    main()
