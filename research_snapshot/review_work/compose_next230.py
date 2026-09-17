"""Fixed equal-weight seed ensembles or disjoint route compositions, never score-fitted weights."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from next230_common import OUT, load_data, load_reference, start_run, finish_run, same_split
from taxiout.artifacts import read_json, sha256, write_json


def compose(name, folds, residual, rome, seeds, rome_seeds):
    _, meta, labels = load_data()
    for fold in folds:
        reference, baseline, idx, split = load_reference(fold, meta)
        config = {'residual': residual, 'rome': rome, 'seeds': seeds, 'rome_seeds': rome_seeds,
                  'weights': 'equal; declared before ensemble score'}
        path, record, reused = start_run(name, fold, 20260910, config, reference, split)
        if reused:
            continue
        result = baseline.copy()
        changed = np.zeros(len(result), dtype=bool)
        components = []
        for candidate, routes in [(residual, ['residual', 'residual_long_proxy']), (rome, ['rome_schedule_residual'])]:
            if not candidate:
                continue
            component_seeds = seeds if candidate == residual else rome_seeds
            paths = [OUT / 'models' / f'{candidate}_{fold}_s{seed}' for seed in component_seeds]
            values = []
            for source in paths:
                evidence = read_json(source / 'manifest.json')
                if evidence['status'] != 'complete' or not same_split(evidence['split'], split):
                    raise ValueError('Component is incomplete or has a different split')
                for file, digest in evidence['outputs'].items():
                    if sha256(source / file) != digest:
                        raise ValueError('Component hash mismatch')
                predictions = pd.read_parquet(source / 'score_predictions.parquet')
                if not np.array_equal(baseline.MVT_ID_mvt, predictions.MVT_ID_mvt):
                    raise ValueError('Component ID mismatch')
                values.append(predictions.prediction_sec.to_numpy())
            mask = baseline.route.isin(routes).to_numpy()
            result.loc[mask, 'prediction_sec'] = np.mean(values, axis=0)[mask]
            changed |= mask
            components.append({'routes': routes, 'paths': [str(p) for p in paths],
                               'manifest_hashes': {str(p): sha256(p / 'manifest.json') for p in paths}})
        write_json(path / 'composition.json', components)
        finish_run(path, record, baseline, result, labels.iloc[idx['score']], changed,
                   {'composition': components, 'weights': 'fixed equal', 'reload_max_abs_delta': 0.0,
                    'runner_sha256': sha256(__file__)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    p.add_argument('--residual')
    p.add_argument('--rome')
    p.add_argument('--seeds', nargs='+', type=int, default=[20260910])
    p.add_argument('--rome-seeds', nargs='+', type=int, default=[20260910])
    a = p.parse_args()
    compose(a.name, a.folds, a.residual, a.rome, a.seeds, a.rome_seeds)
