"""Concurrent direct CPU expert; tune-only gates reuse fit-only residual receipts."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from next230_common import (OUT, config_for, load_data, load_reference, fit, start_run,
                           finish_run, reload_parity, same_split)
from next230_features import fit_blend, blend_weights, apply_blend
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import ID, TARGET

CONFIG = {**config_for('baseline'), 'iterations': 1200, 'depth': 6, 'learning_rate': .05,
          'task_type': 'CPU', 'train_sample': None, 'seed': 20260910}


def protocol():
    path = OUT / 'clock_cpu_protocol.json'
    value = {'config': CONFIG, 'runner_sha256': sha256(__file__),
             'residual': 'capacity_d8_5000', 'shrinkage': 1000,
             'rationale': 'Run direct CPU expert concurrently; reuse stored fit-only residual tune predictions and matching refit.',
             'selection': 'Declared after Rome screens and capacity F1, before any direct CPU score or gate weights.'}
    if path.exists() and read_json(path) != value:
        raise ValueError('Clock CPU protocol changed')
    write_json(path, value)


def train(fold, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    path = OUT / 'clock_cpu_experts' / fold
    if (path / 'expert.json').exists():
        print(f'REUSED DIRECT EXPERT {fold}', flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    status = proxy_status(meta.proxy_sec, CONFIG)
    sf, st, sr = [p[status[p] == 'present'] for p in [idx['fit'], idx['tune'], idx['refit']]]
    y = labels[TARGET].to_numpy(float)
    print(f'CPU DIRECT {fold}: fit={len(sf)} tune={len(st)} refit={len(sr)}', flush=True)
    direct, tuning = fit(x.iloc[sf], y[sf], CONFIG, tuning=(x.iloc[st], y[st]))
    direct.save_model(str(path / 'fit_only_direct.cbm'))
    pd.DataFrame({ID: x.index[st], TARGET: y[st], 'direct': direct.predict(x.iloc[st], thread_count=4)}).to_parquet(path / 'tune_direct.parquet', index=False)
    trees = direct.tree_count_
    del direct
    gc.collect()
    direct, refit = fit(x.iloc[sr], y[sr], CONFIG, trees=trees)
    values, delta = reload_parity(direct, path / 'component.cbm', x.iloc[idx['score']])
    pd.DataFrame({ID: x.index[idx['score']], 'direct': values}).to_parquet(path / 'score_direct.parquet', index=False)
    evidence = {'status': 'complete', 'split': split, 'config': CONFIG, 'trees': trees,
                'training': {'tune': tuning, 'refit': refit}, 'reload_max_abs_delta': delta,
                'runner_sha256': sha256(__file__), 'protocol_sha256': sha256(OUT / 'clock_cpu_protocol.json'),
                'fit_id_hash': object_hash(x.index[sf].tolist()), 'tune_id_hash': object_hash(x.index[st].tolist()),
                'refit_id_hash': object_hash(x.index[sr].tolist()),
                'outputs': {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name != 'expert.json'}}
    write_json(path / 'expert.json', evidence)
    print(f'CPU DIRECT SAVED {fold}: {trees} trees', flush=True)


def calibrate(fold, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    direct_path = OUT / 'clock_cpu_experts' / fold
    expert = read_json(direct_path / 'expert.json')
    residual_path = OUT / 'models' / f'capacity_d8_5000_{fold}_s20260910'
    residual = read_json(residual_path / 'manifest.json')
    if expert['status'] != 'complete' or residual['status'] != 'complete':
        raise ValueError('Wait for both experts to finish')
    if not same_split(split, expert['split']) or not same_split(split, residual['split']):
        raise ValueError('Expert split mismatch')
    for directory, manifest in [(direct_path, expert), (residual_path, residual)]:
        for name, digest in manifest['outputs'].items():
            if sha256(directory / name) != digest:
                raise ValueError('Expert receipt hash mismatch')
    status = proxy_status(meta.proxy_sec, CONFIG)
    st = idx['tune'][status[idx['tune']] == 'present']
    tune_direct = pd.read_parquet(direct_path / 'tune_direct.parquet')
    tune_residual = pd.read_parquet(residual_path / 'tune_predictions.parquet')
    if not np.array_equal(x.index[st], tune_direct[ID]) or not np.array_equal(x.index[st], tune_residual[ID]):
        raise ValueError('Calibration IDs differ')
    y = labels.iloc[st][TARGET].to_numpy(float)
    if not np.array_equal(y, tune_direct[TARGET]) or not np.array_equal(y-meta.iloc[st].proxy_sec.to_numpy(), tune_residual.target):
        raise ValueError('Calibration labels differ')
    gate = fit_blend(y, meta.iloc[st].proxy_sec.to_numpy()+tune_residual.prediction.to_numpy(),
                     tune_direct.direct, meta.iloc[st].ADEP_mvt, shrinkage=1000)
    new_residual = pd.read_parquet(residual_path / 'score_predictions.parquet')
    direct_scores = pd.read_parquet(direct_path / 'score_direct.parquet')
    if not np.array_equal(baseline[ID], new_residual[ID]) or not np.array_equal(baseline[ID], direct_scores[ID]):
        raise ValueError('Score IDs differ')
    changed = baseline.route.eq('residual').to_numpy()
    original = baseline.prediction_sec.to_numpy().copy()
    original[changed] = new_residual.loc[changed, 'prediction_sec']
    for conditioned in [False, True]:
        name = 'clock_cpu_airport' if conditioned else 'clock_cpu_global'
        path, record, reused = start_run(name, fold, 20260910, CONFIG, reference, split)
        if reused:
            continue
        predictions = baseline.copy()
        predictions['prediction_sec'] = apply_blend(original, direct_scores.direct, baseline.route,
            blend_weights(gate, baseline.ADEP_mvt, conditioned))
        write_json(path / 'gate.json', gate)
        finish_run(path, record, baseline, predictions, labels.iloc[idx['score']], changed,
            {'gate': gate, 'conditioned': conditioned, 'direct_expert_path': str(direct_path),
             'residual_expert_path': str(residual_path), 'expert_manifest_sha256': sha256(direct_path/'expert.json'),
             'residual_manifest_sha256': sha256(residual_path/'manifest.json'),
             'runner_sha256': sha256(__file__), 'supplemental_protocol_sha256': sha256(OUT/'clock_cpu_protocol.json'),
             'reload_max_abs_delta': expert['reload_max_abs_delta']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'calibrate'])
    p.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    args = p.parse_args()
    protocol()
    data = load_data()
    for fold in args.folds:
        (train if args.action == 'train' else calibrate)(fold, data)
