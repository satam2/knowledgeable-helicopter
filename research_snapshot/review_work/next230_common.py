"""Immutable next-batch provenance and external artifacts."""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostRegressor

from run_screening import OUT as OLD, RAW, WORKSPACE, config_for
from taxiout.artifacts import environment, object_hash, read_json, sha256, source_hashes, utc_now, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET
from taxiout.splits import make_fold

OUT = external_path(WORKSPACE / 'private_runs/next_230')
SOURCES = ['next230_common.py', 'next230_features.py', 'next230_prepare.py', 'next230_train.py']
RECIPES = {
    'capacity_d8_5000': {'depth': 8, 'iterations': 5000},
    'capacity_d6_5000': {'depth': 6, 'iterations': 5000},
    'capacity_d10_2500': {'depth': 10, 'iterations': 2500},
    'runway_d8_2500': {'depth': 8, 'iterations': 2500, 'features': 'runway'},
    'rome_base_1200': {'depth': 6, 'iterations': 1200, 'l2_leaf_reg': 20, 'task_type': 'CPU'},
    'rome_context_1200': {'depth': 6, 'iterations': 1200, 'l2_leaf_reg': 20, 'task_type': 'CPU', 'features': 'schedule'},
    'clock_experts': {'depth': 8, 'iterations': 2500},
}


def extension_hashes():
    return {f: sha256(Path(__file__).with_name(f)) for f in SOURCES}


def verify_protocol():
    old = read_json(OLD / 'protocol.json')
    if source_hashes() != old['source_hashes']:
        raise ValueError('Frozen reference source changed')
    manifest = verified_manifest(config_for('baseline'))
    protocol_file = OUT / 'protocol.json'
    if not protocol_file.exists():
        write_json(protocol_file, {'created_utc': utc_now(), 'reference': 'H_gpu_full',
            'reference_seasonal_rmse_sec': 299.053253, 'target_sec': 230, 'intermediate_target_sec': 280,
            'source_hashes': source_hashes(), 'extension_hashes': extension_hashes(),
            'raw_hashes': old['raw_hashes'], 'recipes': RECIPES, 'seeds': [20260910, 20260911, 20260912],
            'plan_sha256': sha256(OUT / 'PLAN.md'), 'status': 'declared before new score inspection',
            'environment': environment()})
    protocol = read_json(protocol_file)
    if protocol['extension_hashes'] != extension_hashes() or protocol['recipes'] != RECIPES:
        raise ValueError('New batch source or recipes changed after freeze')
    return manifest


def load_data():
    manifest = verify_protocol()
    x, meta, labels = load_training(config_for('baseline'), manifest)
    if not np.array_equal(x.index, labels[ID]) or not np.array_equal(x.index, meta[ID]):
        raise ValueError('Training ID mismatch')
    return x, meta, labels


def same_split(first, second):
    return object_hash(first) == object_hash(second)


def load_reference(fold, meta):
    record = read_json(OLD / 'refinements' / f'H_gpu_full_{fold}.json')
    path = OLD / 'models' / record['run_id']
    stored = read_json(path / 'manifest.json')
    if record != stored or record['status'] != 'complete':
        raise ValueError('Reference record mismatch')
    file = path / 'score_predictions.parquet'
    if sha256(file) != record['outputs'][file.name]:
        raise ValueError('Reference prediction corruption')
    predictions = pd.read_parquet(file)
    idx, split = make_fold(meta, load_config('configs/folds.yaml')[fold])
    if not same_split(split, record['split']) or not np.array_equal(meta.iloc[idx['score']][ID], predictions[ID]):
        raise ValueError('Reference split mismatch')
    return record, predictions, idx, split


def load_extension(x, kind):
    marker = read_json(OUT / 'features' / 'manifest.json')
    if marker['feature_source_sha256'] != sha256(Path(__file__).with_name('next230_features.py')):
        raise ValueError('Extension source mismatch')
    path = OUT / 'features' / 'extensions.parquet'
    if marker['sha256'] != sha256(path):
        raise ValueError('Extension cache mismatch')
    ext = pd.read_parquet(path).set_index(ID)
    if not np.array_equal(ext.index, x.index):
        raise ValueError('Extension IDs mismatch')
    selected = [c for c in ext if c.startswith('rw_')] if kind == 'runway' else [c for c in ext if not c.startswith('rw_')]
    return pd.concat([x, ext[selected]], axis=1)


def fit(x, y, config, tuning=None, trees=None):
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite labels')
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise MemoryError('Need 4 GB host memory reserve')
    params = dict(iterations=trees or config['iterations'], depth=config['depth'],
                  learning_rate=config['learning_rate'], l2_leaf_reg=config['l2_leaf_reg'],
                  loss_function='RMSE', eval_metric='RMSE', random_seed=config['seed'],
                  thread_count=4, task_type=config['task_type'], verbose=500, allow_writing_files=False)
    if config['task_type'] == 'GPU':
        params.update(devices='0', gpu_ram_part=.25)
    model = CatBoostRegressor(**params)
    options = {'cat_features': list(x.select_dtypes('category').columns)}
    if tuning is not None:
        options.update(eval_set=tuning, early_stopping_rounds=100, use_best_model=True)
    start = time.monotonic()
    model.fit(x, y, **options)
    return model, {'rows': len(x), 'features': len(x.columns), 'trees': model.tree_count_,
                   'runtime_sec': time.monotonic() - start, 'task_type': config['task_type'],
                   'available_ram_after': psutil.virtual_memory().available,
                   'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}


def start_run(name, fold, seed, config, reference, split):
    path = OUT / 'models' / f'{name}_{fold}_s{seed}'
    if path.exists():
        record = read_json(path / 'manifest.json')
        if record['status'] == 'complete':
            if record['extension_hashes'] != extension_hashes():
                raise ValueError('Completed run source changed')
            for f, digest in record['outputs'].items():
                if sha256(path / f) != digest:
                    raise ValueError(f'Corrupt completed run: {f}')
            print(f'REUSED {path.name}: {record["metrics"]["overall"]["rmse_sec"]:.6f}s', flush=True)
            return path, record, True
        raise ValueError(f'Prior incomplete attempt needs diagnosis: {path}')
    path.mkdir(parents=True)
    record = {'status': 'incomplete', 'created_utc': utc_now(), 'candidate': name, 'fold': fold,
              'seed': seed, 'config': config, 'reference_run': reference['run_id'],
              'reference_manifest_sha256': sha256(OLD / 'models' / reference['run_id'] / 'manifest.json'),
              'split': split, 'extension_hashes': extension_hashes(),
              'protocol_sha256': sha256(OUT / 'protocol.json')}
    write_json(path / 'manifest.json', record)
    return path, record, False


def finish_run(path, record, reference, predictions, labels, changed, evidence):
    if not np.array_equal(reference[ID], predictions[ID]) or not np.array_equal(reference[TARGET], labels[TARGET]):
        raise ValueError('Score cohort or labels changed')
    if not np.array_equal(reference.loc[~changed, 'prediction_sec'], predictions.loc[~changed, 'prediction_sec']):
        raise ValueError('Protected routes changed')
    predictions = predictions.drop(columns=[TARGET, 'error_sec', 'squared_error'], errors='ignore')
    predictions = predictions.rename(columns={'new_category': 'reference_new_category'})
    predictions['candidate'], predictions['fold'] = record['candidate'], record['fold']
    metrics, errors = evaluate(predictions, labels)
    errors.to_parquet(path / 'score_predictions.parquet', index=False)
    write_json(path / 'metrics.json', metrics)
    record.update(status='complete', completed_utc=utc_now(), metrics=metrics,
                  score_id_hash=object_hash(errors[ID].tolist()), protected_routes_equal=True,
                  changed_rows=int(changed.sum()), **evidence)
    record['outputs'] = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name != 'manifest.json'}
    write_json(path / 'manifest.json', record)
    write_json(OUT / 'results' / f'{path.name}.json', record)
    print(f'RESULT {path.name}: {metrics["overall"]["rmse_sec"]:.6f}s', flush=True)
    gc.collect()
    return record


def reload_parity(model, path, x):
    model.save_model(str(path))
    saved = CatBoostRegressor()
    saved.load_model(str(path))
    values = model.predict(x, thread_count=4)
    repeated = saved.predict(x, thread_count=4)
    delta = float(np.max(np.abs(values - repeated)))
    if delta > 1e-9:
        raise ValueError('Model reload parity failed')
    return values, delta
