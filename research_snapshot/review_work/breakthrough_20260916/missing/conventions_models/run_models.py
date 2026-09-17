"""Matched full-data ordinary-clock residual control/features/airport specialization."""

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from adapter import fit_bank, predict_bank
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET

OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/conventions_models')
FEATURES = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/source_conventions')
ARMS = ['global_base', 'global_conventions', 'airport_conventions']


def parameters(seed=20260916, threads=2):
    return dict(iterations=5000, depth=8, learning_rate=.05, l2_leaf_reg=8,
        loss_function='RMSE', eval_metric='RMSE', random_seed=seed, task_type='GPU', devices='0',
        gpu_ram_part=.65, thread_count=threads, allow_writing_files=False, verbose=500)


def declare(seed, threads):
    feature_marker = read_json(FEATURES / 'manifest.json')
    assert sha256(FEATURES / 'features.parquet') == feature_marker['feature_sha256']
    assert sha256(HERE.parent / 'source_conventions/prepare_conventions.py') == feature_marker['source_sha256']
    old = read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    assert old['base_config']['l2_leaf_reg'] == 8
    declaration = {'arms': ARMS, 'params': parameters(seed, threads), 'folds': ['F1', 'F3'],
        'availability': 'Observation-only supplied-record clock conventions plus frozen month-isolated causal traffic; no new retrospective batch context.',
        'target': 'Y-(takeoff-NM actual offblock), exactly NMoffblock-airportoffblock disagreement',
        'eligibility': 'Ordinary observed actual clock proxy finite in[0,7200], all original eligible fit/tune/refit rows; no sampling.',
        'protected_routes': 'Every NM-missing, negative-proxy or >7200s proxy score row retains accepted V2 prediction exactly.',
        'comparison': 'global_base vs global_conventions isolates feature block; global_conventions vs airport_conventions isolates airport model allocation. Same max trees/depth, not equal parameter count or compute.',
        'reference_limit': 'These replace the final V2 prediction on eligible routes with a new standalone residual expert; V2 also contains a direct expert and learned gate. Global control is matched across these three arms, not bit-identical to V2.',
        'tuning': '100-round RMSE early stopping on original tune rows; each airport has its own tune count; fresh original fit+tune refit.',
        'variants': ['candidate', 'blend25'], 'blend_weight': .25,
        'resource_policy': 'One GPU process centrally scheduled, sequential airport fits; gpu_ram_part0.65, two CPU threads, at least8GiB available RAM.',
        'source_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), HERE / 'adapter.py', ROOT / 'review_work/campaign_20260916/common.py']},
        'conventions_manifest_sha256': sha256(FEATURES / 'manifest.json'),
        'conventions_feature_sha256': feature_marker['feature_sha256'], 'seed_scope': 'One entire fit/tune/refit seed per arm'}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        record = read_json(path)
        if record['declaration'] != declaration:
            raise ValueError('Frozen conventions-model declaration changed')
        return record
    record = {'created_utc': utc_now(), 'declaration': declaration}
    write_json(path, record)
    return record


def run_arm(arm, fold, x, meta, seed, threads):
    name = f'{arm}_{fold}_s{seed}'
    path = external_path(OUT / 'models' / name)
    if path.exists():
        existing = read_json(path / 'manifest.json')
        if existing['status'] != 'complete':
            raise ValueError('Incomplete GPU run retained; investigate before any retry')
        for filename, digest in existing['outputs'].items():
            assert sha256(path / filename) == digest
        print('REUSED', name, flush=True)
        return
    path.mkdir(parents=True)
    idx, split, _ = common.fold_data(meta, fold, full=True)
    reference, original = common.reference(fold)
    assert object_hash(split) == object_hash(original['split'])
    assert np.array_equal(meta.iloc[idx['score']][ID], reference[ID])
    assert np.array_equal(meta.iloc[idx['score']][TARGET], reference[TARGET])
    proxy = meta.proxy_sec.to_numpy(float)
    ordinary = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    rows = {stage: positions[ordinary[positions]] for stage, positions in idx.items()}
    if arm == 'global_base':
        selected_columns = [column for column in x if not column.startswith('conv_')]
    else:
        selected_columns = list(x)
    frames = {stage: x.iloc[positions][selected_columns].copy() for stage, positions in rows.items()}
    airports = {stage: meta.iloc[positions].ADEP_mvt.astype(str).to_numpy() for stage, positions in rows.items()}
    residual = meta[TARGET].to_numpy(float) - proxy
    record = {'status': 'running', 'name': name, 'arm': arm, 'fold': fold, 'family': 'catboost_gpu_ordinary',
        'target': 'nm_source_correction', 'full': True, 'seed': seed, 'threads': threads,
        'created_utc': utc_now(), 'split': split, 'protocol_sha256': sha256(OUT / 'protocol.json'),
        'fit_ids': {stage: {'n': len(positions), 'hash': object_hash(meta.iloc[positions][ID].tolist())} for stage, positions in rows.items()},
        'features_used': selected_columns, 'per_airport': arm == 'airport_conventions'}
    write_json(path / 'manifest.json', record)
    start = time.monotonic()
    try:
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise MemoryError('Need8GiB available host RAM for GPU model bank')
        tuned, tune_info = fit_bank(frames['fit'], residual[rows['fit']], airports['fit'], path / 'fit_models',
            parameters(seed, threads), per_airport=record['per_airport'],
            tuning=(frames['tune'], residual[rows['tune']], airports['tune']))
        tune_prediction = proxy[rows['tune']] + predict_bank(tuned, frames['tune'], airports['tune'], threads)
        pd.DataFrame({ID: meta.iloc[rows['tune']][ID].to_numpy(), 'prediction_sec': tune_prediction}).to_parquet(path / 'tune_predictions.parquet', index=False)
        write_json(path / 'fit_bank.json', tuned)
        model, refit_info = fit_bank(frames['refit'], residual[rows['refit']], airports['refit'], path / 'refit_models',
            parameters(seed, threads), per_airport=record['per_airport'], selected_steps=tune_info['selected_steps'])
        write_json(path / 'model_bank.json', model)
        inference_start = time.monotonic()
        prediction = proxy[rows['score']] + predict_bank(model, frames['score'], airports['score'], threads)
        inference_time = time.monotonic() - inference_start
        repeated = proxy[rows['score']] + predict_bank(read_json(path / 'model_bank.json'), frames['score'], airports['score'], threads)
        replay = float(np.max(np.abs(prediction - repeated)))
        assert replay <= 1e-9 and np.isfinite(prediction).all()
        mask = ordinary[idx['score']]
        ref_values = reference.prediction_sec.to_numpy()
        candidate = ref_values.copy()
        candidate[mask] = prediction
        reports = {}
        for variant, values in {'candidate': candidate, 'blend25': ref_values + .25 * (candidate - ref_values)}.items():
            assert np.array_equal(values[~mask], ref_values[~mask])
            pred = reference.drop(columns=[TARGET, 'error_sec', 'squared_error', 'label_bin', 'month', 'day'], errors='ignore').copy()
            pred['prediction_sec'] = values
            metrics, errors = evaluate(pred, meta.iloc[idx['score']][[ID, TARGET]])
            errors.to_parquet(path / f'{variant}.parquet', index=False)
            reports[variant] = {'metrics': metrics, 'stability': paired_stability(reference, errors, repetitions=500)}
        record.update(status='complete', completed_utc=utc_now(), tune=tune_info, refit=refit_info, reports=reports,
            changed_rows=int(mask.sum()), complete_score_rows=len(reference), protected_routes_equal=True,
            runtime_sec=time.monotonic() - start, inference_runtime_sec=inference_time, reload_max_abs_delta=replay,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))
        record['outputs'] = {str(file.relative_to(path)): sha256(file) for file in path.rglob('*') if file.is_file() and file.name != 'manifest.json'}
        write_json(path / 'manifest.json', record)
        print('RESULT', name, {key: value['metrics']['overall']['rmse_sec'] for key,value in reports.items()}, flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), runtime_sec=time.monotonic() - start)
        write_json(path / 'manifest.json', record)
        raise
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare(args.seed, args.threads)
    if args.declare_only:
        print('Declared GPU ordinary-only controls; no GPU initialized or training launched', flush=True)
        return
    x, meta = common.load_data()
    extension = pd.read_parquet(FEATURES / 'features.parquet').set_index(ID)
    assert np.array_equal(extension.index, x.index)
    for column in extension.select_dtypes(['object','string']):
        extension[column] = extension[column].astype('category')
    x = pd.concat([x, extension], axis=1)
    for arm in args.arms:
        for fold in args.folds:
            run_arm(arm, fold, x, meta, args.seed, args.threads)


if __name__ == '__main__':
    main()
