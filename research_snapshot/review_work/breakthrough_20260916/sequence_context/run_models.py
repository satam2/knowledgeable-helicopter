"""Prepared bounded ordered-context/static neural comparison; parent schedules GPU."""
import torch
import lightgbm
import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_information as information
import adapter
import cache

core = information.core
OUT = core.external_path(cache.OUT / 'models')
BLOCKS = ['conventions', 'geometry', 'source_past', 'source_twosided', 'surface_T', 'trajectory', 'weather_T']
BASE = core.ROOT / 'private_runs/breakthrough_20260916/information_models' / '__'.join(BLOCKS)
VERIFICATION = core.ROOT / 'private_runs/breakthrough_20260916/missing/sequence_independent_audit/verification.json'


def source_paths():
    return {p.name: p for p in [HERE/'cache.py', HERE/'adapter.py', Path(__file__),
        HERE/'verify_cache.py', HERE/'test_contract.py', HERE/'test_adapter.py', HERE/'test_runner.py',
        Path(information.__file__), Path(information.run_augmented.__file__),
        Path(information.run_augmented.feature_screen.__file__), Path(core.__file__), Path(core.common.__file__)]}


def hashes():
    return {name: cache.sha(path) for name, path in source_paths().items()}


def sampled_rows(index, proxy, seed, limit=200000):
    finite = np.isfinite(proxy)
    eligible = {k: np.asarray(v)[finite[v]] for k, v in index.items()}
    selected = {k: v.copy() for k, v in eligible.items()}
    for j, stage in enumerate(('fit', 'refit')):
        if len(selected[stage]) > limit:
            selected[stage] = np.sort(np.random.default_rng(seed+j).choice(selected[stage], limit, replace=False))
    return selected, eligible


def checked_cache():
    manifest_path = cache.OUT/'manifest.json'
    manifest = core.read_json(manifest_path)
    checked = core.read_json(VERIFICATION)
    if manifest['status'] != 'complete' or checked['status'] != 'passed':
        raise ValueError('Complete independently verified token cache required')
    if checked['manifest_sha256'] != cache.sha(manifest_path) or manifest['source_sha256'] != cache.sha(cache.__file__):
        raise ValueError('Token verification/source is stale')
    if cache.sha(cache.OUT/'neighbors.npy') != manifest['neighbors_sha256']:
        raise ValueError('Neighbor cache bytes changed')
    for record in manifest['records']:
        for kind in ('events', 'queries'):
            if cache.sha(cache.OUT/record[kind+'_file']) != record[kind+'_sha256']:
                raise ValueError('Event or query cache changed')
    return manifest


def load_inputs(manifest):
    cache.memory_guard()
    if psutil.virtual_memory().available < 12*1024**3:
        raise MemoryError('Need12GiB available before loading fullinputs to reserve8GiB plusestimated4GiBworkingmemory')
    baseline = core.read_json(BASE/'lightgbm_aobt_allfinite_s20260916_summary.json')
    x, meta = core.common.load_data()
    standard = [b for b in BLOCKS if b in ['trajectory', *information.run_augmented.BATCH_PATTERNS]]
    x, receipts = information.run_augmented.augment(x, standard)
    x, extra = information.additional(x, [b for b in BLOCKS if b not in standard])
    if list(x) != baseline['features']['columns'] or receipts+extra != baseline['features']['receipts']:
        raise ValueError('Static combined-information comparator differs')
    events, queries = [], []
    for record in manifest['records']:
        cache.memory_guard()
        events.append(pd.read_parquet(cache.OUT/record['events_file'], dtype_backend='pyarrow'))
        queries.append(pd.read_parquet(cache.OUT/record['queries_file'], dtype_backend='pyarrow'))
    events = pd.concat(events, ignore_index=True)
    queries = pd.concat(queries, ignore_index=True)
    np.testing.assert_array_equal(queries[cache.ID], x.index)
    neighbors = np.load(cache.OUT/'neighbors.npy', mmap_mode='r')
    store = adapter.EventStore(events, queries, neighbors)
    cache.memory_guard()
    return x, meta, store


def synthetic(seed, n=2400):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2025-01-01', tz='UTC').value
    times = start + np.arange(n, dtype='int64')*20*10**9
    events = pd.DataFrame({cache.ID: np.arange(n), 'time_ns': times,
        'movement_ns': times-120*10**9, 'phase': np.where(np.arange(n)%2, 'ARR', 'DEP'),
        'runway': np.where(np.arange(n)%2, '01', '02'), 'stand': 'A',
        'aircraft': 'A320', 'wake': 'M', 'operator': 'OP'})
    for c in cache.NUMS:
        events[c] = rng.normal(size=n).astype('float32')
    queries = pd.DataFrame({cache.ID: np.arange(n)+n, 'time_ns': times+1,
        'runway': events.runway, 'stand': 'A', 'operator': 'OP'})
    neighbors = np.arange(n)[:, None]-np.arange(31, -1, -1)[None, :]
    # Context is oldest to newest with right-padding, including no-context coverage.
    for i in range(31):
        valid = neighbors[i][neighbors[i] >= 0]
        neighbors[i] = -1
        neighbors[i, :len(valid)] = valid
    neighbors[0] = -1
    values = {'signal': rng.normal(size=n).astype('float32')}
    for j in range(213):
        values[f'numeric_{j}'] = rng.normal(size=n).astype('float32')
    for j in range(11):
        values[f'category_{j}'] = pd.Categorical(rng.integers(0, 64, size=n).astype(str))
    x = pd.DataFrame(values)
    y = 30 + 2*x.signal.to_numpy() + rng.normal(size=n)
    return x, y, adapter.EventStore(events, queries, neighbors)


def canary(args):
    directory = OUT/'canaries'/f'{args.device}_s{args.seed}'
    if directory.exists():
        raise ValueError('Preserve prior canary; select a newseed')
    cache.memory_guard()
    directory.mkdir(parents=True)
    record = {'status': 'running', 'source_hashes': hashes(), 'device': args.device,
        'private_data_loaded': False, 'created_utc': core.utc_now(), 'variants': {}}
    core.write_json(directory/'manifest.json', record)
    began = time.monotonic()
    try:
        x, target, store = synthetic(args.seed)
        for contextual in (False, True):
            variant = 'context' if contextual else 'static'
            model, fit = adapter.fit(x, target, np.arange(2000), store,
                tune_rows=np.arange(2000, 2200), contextual=contextual, epochs=1,
                device=args.device, threads=args.threads, seed=args.seed)
            refit, refit_info = adapter.fit(x, target, np.arange(2200), store,
                steps=fit['steps'], contextual=contextual,
                device=args.device, threads=args.threads, seed=args.seed)
            joblib.dump(refit, directory/(variant+'.joblib'))
            predicted = adapter.predict(refit, x, np.arange(2200, 2400), store, device=args.device)
            replay = adapter.predict(joblib.load(directory/(variant+'.joblib')), x,
                np.arange(2200, 2400), store, device=args.device)
            delta = float(np.max(np.abs(predicted-replay)))
            if delta > 1e-4:
                raise AssertionError('Serialization replay differs')
            # Scaling is provisional synthetic throughput, not a promised real-data runtime.
            epoch = fit['history'][0]
            record['variants'][variant] = {'fit': fit, 'refit': refit_info, 'replay_delta': delta,
                'projected200Kfit_epoch_seconds': epoch['train_seconds']*100,
                'projected200Ktune_seconds': epoch['tune_seconds']*1000,
                'warning': 'Synthetic querywidth225 and200tune; realcategorymix/full-month timing must be measured separately'}
            del model, refit
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        record.update(status='passed', runtime_sec=time.monotonic()-began,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
        record['outputs'] = {p.name: cache.sha(p) for p in directory.glob('*.joblib')}
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        core.write_json(directory/'manifest.json', record)
        raise
    core.write_json(directory/'manifest.json', record)
    print('CANARY', directory, record['status'], flush=True)


def declare(args, manifest):
    path = OUT/f'protocol_s{args.seed}.json'
    payload = {'source_hashes': hashes(), 'seed': args.seed, 'folds': args.folds, 'threads': args.threads,
        'device': args.device, 'variants': args.variants, 'cache_manifest_sha256': cache.sha(cache.OUT/'manifest.json'),
        'cache_verification_path': str(VERIFICATION), 'cache_verification_sha256': cache.sha(VERIFICATION),
        'baseline_summary_sha256': cache.sha(BASE/'lightgbm_aobt_allfinite_s20260916_summary.json'),
        'fit_refit_sample_cap': 200000, 'tune': 'Every eligible original tune-month row; no sampling',
        'score': 'Every original score row; nonfiniteNM retainsV2 exact',
        'labels': 'Raw Y-P squaredloss; finiteproxyeligibility includesnegativeandlong; no clipping',
        'network': 'GRU64one layer plusstatic64/head64; matchedstatic controlcontextzero; samecontextsupportfeatures',
        'optimizer': {'name': 'AdamW', 'learning_rate': .001, 'weight_decay': .01, 'batch_size': 256, 'max_epochs': 12, 'patience': 3},
        'category_fit_scope': 'Only sampledfit queryrows anduniqueevents referencedbythosequeries; independent refit encoder',
        'availability': manifest['policy'], 'selection': 'Development-exposed folds; no officialscore or newholdout',
        'resource': 'CentralGPU scheduling;8GiBhostfree check beforeloading and everyepoch'}
    if path.exists() and core.read_json(path) != payload:
        raise ValueError('Frozen declaration changed')
    if not path.exists():
        core.write_json(path, payload)
    snapshot = OUT/f'source_s{args.seed}'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, source in source_paths().items():
        dest = snapshot/name
        if dest.exists() and cache.sha(dest) != cache.sha(source):
            raise ValueError('Existing snapshot changed')
        if not dest.exists():
            shutil.copyfile(source, dest)
    return path


def run_one(args, fold, variant, x, meta, store, protocol):
    dest = OUT/f'{variant}_{fold}_s{args.seed}'
    if dest.exists():
        old = core.read_json(dest/'manifest.json')
        if old['status'] != 'complete' or old['source_hashes'] != hashes():
            raise ValueError('Prior incomplete/different run preserved')
        for name, digest in old['outputs'].items():
            if cache.sha(dest/name) != digest:
                raise ValueError('Completed artifact changed')
        return old
    index, split, _ = core.common.fold_data(meta, fold, full=True)
    proxy = meta.proxy_sec.to_numpy(float)
    rows, eligible = sampled_rows(index, proxy, args.seed)
    reference, refrec = core.common.reference(fold)
    if core.object_hash(split) != core.object_hash(refrec['split']):
        raise ValueError('Original purged fold changed')
    np.testing.assert_array_equal(reference[core.ID], meta.iloc[index['score']][core.ID])
    np.testing.assert_array_equal(reference[core.TARGET], meta.iloc[index['score']][core.TARGET])
    np.testing.assert_array_equal(rows['tune'], eligible['tune'])
    dest.mkdir(parents=True)
    record = {'status': 'running', 'created_utc': core.utc_now(), 'source_hashes': hashes(),
        'protocol_sha256': cache.sha(protocol), 'fold': fold, 'variant': variant, 'split': split,
        'original_rows': {k: len(v) for k, v in index.items()}, 'eligible_rows': {k: len(v) for k, v in eligible.items()},
        'sampled_rows': {k: len(v) for k, v in rows.items()},
        'id_hashes': {k: core.object_hash(meta.iloc[v][core.ID].tolist()) for k, v in rows.items()},
        'complete_score_rows': len(reference), 'full_tune_verified': True}
    core.write_json(dest/'manifest.json', record)
    began = time.monotonic()
    try:
        target = meta[core.TARGET].to_numpy(float)-proxy
        model, evidence = adapter.fit(x, target, rows['fit'], store, tune_rows=rows['tune'],
            contextual=variant == 'context', seed=args.seed, threads=args.threads, device=args.device)
        core.write_json(dest/'fit_evidence.json', evidence)
        tune = adapter.predict(model, x, rows['tune'], store, device=args.device)+proxy[rows['tune']]
        pd.DataFrame({core.ID: meta.iloc[rows['tune']][core.ID].to_numpy(), 'prediction_sec': tune}).to_parquet(dest/'tune_predictions.parquet', index=False)
        del model
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        model, refit = adapter.fit(x, target, rows['refit'], store, steps=evidence['steps'],
            contextual=variant == 'context', seed=args.seed, threads=args.threads, device=args.device)
        core.write_json(dest/'refit_evidence.json', refit)
        joblib.dump(model, dest/'model.joblib')
        residual = adapter.predict(model, x, rows['score'], store, device=args.device)
        replay = adapter.predict(joblib.load(dest/'model.joblib'), x, rows['score'][:2048], store, device=args.device)
        delta = float(np.max(np.abs(residual[:len(replay)]-replay)))
        if delta > 1e-4:
            raise AssertionError('Saved model replay differs')
        available = np.isfinite(proxy[index['score']])
        prediction = reference.prediction_sec.to_numpy().copy()
        prediction[available] = residual+proxy[rows['score']]
        reports = {}
        baseline = reference.prediction_sec.to_numpy()
        for name, values in {'candidate': prediction, 'blend25': baseline+.25*(prediction-baseline)}.items():
            np.testing.assert_array_equal(values[~available], baseline[~available])
            frame = reference.drop(columns=[core.TARGET, 'error_sec', 'squared_error', 'label_bin', 'month', 'day'], errors='ignore').copy()
            frame['prediction_sec'] = values
            metrics, errors = core.evaluate(frame, meta.iloc[index['score']][[core.ID, core.TARGET]])
            assert len(errors) == len(reference)
            errors.to_parquet(dest/(name+'.parquet'), index=False)
            reports[name] = {'metrics': metrics, 'stability': core.paired_stability(reference, errors, repetitions=500)}
        if record['source_hashes'] != hashes():
            raise AssertionError('Sources changed during training')
        record.update(status='complete', reports=reports, fit=evidence, refit=refit,
            runtime_sec=time.monotonic()-began, replay_delta=delta,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
        record['outputs'] = {p.name: cache.sha(p) for p in dest.iterdir() if p.is_file() and p.name != 'manifest.json'}
        core.write_json(dest/'manifest.json', record)
        print('RESULT', fold, variant, reports['candidate']['metrics']['overall']['rmse_sec'], flush=True)
        return record
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        core.write_json(dest/'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--variants', nargs='+', choices=['static', 'context'], default=['static', 'context'])
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.canary:
        canary(args)
        return
    manifest = checked_cache()
    protocol = declare(args, manifest)
    if args.prepare_only:
        print('PREPARED', protocol, flush=True)
        return
    canary_path = OUT/'canaries'/f'{args.device}_s{args.seed}'/'manifest.json'
    check = core.read_json(canary_path)
    if check['status'] != 'passed' or check['source_hashes'] != hashes():
        raise ValueError('Matching device/source canary required before fullfit')
    for name, digest in check['outputs'].items():
        if cache.sha(canary_path.parent/name) != digest:
            raise ValueError('Canary artifact changed')
    x, meta, store = load_inputs(manifest)
    for variant in args.variants:
        records = {fold: run_one(args, fold, variant, x, meta, store, protocol) for fold in args.folds}
        if set(records) == {'F1', 'F3'}:
            summary = {name: core.season_score(records['F1']['reports'][name]['metrics']['overall'],
                records['F3']['reports'][name]['metrics']['overall']) for name in ['candidate', 'blend25']}
            core.write_json(OUT/f'{variant}_s{args.seed}_summary.json', {'seasonal_rmse': summary})
            print('SEASONAL', variant, summary, flush=True)


if __name__ == '__main__':
    main()
