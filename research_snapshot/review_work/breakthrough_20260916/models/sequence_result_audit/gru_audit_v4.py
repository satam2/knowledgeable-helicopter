"""Independent sampled-cohort audit and CPU saved-GRU/static replay."""
import torch
import lightgbm
import gc
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import prepare_fit_canary as fixture
import audit

ROOT = fixture.ROOT
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/sequence_context'))
import adapter
import cache
import run_models

BASE = ROOT / 'private_runs/breakthrough_20260916'
MODELS = BASE / 'sequence_context/models'
SEED = 20260919


def features(ids, month, nextmonth, marker):
    path = ROOT / f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-{month}-01_2025-{nextmonth}-01.parquet'
    assert fixture.sha256(path) == fixture.read_json(path.with_suffix('.json'))['sha256']
    x = fixture.selected_frame(path, [c for c in pq.read_schema(path).names if c != audit.ID], ids)
    desired = marker['feature_columns']
    for receipt in marker['anchor']['feature_receipts']:
        root = Path(receipt.get('path', receipt.get('manifest'))).parent
        filename = 'features.parquet' if receipt.get('block') == 'conventions' else 'training_features.parquet'
        path = root / filename
        source = fixture.read_json(root / 'manifest.json')
        expected = source['feature_sha256'] if filename == 'features.parquet' else source['outputs'][filename]
        assert fixture.sha256(path) == expected
        columns = [c for c in desired if c in pq.read_schema(path).names and c not in x]
        if not columns:
            continue
        extra = fixture.selected_frame(path, columns, ids)
        for column in columns:
            if pd.api.types.is_numeric_dtype(extra[column]):
                extra[column] = extra[column].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
            else:
                extra[column] = extra[column].astype('string').fillna('MISSING').astype('category')
        x = pd.concat([x, extra], axis=1)
    return x[desired]


def main():
    torch.set_num_threads(2)
    torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(AssertionError('CPU audit must not initialize CUDA'))
    manifest = run_models.checked_cache()
    protocol = fixture.read_json(MODELS / f'protocol_s{SEED}.json')
    assert protocol['fit_refit_sample_cap'] == 200000 and protocol['seed'] == SEED
    assert protocol['source_hashes'] == run_models.hashes()
    for name, digest in protocol['source_hashes'].items():
        assert fixture.sha256(MODELS / f'source_s{SEED}' / name) == digest
    metapath = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert fixture.sha256(metapath) == fixture.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][metapath.name]
    meta = pd.read_parquet(metapath, columns=[audit.ID, audit.TARGET, 'MVT_TIME_UTC_mvt', 'FLIGHT_ID_mvt', 'proxy_sec'])
    neighbors = np.load(cache.OUT / 'neighbors.npy', mmap_mode='r')
    results = {}
    for fold, month, nextmonth in [('F1', '07', '08'), ('F3', '11', '12')]:
        indices, split, _ = audit.common.fold_data(meta, fold, full=True)
        finite = np.isfinite(meta.proxy_sec.to_numpy())
        eligible = {stage: rows[finite[rows]] for stage, rows in indices.items()}
        sampled = {stage: rows.copy() for stage, rows in eligible.items()}
        for j, stage in enumerate(('fit', 'refit')):
            if len(sampled[stage]) > 200000:
                sampled[stage] = np.sort(np.random.default_rng(SEED + j).choice(sampled[stage], 200000, replace=False))
        expected_ids = {stage: fixture.object_hash(meta.iloc[rows][audit.ID].tolist()) for stage, rows in sampled.items()}
        cache_rec = next(rec for rec in manifest['records'] if rec['file'].startswith(f'training_2025-{month}-'))
        queries = pd.read_parquet(cache.OUT / cache_rec['queries_file'], dtype_backend='pyarrow')
        events = pd.read_parquet(cache.OUT / cache_rec['events_file'], dtype_backend='pyarrow')
        qoffset, eoffset = cache_rec['query_offset'], cache_rec['event_offset']
        n = np.array(neighbors[qoffset:qoffset + len(queries)], dtype=np.int64)
        n[n >= 0] -= eoffset
        assert np.all((n == -1) | ((n >= 0) & (n < len(events))))
        np.testing.assert_array_equal(queries[audit.ID], meta.iloc[indices['score']][audit.ID])
        store = adapter.EventStore(events, queries, n)
        control = fixture.read_json(BASE / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916' / 'manifest.json')
        x = features(pd.Index(queries[audit.ID]), month, nextmonth, control)
        local_proxy = meta.iloc[indices['score']].proxy_sec.to_numpy()
        score_rows = np.flatnonzero(np.isfinite(local_proxy))
        for variant in ('static', 'context'):
            folder = MODELS / f'{variant}_{fold}_s{SEED}'
            rec = fixture.read_json(folder / 'manifest.json')
            assert rec['status'] == 'complete'
            assert rec['source_hashes'] == protocol['source_hashes'] and fixture.object_hash(rec['split']) == fixture.object_hash(split)
            assert rec['id_hashes'] == expected_ids
            assert rec['sampled_rows'] == {stage: len(rows) for stage, rows in sampled.items()}
            assert rec['eligible_rows'] == {stage: len(rows) for stage, rows in eligible.items()}
            assert rec['original_rows'] == {stage: len(rows) for stage, rows in indices.items()}
            assert rec['fit']['fit_rows'] == rec['refit']['fit_rows'] == 200000
            assert rec['fit']['tune_rows'] == len(eligible['tune'])
            best = min(rec['fit']['history'], key=lambda r: r['tune_raw_mse'])['epoch']
            assert best == rec['fit']['steps'] == rec['refit']['steps'] == rec['refit']['epochs_run']
            assert rec['fit']['contextual'] == rec['refit']['contextual'] == (variant == 'context')
            for filename, digest in rec['outputs'].items():
                assert fixture.sha256(folder / filename) == digest
            saved = pd.read_parquet(folder / 'candidate.parquet')
            reference, _ = audit.common.reference(fold)
            np.testing.assert_array_equal(saved[audit.ID], reference[audit.ID])
            np.testing.assert_array_equal(saved[audit.TARGET], reference[audit.TARGET])
            missing = ~np.isfinite(local_proxy)
            np.testing.assert_array_equal(saved.prediction_sec.to_numpy()[missing], reference.prediction_sec.to_numpy()[missing])
            for output in ('candidate', 'blend25'):
                frame = pd.read_parquet(folder / f'{output}.parquet')
                error = frame.prediction_sec.to_numpy() - frame[audit.TARGET].to_numpy()
                report = rec['reports'][output]['metrics']['overall']
                assert report['n'] == len(frame)
                np.testing.assert_allclose(np.square(error).sum(), report['sse'], rtol=1e-12)
            model = joblib.load(folder / 'model.joblib')
            assert model['query_encoder'].columns == list(x)
            assert all(value.device.type == 'cpu' for value in model['state'].values())
            encoded = store.encoded(model['event_encoder'])
            random_rows = np.sort(np.random.default_rng(SEED).choice(score_rows, 4096, replace=False))
            began = time.monotonic()
            probe = adapter.predict(model, x, random_rows, store, device='cpu', event_arrays=encoded)
            seconds = time.monotonic() - began
            projected = seconds * len(score_rows) / len(random_rows)
            if projected < 120:
                chosen = score_rows
                prediction = adapter.predict(model, x, chosen, store, device='cpu', event_arrays=encoded) + local_proxy[chosen]
                scope = 'All original eligible score rows independently replayedCPU'
            else:
                chosen = random_rows
                prediction = probe + local_proxy[chosen]
                scope = '4096 deterministic random score rows; projected fullCPU replay exceeds120sec'
            delta = np.abs(prediction - saved.prediction_sec.to_numpy()[chosen])
            assert np.isfinite(prediction).all()
            print('CROSS_DEVICE_DELTA', variant, fold, float(delta.max()), 'large_rows', [(float(saved.iloc[chosen[i]][audit.ID]), float(prediction[i]), float(saved.iloc[chosen[i]].prediction_sec), float(delta[i])) for i in np.argsort(delta)[-5:]], flush=True)
            cpu_full = saved.prediction_sec.to_numpy().copy()
            cpu_full[chosen] = prediction
            cpu_rmse = float(np.sqrt(np.square(cpu_full - saved[audit.TARGET].to_numpy()).mean()))
            detail = {'delta_percentiles_sec': np.quantile(delta, [0, .5, .9, .99, .999, 1]).tolist(),
                      'rows_over_005sec': int((delta > .05).sum()), 'rows_over_05sec': int((delta > .5).sum()),
                      'cpu_fullscore_rmse': cpu_rmse, 'cpu_minus_saved_rmse': cpu_rmse - rec['reports']['candidate']['metrics']['overall']['rmse_sec']}
            if variant == 'context':
                worst = chosen[np.argsort(delta)[-16:]]
                probe_rows = np.unique(np.r_[worst, np.random.default_rng(SEED).choice(score_rows, 128, replace=False)])
                q = model['query_encoder'].transform(x.iloc[probe_rows])
                event = store.batch(probe_rows, encoded)
                fixture_path = audit.OUT / f'gru_{fold}_cross_device_fixture.joblib'
                joblib.dump({'architecture': model['architecture'], 'state': model['state'], 'query': q, 'event': event,
                    'target_scale': model['target_scale'], 'target_mean': model['target_mean'],
                    'proxy': local_proxy[probe_rows], 'saved_gpu_prediction': saved.prediction_sec.to_numpy()[probe_rows],
                    'cpu_prediction': cpu_full[probe_rows], 'ids': saved[audit.ID].to_numpy()[probe_rows]}, fixture_path)
                detail['cross_device_fixture_sha256'] = fixture.sha256(fixture_path)
                detail['cross_device_fixture_rows'] = len(probe_rows)
            result = {'status': 'passed' if delta.max() < .05 else 'tolerance_exceeded', **detail, 'score_rows': len(saved), 'replay_rows': len(chosen), 'replay_scope': scope,
                'cpu_gpu_max_abs_delta_sec': float(delta.max()), 'cpu_gpu_tolerance_sec': .05,
                'producer_saved_replay_rows': 2048, 'producer_saved_replay_delta': rec['replay_delta'],
                'CPU_probe_rows': len(random_rows), 'CPU_probe_seconds': seconds, 'CPU_projected_full_seconds': projected,
                'replay_seconds_including_probe': time.monotonic() - began, 'selected_epochs': best,
                'candidate_rmse': rec['reports']['candidate']['metrics']['overall']['rmse_sec'],
                'original_sampled_cohorts_verified': True, 'seed': SEED, 'fit_refit_rows': 200000,
                'missing_v2_exact_rows': int(missing.sum()), 'manifest_sha256': fixture.sha256(folder / 'manifest.json')}
            results[f'{variant}_{fold}'] = result
            fixture.write_json(audit.OUT / 'gru_audit_v4_progress.json', results)
            print('GRU_AUDIT', variant, fold, result, flush=True)
            del model, encoded, saved, frame, prediction
            gc.collect()
        del x, store, events, queries, n
        gc.collect()
    assert not torch.cuda.is_initialized()
    fixture.write_json(audit.OUT / 'gru_audit_v4.json', {'status': 'passed' if all(r['status'] == 'passed' for r in results.values()) else 'cross_device_tolerance_exceeded', 'source_sha256': fixture.sha256(__file__),
        'models': results, 'cuda_initialized': False,
        'cache_manifest_sha256': fixture.sha256(cache.OUT / 'manifest.json'),
        'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)})


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
