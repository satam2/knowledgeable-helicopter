"""Independent score/cohort audit and bounded native saved-model replay."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import sys
import gc
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/missing'))
import run_missing_models as base
import run_id_context as context

OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit/missing_information'
CONTROL = ROOT / 'private_runs/breakthrough_20260916/missing/id_context_v1/models'
ARMS = {
    'opdi': (ROOT / 'private_runs/breakthrough_20260916/models/opdi_missing_v3',
             ROOT / 'private_runs/breakthrough_20260916/missing/opdi_rotation'),
    'monthly_arrival': (ROOT / 'private_runs/breakthrough_20260916/missing/monthly_arrival_model',
                        ROOT / 'private_runs/breakthrough_20260916/retrospective_research/monthly_arrival')}


def stats(pred, target):
    error = pred - target
    return {'rmse_sec': float(np.sqrt(np.mean(error**2))),
        'mae_sec': float(np.mean(np.abs(error))), 'bias_sec': float(np.mean(error))}


def native_replay(model, features, times):
    assert model['kind'] == 'historical_template'
    prior = model['prior'].transform(features, times)
    matrix = pd.concat([features, prior], axis=1)
    return np.asarray(model['model'].predict(matrix, thread_count=1), float) + prior.template_mean_sec.to_numpy(float)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'audit.json').exists()
    assert psutil.virtual_memory().available >= 8 * 1024**3
    x, meta = base.load_data()
    old_declaration = base.read_json(ROOT / 'private_runs/breakthrough_20260916/missing/id_context_v1/protocol.json')['declaration']
    assert base.sha256(context.CACHE / 'audit.json') == old_declaration['id_context_audit_sha256']
    assert base.sha256(context.CACHE / 'features.parquet') == old_declaration['id_context_feature_sha256']
    peer = pd.read_parquet(context.CACHE / 'features.parquet').set_index(base.ID)
    np.testing.assert_array_equal(peer.index, meta[base.ID])
    times = meta.set_index(base.ID).loc[x.index, base.MOVEMENT]
    x = pd.concat([x, context.id_context_features(x, peer.loc[x.index], times)], axis=1)
    del peer
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    results = {}
    for label, (folder, cache) in ARMS.items():
        declaration = base.read_json(folder / 'protocol.json')['declaration']
        for source, digest in declaration['source_hashes'].items():
            assert base.sha256(ROOT / source) == digest, source
        cm = base.read_json(cache / 'manifest.json')
        cv = base.read_json(cache / 'verification.json')
        co = base.read_json(cache / 'independent_oracle.json')
        mh = base.sha256(cache / 'manifest.json')
        assert cm['status'] == 'complete' and cv['status'] == co['status'] == 'passed'
        assert cv['manifest_sha256'] == co['manifest_sha256'] == mh
        cachepath = cache / 'training_features.parquet'
        assert base.sha256(cachepath) == cm['outputs'][cachepath.name]
        extension = pq.read_table(cachepath, filters=[(base.ID, 'in', x.index.tolist())], use_threads=False).to_pandas().set_index(base.ID)
        np.testing.assert_array_equal(extension.index, x.index)
        assert list(extension) == declaration['added_features']
        features = pd.concat([x, extension.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')], axis=1)
        del extension
        folds = {}
        for fold in ('F1', 'F3'):
            dest = folder / f'models/historical_template_{fold}_s20260916'
            control = CONTROL / dest.name
            record, previous = base.read_json(dest / 'manifest.json'), base.read_json(control / 'manifest.json')
            assert record['status'] == previous['status'] == 'complete'
            assert record['protocol_sha256'] == base.sha256(folder / 'protocol.json')
            assert record['seed'] == previous['seed'] == 20260916 and record['threads'] == previous['threads'] == 2
            assert record['features_used'] == list(features) and previous['features_used'] == list(x)
            for filename, digest in record['outputs'].items():
                assert base.sha256(dest / filename) == digest, filename
            idx, split, _ = base.common.fold_data(meta, fold, full=True)
            identities = {stage: {'n': int(missing[rows].sum()),
                'hash': base.object_hash(meta.iloc[rows[missing[rows]]][base.ID].tolist())} for stage, rows in idx.items()}
            assert identities == record['fit_ids'] == previous['fit_ids']
            assert base.object_hash(split) == base.object_hash(record['split']) == base.object_hash(previous['split'])
            assert record['refit']['steps'] == record['tune']['steps']
            ref, _ = base.common.reference(fold)
            target = meta.iloc[idx['score']][base.TARGET].to_numpy(float)
            changed = missing[idx['score']]
            predictions = {}
            comparisons = {}
            for variant in ('candidate', 'blend25'):
                pred = pd.read_parquet(dest / f'{variant}.parquet')
                old = pd.read_parquet(control / f'{variant}.parquet')
                np.testing.assert_array_equal(pred[base.ID], ref[base.ID])
                np.testing.assert_array_equal(old[base.ID], ref[base.ID])
                np.testing.assert_array_equal(pred[base.TARGET], target)
                values = pred.prediction_sec.to_numpy(float)
                previous_values = old.prediction_sec.to_numpy(float)
                assert np.isfinite(values).all()
                np.testing.assert_array_equal(values[~changed], ref.prediction_sec.to_numpy()[~changed])
                measured = stats(values, target)
                reported = record['reports'][variant]['metrics']['overall']
                for name, value in measured.items():
                    assert abs(value - reported[name]) < 1e-8, name
                predictions[variant] = values
                delta = measured['rmse_sec'] - stats(previous_values, target)['rmse_sec']
                days = pd.to_datetime(meta.iloc[idx['score']][base.MOVEMENT], utc=True).dt.strftime('%Y-%m-%d').to_numpy()
                removals = []
                for day in np.unique(days):
                    keep = days != day
                    removals.append(stats(values[keep], target[keep])['rmse_sec'] - stats(previous_values[keep], target[keep])['rmse_sec'])
                comparisons[variant] = {'metrics': measured, 'delta_previous_same_variant': delta,
                    'missing_rmse': stats(values[changed], target[changed])['rmse_sec'],
                    'day_removal_delta_range': [min(removals), max(removals)]}
            expected_blend = ref.prediction_sec.to_numpy() + .25 * (predictions['candidate'] - ref.prediction_sec.to_numpy())
            np.testing.assert_array_equal(predictions['blend25'], expected_blend)
            score_positions = np.flatnonzero(changed)
            rng = np.random.default_rng(20260916)
            random_positions = rng.choice(score_positions, min(112, len(score_positions)), replace=False)
            worst = score_positions[np.argsort(np.abs(predictions['candidate'][score_positions] - target[score_positions]))[-16:]]
            positions = np.unique(np.r_[random_positions, worst])
            ids = ref.iloc[positions][base.ID]
            model = joblib.load(dest / 'model.joblib')
            native = native_replay(model, features.loc[ids], times.loc[ids])
            delta = float(np.max(np.abs(native - predictions['candidate'][positions])))
            assert delta <= 1e-9, delta
            options = model['model'].get_params()
            for key, value in declaration['parameters'].items():
                if key != 'iterations':
                    assert options[key] == value, key
            assert int(model['model'].tree_count_) == record['refit']['steps']
            assert model['prior'].history_n == identities['refit']['n']
            folds[fold] = {'status': 'passed', 'manifest_sha256': base.sha256(dest / 'manifest.json'),
                'cohort_ids': identities, 'full_score_rows': len(ref), 'protected_rows': int((~changed).sum()),
                'replay_rows': len(positions), 'replay_max_abs_delta': delta, 'selected_trees': record['tune']['steps'],
                'comparisons': comparisons}
            print('AUDITED', label, fold, 'replay', delta, 'results', comparisons, flush=True)
            del model, pred, old
            gc.collect()
        seasonal = {variant: float(np.sqrt(sum(weight * folds[fold]['comparisons'][variant]['metrics']['rmse_sec']**2
            for fold, weight in [('F1', 192122/344841), ('F3', 152719/344841)]))) for variant in ('candidate', 'blend25')}
        results[label] = {'folds': folds, 'seasonal_rmse': seasonal, 'cache_manifest_sha256': mh}
        del features
        gc.collect()
    memory = psutil.Process().memory_info()
    output = {'status': 'passed', 'source_sha256': base.sha256(__file__), 'results': results,
        'peak_rss_bytes': getattr(memory, 'peak_wset', memory.rss), 'runtime_sec': time.monotonic() - started,
        'replay_scope': 'Independent saved native CatBoost calls plus stored prior transform; shared original row-feature implementation, independently audited new caches; bounded score rows only.',
        'gpu_used': False, 'promotion': 'No new composition; retain split-inconsistent experimental results.'}
    base.write_json(OUT / 'audit.json', output)
    print('COMPLETE', {key: value['seasonal_rmse'] for key, value in results.items()}, flush=True)


if __name__ == '__main__':
    main()
