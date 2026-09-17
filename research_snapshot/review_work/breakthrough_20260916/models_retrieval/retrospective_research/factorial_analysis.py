"""Predeclared four-arm retrospective context analysis; never trains a model."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path

BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = external_path(BASE / 'retrospective_research/factorial_analysis')
CONTROL = BASE / 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
ARMS = {'past': ['past'], 'past_future': ['future', 'past'],
        'past_day': ['day', 'past'], 'all': ['day', 'future', 'past']}
CONTRASTS = {
    'future_without_day': {'past_future': 1, 'past': -1},
    'future_with_day': {'all': 1, 'past_day': -1},
    'day_without_future': {'past_day': 1, 'past': -1},
    'day_with_future': {'all': 1, 'past_future': -1},
    'joint_future_day': {'all': 1, 'past': -1},
    'future_by_day_interaction': {'all': 1, 'past_future': -1, 'past_day': -1, 'past': 1},
}
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def relative(path):
    return str(Path(path).relative_to(ROOT)).replace('\\', '/')


def arm_root(blocks):
    return BASE / 'retrospective_models' / '__'.join(blocks)


def folder(root, fold):
    return root / f'lightgbm_aobt_allfinite_{fold}_s20260916'


def guard():
    import psutil
    if psutil.virtual_memory().available < 8 * 1024**3 or psutil.Process().memory_info().rss > 2 * 1024**3:
        raise MemoryError('Analysis requires8GiB host reserve and at most2GiB RSS')


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        raise FileExistsError('Preserve existing analysis declaration')
    pins = [Path(__file__), ROOT / 'review_work/breakthrough_20260916/retrospective_models/run.py',
            ROOT / 'review_work/breakthrough_20260916/models/run_full.py',
            ROOT / 'review_work/campaign_20260916/lgbm_adapter.py',
            BASE / 'retrospective_research/manifest.json',
            BASE / 'retrospective_research/verification.json',
            BASE / 'models/sequence_result_audit/retrospective_oracle.json']
    control_records = {}
    for fold in WEIGHTS:
        control_path = folder(CONTROL, fold) / 'manifest.json'
        control = read(control_path)
        assert control['status'] == 'complete' and control['fit']['steps'] == 600
        control_records[fold] = {'manifest': relative(control_path), 'sha256': sha(control_path),
                                'fit_ids': control['fit_ids'], 'fit_params': control['fit']['params'],
                                'refit_params': control['refit']['params'], 'split': control['split']}
    current = {}
    for name, blocks in ARMS.items():
        current[name] = {}
        for fold in WEIGHTS:
            marker = folder(arm_root(blocks), fold) / 'manifest.json'
            current[name][fold] = {'exists': marker.exists(),
                                   'status': read(marker).get('status') if marker.exists() else 'not_launched'}
            if name in ('past_future', 'past_day') and marker.exists():
                raise ValueError('Missing-arm outcome/attempt already exists; cannot claim pre-run declaration')
            if marker.exists():
                current[name][fold]['sha256'] = sha(marker)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Mechanistic2x2 conditional on existingpast context, predeclared before newfuture/day arms; knownpast/all endpoints are development-exposed.',
        'arms': ARMS, 'folds': list(WEIGHTS), 'seasonal_weights': WEIGHTS,
        'contrasts_sec2': CONTRASTS,
        'interaction': 'MSE(all)-MSE(past_future)-MSE(past_day)+MSE(past). Negative=superadditive error reduction on the MSE scale; not causal interaction.',
        'parameters': 'Frozen LightGBM600, original seed20260916,4CPU, full original finite-NM fit/tune/refit; complete original score cohorts; missingV2 exact.',
        'checks': ['completeall8manifests', 'source/cache/protocol/snapshot hashes', 'exactbase225plusdeclaredblocks',
                   'samefullfit/tune/refit/score IDcounts/hashes andsplits', 'matched600treecap fitparameters; originaltune-stopping/freshrefit rule',
                   'scoreIDs andrawlabels exactV2 reference', 'missingV2 exact', 'independentMSE arithmetic',
                   'producerfullsavedreplay receipt<=1e-4'],
        'statistics': 'All six paired rowwise squared-error contrasts, day means/SSE and leave-one-day-out; seasonal weightedMSE contrast and seasonal leave-one-day-out. RMSE perarm only; no additiveRMSEinteraction.',
        'decision': 'Excludedfromfrozenfinal9; no newcandidate, weights, score-basedpromotion or thresholdsearch.',
        'limits': 'Adaptiveexposeddevelopment; labelsreadonly forauthorizedevaluation; day-removal isdescriptive, not independentconfirmation. Doesnotseparate ARR vs DEP within a block or establish publication-timecausality.',
        'resource_policy': 'NoGPU/no modelreload/no network;1CPU numerical/Arrow;<=2GiBRSS and>=8GiBhostavailable.',
        'pins': {relative(p): sha(p) for p in pins}, 'controls': control_records,
        'states_at_declaration': current,
    }
    write_once(path, protocol)
    print(json.dumps({'status': 'prepared', 'path': str(path), 'sha256': sha(path),
                      'created_utc': protocol['created_utc'], 'states': current}, indent=2))


def contrast_summary(errors, days):
    import numpy as np
    results = {}
    unique, inverse = np.unique(days, return_inverse=True)
    counts = np.bincount(inverse)
    for name, weights in CONTRASTS.items():
        values = sum(weight * errors[arm] for arm, weight in weights.items())
        daily = np.bincount(inverse, weights=values)
        total = float(values.sum())
        leave = (total - daily) / (len(values) - counts)
        results[name] = {'delta_mse_sec2': float(values.mean()), 'paired_rows': len(values),
                         'leave_one_day_out_range_sec2': [float(leave.min()), float(leave.max())],
                         'daily': [{'day': str(day), 'n': int(n), 'delta_sse_sec2': float(s),
                                    'delta_mse_sec2': float(s / n), 'leave_one_day_out_sec2': float(l)}
                                   for day, n, s, l in zip(unique, counts, daily, leave)]}
    return results


def analyze():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    from taxiout.schema import ID, TARGET
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    protocol_path = OUT / 'protocol.json'
    protocol = read(protocol_path)
    if (OUT / 'analysis.json').exists():
        raise FileExistsError('Preserve completed analysis')
    for path, expected in protocol['pins'].items():
        assert sha(ROOT / path) == expected, path
    cache = read(BASE / 'retrospective_research/manifest.json')
    # Require every result before loading any private predictions.
    for blocks in ARMS.values():
        for fold in WEIGHTS:
            path = folder(arm_root(blocks), fold) / 'manifest.json'
            if not path.exists() or read(path)['status'] != 'complete':
                raise RuntimeError(f'Await immutable complete arm: {path}')
    results = {}
    for fold in WEIGHTS:
        guard()
        control_path = ROOT / protocol['controls'][fold]['manifest']
        assert sha(control_path) == protocol['controls'][fold]['sha256']
        control = read(control_path)
        reference_root = ROOT / f'private_runs/next_230/models/clock_and_rome_ensemble_{fold}_s20260910'
        reference_manifest = read(reference_root / 'manifest.json')
        reference_path = reference_root / 'score_predictions.parquet'
        assert reference_manifest['status'] == 'complete'
        assert sha(reference_path) == reference_manifest['outputs'][reference_path.name]
        ref = pq.read_table(reference_path, columns=[ID, TARGET, 'prediction_sec', 'proxy_sec', 'day'], use_threads=False).to_pandas()
        assert ref[ID].is_unique and len(ref)
        missing = ~np.isfinite(ref.proxy_sec.to_numpy(float))
        errors, metrics, receipts = {}, {}, {}
        for name, blocks in ARMS.items():
            guard()
            root = arm_root(blocks)
            location = folder(root, fold)
            marker = read(location / 'manifest.json')
            payload = read(root / 'matched_protocol.json')
            assert marker['anchor']['retrospective_context'] == payload
            assert payload['blocks'] == blocks
            expected = [c for block in blocks for c in cache['feature_groups'][block]]
            assert payload['added_columns'] == expected
            assert payload['base_columns'] == control['feature_columns']
            assert marker['feature_columns'] == control['feature_columns'] + expected
            assert payload['controls'][fold] == sha(control_path)
            assert payload['cache_manifest_sha256'] == sha(BASE / 'retrospective_research/manifest.json')
            assert payload['cache_verification_sha256'] == sha(BASE / 'retrospective_research/verification.json')
            assert marker['anchor']['feature_receipts'] == control['anchor']['feature_receipts']
            assert marker['fit_ids'] == control['fit_ids'] and marker['split'] == control['split']
            assert marker['seed'] == 20260916 and marker['threads'] == 4
            assert marker['training_scope'] == 'full eligible'
            assert 1 <= marker['fit']['steps'] <= 600
            assert marker['refit']['steps'] == marker['fit']['steps']
            for stage in ('fit', 'refit'):
                assert marker[stage]['rows'] == control[stage]['rows']
                expected_params = dict(control[stage]['params'])
                if stage == 'refit':
                    expected_params['n_estimators'] = marker['fit']['steps']
                assert marker[stage]['params'] == expected_params
            producer_protocol = root / 'protocols/lightgbm_aobt_allfinite_s20260916.json'
            assert sha(producer_protocol) == marker['protocol_sha256']
            assert marker['source_hashes'] == payload['source_hashes']
            snapshot = root / 'source_snapshots/lightgbm_aobt_allfinite_s20260916'
            for source, expected_hash in marker['source_hashes'].items():
                assert sha(snapshot / Path(source).name) == expected_hash
            assert 0 <= marker['reload_max_abs_delta'] <= 1e-4
            prediction = location / 'candidate.parquet'
            assert sha(prediction) == marker['outputs'][prediction.name]
            frame = pq.read_table(prediction, columns=[ID, TARGET, 'prediction_sec', 'day'], use_threads=False).to_pandas()
            assert len(frame) == marker['complete_score_rows'] == len(ref) and frame[ID].is_unique
            for column in (ID, TARGET, 'day'):
                np.testing.assert_array_equal(frame[column], ref[column])
            values = frame.prediction_sec.to_numpy(float)
            labels = ref[TARGET].to_numpy(float)
            assert np.isfinite(values).all() and np.isfinite(labels).all()
            np.testing.assert_array_equal(values[missing], ref.prediction_sec.to_numpy(float)[missing])
            squared = np.square(values - labels)
            errors[name] = squared
            reported = marker['reports']['candidate']['metrics']['overall']
            assert reported['n'] == len(frame)
            np.testing.assert_allclose(squared.sum(), reported['sse'], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(squared.mean()), reported['rmse_sec'], rtol=1e-12)
            metrics[name] = {'n': len(frame), 'mse_sec2': float(squared.mean()),
                             'rmse_sec': float(np.sqrt(squared.mean()))}
            receipts[name] = {'manifest': relative(location / 'manifest.json'),
                              'manifest_sha256': sha(location / 'manifest.json'),
                              'candidate_sha256': sha(prediction), 'feature_count': len(marker['feature_columns']),
                              'producer_full_saved_replay_delta': marker['reload_max_abs_delta']}
        results[fold] = {'arms': metrics, 'contrasts': contrast_summary(errors, ref.day.to_numpy(str)),
                         'receipts': receipts, 'rows': len(ref), 'missing_V2_exact_rows': int(missing.sum()),
                         'reference_manifest_sha256': sha(reference_root / 'manifest.json'),
                         'reference_predictions_sha256': sha(reference_path), 'all_contract_checks_passed': True}
        del errors, ref, frame
    seasonal = {name: sum(WEIGHTS[fold] * results[fold]['arms'][name]['mse_sec2'] for fold in WEIGHTS) for name in ARMS}
    contrasts = {}
    for name, weights in CONTRASTS.items():
        value = sum(weight * seasonal[arm] for arm, weight in weights.items())
        variations = []
        for fold in WEIGHTS:
            current = results[fold]['contrasts'][name]
            for day in current['daily']:
                variations.append({'removed_fold': fold, 'removed_day': day['day'],
                                   'delta_mse_sec2': value + WEIGHTS[fold] * (day['leave_one_day_out_sec2'] - current['delta_mse_sec2'])})
        contrasts[name] = {'delta_mse_sec2': value, 'seasonal_leave_one_day_out': variations,
                           'seasonal_leave_one_day_out_range_sec2': [min(v['delta_mse_sec2'] for v in variations), max(v['delta_mse_sec2'] for v in variations)]}
    guard()
    output = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(protocol_path), 'source_sha256': sha(__file__), 'folds': results,
              'seasonal_mse_sec2': seasonal, 'seasonal_rmse_sec': {k: float(np.sqrt(v)) for k, v in seasonal.items()},
              'seasonal_contrasts': contrasts,
              'scope': 'Complete original score rows, labels, cohort receipts, settings and savedprediction metrics independently verified; no model reload. Producer full savedreplay receipt checked.',
              'limits': protocol['limits'], 'decision': protocol['decision'],
              'rss_bytes_at_finish': psutil.Process().memory_info().rss, 'cuda_used': False, 'network_used': False}
    write_once(OUT / 'analysis.json', output)
    print(json.dumps({'status': 'passed', 'seasonal_rmse_sec': output['seasonal_rmse_sec'],
                      'contrasts_sec2': {k: v['delta_mse_sec2'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_only else analyze()
