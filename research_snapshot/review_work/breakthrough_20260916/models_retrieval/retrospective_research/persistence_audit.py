"""Tune-only diagnostic of residual temporal persistence, never a feature cache."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.schema import ID, FLIGHT_ID, PHASE, MOVEMENT, TARGET, CLOCKS
from taxiout.paths import external_path
BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = external_path(BASE / 'retrospective_research/persistence_audit')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def guard():
    if psutil.Process().memory_info().rss > 2 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Diagnostic resource reserve exceeded')


def load_predictions(relative):
    directory = BASE / relative
    marker = read(directory / 'manifest.json')
    assert marker['status'] == 'complete'
    path = directory / 'tune_predictions.parquet'
    assert sha(path) == marker['outputs'][path.name]
    data = pq.read_table(path, use_threads=False).to_pandas()
    assert data[ID].is_unique
    return data, {'manifest': str((directory / 'manifest.json').relative_to(ROOT)), 'manifest_sha256': sha(directory / 'manifest.json'), 'prediction_sha256': sha(path), 'fit_ids': marker['fit_ids']}


def paired(values, left, right):
    x, y = values[left], values[right]
    if len(x) < 3:
        return {'n': len(x)}
    return {'n': len(x), 'pearson': float(np.corrcoef(x, y)[0, 1]),
        'spearman': float(spearmanr(x, y).statistic), 'mean_product_sec2': float(np.mean(x * y)),
        'same_sign_share': float(np.mean((x > 0) == (y > 0))),
        'query_rmse_sec': float(np.sqrt(np.mean(y * y))),
        'peer_half_correction_rmse_sec': float(np.sqrt(np.mean((y - .5 * x)**2))),
        'peer_half_correction_unavailable_diagnostic_only': True}


def pairs(frame, lag):
    left, right = [], []
    for _, positions in frame.groupby(['airport', 'day'], sort=False, observed=True).indices.items():
        positions = positions[np.argsort(frame.time.to_numpy()[positions], kind='stable')]
        keys = frame.flightkey.to_numpy()[positions]
        times = frame.time.to_numpy()[positions]
        for j in range(lag, len(positions)):
            i = j - lag
            if times[i] < times[j] and keys[i] != keys[j] and times[j] - times[i] <= 3600:
                left.append(positions[i])
                right.append(positions[j])
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def permuted_null(values, left, right, groups, seed, repetitions=25):
    rng = np.random.default_rng(seed)
    products = []
    correlations = []
    for _ in range(repetitions):
        order = np.arange(len(values))
        for positions in groups:
            order[positions] = rng.permutation(positions)
        permuted = values[order]
        a, b = permuted[left], permuted[right]
        products.append(float(np.mean(a * b)))
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return {'repetitions': repetitions, 'pearson_mean': float(np.mean(correlations)),
        'pearson_min': min(correlations), 'pearson_max': max(correlations),
        'mean_product_sec2': float(np.mean(products)), 'max_product_sec2': max(products),
        'interpretation': 'Exploratory finite permutation range, not calibrated significance'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise ValueError('Prior persistence attempt preserved')
    write(OUT / 'protocol.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'label_scope': 'June and October original tune only; no July/November/December or ranking target read',
        'models': 'Previously fitted115ownclockcontrol,118priorfrequencycontext and225combinedLGB600 tune predictions',
        'meaning': 'Peer residual requires hidden labels and is DIAGNOSTIC ONLY, cannot be an inference feature or achievable score',
        'pairs': 'SameairportUTCday nearest row lags1/3/8, within60minutes; sameflight and tied events excluded',
        'null': '25fixedseed permutations of residuals withinairportday and withinairportdayoperatorproxyminute',
        'targets': 'Unmodified raw labels; all ordinary finiteNM eligible tune rows from original flightpurged models',
        'resources': 'oneCPUthread,2GiBRSSceiling,8GiBhostreserve; no network'})
    frozen = read(ROOT / 'private_runs/submission_v2/protocol.json')
    records = {}
    started = time.monotonic()
    for fold, month in [('F1', '06'), ('F3', '10')]:
        guard()
        path = next(RAW.glob(f'training_2025-{month}-01_*.parquet'))
        assert sha(path) == frozen['raw_hashes'][path.name]
        columns = [ID, FLIGHT_ID, PHASE, MOVEMENT, TARGET, 'ADEP_mvt', 'AIRCRAFT_OPERATOR_flt', *CLOCKS]
        raw = pq.read_table(path, columns=columns, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
        own, own_receipt = load_predictions(f'missing/provenance_audit/matched_models/control/lightgbm_ordinary_source_residual_{fold}_s20260916')
        context, context_receipt = load_predictions(f'missing/provenance_audit/matched_models/context/lightgbm_ordinary_source_residual_{fold}_s20260916')
        combined, combined_receipt = load_predictions(f'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T/lightgbm_aobt_allfinite_{fold}_s20260916')
        assert np.array_equal(own[ID], context[ID])
        frame = raw.set_index(ID).loc[own[ID]].reset_index()
        assert len(frame) == own_receipt['fit_ids']['tune']['n']
        expected_hash = hashlib.sha256(json.dumps(frame[ID].tolist(), sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        # ID equality with the original hashed prediction artifact binds this cohort.
        frame['airport'] = frame.ADEP_mvt.astype('string').fillna('<missing>')
        timestamp = pd.to_datetime(frame[MOVEMENT], utc=True)
        frame['day'] = timestamp.dt.strftime('%Y-%m-%d')
        frame['time'] = timestamp.dt.as_unit('ns').astype('int64') / 1e9
        frame['operator'] = frame.AIRCRAFT_OPERATOR_flt.astype('string').fillna('<missing>')
        frame['flightkey'] = [('flight', float(f)) if pd.notna(f) else ('movement', float(i)) for f, i in zip(frame[FLIGHT_ID], frame[ID])]
        proxies = np.column_stack([(timestamp - pd.to_datetime(frame[c], utc=True)).dt.total_seconds() for c in CLOCKS])
        frame['proxyminute'] = np.floor((proxies[:, 0] + 30) / 60).astype(int)
        frame['order'] = pd.DataFrame(proxies).rank(axis=1, method='dense').fillna(0).astype(int).astype(str).agg('|'.join, axis=1)
        frame['raw_source_residual'] = frame[TARGET].to_numpy(float) - proxies[:, 0]
        for name, prediction in [('own', own), ('context', context), ('combined', combined)]:
            frame[name] = frame[TARGET].to_numpy(float) - prediction.set_index(ID).loc[frame[ID], 'prediction_sec'].to_numpy(float)
        guard()
        groups = {name: list(frame.groupby(keys, observed=True, sort=False).indices.values()) for name, keys in [
            ('airport_day', ['airport', 'day']), ('airport_day_operator_proxyminute', ['airport', 'day', 'operator', 'proxyminute'])]}
        output = {'rows': len(frame), 'receipts': {'own': own_receipt, 'context': context_receipt, 'combined': combined_receipt}, 'lags': {}}
        for lag in [1, 3, 8]:
            left, right = pairs(frame, lag)
            conditions = {'all': np.ones(len(left), bool),
                'same_proxyminute': frame.proxyminute.to_numpy()[left] == frame.proxyminute.to_numpy()[right],
                'same_operator': frame.operator.to_numpy()[left] == frame.operator.to_numpy()[right],
                'same_clock_order': frame.order.to_numpy()[left] == frame.order.to_numpy()[right],
                'within15min': frame.time.to_numpy()[right] - frame.time.to_numpy()[left] <= 900}
            bymodel = {}
            for model in ['raw_source_residual', 'own', 'context', 'combined']:
                values = frame[model].to_numpy(float)
                result = {name: paired(values, left[mask], right[mask]) for name, mask in conditions.items()}
                if lag == 1:
                    result['permutation'] = {name: permuted_null(values, left, right, parts, 20260916) for name, parts in groups.items()}
                    result['by_airport'] = {airport: paired(values, left[frame.airport.to_numpy()[right] == airport], right[frame.airport.to_numpy()[right] == airport]) for airport in sorted(frame.airport.unique())}
                bymodel[model] = result
            output['lags'][str(lag)] = bymodel
            print(fold, 'lag', lag, {k: v['all']['pearson'] for k, v in bymodel.items()}, flush=True)
        records[fold] = output
        write(OUT / 'progress.json', records)
        guard()
    write(OUT / 'analysis.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol.json'), 'records': records,
        'runtime_sec': time.monotonic() - started, 'rss_bytes': psutil.Process().memory_info().rss,
        'label_aware_diagnostic_only': True, 'feature_cache_created': False, 'model_trained': False})
    print('COMPLETE', OUT / 'analysis.json', flush=True)


if __name__ == '__main__':
    main()
