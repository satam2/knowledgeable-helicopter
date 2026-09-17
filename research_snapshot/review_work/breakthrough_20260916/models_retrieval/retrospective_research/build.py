"""Observation-only retrospective peers with explicit prior/future/day groups."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, CLOCKS

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/retrospective_research')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
COLS = [ID, FLIGHT_ID, PHASE, MOVEMENT, 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', *CLOCKS]
NAMES = ['nm', 'est', 'init', 'last', 'sched']
DEP_STATS = [*[f'{n}_mean_sec' for n in NAMES], 'nm_std_sec', 'nm_missing_share', 'count', 'nm_minus_est_mean_sec', 'nm_minus_est_std_sec']
ARR_STATS = ['count', 'mean_sec', 'std_sec', 'over_1200_share']
GROUPS = {mode: [*[f'retro_{mode}_arr_{scope}_{s}' for scope in (['airport'] if mode == 'day' else ['airport', 'runway']) for s in ARR_STATS], *[f'retro_{mode}_dep_airport_{s}' for s in DEP_STATS]] for mode in ['past', 'future', 'day']}
POLICY = {
    'availability': 'Retrospective supplied final batch; no real-time publication claim',
    'windows': {'past': '[T-60min,T)', 'future': '(T,T+60min]', 'day': 'UTC [daystart,dayend), including future and tied peers'},
    'month_isolation': 'Query departure movement UTC month equals peer departure takeoff or peer ARR landing UTC month',
    'arrival_time': 'Completion BLOCK read only after Arrow PHASE=ARR predicate; duration=completion-landing; finite nonnegative arrivals only; no upper truncation',
    'departure_values': 'Uncensored finite takeoff-minus-five-supplied-clock moments; NM-minus-EOBT moments; no departure block or target loaded',
    'deduplication': 'Same nonnull flight, airport, phase and movement timestamp keeps smallest movement ID; missing flight IDs remain distinct',
    'exclusions': 'Every same nonnull flight excluded; own movement excluded even with missing flight; month and airport isolated',
    'missing': 'Counts zero; unsupported means/std/shares NaN; every value float32',
    'label_derived_features': False, 'gpu': False, 'cpu_threads': 1,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def ns(series):
    return pd.to_datetime(series, utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def prepare(observed, arrival_blocks=None, dedup=True):
    # Selecting this whitelist makes poisoned hidden DEP columns inert in tests.
    x = observed[COLS].copy()
    x['airport'] = x.ADEP_mvt.where(x[PHASE].eq('DEP'), x.ADES_mvt).astype('string').fillna('<missing>')
    x['runway'] = x.RUNWAY_mvt.astype('string').fillna('<missing>')
    x['time'] = ns(x[MOVEMENT])
    x['month'] = pd.to_datetime(x[MOVEMENT], utc=True).dt.strftime('%Y-%m')
    x['flightkey'] = [('flight', float(f)) if pd.notna(f) else ('movement', float(i)) for f, i in zip(x[FLIGHT_ID], x[ID])]
    if dedup:
        x = x.sort_values(ID, kind='stable').drop_duplicates(['flightkey', 'airport', PHASE, 'time'])
    x = x.reset_index(drop=True)
    for name, clock in zip(NAMES, CLOCKS):
        x[name] = (pd.to_datetime(x[MOVEMENT], utc=True) - pd.to_datetime(x[clock], utc=True)).dt.total_seconds()
    x['gap'] = x['est'] - x['nm']
    if arrival_blocks is not None:
        if not arrival_blocks[PHASE].eq('ARR').all():
            raise ValueError('Arrival block boundary rejected non-ARR row')
        blocks = arrival_blocks.set_index(ID)[BLOCK]
        arr = x[PHASE].eq('ARR')
        completion = pd.to_datetime(x.loc[arr, ID].map(blocks), utc=True)
        x['completion'] = np.int64(np.iinfo(np.int64).min)
        x.loc[arr, 'completion'] = ns(completion)
        x['duration'] = np.nan
        x.loc[arr, 'duration'] = (completion - pd.to_datetime(x.loc[arr, MOVEMENT], utc=True)).dt.total_seconds().to_numpy()
    return x


def matrix_for(events, kind):
    if kind == 'arr':
        v = events.duration.to_numpy(float)
        return np.column_stack([np.ones(len(events)), v, v * v, v > 1200])
    result = [np.ones(len(events))]
    for name in NAMES:
        v = events[name].to_numpy(float)
        result.extend([np.isfinite(v).astype(float), np.where(np.isfinite(v), v, 0)])
    nm = events.nm.to_numpy(float)
    gap = events.gap.to_numpy(float)
    result.extend([np.where(np.isfinite(nm), nm * nm, 0), np.isfinite(gap), np.where(np.isfinite(gap), gap, 0), np.where(np.isfinite(gap), gap * gap, 0)])
    return np.column_stack(result)


def bounds(times, query_times, mode):
    hour = np.int64(3600 * 10**9)
    if mode == 'past':
        return np.searchsorted(times, query_times - hour, side='left'), np.searchsorted(times, query_times, side='left')
    if mode == 'future':
        return np.searchsorted(times, query_times, side='right'), np.searchsorted(times, query_times + hour, side='right')
    day = np.int64(86400 * 10**9)
    start = (query_times // day) * day
    return np.searchsorted(times, start, side='left'), np.searchsorted(times, start + day, side='left')


def safe_divide(a, b):
    return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)


def reduce_sums(sums, kind):
    if kind == 'arr':
        count = np.maximum(0, sums[:, 0])
        mean = safe_divide(sums[:, 1], count)
        return dict(zip(ARR_STATS, [count, mean, np.sqrt(np.maximum(0, safe_divide(sums[:, 2], count) - mean * mean)), safe_divide(sums[:, 3], count)]))
    result = {}
    for j, name in enumerate(NAMES):
        result[f'{name}_mean_sec'] = safe_divide(sums[:, 2 + 2 * j], sums[:, 1 + 2 * j])
    result['nm_std_sec'] = np.sqrt(np.maximum(0, safe_divide(sums[:, 11], sums[:, 1]) - result['nm_mean_sec'] ** 2))
    result['nm_missing_share'] = safe_divide(sums[:, 0] - sums[:, 1], sums[:, 0])
    result['count'] = np.maximum(0, sums[:, 0])
    result['nm_minus_est_mean_sec'] = safe_divide(sums[:, 13], sums[:, 12])
    result['nm_minus_est_std_sec'] = np.sqrt(np.maximum(0, safe_divide(sums[:, 14], sums[:, 12]) - result['nm_minus_est_mean_sec'] ** 2))
    return result


def summarize(events, queries, kind, mode, brute=False):
    timefield = 'completion' if kind == 'arr' else 'time'
    e = events.sort_values(timefield, kind='stable').reset_index(drop=True)
    times = e[timefield].to_numpy(np.int64)
    values = matrix_for(e, kind)
    lo, hi = bounds(times, queries.time.to_numpy(np.int64), mode)
    if brute:
        sums = np.zeros((len(queries), values.shape[1]))
        for i, row in enumerate(queries.itertuples()):
            allowed = np.array([row.flightkey != key and getattr(row, ID) != eid for key, eid in zip(e.flightkey, e[ID])])
            allowed &= (np.arange(len(e)) >= lo[i]) & (np.arange(len(e)) < hi[i])
            sums[i] = values[allowed].sum(axis=0)
    else:
        prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
        sums = prefix[hi] - prefix[lo]
        excluded = defaultdict(list)
        for i, key in enumerate(e.flightkey):
            excluded[key].append(i)
        ids = dict(zip(e[ID], range(len(e))))
        for i, (key, ident) in enumerate(zip(queries.flightkey, queries[ID])):
            positions = excluded.get(key, ())
            for pos in positions:
                if lo[i] <= pos < hi[i]:
                    sums[i] -= values[pos]
            own = ids.get(ident)
            if own is not None and own not in positions and lo[i] <= own < hi[i]:
                sums[i] -= values[own]
    return reduce_sums(sums, kind)


def build(observed_queries, observed_context, arrival_blocks, brute=False):
    q = prepare(observed_queries, dedup=False)
    c = prepare(observed_context, arrival_blocks)
    out = pd.DataFrame(np.nan, index=pd.Index(q[ID], name=ID), columns=sum(GROUPS.values(), []), dtype=np.float32)
    for (airport, month), positions in q.groupby(['airport', 'month'], observed=True).indices.items():
        query = q.iloc[positions].reset_index(drop=True)
        local = c.loc[c.airport.eq(airport) & c.month.eq(month)]
        dep = local.loc[local[PHASE].eq('DEP')]
        arr = local.loc[local[PHASE].eq('ARR') & local.duration.ge(0) & np.isfinite(local.duration)]
        for mode in GROUPS:
            for kind, scope, part, events in [('dep', 'airport', query, dep), ('arr', 'airport', query, arr)]:
                for stat, values in summarize(events, part, kind, mode, brute).items():
                    out.loc[part[ID], f'retro_{mode}_{kind}_{scope}_{stat}'] = values.astype(np.float32)
            if mode != 'day':
                for runway, part in query.groupby('runway', observed=True):
                    for stat, values in summarize(arr.loc[arr.runway.eq(runway)], part, 'arr', mode, brute).items():
                        out.loc[part[ID], f'retro_{mode}_arr_runway_{stat}'] = values.astype(np.float32)
    return out


def tests():
    t = pd.Timestamp('2025-07-01T12:00Z')
    rows = []
    for i, flight, phase, minute, duration in [(1, 1, 'DEP', 0, 0), (2, 2, 'DEP', -5, 0), (3, 3, 'DEP', 5, 0), (4, 1, 'DEP', 10, 0), (5, 5, 'ARR', -20, 600), (6, 6, 'ARR', 5, 600), (7, 7, 'ARR', -10, 600), (8, 2, 'DEP', -5, 0), (9, 1, 'ARR', 5, 900), (10, np.nan, 'DEP', 0, 0), (11, 11, 'ARR', -10, -1), (12, 12, 'ARR', 60, 1)]:
        movement = t + pd.Timedelta(minutes=minute)
        row = {ID: float(i), FLIGHT_ID: float(flight), PHASE: phase, MOVEMENT: movement, 'ADEP_mvt': 'AAA' if phase == 'DEP' else 'BBB', 'ADES_mvt': 'BBB' if phase == 'DEP' else 'AAA', 'RUNWAY_mvt': 'R1', BLOCK: movement + pd.Timedelta(seconds=duration)}
        row.update({clock: movement - pd.Timedelta(seconds=600 + i * (j + 1)) for j, clock in enumerate(CLOCKS)})
        rows.append(row)
    raw = pd.DataFrame(rows)
    raw.loc[raw[ID].eq(3), CLOCKS[0]] = pd.NaT
    dep = raw.loc[raw[PHASE].eq('DEP')]
    arr = raw.loc[raw[PHASE].eq('ARR'), [ID, PHASE, BLOCK]]
    got = build(dep, raw, arr)
    pd.testing.assert_frame_equal(got, build(dep, raw, arr, brute=True), rtol=1e-5, atol=0.01)
    assert got.loc[1, 'retro_past_arr_airport_count'] == 1
    assert got.loc[1, 'retro_future_arr_airport_count'] == 1
    assert got.loc[1, 'retro_day_arr_airport_count'] == 4
    assert got.loc[1, 'retro_past_dep_airport_count'] == 1
    assert got.loc[1, 'retro_future_dep_airport_count'] == 1
    assert got.loc[1, 'retro_future_dep_airport_nm_missing_share'] == 1
    poisoned = raw.copy()
    poisoned.loc[poisoned[PHASE].eq('DEP'), BLOCK] = t + pd.Timedelta(days=400)
    poisoned['TAXITIME_SEC_mvt'] = -1e10
    pd.testing.assert_frame_equal(got, build(poisoned.loc[poisoned[PHASE].eq('DEP')], poisoned, arr))
    changed = arr.copy()
    changed.loc[changed[ID].eq(6), BLOCK] += pd.Timedelta(seconds=30)
    changed_result = build(dep, raw, changed)
    assert changed_result.loc[1, 'retro_future_arr_airport_mean_sec'] == 630
    assert got.loc[1, 'retro_past_arr_airport_mean_sec'] == changed_result.loc[1, 'retro_past_arr_airport_mean_sec']
    other = raw.copy()
    other[MOVEMENT] -= pd.Timedelta(days=31)
    other[ID] += 100
    other[FLIGHT_ID] += 100
    otherarr = other.loc[other[PHASE].eq('ARR'), [ID, PHASE, BLOCK]]
    pd.testing.assert_frame_equal(got, build(dep, pd.concat([raw, other]), pd.concat([arr, otherarr])))
    try:
        build(dep, raw, raw[[ID, PHASE, BLOCK]])
        raise AssertionError('DEP blocks accepted')
    except ValueError as exc:
        assert 'boundary' in str(exc)
    assert len(got.columns) == 50 and got.columns.is_unique
    print('TESTS_PASS brute-prefix equality; strict boundaries; duplicate/sameflight/self exclusion; month isolation; hidden DEP poison; ARR boundary; future perturbation', flush=True)
    return {'status': 'passed', 'test_count': 9}


def guard():
    available = psutil.virtual_memory().available
    if available < 8 * 1024**3:
        raise MemoryError(f'Host reserve below 8GiB: {available}')
    if psutil.Process().memory_info().rss > 2 * 1024**3:
        raise MemoryError('Builder exceeds 2GiB RSS')


def load(path):
    observed = pq.read_table(path, columns=COLS, use_threads=False).to_pandas()
    arrival = pq.read_table(path, columns=[ID, PHASE, BLOCK], filters=[(PHASE, '=', 'ARR')], use_threads=False).to_pandas()
    assert arrival[PHASE].eq('ARR').all()
    return observed, arrival


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    verification = tests()
    if args.test_only:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.glob('*.parquet')) or (OUT / 'manifest.json').exists():
        raise ValueError('Preserve existing cache; use new version for retries')
    sourcehash = sha(__file__)
    frozen = json.loads((ROOT / 'private_runs/submission_v2/protocol.json').read_text())
    write_json(OUT / 'protocol.json', {'policy': POLICY, 'feature_groups': GROUPS, 'source_sha256': sourcehash})
    paths = sorted(RAW.glob('training_*.parquet'))
    for path in [*paths, RAW / 'ranking.parquet']:
        assert sha(path) == frozen['raw_hashes'][path.name]
    records = []
    writer = None
    started = time.monotonic()
    rows_total = 0
    try:
        for i, path in enumerate(paths):
            guard()
            tic = time.monotonic()
            surrounding = paths[max(0, i - 1):min(len(paths), i + 2)]
            packs = [load(p) for p in surrounding]
            observed = packs[surrounding.index(path)][0]
            dep = observed.loc[observed[PHASE].eq('DEP')]
            features = build(dep, pd.concat([a for a, _ in packs], ignore_index=True), pd.concat([a for _, a in packs], ignore_index=True))
            assert np.array_equal(features.index, dep[ID])
            guard()
            table = pa.Table.from_pandas(features.reset_index(), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT / 'training_features.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            rows_total += len(features)
            rec = {'file': path.name, 'rows': len(features), 'seconds': time.monotonic() - tic, 'rss_bytes': psutil.Process().memory_info().rss, 'available_bytes': psutil.virtual_memory().available, 'future_arr_coverage': float(features.retro_future_arr_airport_mean_sec.notna().mean()), 'past_arr_coverage': float(features.retro_past_arr_airport_mean_sec.notna().mean()), 'day_dep_coverage': float(features.retro_day_dep_airport_nm_mean_sec.notna().mean())}
            records.append(rec)
            write_json(OUT / 'progress.json', records)
            print(rec, flush=True)
            del packs, observed, dep, features, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    guard()
    observed, arrivals = load(RAW / 'ranking.parquet')
    dep = observed.loc[observed[PHASE].eq('DEP')]
    features = build(dep, observed, arrivals)
    assert np.array_equal(features.index, dep[ID])
    features.reset_index().to_parquet(OUT / 'ranking_features.parquet', index=False)
    expected = pq.read_table(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    actual = pq.read_table(OUT / 'training_features.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    assert np.array_equal(expected, actual)
    guard()
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sourcehash, 'policy': POLICY, 'features': sum(GROUPS.values(), []), 'feature_groups': GROUPS, 'training_rows': rows_total, 'ranking_rows': len(features), 'records': records, 'outputs': {p.name: sha(p) for p in OUT.glob('*.parquet')}, 'runtime_sec': time.monotonic() - started, 'training_id_order_verified': True, 'prediction_gain_tested': False}
    write_json(OUT / 'manifest.json', manifest)
    verification.update({'manifest_sha256': sha(OUT / 'manifest.json'), 'source_sha256': sourcehash, 'training_id_order_verified': True, 'raw_hashes_verified': True, 'feature_count': 50})
    write_json(OUT / 'verification.json', verification)
    print('COMPLETE', rows_total, len(features), flush=True)


if __name__ == '__main__':
    main()
