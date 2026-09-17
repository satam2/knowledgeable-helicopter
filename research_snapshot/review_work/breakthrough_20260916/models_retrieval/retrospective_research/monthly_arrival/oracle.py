"""Independent raw enumeration of300 monthly ARR queries; no builder imports."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
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

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
ID, FID, PHASE = 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt'
TIME, BLOCK = 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
CACHE = external_path(ROOT / 'private_runs/breakthrough_20260916/retrospective_research/monthly_arrival')
OUT = external_path(CACHE / 'independent_oracle')
DEP = [ID, FID, PHASE, TIME, 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt']
ARR = [ID, FID, PHASE, TIME, BLOCK, 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt']
NAMES = [f'monthly_arr_{scope}_{stat}' for scope in ['airport', 'stand', 'runway']
         for stat in ['count', 'mean_sec', 'median_sec', 'q10_sec', 'q90_sec']]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard():
    if psutil.Process().memory_info().rss > 2 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Oracle sampled memory guard exceeded')


def deduplicate(frame):
    x = frame[ARR].drop_duplicates().copy()
    assert x[PHASE].eq('ARR').all() and x[ID].notna().all() and x[ID].is_unique
    x = x.sort_values(ID, kind='stable')
    keep, seen = [], set()
    for position, (flight, airport, landing) in enumerate(zip(x[FID], x.ADES_mvt, x[TIME])):
        if pd.notna(flight):
            key = (flight, str(airport), landing)
            if key in seen:
                continue
            seen.add(key)
        keep.append(position)
    return x.iloc[keep].reset_index(drop=True)


def enumerate_query(query, arrivals):
    durations = (pd.to_datetime(arrivals[BLOCK], utc=True) - pd.to_datetime(arrivals[TIME], utc=True)).dt.total_seconds()
    allowed = arrivals.ADES_mvt.astype('string').eq(str(query.ADEP_mvt)).fillna(False)
    allowed &= arrivals[TIME].dt.strftime('%Y-%m').eq(query[TIME].strftime('%Y-%m'))
    allowed &= arrivals[ID].ne(query[ID]) & np.isfinite(durations)
    if pd.notna(query[FID]):
        allowed &= arrivals[FID].isna() | arrivals[FID].ne(query[FID])
    result = []
    for field in [None, 'STAND_mvt', 'RUNWAY_mvt']:
        mask = allowed.copy()
        if field:
            if pd.isna(query[field]):
                mask &= False
            else:
                mask &= arrivals[field].astype('string').eq(str(query[field])).fillna(False)
        values = durations.loc[mask].to_numpy(float)
        result.extend([len(values), values.mean(), *np.quantile(values, [.5, .1, .9], method='linear')]
                      if len(values) else [0, np.nan, np.nan, np.nan, np.nan])
    return np.asarray(result, dtype=np.float32)


def choose(frame, count):
    ordered = frame.sort_values(['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', TIME, ID], kind='stable', na_position='first')
    chosen = list(ordered.iloc[np.linspace(0, len(ordered) - 1, count, dtype=int)].index)
    # Reserve explicitmissingflight/stand/runway rows when those cases exist.
    for j, field in enumerate([FID, 'STAND_mvt', 'RUNWAY_mvt']):
        missing = ordered.loc[ordered[field].isna()]
        if len(missing):
            chosen[j] = missing.index[0]
    chosen = list(dict.fromkeys(chosen))
    for i in ordered.index:
        if len(chosen) == count:
            break
        if i not in chosen:
            chosen.append(i)
    return frame.loc[chosen]


def main():
    marker = CACHE / 'manifest.json'
    if not marker.exists() or read(marker)['status'] != 'complete':
        raise RuntimeError('Await complete monthly arrival cache')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = read(marker)
    manifest_digest = sha(marker)
    assert manifest['features'] == NAMES
    protocol = read(CACHE / 'protocol.json')
    assert sha(CACHE / 'protocol.json') == manifest['protocol_sha256']
    for path, expected in manifest['source_hashes'].items():
        assert sha(ROOT / path) == sha(CACHE / 'source' / Path(path).name) == expected
    for filename, expected in manifest['outputs'].items():
        assert sha(CACHE / filename) == expected
    paths = sorted(RAW.glob('training_*.parquet'))
    ranking = RAW / 'ranking.parquet'
    assert len(paths) == 12
    for path in [*paths, ranking]:
        assert sha(path) == protocol['raw_hashes'][path.name]
    reports, rows = [], 0
    for role, files in [('training', paths), ('ranking', [ranking])]:
        actual_ids = pq.read_table(CACHE / f'{role}_features.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
        cursor = 0
        for path in files:
            guard()
            queries = pq.read_table(path, columns=DEP, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
            assert queries[ID].notna().all() and queries[ID].is_unique and queries[PHASE].eq('DEP').all()
            np.testing.assert_array_equal(actual_ids[cursor:cursor + len(queries)], queries[ID])
            cursor += len(queries)
            selected = choose(queries, 20 if role == 'training' else 60)
            months = sorted(selected[TIME].dt.strftime('%Y-%m').unique())
            filters = [[(PHASE, '=', 'ARR'), (TIME, '>=', pd.Timestamp(m + '-01', tz='UTC').to_pydatetime()),
                        (TIME, '<', (pd.Timestamp(m + '-01', tz='UTC') + pd.offsets.MonthBegin(1)).to_pydatetime())] for m in months]
            parts = []
            for peer_path in files:
                guard()
                part = pq.read_table(peer_path, columns=ARR, filters=filters, use_threads=False).to_pandas()
                assert part[PHASE].eq('ARR').all()
                parts.append(part)
            raw_arrivals = pd.concat(parts, ignore_index=True)
            arrivals = deduplicate(raw_arrivals)
            expected = np.vstack([enumerate_query(row, arrivals) for _, row in selected.iterrows()])
            saved = pq.read_table(CACHE / f'{role}_features.parquet', filters=[(ID, 'in', selected[ID].tolist())], use_threads=False).to_pandas().set_index(ID)
            saved = saved.loc[selected[ID], NAMES].to_numpy(np.float32)
            np.testing.assert_allclose(saved, expected, rtol=1e-6, atol=.0001, equal_nan=True)
            equal = (saved == expected) | (np.isnan(saved) & np.isnan(expected))
            deltas = np.abs(saved - expected)
            finite = deltas[np.isfinite(deltas)]
            reports.append({'file': path.name, 'queries': len(selected), 'cells': expected.size,
                            'exact_float32_cells': int(equal.sum()), 'max_abs_delta': float(finite.max()) if len(finite) else 0,
                            'months': months, 'airports': sorted(selected.ADEP_mvt.astype(str).unique()),
                            'missing_query_groups': {field: int(selected[field].isna().sum()) for field in [FID, 'STAND_mvt', 'RUNWAY_mvt']},
                            'raw_arrivals': len(raw_arrivals), 'deduplicated_arrivals': len(arrivals)})
            rows += len(selected)
            (OUT / 'progress.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
            print('ORACLE', reports[-1], flush=True)
            del queries, selected, parts, arrivals, raw_arrivals, expected, saved
            gc.collect()
        assert cursor == len(actual_ids)
        del actual_ids
    assert rows == 300
    for path in [*paths, ranking]:
        assert sha(path) == protocol['raw_hashes'][path.name]
    assert sha(marker) == manifest_digest
    guard()
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'manifest_sha256': sha(marker), 'protocol_sha256': sha(CACHE / 'protocol.json'),
              'queries': rows, 'cells': rows * 15, 'reports': reports,
              'all_training_ranking_IDs_exact': True, 'raw_hashes_before_after_verified': True,
              'private_columns': {'DEP': DEP, 'ARR': ARR}, 'ARRpredicate_checked': True,
              'rtol': 1e-6, 'atol': .0001, 'model_or_GPU_or_network_used': False,
              'runtime_seconds': time.monotonic() - started, 'rss_bytes': psutil.Process().memory_info().rss}
    target = CACHE / 'independent_oracle.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print('ORACLE_PASSED', rows, rows * 15, flush=True)


if __name__ == '__main__':
    main()
