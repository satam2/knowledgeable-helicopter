"""Independent public-flight cache oracle; never import the producer builder."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import json
from pathlib import Path
import time
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / 'private_runs/breakthrough_20260916/missing/opdi_rotation'
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
ID = 'MVT_ID_mvt'
TIME = 'MVT_TIME_UTC_mvt'
FEATURES = ['opdi_match', 'opdi_match_ambiguous', 'opdi_match_offset_sec',
    'opdi_previous_leg_available', 'opdi_previous_same_airport',
    'opdi_ground_interval_sec', 'opdi_previous_leg_duration_sec',
    'opdi_previous_leg_age_at_takeoff_sec']
FIELDS = [ID, TIME, 'PHASE_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'AOBT_3_flt']
PUBLIC = ['id', 'icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalize(value):
    return '' if pd.isna(value) else ''.join(str(value).upper().split())


def oracle(row, public):
    result = np.array([0., 0., np.nan, 0., 0., np.nan, np.nan, np.nan], dtype=np.float32)
    key = tuple(normalize(row[name]) for name in ['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'])
    if not all(key):
        return result
    takeoff = row[TIME]
    candidates = public.loc[(public.callsign == key[0]) & (public.origin == key[1])
        & (public.destination == key[2]) & public.aircraft.ne('')
        & ((takeoff - public['first']).dt.total_seconds().abs() <= 600)]
    if len(candidates) > 1:
        result[1] = 1
        return result
    if len(candidates) != 1:
        return result
    current = candidates.iloc[0]
    result[0], result[2] = 1, (takeoff - current['first']).total_seconds()
    before = public.loc[(public.aircraft == current.aircraft) & (public['first'] < current['first'])]
    if (before['last'] >= current['first']).any():
        return result
    completed = before.loc[(before['last'] < current['first']) & (before['last'] < takeoff)]
    if completed.empty:
        return result
    latest = completed.loc[completed['last'] == completed['last'].max()]
    if len(latest) != 1:
        return result
    prior = latest.iloc[0]
    result[3] = 1
    if prior.destination != key[1]:
        return result
    result[4:] = [1, (current['first'] - prior['last']).total_seconds(),
        (prior['last'] - prior['first']).total_seconds(), (takeoff - prior['last']).total_seconds()]
    return result


def guard():
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert peak < 2 * 1024**3, peak
    assert psutil.virtual_memory().available >= 8 * 1024**3
    return int(peak)


def main():
    started = time.monotonic()
    output = CACHE / 'independent_oracle.json'
    assert not output.exists(), 'Preserve prior oracle receipt'
    manifest = read(CACHE / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['features'] == FEATURES
    manifest_hash = sha(CACHE / 'manifest.json')
    for name, expected in manifest['outputs'].items():
        assert sha(CACHE / name) == expected, name
    for key in ['builder_source_path', 'builder_base_path']:
        expected = manifest['source_sha256' if key == 'builder_source_path' else 'builder_base_sha256']
        assert sha(manifest[key]) == expected, key
    if 'builder_v3_path' in manifest:
        assert sha(manifest['builder_v3_path']) == manifest['builder_v3_sha256']
    acquisition = read(CACHE / 'acquisition.json')
    assert sha(CACHE / 'acquisition.json') == manifest['acquisition_sha256']
    frozen = read(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    cache_frames = {role: pd.read_parquet(CACHE / f'{role}_features.parquet').set_index(ID)
                    for role in ('training', 'ranking')}
    query_parts = []
    for path in sorted(RAW.glob('training_*.parquet')) + [RAW / 'ranking.parquet']:
        assert sha(path) == frozen[path.name], path.name
        raw = pq.read_table(path, columns=FIELDS, filters=[('PHASE_mvt', '=', 'DEP')], use_threads=False).to_pandas()
        missing = raw.loc[pd.to_datetime(raw.AOBT_3_flt, utc=True, errors='coerce').isna()].copy()
        missing[TIME] = pd.to_datetime(missing[TIME], utc=True)
        missing['role'] = 'ranking' if path.name == 'ranking.parquet' else 'training'
        query_parts.append(missing.drop(columns=['PHASE_mvt', 'AOBT_3_flt']))
        del raw, missing
        guard()
    query = pd.concat(query_parts, ignore_index=True).set_index(ID)
    del query_parts
    assert query.index.is_unique and not query.index.isna().any()
    for role, frame in cache_frames.items():
        expected_ids = query.loc[query.role.eq(role)].index
        np.testing.assert_array_equal(frame.index, expected_ids)
        assert list(frame) == FEATURES and all(dtype == np.float32 for dtype in frame.dtypes)
        for flag in FEATURES[:2] + FEATURES[3:5]:
            assert frame[flag].isin([0, 1]).all(), flag
        assert not np.isinf(frame.to_numpy()).any()
        assert not (frame.opdi_match.eq(1) & frame.opdi_match_ambiguous.eq(1)).any()
        assert frame.loc[frame.opdi_match.eq(1), 'opdi_match_offset_sec'].abs().le(600).all()
        assert frame.loc[frame.opdi_previous_same_airport.eq(1), FEATURES[5:]].gt(0).all().all()
        assert frame.loc[frame.opdi_previous_same_airport.eq(0), FEATURES[5:]].isna().all().all()
        assert not (frame.opdi_previous_leg_available.eq(1) & frame.opdi_match.eq(0)).any()
        assert not (frame.opdi_previous_same_airport.eq(1) & frame.opdi_previous_leg_available.eq(0)).any()
    full = pd.concat(cache_frames.values())
    query['month'] = query[TIME].dt.strftime('%Y%m')
    source = {record['month']: record for record in acquisition['sources']}
    records, row_count, finite_cells, missing_cells, peak = [], 0, 0, 0, guard()
    for month, month_queries in query.groupby('month', sort=True):
        candidates = full.loc[month_queries.index]
        chosen = []
        masks = [candidates.opdi_previous_same_airport.eq(1),
            candidates.opdi_match.eq(1) & candidates.opdi_previous_same_airport.eq(0),
            candidates.opdi_match_ambiguous.eq(1), candidates.opdi_match.eq(0)]
        for mask in masks:
            ids = candidates.index[mask]
            if len(ids):
                rng = np.random.default_rng(20260916 + int(month))
                chosen.extend(rng.choice(ids.to_numpy(), min(12, len(ids)), replace=False).tolist())
        selected = month_queries.loc[list(dict.fromkeys(chosen))]
        receipt = source[month]
        path = Path(receipt['local_path'])
        assert sha(path) == receipt['sha256'], month
        raw = pq.read_table(path, columns=PUBLIC, use_threads=False).to_pandas()
        public = pd.DataFrame({
            'callsign': raw.flt_id.map(normalize), 'origin': raw.adep.map(normalize),
            'destination': raw.ades.map(normalize), 'aircraft': raw.icao24.map(normalize),
            'first': pd.to_datetime(raw.first_seen, utc=True, errors='coerce'),
            'last': pd.to_datetime(raw.last_seen, utc=True, errors='coerce')})
        del raw
        public = public.loc[public['first'].dt.strftime('%Y%m').eq(month) & (public['last'] > public['first'])]
        compared = []
        for ident, row in selected.iterrows():
            expected = oracle(row, public)
            actual = full.loc[ident].to_numpy(np.float32)
            np.testing.assert_array_equal(actual, expected, err_msg=f'{month} query {ident}')
            finite_cells += int(np.isfinite(expected).sum())
            missing_cells += int(np.isnan(expected).sum())
            compared.append(str(ident))
        records.append({'month': month, 'sample_rows': len(compared), 'sample_ids': compared,
            'available_queries': len(month_queries), 'public_sha256': receipt['sha256']})
        row_count += len(compared)
        peak = max(peak, guard())
        print('OPDI_ORACLE', month, len(compared), 'exact match', flush=True)
        del public
        gc.collect()
    assert sha(CACHE / 'manifest.json') == manifest_hash
    for name, expected in manifest['outputs'].items():
        assert sha(CACHE / name) == expected
    result = {'status': 'passed', 'manifest_sha256': manifest_hash,
        'source_sha256': sha(__file__), 'independence': 'Separate raw reconstruction, no producer imports; stratified deterministic samples across every month.',
        'raw_training_and_ranking_ID_order_exact': True, 'training_rows': len(cache_frames['training']),
        'ranking_rows': len(cache_frames['ranking']), 'all_cache_structural_checks': True,
        'sample_rows': row_count, 'finite_cells': finite_cells, 'missing_cells': missing_cells,
        'max_abs_delta': 0., 'records': records, 'peak_rss_bytes': peak,
        'runtime_sec': time.monotonic() - started, 'identity_truth_verified': False,
        'limitation': 'Verifies the declared deterministic linkage algorithm, not real aircraft identity or exact surface times.'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    verification = CACHE / 'verification.json'
    assert not verification.exists(), 'Preserve previous verification receipt'
    verification.write_text(json.dumps({'status': 'passed', 'manifest_sha256': manifest_hash,
        'independent_oracle_sha256': sha(output), 'source_sha256': sha(__file__),
        'scope': 'Complete cache hashes, original raw missing-cohort order, all-row structural checks, independent raw sample oracle.',
        'sample_rows': row_count, 'sample_cells': finite_cells + missing_cells,
        'max_abs_delta': 0., 'identity_truth_verified': False}, indent=2), encoding='utf-8')
    print('PASS', row_count, 'rows;', finite_cells + missing_cells, 'cells; peak', peak, flush=True)


if __name__ == '__main__':
    main()
