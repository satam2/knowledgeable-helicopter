"""Stream each confidential input in place; write only external feature artifacts."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'

import gc
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.availability import make_observations
from taxiout.io import concat_frames
from taxiout.paths import external_path
from taxiout.schema import BLOCK, CLOCKS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, utc
from arrival_features import POLICY, aviation_features, extract_completed_arrivals

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
OUT = external_path(ROOT / 'private_runs/campaign_20260916/aviation')
COLS = [ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, TARGET, 'ADEP_mvt', 'ADES_mvt',
        'RUNWAY_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt', *CLOCKS]


def profile(raw):
    arr = raw.loc[raw[PHASE].eq('ARR')]
    dep = raw.loc[raw[PHASE].eq('DEP')]
    label = pd.to_numeric(arr[TARGET], errors='coerce')
    block, move = utc(arr[BLOCK]), utc(arr[MOVEMENT], required=True)
    seconds = (block - move).dt.total_seconds()
    comparable = seconds.notna() & label.notna()
    return {'arrivals': len(arr), 'departures': len(dep),
        'arrival_block_present': int(block.notna().sum()), 'arrival_label_present': int(label.notna().sum()),
        'arrival_identity_exact': bool(np.array_equal(label[comparable], seconds[comparable])),
        'arrival_negative_duration': int((seconds < 0).sum()),
        'arrival_duration_over_7200': int((seconds > 7200).sum()),
        'departure_label_present': int(dep[TARGET].notna().sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / 'manifest.json'
    if final.exists():
        manifest = read_json(final)
        for name, digest in manifest['sources'].items():
            if sha256(HERE / name) != digest:
                raise ValueError('Completed feature source changed')
        if sha256(OUT / 'features.parquet') != manifest['feature_sha256']:
            raise ValueError('Completed feature cache changed')
        print('Verified completed aviation feature cache', flush=True)
        return
    existing = [path for path in OUT.iterdir() if path.is_file()]
    if existing:
        failed = OUT / ('failed_attempt_' + utc_now().replace(':', '-'))
        failed.mkdir()
        for path in existing:
            path.rename(failed / path.name)
    tests = subprocess.run([sys.executable, '-B', '-u', str(HERE / 'test_arrival_features.py')], capture_output=True, text=True)
    (OUT / 'test_results.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    if tests.returncode:
        raise RuntimeError('Aviation contract tests failed; inspect test_results.txt')
    expected = read_json(ROOT / 'private_runs/aviation_spike/protocol.json')['raw_hashes']
    sources = {p.name: sha256(p) for p in HERE.glob('*.py')}
    write_json(OUT / 'protocol.json', {'created_utc': utc_now(), 'sources': sources, 'policy': POLICY,
        'hypothesis': 'Completed arrival taxi-in durations and bounded arrival inventory provide surface state beyond historical throughput; recording precision identifies source conventions.',
        'columns_separate': ['arr_', 'surface_', 'seq_', 'precision_'],
        'label_derived_reference': 'q10 not in global cache; must fit within each fold and cross-fit training months',
        'resource_budget': {'cpu_threads': 1, 'max_peak_rss_gib': 4, 'gpu': False},
        'departure_queue_limit': 'No departure queue or pushback inventory inferred from future takeoff rows.',
        'real_time_limit': 'Observed final NM timestamps do not prove real-time publication.',
        'inventory_limit': 'Observed arrivals landed within 120 minutes, still lacking a valid prior completion; month isolated; incomplete or anomalous arrivals may remain until expiration.',
        'duplicate_policy': 'Smallest movement ID for identical flight/airport/landing, using no completion information; exclude all query-flight counterparts.',
        'prediction_event': 'takeoff', 'matching_training_labels_used': False})
    total_start = time.monotonic()
    records = []
    ids = []
    writer = None
    paths = sorted(RAW.glob('training_*.parquet'))
    file_months = {}
    for path in paths:
        movements = pq.read_table(path, columns=[MOVEMENT], use_threads=False).to_pandas()[MOVEMENT]
        file_months[path] = set(utc(movements, required=True).dt.strftime('%Y-%m').unique())
    try:
        for path in paths:
            started = time.monotonic()
            digest = sha256(path)
            if digest != expected[path.name]:
                raise ValueError(f'Raw hash changed: {path.name}')
            raw = pq.read_table(path, columns=COLS, use_threads=False).to_pandas(strings_to_categorical=True)
            months = file_months[path]
            companions = []
            # Raw filenames can straddle UTC month boundaries. Match the frozen
            # pipeline's event-month context across all overlapping input packs.
            for other in paths:
                if other != path and months & file_months[other]:
                    if sha256(other) != expected[other.name]:
                        raise ValueError('Companion raw hash changed')
                    extra = pq.read_table(other, columns=COLS, use_threads=False).to_pandas(strings_to_categorical=True)
                    companions.append(extra.loc[utc(extra[MOVEMENT], required=True).dt.strftime('%Y-%m').isin(months)])
            combined = concat_frames([raw, *companions]) if companions else raw
            obs, _, _ = make_observations(combined)
            dep = obs.loc[obs[PHASE].eq('DEP') & obs[ID].isin(raw[ID])]
            arr = extract_completed_arrivals(combined)
            x = aviation_features(dep, obs, arr)
            table = pa.Table.from_pandas(x.reset_index(), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT / 'features.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            ids.append(x.index.to_numpy(copy=True))
            record = {'file': path.name, 'raw_sha256': digest, **profile(raw),
                'rows': len(x), 'columns': list(x.columns), 'runtime_sec': time.monotonic() - started,
                'completed_arrivals_30m_support_pct': float(x.arr_airport_count_30m.gt(0).mean() * 100),
                'completed_arrivals_30m_median_n': float(x.arr_airport_count_30m.median()),
                'arrival_inventory_mean': float(x.surface_arrivals_open_120m.mean()),
                'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}
            records.append(record)
            write_json(OUT / 'progress.json', records)
            print(f'{path.name}: {len(x):,} departures, {len(x.columns)} features, {record["runtime_sec"]:.1f}s, {record["peak_rss_bytes"] / 1024**3:.2f} GiB peak', flush=True)
            if record['peak_rss_bytes'] > 4 * 1024**3:
                raise MemoryError('Aviation build exceeded approved 4 GiB RSS')
            del raw, obs, dep, arr, x, table, combined, companions
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    observed_ids = np.concatenate(ids)
    reference = pd.read_parquet(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID])
    if not np.array_equal(reference[ID].to_numpy(), observed_ids):
        raise ValueError('All-training cache does not preserve original audited IDs/order')
    ranking_path = RAW / 'ranking.parquet'
    if sha256(ranking_path) != expected[ranking_path.name]:
        raise ValueError('Ranking raw hash changed')
    ranking = pq.read_table(ranking_path, columns=COLS, use_threads=False).to_pandas(strings_to_categorical=True)
    ranking_profile = profile(ranking)
    extract_completed_arrivals(ranking)
    write_json(final, {'status': 'complete', 'created_utc': utc_now(), 'policy': POLICY, 'sources': sources,
        'feature_path': str(OUT / 'features.parquet'), 'feature_sha256': sha256(OUT / 'features.parquet'),
        'rows': len(observed_ids), 'original_audited_id_order_equal': True, 'months': records,
        'ranking_availability': ranking_profile, 'runtime_sec': time.monotonic() - total_start,
        'scope': 'Feature availability and causality verified; no accuracy result, model or ranking feature cache.'})
    print(f'COMPLETE: {len(observed_ids):,} rows; ranking arrival availability verified', flush=True)


if __name__ == '__main__':
    main()
