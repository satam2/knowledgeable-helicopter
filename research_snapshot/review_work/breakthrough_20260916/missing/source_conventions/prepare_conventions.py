"""Label-free observation-source conventions, frozen inputs and external cache."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import itertools
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.availability import assert_observations, make_observations
from taxiout.paths import external_path
from taxiout.schema import CLOCKS, ID, MOVEMENT, PHASE, utc

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/source_conventions')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
NAMES = ['nm', 'est', 'init', 'last', 'sched']


def convention_features(observations):
    assert_observations(observations)
    if not observations[PHASE].eq('DEP').all():
        raise ValueError('Convention features require departures')
    query = utc(observations[MOVEMENT], required=True)
    clocks = [utc(observations[column]) for column in CLOCKS]
    proxies = np.column_stack([(query - clock).dt.total_seconds().to_numpy() for clock in clocks])
    finite = np.isfinite(proxies)
    result = {}
    for i, name in enumerate(NAMES):
        value = clocks[i]
        result[f'conv_{name}_second'] = value.dt.second.to_numpy()
        result[f'conv_{name}_minute_aligned'] = (value.notna() & value.dt.second.eq(0) & value.dt.microsecond.eq(0)).to_numpy()
        result[f'conv_{name}_five_min_aligned'] = (value.notna() & value.dt.minute.mod(5).eq(0) & value.dt.second.eq(0) & value.dt.microsecond.eq(0)).to_numpy()
        result[f'conv_{name}_calendar_day_delta'] = (query.dt.normalize() - value.dt.normalize()).dt.total_seconds().to_numpy() / 86400
    for i, j in itertools.combinations(range(5), 2):
        # Clock_i - clock_j is proxy_j - proxy_i, reversing duration order.
        delta = proxies[:, j] - proxies[:, i]
        base = f'conv_{NAMES[i]}_minus_{NAMES[j]}'
        result[base + '_sec'] = delta
        result[base + '_signedlog'] = np.sign(delta) * np.log1p(np.abs(delta))
        result[base + '_nearest_hour_residual'] = delta - np.rint(delta / 3600) * 3600
        result[base + '_nearest_day_residual'] = delta - np.rint(delta / 86400) * 86400
        result[base + '_missing'] = ~np.isfinite(delta)
    result['conv_available_clock_count'] = finite.sum(axis=1)
    result['conv_est_init_last_same'] = finite[:, 1:4].all(axis=1) & (proxies[:, 1] == proxies[:, 2]) & (proxies[:, 2] == proxies[:, 3])
    result['conv_nm_matches_any_planning'] = ((proxies[:, :1] == proxies[:, 1:4]) & finite[:, :1] & finite[:, 1:4]).any(axis=1)
    result['conv_schedule_matches_any_nm'] = ((proxies[:, 4:5] == proxies[:, :4]) & finite[:, 4:5] & finite[:, :4]).any(axis=1)
    sorted_proxy = np.sort(np.where(finite, proxies, np.inf), axis=1)
    distinct = np.isfinite(sorted_proxy[:, 0]).astype(np.int16)
    distinct += np.sum(np.isfinite(sorted_proxy[:, 1:]) & (sorted_proxy[:, 1:] != sorted_proxy[:, :-1]), axis=1)
    result['conv_distinct_clock_count'] = distinct
    for name in ['ADEP', 'ADES', 'AIRCRAFT_TYPE']:
        a, b = observations[name + '_mvt'], observations[name + '_flt']
        result[f'conv_{name}_source_missing'] = b.isna().to_numpy()
        result[f'conv_{name}_source_disagree'] = (a.notna() & b.notna() & a.astype('string').ne(b.astype('string'))).fillna(False).to_numpy()
    numeric = pd.DataFrame({key: np.asarray(value, dtype=np.float32) for key, value in result.items()}, index=pd.Index(observations[ID], name=ID))
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(-999999)
    # Equal values receive the same rank; missing clocks retain an explicit code.
    ranks = pd.DataFrame(proxies).rank(axis=1, method='dense').fillna(0).astype(int)
    numeric['conv_clock_rank_signature'] = ranks.astype(str).agg('|'.join, axis=1).to_numpy()
    eq = np.column_stack([finite[:, i] & finite[:, j] & (proxies[:, i] == proxies[:, j]) for i, j in itertools.combinations(range(5), 2)])
    numeric['conv_clock_equality_signature'] = pd.DataFrame(eq.astype(np.int8)).astype(str).agg(''.join, axis=1).to_numpy()
    delta = proxies[:, 1] - proxies[:, 0]
    numeric['conv_nm_est_gap_bucket'] = pd.cut(delta, [-np.inf, -7200, -1800, -300, -60, 60, 300, 1800, 7200, np.inf],
        labels=['ltm2h','m2h_m30m','m30m_m5m','m5m_m1m','near1m','p1m_p5m','p5m_p30m','p30m_p2h','gtp2h']).astype('string').fillna('missing').to_numpy()
    numeric['conv_airport_order'] = observations.ADEP_mvt.astype('string').fillna('missing').to_numpy() + '|' + numeric.conv_clock_rank_signature
    for name in numeric.select_dtypes(['object','string']):
        numeric[name] = numeric[name].astype('category')
    return numeric


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / 'manifest.json'
    if marker.exists():
        report = read_json(marker)
        assert report['source_sha256'] == sha256(__file__)
        assert report['feature_sha256'] == sha256(OUT / 'features.parquet')
        print('Verified completed source-convention cache', flush=True)
        return
    started = time.monotonic()
    frozen = read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    write_json(OUT / 'protocol.json', {'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'hypothesis': 'Clock ordering/equality, signed source differences and precision expose source conventions omitted by two-clock trust/base features.',
        'availability': 'Current supplied record fields only; hidden departure target/block not read in feature path; final timestamp publication unverified.',
        'label_derived_features': False, 'gpu': False, 'cpu_threads': 1})
    writer, ids, records = None, [], []
    try:
        for path in sorted(RAW.glob('training_*.parquet')):
            assert sha256(path) == frozen['raw_hashes'][path.name]
            cols = [ID, PHASE, MOVEMENT, *CLOCKS, 'ADEP_mvt','ADEP_flt','ADES_mvt','ADES_flt','AIRCRAFT_TYPE_mvt','AIRCRAFT_TYPE_flt']
            raw = pq.read_table(path, columns=cols, use_threads=False).to_pandas(strings_to_categorical=True)
            obs, _, _ = make_observations(raw)
            dep = obs.loc[obs[PHASE].eq('DEP')]
            features = convention_features(dep)
            ids.append(features.index.to_numpy())
            serial = features.reset_index()
            for column in serial.select_dtypes('category'):
                serial[column] = serial[column].astype('string')
            table = pa.Table.from_pandas(serial, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT / 'features.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            records.append({'file': path.name, 'rows': len(features), 'rank_patterns': int(features.conv_clock_rank_signature.nunique()),
                'all_three_planning_clocks_equal_pct': float(features.conv_est_init_last_same.mean() * 100),
                'nm_matches_any_planning_pct': float(features.conv_nm_matches_any_planning.mean() * 100)})
            print(path.name, len(features), 'rows', len(features.columns), 'features', flush=True)
    finally:
        if writer is not None:
            writer.close()
    original = pd.read_parquet(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID])
    assert np.array_equal(np.concatenate(ids), original[ID])
    write_json(marker, {'status': 'complete', 'source_sha256': sha256(__file__), 'created_utc': utc_now(),
        'feature_sha256': sha256(OUT / 'features.parquet'), 'columns': list(features), 'rows': len(original),
        'months': records, 'original_audited_id_order_equal': True, 'runtime_sec': time.monotonic() - started,
        'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)})


if __name__ == '__main__':
    main()
