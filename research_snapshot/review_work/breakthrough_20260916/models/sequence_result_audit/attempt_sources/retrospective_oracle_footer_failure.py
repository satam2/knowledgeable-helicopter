"""Independent direct-mask oracle for the frozen retrospective feature cache."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'review_work/breakthrough_20260916/models_retrieval/retrospective_research/build.py'
CACHE = ROOT / 'private_runs/breakthrough_20260916/retrospective_research'
OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit'
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
ID, FLIGHT, PHASE = 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt'
MOVE, BLOCK = 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
NAMES = ['nm', 'est', 'init', 'last', 'sched']
COLS = [ID, FLIGHT, PHASE, MOVE, 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', *CLOCKS]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_load(path):
    observed = pq.read_table(path, columns=COLS, use_threads=False).to_pandas()
    arrivals = pq.read_table(path, columns=[ID, PHASE, BLOCK], filters=[(PHASE, '=', 'ARR')], use_threads=False).to_pandas()
    return observed, arrivals


def context(raw, blocks):
    x = raw[COLS].copy()
    x['airport'] = np.where(x[PHASE].eq('DEP'), x.ADEP_mvt, x.ADES_mvt)
    x['airport'] = x.airport.fillna('<missing>')
    x['runway'] = x.RUNWAY_mvt.fillna('<missing>')
    x['move'] = pd.to_datetime(x[MOVE], utc=True)
    x['month'] = x.move.dt.strftime('%Y-%m')
    x['key'] = [('f', f) if pd.notna(f) else ('i', i) for f, i in zip(x[FLIGHT], x[ID])]
    x = x.sort_values(ID).drop_duplicates(['key', 'airport', PHASE, 'move'])
    x['end'] = pd.to_datetime(x[ID].map(blocks.set_index(ID)[BLOCK]), utc=True)
    x['duration'] = (x.end - x.move).dt.total_seconds()
    for name, clock in zip(NAMES, CLOCKS):
        x[name] = (x.move - pd.to_datetime(x[clock], utc=True)).dt.total_seconds()
    x['gap'] = (pd.to_datetime(x[CLOCKS[0]], utc=True) - pd.to_datetime(x[CLOCKS[1]], utc=True)).dt.total_seconds()
    return x


def moments(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.mean(v)), float(np.std(v, ddof=0))) if len(v) else (np.nan, np.nan)


def oracle(queries, raw, blocks, features):
    c = context(raw, blocks)
    records = []
    for _, query in queries.iterrows():
        t = pd.Timestamp(query[MOVE])
        airport = query.ADEP_mvt if pd.notna(query.ADEP_mvt) else '<missing>'
        runway = query.RUNWAY_mvt if pd.notna(query.RUNWAY_mvt) else '<missing>'
        eligible = c.airport.eq(airport) & c.month.eq(t.strftime('%Y-%m')) & c[ID].ne(query[ID])
        if pd.notna(query[FLIGHT]):
            eligible &= ~c[FLIGHT].eq(query[FLIGHT])
        local = c.loc[eligible]
        result = {}
        for mode in ('past', 'future', 'day'):
            for kind in ('arr', 'dep'):
                event = local.loc[local[PHASE].eq(kind.upper())]
                times = event.end if kind == 'arr' else event.move
                if mode == 'past':
                    mask = times.ge(t - pd.Timedelta(hours=1)) & times.lt(t)
                elif mode == 'future':
                    mask = times.gt(t) & times.le(t + pd.Timedelta(hours=1))
                else:
                    mask = times.ge(t.normalize()) & times.lt(t.normalize() + pd.Timedelta(days=1))
                if kind == 'arr':
                    mask &= np.isfinite(event.duration) & event.duration.ge(0)
                selected = event.loc[mask]
                for scope in (('airport', 'runway') if kind == 'arr' and mode != 'day' else ('airport',)):
                    part = selected if scope == 'airport' else selected.loc[selected.runway.eq(runway)]
                    prefix = f'retro_{mode}_{kind}_{scope}_'
                    result[prefix + 'count'] = len(part)
                    if kind == 'arr':
                        result[prefix + 'mean_sec'], result[prefix + 'std_sec'] = moments(part.duration)
                        result[prefix + 'over_1200_share'] = float(part.duration.gt(1200).mean()) if len(part) else np.nan
                    else:
                        for name in NAMES:
                            result[prefix + name + '_mean_sec'] = moments(part[name])[0]
                        result[prefix + 'nm_std_sec'] = moments(part.nm)[1]
                        result[prefix + 'nm_missing_share'] = float((~np.isfinite(part.nm)).mean()) if len(part) else np.nan
                        result[prefix + 'nm_minus_est_mean_sec'], result[prefix + 'nm_minus_est_std_sec'] = moments(part.gap)
        records.append(result)
    return pd.DataFrame(records, index=queries[ID], columns=features).astype('float32')


def compare(got, expected):
    assert list(got.columns) == list(expected.columns)
    assert np.array_equal(got.index, expected.index)
    a, b = got.to_numpy(), expected.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), 'NaN support mismatch'
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=0.005, equal_nan=True)
    return float(np.nanmax(np.abs(a - b)))


def synthetic(builder, features):
    t = pd.Timestamp('2025-07-01T00:30:00Z')
    rows = []
    specs = [('DEP', s, 0) for s in (-3601, -3600, -1, 0, 1, 3600, 3601)]
    specs += [('ARR', s - 600, 600) for s in (-3601, -3600, -1, 0, 1, 3600, 3601)]
    specs += [('ARR', -601, -1), ('ARR', -300, 1300)]
    for i, (phase, delta, duration) in enumerate(specs, 2):
        move = t + pd.Timedelta(seconds=delta)
        row = {ID: float(i), FLIGHT: float(i), PHASE: phase, MOVE: move, BLOCK: move + pd.Timedelta(seconds=duration), 'ADEP_mvt': 'AAA' if phase == 'DEP' else 'BBB', 'ADES_mvt': 'BBB' if phase == 'DEP' else 'AAA', 'RUNWAY_mvt': 'R'}
        row.update({clock: move - pd.Timedelta(seconds=600 + i * (j + 1)) for j, clock in enumerate(CLOCKS)})
        rows.append(row)
    query = rows[3].copy()
    query[ID], query[FLIGHT] = 1., 1.
    rows.append(query)
    sameflight = rows[4].copy()
    sameflight[ID], sameflight[FLIGHT] = 100., 1.
    rows.append(sameflight)
    duplicate = rows[4].copy()
    duplicate[ID] = 101.
    rows.append(duplicate)
    noflight = rows[4].copy()
    noflight[ID], noflight[FLIGHT], noflight['RUNWAY_mvt'] = 102., np.nan, None
    rows.append(noflight)
    raw = pd.DataFrame(rows)
    raw.loc[raw[ID].eq(6.), CLOCKS[0]] = pd.NaT
    queries = raw.loc[raw[ID].isin([1., 102.])]
    blocks = raw.loc[raw[PHASE].eq('ARR'), [ID, PHASE, BLOCK]]
    expected = oracle(queries, raw, blocks, features)
    got = builder.build(queries, raw, blocks)
    delta = compare(got, expected)
    poison = raw.copy()
    poison.loc[poison[PHASE].eq('DEP'), BLOCK] = t + pd.Timedelta(days=1000)
    poison['TAXITIME_SEC_mvt'] = 1e20
    pd.testing.assert_frame_equal(got, builder.build(poison.loc[poison[ID].isin(queries[ID])], poison, blocks))
    return {'status': 'passed', 'max_absolute_delta': delta, 'checks': ['exact_hour_and_tied_boundaries', 'UTC_day_and_month_boundary', 'same_flight_and_own_ID_exclusion', 'duplicate_smallest_ID', 'missing_flight_and_runway', 'negative_ARR_rejected', 'hidden_DEP_block_and_target_poison']}


def main():
    start = time.monotonic()
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    assert sha(SOURCE) == manifest['source_sha256']
    for name, digest in manifest['outputs'].items():
        assert sha(CACHE / name) == digest
    spec = importlib.util.spec_from_file_location('reviewed_retro_builder', SOURCE)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    result = {'source_sha256': sha(SOURCE), 'manifest_sha256': sha(CACHE / 'manifest.json'), 'synthetic': synthetic(builder, manifest['features']), 'files': []}
    paths = sorted(RAW.glob('training_*.parquet'))
    for index, path in enumerate([*paths, RAW / 'ranking.parquet']):
        if index < len(paths):
            surrounding = paths[max(0, index - 1):min(len(paths), index + 2)]
            packs = [raw_load(p) for p in surrounding]
            observed = packs[surrounding.index(path)][0]
            cache_path = CACHE / 'training_features.parquet'
        else:
            packs = [raw_load(path)]
            observed = packs[0][0]
            cache_path = CACHE / 'ranking_features.parquet'
        dep = observed.loc[observed[PHASE].eq('DEP')]
        ordered = dep.sort_values(MOVE)
        queries = pd.concat([ordered.iloc[np.linspace(0, len(ordered) - 1, 16).astype(int)], ordered.iloc[:2], ordered.iloc[-2:]]).drop_duplicates(ID)
        raw = pd.concat([p[0] for p in packs], ignore_index=True)
        blocks = pd.concat([p[1] for p in packs], ignore_index=True)
        expected = oracle(queries, raw, blocks, manifest['features'])
        saved = pq.read_table(cache_path, filters=[(ID, 'in', queries[ID].tolist())], use_threads=False).to_pandas().set_index(ID).loc[queries[ID], manifest['features']]
        delta = compare(saved, expected)
        rec = {'file': path.name, 'query_rows': len(queries), 'feature_cells': len(queries) * 50, 'max_absolute_delta': delta, 'query_ids': queries[ID].tolist()}
        result['files'].append(rec)
        print(json.dumps(rec), flush=True)
        del packs, observed, dep, ordered, queries, raw, blocks, expected, saved
        gc.collect()
    result.update(status='passed', runtime_sec=time.monotonic() - start, peak_rss_bytes=psutil.Process().memory_info().get('peak_wset', psutil.Process().memory_info().rss), independent_mask_and_direct_moments=True, cache_full_output_hashes_verified=True, gpu_used=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'retrospective_oracle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
