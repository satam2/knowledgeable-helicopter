"""Independent raw-filter oracle; no model or departure outcome reads."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cache


def main():
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    began = time.monotonic()
    manifest_path = cache.OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete'
    assert manifest['source_sha256'] == cache.sha(cache.__file__)
    assert cache.sha(cache.OUT / 'neighbors.npy') == manifest['neighbors_sha256']
    neighbors = np.load(cache.OUT / 'neighbors.npy', mmap_mode='r')
    audit = json.loads((cache.ROOT / 'private_runs/screening_230/reports/data_audit.json').read_text())
    meta_path = cache.ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert cache.sha(meta_path) == audit['artifacts'][meta_path.name]
    expected_ids = pq.read_table(meta_path, columns=[cache.ID]).column(0).to_numpy()
    checks = []
    files = sorted(cache.RAW.glob('training_*.parquet'))
    for i, record in enumerate(manifest['records']):
        cache.memory_guard()
        assert cache.sha(cache.OUT / record['events_file']) == record['events_sha256']
        assert cache.sha(cache.OUT / record['queries_file']) == record['queries_sha256']
        events = pd.read_parquet(cache.OUT / record['events_file'])
        queries = pd.read_parquet(cache.OUT / record['queries_file'])
        offset = record['query_offset']
        np.testing.assert_array_equal(queries[cache.ID], expected_ids[offset:offset+len(queries)])
        parts, completions = [], []
        for j in range(max(0, i-1), min(len(files), i+2)):
            assert cache.sha(files[j]) == manifest['raw_hashes'][files[j].name]
            parts.append(pq.read_table(files[j], columns=cache.PUBLIC).to_pandas())
            completions.append(pq.read_table(files[j], columns=[cache.ID, cache.BLOCK],
                                            filters=[(cache.PHASE, '=', 'ARR')]).to_pandas())
        raw = pd.concat(parts, ignore_index=True).drop_duplicates(cache.ID)
        arr = pd.concat(completions, ignore_index=True).drop_duplicates(cache.ID).set_index(cache.ID)[cache.BLOCK]
        raw = raw.sort_values(cache.ID).reset_index(drop=True)
        movement = pd.to_datetime(raw[cache.MOVEMENT], utc=True)
        completion = pd.to_datetime(raw[cache.ID].map(arr), utc=True)
        isdep = raw[cache.PHASE].eq('DEP').to_numpy()
        airport = raw.ADEP_mvt.where(isdep, raw.ADES_mvt).astype('string').fillna('<missing>')
        month = movement.dt.strftime('%Y-%m').to_numpy()
        eventtime = movement.where(isdep, completion)
        seconds = (completion-movement).dt.total_seconds()
        accepted = isdep | ((~isdep) & seconds.notna().to_numpy() & seconds.ge(0).to_numpy())
        duplicate = pd.DataFrame({'flight': raw[cache.FLIGHT], 'airport': airport,
            'phase': raw[cache.PHASE], 'movement': movement}).duplicated()
        accepted &= ~(raw[cache.FLIGHT].notna().to_numpy() & duplicate.to_numpy())
        ids = raw[cache.ID].to_numpy()
        flights = raw[cache.FLIGHT].to_numpy(float)
        rng = np.random.default_rng(20260916+i)
        sample = np.unique(np.r_[0, len(queries)-1, rng.choice(len(queries), 48, replace=False)])
        for qpos in sample:
            q = queries.iloc[qpos]
            qt = pd.Timestamp(q.time_ns, tz='UTC')
            ok = accepted & airport.eq(q.airport).to_numpy() & (month == q.month)
            ok &= eventtime.lt(qt).to_numpy() & eventtime.ge(qt-pd.Timedelta(hours=1)).to_numpy()
            ok &= ids != q[cache.ID]
            if np.isfinite(q.flight):
                ok &= flights != q.flight
            pieces = []
            for phase in (True, False):
                candidates = np.flatnonzero(ok & (isdep == phase))
                ordered = sorted(candidates, key=lambda k: (eventtime.iloc[k].value, ids[k]), reverse=True)[:16]
                pieces.extend(ordered)
            pieces = sorted(pieces, key=lambda k: eventtime.iloc[k].value)
            observed = neighbors[offset+qpos]
            local = observed[observed >= 0] - record['event_offset']
            np.testing.assert_array_equal(events.iloc[local][cache.ID], ids[pieces])
            for k in local:
                event = events.iloc[k]
                r = raw.loc[raw[cache.ID].eq(event[cache.ID])].iloc[0]
                if event.phase == 'DEP':
                    for clock in cache.CLOCKS:
                        expected = (pd.Timestamp(r[cache.MOVEMENT])-pd.Timestamp(r[clock])).total_seconds() if pd.notna(r[clock]) else np.nan
                        np.testing.assert_allclose(event['offset_'+clock], expected, rtol=1e-6, equal_nan=True)
                else:
                    expected = (pd.Timestamp(arr.loc[event[cache.ID]])-pd.Timestamp(r[cache.MOVEMENT])).total_seconds()
                    np.testing.assert_allclose(event.arrival_duration, expected, rtol=1e-6)
        checks.append({'file': record['file'], 'sample_queries': len(sample), 'all_IDs_order_verified': len(queries)})
        print('VERIFIED', record['file'], len(sample), flush=True)
        del events, queries, raw, parts, completions
        gc.collect()
    result = {'status': 'passed', 'manifest_sha256': cache.sha(manifest_path),
        'verifier_sha256': cache.sha(__file__), 'queries': sum(v['sample_queries'] for v in checks),
        'checks': checks, 'departure_outcomes_read': False, 'method': 'Raw independent eligibility predicates and sorted lists, no builder event/selection helpers',
        'runtime_sec': time.monotonic()-began}
    cache.write_json(cache.OUT / 'verification.json', result)
    print('PASSED', result['queries'], result['runtime_sec'], flush=True)


if __name__ == '__main__':
    main()
