"""Raw-bound ranking cache replay and independent flattened-token arithmetic."""
from preflight import ROOT, OUT, RAW, ID, read, sha
import gc
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/final_ple387'))
import prepare_ranking as producer
from threadpoolctl import threadpool_limits


def guard():
    peak = psutil.Process().memory_info().peak_wset
    assert peak < 3 * 1024**3, peak
    assert psutil.virtual_memory().available >= 8 * 1024**3
    return peak


def main():
    dest = OUT / 'ranking_features_receipt.json'
    assert not dest.exists()
    started = time.monotonic()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    cache = producer.OUT
    cache8 = cache.parent / 'ranking_flat8_v1'
    markers = [read(p / 'manifest.json') for p in (cache, cache8)]
    for folder, marker in zip((cache, cache8), markers):
        assert marker['status'] == 'complete'
        assert marker['protocol_sha256'] == sha(folder / 'protocol.json')
        for name, digest in marker['outputs'].items():
            assert sha(folder / name) == digest
    protocol = read(cache / 'protocol.json')
    assert protocol['raw_sha256'] == sha(RAW / 'ranking.parquet')
    for path, digest in protocol['sources'].items():
        assert sha(ROOT / path) == digest
    assert protocol['source_sha256'] == sha(producer.__file__)
    meta = pd.read_parquet(ROOT / 'private_runs/submission_v2/ranking_meta.parquet')
    base = pd.read_parquet(cache / 'ranking_base.parquet')
    pd.testing.assert_frame_equal(base, pd.read_parquet(ROOT / 'private_runs/submission_v2/ranking_base.parquet').reset_index())
    del base
    conv = producer.conv
    columns = [ID, conv.PHASE, conv.MOVEMENT, *conv.CLOCKS, 'ADEP_mvt', 'ADEP_flt', 'ADES_mvt', 'ADES_flt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_TYPE_flt']
    raw = pq.read_table(RAW / 'ranking.parquet', columns=columns, use_threads=False).to_pandas(strings_to_categorical=True)
    obs = conv.make_observations(raw)[0]
    expected = conv.convention_features(obs.loc[obs[conv.PHASE].eq('DEP')]).reset_index()
    for name in expected.select_dtypes('category'):
        expected[name] = expected[name].astype('string')
    pd.testing.assert_frame_equal(expected, pd.read_parquet(cache / 'ranking_conventions.parquet'))
    del raw, obs, expected
    gc.collect(); guard()
    sequence = producer.sequence
    public = pq.read_table(RAW / 'ranking.parquet', columns=sequence.PUBLIC, use_threads=False).to_pandas()
    arrivals = pq.read_table(RAW / 'ranking.parquet', columns=[ID, sequence.BLOCK], filters=[(sequence.PHASE, '=', 'ARR')], use_threads=False).to_pandas()
    queries = sequence.query_frame(public.loc[public[sequence.PHASE].eq('DEP')])
    events = sequence.events_from_public(public, arrivals)
    pd.testing.assert_frame_equal(queries, pd.read_parquet(cache / 'ranking_queries.parquet'))
    pd.testing.assert_frame_equal(events, pd.read_parquet(cache / 'ranking_events.parquet'))
    np.testing.assert_array_equal(queries[ID], meta[ID])
    neighbors = np.load(cache / 'ranking_neighbors.npy')
    np.testing.assert_array_equal(sequence.neighbor_indices(events, queries), neighbors)
    del public, arrivals
    gc.collect(); guard()
    # Independent nearest-event selection on a fixed, label-free spread of queries.
    sample = np.unique(np.linspace(0, len(queries) - 1, 1024, dtype=int))
    for position in sample:
        q = queries.iloc[position]
        eligible = events.loc[(events.airport == q.airport) & (events.month == q.month) &
            (events.time_ns < q.time_ns) & (events.time_ns >= q.time_ns - 3600 * 10**9) & (events[ID] != q[ID])]
        if np.isfinite(q.flight):
            eligible = eligible.loc[eligible.flight != q.flight]
        chosen = []
        for phase in ('DEP', 'ARR'):
            chosen.extend(eligible.loc[eligible.phase == phase].sort_values(['time_ns', ID], ascending=False).head(16).index.tolist())
        chosen.sort(key=lambda index: events.time_ns.iloc[index])
        wanted = chosen + [-1] * (32 - len(chosen))
        np.testing.assert_array_equal(neighbors[position], wanted)
    numeric = events[sequence.NUMS].to_numpy('float32')
    event_time, movement = events.time_ns.to_numpy(), events.movement_ns.to_numpy()
    dep = events.phase.eq('DEP').to_numpy()
    eq_names = ['runway', 'stand', 'operator']
    ev_strings = {c:events[c].astype('string').replace({'<missing>':pd.NA, 'MISSING':pd.NA, 'm:':pd.NA}) for c in eq_names}
    qu_strings = {c:queries[c].astype('string').replace({'<missing>':pd.NA, 'MISSING':pd.NA, 'm:':pd.NA}) for c in eq_names}
    fields = [*sequence.NUMS, 'age_sec', 'landing_age_sec', 'same_runway', 'same_stand', 'same_operator', 'padding_missing']
    checked = {}
    for depth, path in [(4, cache / 'ranking_flat112.parquet'), (8, cache8 / 'ranking_flat224.parquet')]:
        start = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, use_threads=False):
            saved = batch.to_pandas()
            block = neighbors[start:start + len(saved)]
            result = np.empty((len(saved), depth * 2, len(fields)), dtype='float32')
            for phase_index, is_dep in enumerate((True, False)):
                offsets = np.arange(32)[None, :]
                order = np.sort(np.where((block >= 0) & (dep[np.maximum(block, 0)] == is_dep), offsets, -1), axis=1)[:, -depth:][:, ::-1]
                chosen = np.take_along_axis(block, np.maximum(order, 0), axis=1)
                valid = order >= 0
                safe = np.maximum(chosen, 0)
                part = result[:, phase_index * depth:(phase_index + 1) * depth]
                part[:, :, :len(sequence.NUMS)] = numeric[safe]
                times = queries.time_ns.iloc[start:start+len(saved)].to_numpy()[:, None]
                part[:, :, len(sequence.NUMS)] = (times - event_time[safe]) / 1e9
                part[:, :, len(sequence.NUMS) + 1] = np.where(is_dep, np.nan, (times - movement[safe]) / 1e9)
                part[:, :, :len(sequence.NUMS) + 2][~valid] = np.nan
                for j, name in enumerate(eq_names):
                    ev = ev_strings[name].fillna('__ABSENT_EVENT__').to_numpy()[safe]
                    qu = qu_strings[name].iloc[start:start+len(saved)].fillna('__ABSENT_QUERY__').to_numpy()[:, None]
                    part[:, :, len(sequence.NUMS) + 2 + j] = valid & (ev == qu)
                part[:, :, -1] = ~valid
            np.testing.assert_array_equal(saved[ID], queries[ID].iloc[start:start+len(saved)])
            names = [f'flat_{phase}{rank}_{field}' for phase in ('dep', 'arr') for rank in range(1, depth+1) for field in fields]
            assert list(saved.columns) == [ID, *names]
            np.testing.assert_array_equal(saved[names].to_numpy('float32'), result.reshape(len(saved), -1))
            start += len(saved)
            guard()
        assert start == 344841
        checked[str(depth)] = start * depth * 2 * len(fields)
        print('FLAT_VERIFIED', depth, start, flush=True)
    receipt = dict(status='passed', rows=344841, conventions_fields=85,
        raw_event_query_replay='All rows, frozen producer functions; arrival-only completion read',
        neighbor_replay='All rows frozen function; independent direct nearest16 scan on1024 fixed-spread queries',
        independent_flatten_values=checked, source_manifest_sha256=sha(cache / 'manifest.json'),
        flat8_manifest_sha256=sha(cache8 / 'manifest.json'), verifier_sha256=sha(__file__),
        peak_bytes=guard(), elapsed_sec=time.monotonic()-started, cpu_threads=1, gpu_used=False,
        target_or_departure_block_read=False)
    dest.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
