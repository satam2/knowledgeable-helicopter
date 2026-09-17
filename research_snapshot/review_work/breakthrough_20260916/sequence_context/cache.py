"""Ordered supplied-event cache, with no departure outcome column reads."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ID = 'MVT_ID_mvt'
FLIGHT = 'FLIGHT_ID_mvt'
PHASE = 'PHASE_mvt'
MOVEMENT = 'MVT_TIME_UTC_mvt'
BLOCK = 'BLOCK_TIME_UTC_mvt'
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
CATS = ['phase', 'runway', 'stand', 'aircraft', 'wake', 'operator']
NUMS = ['offset_' + c for c in CLOCKS] + ['aobt_second', 'movement_second', 'arrival_duration']
PUBLIC = [ID, FLIGHT, PHASE, MOVEMENT, 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt',
          'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', *CLOCKS]
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
OUT = ROOT / 'private_runs/breakthrough_20260916/sequence_context'
WIDTH_NS = 3600 * 10**9
MIN_RAM = 8 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def memory_guard():
    if psutil.virtual_memory().available < MIN_RAM:
        raise MemoryError('Sequence preparation requires8GiB free host RAM')


def ns(values):
    return pd.to_datetime(values, utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def strings(values):
    return values.astype('string').fillna('<missing>').astype(str).to_numpy()


def query_frame(raw):
    t = pd.to_datetime(raw[MOVEMENT], utc=True)
    if t.isna().any() or raw[ID].isna().any() or not raw[ID].is_unique:
        raise ValueError('Invalid query identity/time')
    return pd.DataFrame({ID: raw[ID].to_numpy(), 'flight': raw[FLIGHT].to_numpy(dtype=float),
        'airport': strings(raw.ADEP_mvt), 'month': t.dt.strftime('%Y-%m').to_numpy(),
        'time_ns': ns(t), 'runway': strings(raw.RUNWAY_mvt), 'stand': strings(raw.STAND_mvt),
        'operator': strings(raw.AIRCRAFT_OPERATOR_flt)})


def events_from_public(raw, arrivals):
    raw = raw[PUBLIC].copy()
    if not raw[ID].is_unique or raw[ID].isna().any():
        raise ValueError('Movement IDs are not unique')
    if not arrivals[ID].is_unique or not arrivals[ID].isin(raw.loc[raw[PHASE].eq('ARR'), ID]).all():
        raise ValueError('Completion input contains non-arrival IDs')
    t = pd.to_datetime(raw[MOVEMENT], utc=True)
    if t.isna().any():
        raise ValueError('Missing movement time')
    dep = raw[PHASE].eq('DEP').to_numpy()
    completed = raw[ID].map(arrivals.set_index(ID)[BLOCK])
    completed = pd.to_datetime(completed, utc=True)
    duration = (completed - t).dt.total_seconds().to_numpy()
    valid_arr = (~dep) & completed.notna().to_numpy() & np.isfinite(duration) & (duration >= 0)
    # Never inspect a departure block column: only the separately filtered ARR table is joined.
    event_time = t.where(dep, completed)
    result = pd.DataFrame({ID: raw[ID].to_numpy(), 'flight': raw[FLIGHT].to_numpy(dtype=float),
        'airport': strings(raw.ADEP_mvt.where(dep, raw.ADES_mvt)),
        'month': t.dt.strftime('%Y-%m').to_numpy(), 'time_ns': ns(event_time),
        'movement_ns': ns(t), 'phase': np.where(dep, 'DEP', 'ARR'),
        'runway': strings(raw.RUNWAY_mvt), 'stand': strings(raw.STAND_mvt),
        'aircraft': strings(raw.AIRCRAFT_TYPE_mvt), 'wake': strings(raw.WK_TBL_CAT_flt),
        'operator': strings(raw.AIRCRAFT_OPERATOR_flt)})
    for clock in CLOCKS:
        values = (t - pd.to_datetime(raw[clock], utc=True)).dt.total_seconds().to_numpy()
        result['offset_' + clock] = np.where(dep, values, np.nan).astype('float32')
    result['aobt_second'] = np.where(dep, pd.to_datetime(raw[CLOCKS[0]], utc=True).dt.second, np.nan).astype('float32')
    result['movement_second'] = t.dt.second.to_numpy(dtype='float32')
    result['arrival_duration'] = np.where(valid_arr, duration, np.nan).astype('float32')
    result = result.loc[dep | valid_arr].sort_values(ID, kind='stable')
    present = result.flight.notna()
    duplicate = result.loc[present].duplicated(['flight', 'airport', 'phase', 'movement_ns'])
    result = result.drop(index=duplicate.index[duplicate]).reset_index(drop=True)
    return result


def neighbor_indices(events, queries):
    result = np.full((len(queries), 32), -1, dtype=np.int32)
    qtime = queries.time_ns.to_numpy(np.int64)
    qflight = queries.flight.to_numpy(float)
    qids = queries[ID].to_numpy()
    times = events.time_ns.to_numpy(np.int64)
    eflight = events.flight.to_numpy(float)
    eids = events[ID].to_numpy()
    for phase_number, phase in enumerate(['DEP', 'ARR']):
        grouped = events.loc[events.phase.eq(phase)].groupby(['airport', 'month'], sort=False).groups
        for key, qpos in queries.groupby(['airport', 'month'], sort=False).indices.items():
            ids = grouped.get(key)
            if ids is None:
                continue
            ids = np.asarray(ids, dtype=np.int64)
            order = np.lexsort((eids[ids], times[ids]))
            ids = ids[order]
            ordered_times = times[ids]
            for start in range(0, len(qpos), 4096):
                qp = qpos[start:start+4096]
                hi = np.searchsorted(ordered_times, qtime[qp], side='left')
                slots = hi[:, None] - 1 - np.arange(32)[None, :]
                candidates = ids[np.maximum(slots, 0)]
                valid = (slots >= 0) & (times[candidates] >= qtime[qp, None] - WIDTH_NS)
                valid &= eids[candidates] != qids[qp, None]
                valid &= ~(np.isfinite(qflight[qp, None]) & (eflight[candidates] == qflight[qp, None]))
                # Stable compaction keeps the closest16 eligible events without sampling context.
                compact = np.argsort(~valid, axis=1, kind='stable')[:, :16]
                chosen = np.take_along_axis(candidates, compact, axis=1)
                ok = np.take_along_axis(valid, compact, axis=1)
                result[qp, phase_number*16:(phase_number+1)*16] = np.where(ok, chosen, -1)
                # A pathological long run of excluded counterparts may require looking beyond32.
                retry = (valid.sum(axis=1) < 16) & (slots[:, -1] > 0)
                retry &= times[candidates[:, -1]] >= qtime[qp] - WIDTH_NS
                for r in np.flatnonzero(retry):
                    old = ids[:hi[r]][::-1]
                    keep = (times[old] >= qtime[qp[r]] - WIDTH_NS) & (eids[old] != qids[qp[r]])
                    if np.isfinite(qflight[qp[r]]):
                        keep &= eflight[old] != qflight[qp[r]]
                    chosen = old[keep][:16]
                    result[qp[r], phase_number*16:(phase_number+1)*16] = -1
                    result[qp[r], phase_number*16:phase_number*16+len(chosen)] = chosen
    valid = result >= 0
    keys = np.where(valid, times[np.maximum(result, 0)], np.iinfo(np.int64).max)
    # Stable ties inherit deterministic phase/time/ID selection, and are explicitly encoded downstream.
    order = np.argsort(keys, axis=1, kind='stable')
    return np.take_along_axis(result, order, axis=1)


def read_public(path):
    public = pq.read_table(path, columns=PUBLIC).to_pandas()
    arrival = pq.read_table(path, columns=[ID, BLOCK], filters=[(PHASE, '=', 'ARR')]).to_pandas()
    if not arrival[ID].isin(public.loc[public[PHASE].eq('ARR'), ID]).all():
        raise AssertionError('ARR predicate boundary failed')
    return public, arrival


def build():
    os.environ['OMP_NUM_THREADS'] = '2'
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    frozen = json.loads((ROOT / 'private_runs/submission_v2/protocol.json').read_text())
    files = sorted(RAW.glob('training_*.parquet'))
    if len(files) != 12:
        raise ValueError('Expected12 training packs')
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        raise ValueError('Existing sequence attempt preserved; inspect before new version')
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {'status': 'building', 'source_sha256': sha(__file__),
        'raw_hashes': {p.name: frozen['raw_hashes'][p.name] for p in files},
        'columns_read_public': PUBLIC, 'columns_read_arrival_filtered': [ID, BLOCK],
        'arrival_predicate': [PHASE, '=', 'ARR'], 'departure_outcomes_read': [],
        'policy': 'last16DEP bytakeoff and16completedARR byinblock, strict<queryT, within60m, sameairport/UTCmovementmonth, excludes self/sameflight; finalNM retrospective',
        'event_numeric': NUMS, 'event_categories': CATS, 'sequence_length': 32,
        'minimum_available_ram_gib': 8, 'records': []}
    write_json(manifest_path, declaration)
    started = time.monotonic()
    try:
        for path in files:
            memory_guard()
            if sha(path) != frozen['raw_hashes'][path.name]:
                raise ValueError('Raw input hash mismatch')
        # Builder preserves raw departure order; the independent verifier binds it to baseline IDs.
        total = 2085047
        neighbors = np.lib.format.open_memmap(OUT / 'neighbors.npy', mode='w+', dtype='int32', shape=(total, 32))
        offset = 0
        event_offset = 0
        for i, path in enumerate(files):
            memory_guard()
            own, own_arr = read_public(path)
            query = query_frame(own.loc[own[PHASE].eq('DEP')])
            months = set(query.month)
            raws, arrivals = [own], [own_arr]
            for j in [i-1, i+1]:
                if 0 <= j < len(files):
                    extra, completion = read_public(files[j])
                    raw_month = pd.to_datetime(extra[MOVEMENT], utc=True).dt.strftime('%Y-%m')
                    extra = extra.loc[raw_month.isin(months)]
                    raws.append(extra)
                    arrivals.append(completion.loc[completion[ID].isin(extra[ID])])
            raw = pd.concat(raws, ignore_index=True).drop_duplicates(ID)
            arr = pd.concat(arrivals, ignore_index=True).drop_duplicates(ID)
            events = events_from_public(raw, arr)
            events = events.loc[events.month.isin(months)].reset_index(drop=True)
            local = neighbor_indices(events, query)
            neighbors[offset:offset+len(query)] = np.where(local >= 0, local + event_offset, -1)
            key = path.stem
            ep, qp = OUT / (key + '_events.parquet'), OUT / (key + '_queries.parquet')
            events.to_parquet(ep, index=False)
            query.to_parquet(qp, index=False)
            declaration['records'].append({'file': path.name, 'query_rows': len(query), 'event_rows': len(events),
                'query_offset': offset, 'event_offset': event_offset, 'events_file': ep.name, 'queries_file': qp.name,
                'events_sha256': sha(ep), 'queries_sha256': sha(qp),
                'no_context_rows': int((local < 0).all(axis=1).sum())})
            offset += len(query)
            event_offset += len(events)
            neighbors.flush()
            write_json(manifest_path, declaration)
            print('CACHE', key, len(query), len(events), 'elapsed', round(time.monotonic()-started, 1), flush=True)
            del own, own_arr, raw, arr, raws, arrivals, events, query, local
            gc.collect()
        if offset != total:
            raise AssertionError('Training query count changed')
        del neighbors
        declaration.update(status='complete', training_rows=offset, event_rows=event_offset,
            neighbors_sha256=sha(OUT / 'neighbors.npy'), runtime_sec=time.monotonic()-started,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))
        write_json(manifest_path, declaration)
    except Exception as error:
        declaration.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        write_json(manifest_path, declaration)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
