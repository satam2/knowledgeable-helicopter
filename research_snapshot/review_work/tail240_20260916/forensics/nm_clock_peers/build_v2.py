"""Observed NM-offblock peer cohorts; no private departure outcomes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
ID, FID, TIME, PHASE = 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt'
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
NAMES = ['nm', 'est', 'init', 'last', 'sched']
INPUTS = [ID, FID, TIME, PHASE, 'ADEP_mvt', 'STAND_mvt', *CLOCKS]
STATS = ['count', *[n + '_mean_sec' for n in NAMES], 'nm_std_sec', 'exact_nm_tie_fraction',
         *[n + '_finite_count' for n in NAMES[1:]]]
COLUMNS = ['nmpeer_anchor_missing'] + [f'nmpeer_{scope}_{width}m_{s}'
    for scope in ('airport', 'stand') for width in (15, 60) for s in STATS]
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
OUT = ROOT / 'private_runs/tail240_20260916/forensics/nm_clock_peers/cache_v2'
MISSING = np.iinfo(np.int64).min
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert memory.rss < 2 * 1024**3 and peak < 2 * 1024**3, (memory.rss, peak)
    assert psutil.virtual_memory().available >= 8 * 1024**3
    return peak


def ns(values):
    return pd.to_datetime(values, utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def prepare(raw):
    x = raw.loc[raw[PHASE].eq('DEP'), INPUTS].copy()
    assert x[ID].notna().all() and x[ID].is_unique
    t = pd.to_datetime(x[TIME], utc=True)
    assert t.notna().all()
    out = pd.DataFrame({ID:x[ID].to_numpy(), 'flight':x[FID].to_numpy(float),
        'time':ns(t), 'nm_time':ns(x[CLOCKS[0]]), 'month':t.dt.strftime('%Y-%m').to_numpy()})
    for short, column in [('airport', 'ADEP_mvt'), ('stand', 'STAND_mvt')]:
        out[short] = x[column].astype('string').fillna('').str.strip().str.upper().to_numpy()
    out['flightcode'] = pd.factorize([('f', float(f)) if pd.notna(f) else ('m', float(i))
        for f, i in zip(out.flight, out[ID])], sort=False)[0]
    for name, column in zip(NAMES, CLOCKS):
        out[name] = (t - pd.to_datetime(x[column], utc=True)).dt.total_seconds().to_numpy()
    return out


def prefix(values):
    return np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)])


def keys(flight, time):
    value = np.empty(len(time), dtype=[('flight', 'i8'), ('time', 'i8')])
    value['flight'], value['time'] = flight, time
    return value


def summarize(query, events, width):
    result = np.full((len(query), len(STATS)), np.nan)
    result[:, 0] = 0.
    result[:, 8:12] = 0.
    if events.empty:
        return result
    e = events.sort_values(['nm_time', ID], kind='stable')
    times, codes = e.nm_time.to_numpy(np.int64), e.flightcode.to_numpy(np.int64)
    qt, qf = query.nm_time.to_numpy(np.int64), query.flightcode.to_numpy(np.int64)
    values = e[NAMES].to_numpy(float)
    valid = np.isfinite(values)
    sufficient = np.column_stack([np.ones(len(e)), valid.astype(float), np.where(valid, values, 0.), values[:, 0]**2])
    sums = prefix(sufficient)
    span = np.int64(width * 60 * 10**9)
    lo, hi = np.searchsorted(times, qt-span, 'left'), np.searchsorted(times, qt+span, 'right')
    total = sums[hi] - sums[lo]
    order = np.lexsort((times, codes))
    ordered_keys = keys(codes[order], times[order])
    fs = prefix(sufficient[order])
    flo = np.searchsorted(ordered_keys, keys(qf, qt-span), 'left')
    fhi = np.searchsorted(ordered_keys, keys(qf, qt+span), 'right')
    total -= fs[fhi] - fs[flo]
    tie = (np.searchsorted(times, qt, 'right') - np.searchsorted(times, qt, 'left')).astype(float)
    tie -= np.searchsorted(ordered_keys, keys(qf, qt), 'right') - np.searchsorted(ordered_keys, keys(qf, qt), 'left')
    assert (total[:, :6] >= 0).all() and (tie >= 0).all()
    count = total[:, 0]
    means = np.divide(total[:, 6:11], total[:, 1:6], out=np.full((len(query), 5), np.nan), where=total[:, 1:6] > 0)
    second = np.divide(total[:, 11], count, out=np.full(len(query), np.nan), where=count > 0)
    result[:, 0], result[:, 1:6] = count, means
    result[:, 6] = np.sqrt(np.maximum(0., second - means[:, 0]**2))
    result[:, 7] = np.divide(tie, count, out=np.full(len(query), np.nan), where=count > 0)
    result[:, 8:12] = total[:, 2:6]
    return result


def build_frame(x):
    result = np.full((len(x), len(COLUMNS)), np.nan, dtype=np.float32)
    missing = x.nm_time.eq(MISSING)
    result[:, 0] = missing.to_numpy(np.float32)
    # Airport/flight/movement deduplication precedes stand grouping; missing flight IDs are movement-specific.
    peers = x.sort_values(ID, kind='stable').drop_duplicates(['airport', 'flightcode', 'time'])
    peers = peers.loc[peers.nm_time.ne(MISSING)]
    for si, scope in enumerate(('airport', 'stand')):
        groups = ['airport', 'month'] + (['stand'] if scope == 'stand' else [])
        qvalid = ~missing & x.airport.ne('')
        pvalid = peers.airport.ne('')
        if scope == 'stand':
            qvalid &= x.stand.ne('')
            pvalid &= peers.stand.ne('')
        pool = peers.loc[pvalid]
        grouped = pool.groupby(groups, sort=False, observed=True).indices
        for wi, width in enumerate((15, 60)):
            offset = 1 + (si * 2 + wi) * len(STATS)
            result[:, offset] = 0.
            result[:, offset+8:offset+12] = 0.
            for key, positions in x.loc[qvalid].groupby(groups, sort=False, observed=True).groups.items():
                events = pool.iloc[grouped.get(key, np.array([], dtype=int))]
                result[np.asarray(positions), offset:offset+12] = summarize(x.loc[positions], events, width).astype(np.float32)
    return pd.DataFrame(result, columns=COLUMNS).assign(**{ID:x[ID].to_numpy()})[[ID, *COLUMNS]]


def declare():
    assert len(COLUMNS) == 49 and len(set(COLUMNS)) == 49
    record = dict(source_sha256=common.sha256(__file__), feature_columns=COLUMNS, input_columns=INPUTS,
        raw_hashes=common.read_json(ROOT/'private_runs/submission_v2/protocol.json')['raw_hashes'],
        scope='All original 2025 training departure IDs in raw input order. No ranking cache, model, labels, DEP block or ARR fields.',
        peers='Same departure airport and logical UTC movement month; airport scope and known normalized stand scope. Peer membership sorted on observed NM AOBT N, not takeoff T. Inclusive [queryN-15m,queryN+15m] and [queryN-60m,queryN+60m]. Final retrospective clocks.',
        query_missing='Missing N or unknown grouping key: count and finite-clock counts0; moments/tie fraction NaN. Single explicit query-N-missing flag. No fallback anchor.',
        exclusions='Exclude own movement and all same nonnull FLIGHT_ID. Missing flight IDs are distinct movement identities. Before scope grouping, deduplicate airport/flight/movement timestamp by smallest MVT_ID; all raw query IDs retained.',
        moments='Count, arithmetic mean of five peer takeoff-minus-observed-clocks, population std of peer P=T-N, exact peerN=queryN tie fraction, four nonNM finite-clock counts. Keep negative and long P without clipping. Nonfinite unavailable values omitted perclock; peers require finiteN.',
        chronology='Context logical movement month is retained even if peer N lies outside it. No cross-month carry-in. All query/peer times exact nanoseconds. Ties and both window boundaries included before sameflight exclusion.',
        novelty='N-anchored wave counts already exist and were tested with T. This is N-selected raw peer-clock membership/moments; no claim that all N-anchor features are new. Incremental comparison to strongest387 must follow separate declaration.',
        resources='1CPU; current and OS historical peak below2GiB; >=8GiB host reserve. Full raw hashes verified once, perlogicalmonth Arrow DEP/time predicates across all12packs,49float32 memmap,output streamed.',
        availability='Retrospective observed offblock-event cohort, not real-time data or true surface queue.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print('DECLARED_NM_PEERS49', flush=True)
        return
    assert not (OUT/'features.parquet').exists() and not (OUT/'matrix.float32').exists()
    guard()
    files = sorted(RAW.glob('training_*.parquet'))
    assert len(files) == 12
    all_ids = []
    for path in files:
        assert common.sha256(path) == protocol['raw_hashes'][path.name]
        all_ids.append(pq.read_table(path, columns=[ID], filters=[(PHASE,'=','DEP')], use_threads=False).column(ID).to_numpy())
    ids = np.concatenate(all_ids)
    del all_ids
    assert len(ids) == 2085047 and pd.Index(ids).is_unique
    audit_ids = pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    np.testing.assert_array_equal(ids, audit_ids)
    del audit_ids
    index = pd.Index(ids)
    matrix = np.memmap(OUT/'matrix.float32', mode='w+', dtype='float32', shape=(len(ids), len(COLUMNS)))
    seen = np.zeros(len(ids), dtype=bool)
    records = []
    for month in pd.date_range('2025-01-01', '2025-12-01', freq='MS', tz='UTC'):
        end = month + pd.offsets.MonthBegin(1)
        parts = [pq.read_table(path, columns=INPUTS,
            filters=[(PHASE,'=','DEP'),(TIME,'>=',month.to_pydatetime()),(TIME,'<',end.to_pydatetime())],
            use_threads=False).to_pandas() for path in files]
        raw = pd.concat(parts, ignore_index=True)
        del parts
        x = prepare(raw)
        del raw
        guard()
        features = build_frame(x)
        positions = index.get_indexer(features[ID])
        assert (positions >= 0).all() and not seen[positions].any()
        seen[positions] = True
        matrix[positions] = features[COLUMNS].to_numpy(np.float32)
        matrix.flush()
        records.append(dict(month=month.strftime('%Y-%m'), rows=len(features), missing_anchor=int(features[COLUMNS[0]].sum()),
            supported={scope+'_'+str(w):int(features[f'nmpeer_{scope}_{w}m_count'].gt(0).sum()) for scope in ('airport','stand') for w in (15,60)}, peak_bytes=guard()))
        common.write_json(OUT/'progress.json', records)
        print('BUILT', records[-1], flush=True)
        del x, features
        gc.collect()
    assert seen.all()
    writer = None
    try:
        for start in range(0, len(ids), 8192):
            stop = min(len(ids), start+8192)
            frame = pd.DataFrame(np.asarray(matrix[start:stop]), columns=COLUMNS)
            frame.insert(0, ID, ids[start:stop])
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT/'features.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            guard()
    finally:
        if writer is not None:
            writer.close()
    del frame, table, writer
    gc.collect()
    del matrix
    gc.collect()
    (OUT/'matrix.float32').unlink()
    common.write_json(OUT/'manifest.json', dict(status='complete', source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'), feature_columns=COLUMNS, rows=len(ids), id_hash=common.object_hash(ids.tolist()),
        raw_hashes={p.name:protocol['raw_hashes'][p.name] for p in files}, records=records,
        outputs={'features.parquet':common.sha256(OUT/'features.parquet')}, peak_bytes=guard(), private_targets_read=False))
    print('COMPLETE_NM_PEERS49', len(ids), flush=True)


if __name__ == '__main__':
    main()


