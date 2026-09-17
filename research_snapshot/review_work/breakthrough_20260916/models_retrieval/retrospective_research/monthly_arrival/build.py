"""Raw finite monthly arrival-duration context; no departure block/target reads."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
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
from taxiout.schema import ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK

RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/retrospective_research/monthly_arrival')
QUERY_COLUMNS = [ID, FLIGHT_ID, PHASE, MOVEMENT, 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt']
ARR_COLUMNS = [ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt']
STATS = ['count', 'mean_sec', 'median_sec', 'q10_sec', 'q90_sec']
GROUPS = {scope: [f'monthly_arr_{scope}_{stat}' for stat in STATS] for scope in ['airport', 'stand', 'runway']}
FEATURES = sum(GROUPS.values(), [])
POLICY = {
    'availability': 'Explicit retrospective final supplied batch. Query DEPmovementUTCmonth equals peer ARRlandingUTCmonth; future/tiedcompletion allowed; not completed-before-query or real-time.',
    'duration': 'ARR BLOCK minus landing; every raw finite value including zero and negative anomalies; no upper/lower clipping or target-basedfilter. Missingblock/landing unavailable.',
    'read_boundary': 'DEPmetadata whitelist neverincludesBLOCK orTAXITIME. ARRblock readonlyunderArrow PHASE=ARR predicate and checkedagain.',
    'groups': 'Airport-month; airport-month-stand; airport-month-runway. Missing groupidentifiers havezero count andNaN moments; no implicitairportfallback.',
    'exclusions': 'Same nonmissingflightID or ownmovementID excluded. MissingflightIDs donotmatch oneanother.',
    'duplicates': 'Nonnull unique movementIDs required afterdropping identical whitelisted duplicate records; conflicting sameID rejected. Same nonmissingflight+arrivalairport+landingtimestamp keepssmallestmovementID BEFORE durationavailabilityfilter, so a missingblock smallestID suppresses largerID duplicates; missingflight rowsremaindistinct.',
    'quantiles': 'Exact linear interpolation at(n-1)*q over sorted remaining rawdurations, afterqueryexclusions. Median q=.5.',
    'outputs': 'All15features float32; original fullDEPorder retained fortraining andranking; unsupportedmomentsNaN/count0.',
    'scope': 'Post-publicsource information follow-up; ownimplementation; no copiedupstreamcode; excludedfromfinal9; no predictivegainclaimed.',
    'resources': '1CPU/Arrowthread; sampled guards rejectRSS>2GiB oravailablehost<8GiB; not an enforced OSpeakmemorylimit; noGPU/network/modeltraining.',
}
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def guard():
    if psutil.Process().memory_info().rss > 2 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Monthly arrival builder resource reserve exceeded')


def tokens(series):
    return series.astype('string')


def prepare_queries(frame):
    query = frame[QUERY_COLUMNS].copy()
    if not query[PHASE].eq('DEP').all() or query[ID].isna().any() or not query[ID].is_unique or query[MOVEMENT].isna().any():
        raise ValueError('Require original unique departure query IDs')
    query['month'] = pd.to_datetime(query[MOVEMENT], utc=True).dt.strftime('%Y-%m')
    query['airport'] = tokens(query.ADEP_mvt)
    query['stand'] = tokens(query.STAND_mvt)
    query['runway'] = tokens(query.RUNWAY_mvt)
    return query.reset_index(drop=True)


def prepare_arrivals(frame):
    arrivals = frame[ARR_COLUMNS].copy()
    if not arrivals[PHASE].eq('ARR').all():
        raise ValueError('ARR-only block boundary rejected nonARR row')
    if arrivals[ID].isna().any():
        raise ValueError('Require nonnull arrival movement IDs')
    arrivals = arrivals.drop_duplicates()
    if not arrivals[ID].is_unique:
        raise ValueError('Conflicting duplicate arrival movement ID')
    arrivals['month'] = pd.to_datetime(arrivals[MOVEMENT], utc=True).dt.strftime('%Y-%m')
    arrivals['airport'] = tokens(arrivals.ADES_mvt)
    arrivals['stand'] = tokens(arrivals.STAND_mvt)
    arrivals['runway'] = tokens(arrivals.RUNWAY_mvt)
    arrivals['duration'] = (pd.to_datetime(arrivals[BLOCK], utc=True) - pd.to_datetime(arrivals[MOVEMENT], utc=True)).dt.total_seconds()
    arrivals = arrivals.sort_values(ID, kind='stable')
    known = arrivals[FLIGHT_ID].notna()
    duplicate = arrivals.loc[known].duplicated([FLIGHT_ID, 'airport', MOVEMENT], keep='first')
    arrivals = arrivals.drop(index=duplicate.index[duplicate])
    return arrivals.loc[np.isfinite(arrivals.duration)].reset_index(drop=True)


def remaining_rank(rank, removed):
    position = rank
    while True:
        updated = rank + int(np.searchsorted(removed, position, side='right'))
        if updated == position:
            return position
        position = updated


def summarize(values, removed=()):
    removed = np.unique(np.asarray(removed, dtype=np.int64))
    count = len(values) - len(removed)
    if not count:
        return np.array([0., np.nan, np.nan, np.nan, np.nan])
    total = values.sum(dtype=np.float64) - values[removed].sum(dtype=np.float64)
    quantiles = []
    for quantile in [.5, .1, .9]:
        location = (count - 1) * quantile
        lower, upper = int(np.floor(location)), int(np.ceil(location))
        left = values[remaining_rank(lower, removed)]
        right = values[remaining_rank(upper, removed)]
        quantiles.append(left + (right - left) * (location - lower))
    return np.array([count, total / count, *quantiles])


def build(query_frame, arrival_frame):
    query, arrivals = prepare_queries(query_frame), prepare_arrivals(arrival_frame)
    output = pd.DataFrame(np.nan, index=pd.Index(query[ID], name=ID), columns=FEATURES, dtype='float32')
    for scope, names in GROUPS.items():
        guard()
        output[names[0]] = np.float32(0)
        keys = ['airport', 'month'] + ([] if scope == 'airport' else [scope])
        arrival_groups = arrivals.groupby(keys, observed=True, dropna=True).indices
        for key, positions in query.groupby(keys, observed=True, dropna=True).indices.items():
            if key not in arrival_groups:
                continue
            events = arrivals.iloc[arrival_groups[key]].sort_values(['duration', ID], kind='stable')
            values = events.duration.to_numpy(np.float64)
            result = np.tile(summarize(values), (len(positions), 1))
            flight_positions = events.reset_index(drop=True).groupby(FLIGHT_ID, dropna=True, observed=True).indices
            id_positions = dict(zip(events[ID], range(len(events))))
            subset = query.iloc[positions]
            for row, (flight, ident) in enumerate(zip(subset[FLIGHT_ID], subset[ID])):
                excluded = list(flight_positions.get(flight, ())) if pd.notna(flight) else []
                own = id_positions.get(ident)
                if own is not None:
                    excluded.append(own)
                if excluded:
                    result[row] = summarize(values, excluded)
            output.loc[subset[ID], names] = result.astype(np.float32)
        guard()
    return output


def sources():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [Path(__file__), Path(__file__).with_name('test_build.py')]}


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'protocol.json'
    payload = {'policy': POLICY, 'feature_groups': GROUPS, 'features': FEATURES, 'source_hashes': sources()}
    if protocol.exists():
        if json.loads(protocol.read_text())['declaration'] != payload:
            raise ValueError('Preserve frozen declaration; use newversion')
    else:
        frozen = json.loads((ROOT / 'private_runs/submission_v2/protocol.json').read_text())
        write(protocol, {'created_utc': datetime.now(timezone.utc).isoformat(), 'declaration': payload,
                         'raw_hashes': frozen['raw_hashes'], 'status': 'prepared_not_executed'})
        snapshot = OUT / 'source'
        snapshot.mkdir()
        for source in payload['source_hashes']:
            p = ROOT / source
            (snapshot / p.name).write_bytes(p.read_bytes())
    return protocol


def load_queries(path):
    return pq.read_table(path, columns=QUERY_COLUMNS, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()


def load_arrivals(paths, months):
    filters = []
    for month in months:
        start = pd.Timestamp(month + '-01', tz='UTC')
        end = start + pd.offsets.MonthBegin(1)
        filters.append([(PHASE, '=', 'ARR'), (MOVEMENT, '>=', start.to_pydatetime()), (MOVEMENT, '<', end.to_pydatetime())])
    parts = []
    # Predicate each raw pack, so movement-month boundary spillovers are included.
    for path in paths:
        guard()
        part = pq.read_table(path, columns=ARR_COLUMNS, filters=filters, use_threads=False).to_pandas()
        if not part[PHASE].eq('ARR').all():
            raise AssertionError('Arrow ARR predicate failed')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.prepare_only:
        print('PREPARED', protocol, 'No private rows loaded', flush=True)
        return
    if (OUT / 'manifest.json').exists() or any(OUT.glob('*.parquet')):
        raise ValueError('Preserve existing completed/partial cache')
    start = time.monotonic()
    original_sources = sources()
    frozen = json.loads(protocol.read_text())
    training = sorted(RAW.glob('training_*.parquet'))
    assert len(training) == 12
    ranking = RAW / 'ranking.parquet'
    for path in [*training, ranking]:
        guard()
        assert sha(path) == frozen['raw_hashes'][path.name]
    records, totals = [], {}
    for role, paths in [('training', training), ('ranking', [ranking])]:
        writer = None
        count = 0
        try:
            for path in paths:
                tic = time.monotonic()
                query = load_queries(path)
                months = sorted(pd.to_datetime(query[MOVEMENT], utc=True).dt.strftime('%Y-%m').dropna().unique())
                arrivals = load_arrivals(paths, months)
                output = build(query, arrivals)
                np.testing.assert_array_equal(output.index, query[ID])
                assert list(output) == FEATURES and all(dtype == np.float32 for dtype in output.dtypes)
                table = pa.Table.from_pandas(output.reset_index(), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(OUT / f'{role}_features.parquet', table.schema, compression='zstd')
                writer.write_table(table)
                count += len(output)
                records.append({'file': path.name, 'role': role, 'rows': len(output), 'months': months,
                                'seconds': time.monotonic() - tic, 'rss_bytes': psutil.Process().memory_info().rss,
                                'available_bytes': psutil.virtual_memory().available,
                                'raw_ARR_negative_durations': int(((pd.to_datetime(arrivals[BLOCK], utc=True) - pd.to_datetime(arrivals[MOVEMENT], utc=True)).dt.total_seconds() < 0).sum()),
                                'coverage': {scope: float(output[names[0]].gt(0).mean()) for scope, names in GROUPS.items()}})
                write(OUT / 'progress.json', records)
                print('PACK', records[-1], flush=True)
                del query, arrivals, output, table
                gc.collect()
                guard()
        finally:
            if writer:
                writer.close()
        totals[role] = count
    expected = pq.read_table(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    actual = pq.read_table(OUT / 'training_features.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    np.testing.assert_array_equal(actual, expected)
    expected_ranking = pq.read_table(ranking, columns=[ID], filters=[(PHASE, '=', 'DEP')], use_threads=False).column(ID).to_numpy()
    actual_ranking = pq.read_table(OUT / 'ranking_features.parquet', columns=[ID], use_threads=False).column(ID).to_numpy()
    np.testing.assert_array_equal(actual_ranking, expected_ranking)
    assert totals == {'training': 2085047, 'ranking': 344841}
    assert sources() == original_sources
    for path in [*training, ranking]:
        guard()
        assert sha(path) == frozen['raw_hashes'][path.name]
    guard()
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': sha(protocol), 'source_hashes': original_sources, 'policy': POLICY,
                'features': FEATURES, 'feature_groups': GROUPS, 'training_rows': totals['training'], 'ranking_rows': totals['ranking'],
                'records': records, 'outputs': {p.name: sha(p) for p in OUT.glob('*.parquet')},
                'runtime_sec': time.monotonic() - start, 'original_ID_order_verified': True, 'prediction_gain_tested': False}
    write(OUT / 'manifest.json', manifest)
    write(OUT / 'verification.json', {'status': 'passed', 'manifest_sha256': sha(OUT / 'manifest.json'),
                                     'original_ID_order_verified': True, 'raw_hashes_verified': True,
                                     'independent_raw_oracle': 'Pending separately owned validation; not claimed by builder'})
    print('COMPLETE', totals, flush=True)


if __name__ == '__main__':
    main()
