"""Independent target-free surface join and event-feature reconstruction."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import bisect
from collections import Counter
import numpy as np
import pandas as pd
import psutil
import validate_candidate as v

BASE = v.ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot'
OUT = v.ROOT / 'private_runs/tail240_20260916/validation/surface_join_v2'
ID, TIME = v.ID, v.common.MOVEMENT


def guard():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, 'peak_wset', 0))
    assert peak < 2 * 1024**3 and psutil.virtual_memory().available >= 8 * 1024**3
    return peak


def normalize(s):
    return s.astype('string').fillna('').str.upper().str.replace(r'\s+', '', regex=True)


def main():
    assert psutil.virtual_memory().available >= 10 * 1024**3
    OUT.mkdir(parents=True, exist_ok=False)
    joined = BASE / 'join_v2'
    source = v.ROOT / 'review_work/tail240_20260916/forensics/surface_join_v2.py'
    marker = v.read_json(joined / 'manifest.json')
    protocol = v.read_json(joined / 'protocol.json')
    assert marker['status'] == 'complete'
    assert marker['protocol_sha256'] == v.sha256(joined / 'protocol.json')
    assert marker['source_sha256'] == protocol['source_sha256'] == v.sha256(source)
    for name, digest in marker['outputs'].items():
        assert v.sha256(joined / name) == digest
    inspection = BASE / 'inspection_v2'
    assert v.sha256(inspection / 'manifest.json') == protocol['inspection_manifest_sha256']
    for name, digest in v.read_json(inspection / 'manifest.json')['outputs'].items():
        assert v.sha256(inspection / name) == digest
    allowed = [ID, TIME, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt']
    assert protocol['query_columns'] == allowed
    raw = v.common.RAW / 'training_2025-01-01_2025-02-01.parquet'
    assert v.sha256(raw) == protocol['raw_sha256']
    q = pd.read_parquet(raw, columns=allowed)
    q = q.loc[q.PHASE_mvt.eq('DEP') & q[TIME].ge(pd.Timestamp('2025-01-05', tz='UTC')) & q[TIME].lt(pd.Timestamp('2025-01-15', tz='UTC'))].copy()
    assert q[ID].is_unique
    for name in ['ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt']:
        q[name] = normalize(q[name])
    q['match_mode'] = np.where(q.CALLSIGN_flt.eq(''), 'movement_fallback', 'nm_preferred')
    q['match_code'] = q.CALLSIGN_flt.mask(q.CALLSIGN_flt.eq(''), q.FLIGHT_mvt)
    public = pd.read_parquet(inspection / 'public_origin_flights.parquet')
    assert public.flight_id_exact.is_unique and public.flight_version.eq('v2.0.0').all()
    for name in ['flt_id', 'adep', 'ades', 'icao24']:
        public[name] = normalize(public[name])
    public = public.loc[public.first_seen.notna() & public.last_seen.gt(public.first_seen) & public.icao24.ne('') & public.flt_id.ne('')].copy()
    wanted = set(zip(q.match_code, q.ADEP_mvt, q.ADES_mvt))
    lookup = {}
    for row in public.itertuples(index=False):
        key = (row.flt_id, row.adep, row.ades)
        if key in wanted:
            lookup.setdefault(key, []).append((pd.Timestamp(row.first_seen).value, str(row.flight_id_exact)))
    for key in lookup:
        lookup[key].sort()
    counts, ids = [], []
    width = 1800 * 10**9
    for row in q.itertuples(index=False):
        possibilities = lookup.get((row.match_code, row.ADEP_mvt, row.ADES_mvt), [])
        now = pd.Timestamp(getattr(row, TIME)).value
        # Tuple bounds retain exact public identifiers; no uint64-to-float path.
        lo = bisect.bisect_left(possibilities, (now - width, ''))
        hi = bisect.bisect_right(possibilities, (now + width, '\uffff'))
        counts.append(hi - lo)
        ids.append(possibilities[lo][1] if hi - lo == 1 else '')
    reuse = Counter(i for i in ids if i)
    q['candidate_count'] = counts
    q['public_match_reused'] = [bool(i and reuse[i] > 1) for i in ids]
    q['flight_id_exact'] = [i if not i or reuse[i] == 1 else '' for i in ids]
    q = q.merge(public[['flight_id_exact', 'first_seen', 'last_seen', 'icao24', 'flight_version']], on='flight_id_exact', how='left', validate='many_to_one', sort=False)
    q['public_matched'] = q.flight_id_exact.ne('')
    actual = pd.read_parquet(joined / 'queries.parquet')
    pd.testing.assert_frame_equal(q[actual.columns], actual, check_dtype=False, check_index_type=False)
    assert v.object_hash(q[ID].tolist()) == marker['query_id_hash']
    guard()
    events = pd.read_parquet(inspection / 'airport_events.parquet', columns=['event_id_exact', 'flight_id_exact', 'type', 'event_time', 'event_airport', 'distance_to_airport_m', 'source', 'version', 'info'])
    assert events.event_id_exact.is_unique and events.version.eq('v0.0.2').all()
    events = events.merge(q.loc[q.public_matched, [ID, TIME, 'ADEP_mvt', 'flight_id_exact', 'first_seen', 'last_seen']], on='flight_id_exact', how='inner', validate='many_to_one')
    events = events.loc[events.event_airport.eq(events.ADEP_mvt)].copy()
    assert events.distance_to_airport_m.le(10000).all()
    events['within_segment'] = events.event_time.ge(events.first_seen) & events.event_time.le(events.last_seen)
    events['no_later_than_takeoff'] = events.event_time.le(events[TIME])
    events['offset_sec'] = (events[TIME] - events.event_time).dt.total_seconds()
    events['eligible_parking_exit'] = events.type.eq('exit-parking_position') & events.within_segment & events.no_later_than_takeoff
    observed = pd.read_parquet(joined / 'event_witnesses.parquet')
    pd.testing.assert_frame_equal(events[observed.columns].reset_index(drop=True), observed.reset_index(drop=True), check_dtype=False)
    features = q[[ID, 'ADEP_mvt', 'match_mode', 'candidate_count', 'public_match_reused', 'public_matched']].copy().set_index(ID)
    features['opdi_first_seen_offset_sec'] = (q[TIME] - q.first_seen).dt.total_seconds().to_numpy()
    for name, subset in [('matched_event_count', events), ('parking_exit_count', events.loc[events.type.eq('exit-parking_position')]), ('eligible_parking_exit_count', events.loc[events.eligible_parking_exit])]:
        features[name] = subset.groupby(ID).size().reindex(features.index).fillna(0).astype('int32')
    features['parking_exit_chronology_rejected_count'] = features.parking_exit_count - features.eligible_parking_exit_count
    stats = events.loc[events.eligible_parking_exit].groupby(ID).offset_sec.agg(['min', 'max'])
    features['exit_offset_min_sec'] = stats['min'].reindex(features.index)
    features['exit_offset_max_sec'] = stats['max'].reindex(features.index)
    features['unique_exit_offset_sec'] = features.exit_offset_min_sec.where(features.eligible_parking_exit_count.eq(1))
    observed = pd.read_parquet(joined / 'features.parquet')
    pd.testing.assert_frame_equal(features.reset_index()[observed.columns], observed, check_dtype=False)
    coverage = []
    for (airport, mode), group in features.groupby(['ADEP_mvt', 'match_mode']):
        coverage.append(dict(airport=airport, mode=mode, queries=len(group), matched=int(group.public_matched.sum()), ambiguous_public_candidates=int(group.candidate_count.gt(1).sum()), reused_public_matches=int(group.public_match_reused.sum()), any_parking_exit=int(group.parking_exit_count.gt(0).sum()), eligible_exit=int(group.eligible_parking_exit_count.gt(0).sum()), unique_eligible_exit=int(group.eligible_parking_exit_count.eq(1).sum()), multiple_eligible_exits=int(group.eligible_parking_exit_count.gt(1).sum()), chronology_rejected_events=int(group.parking_exit_chronology_rejected_count.sum())))
    assert coverage == marker['coverage']
    assert len(q) == marker['queries'] and int(q.public_matched.sum()) == marker['total_matched']
    assert int(features.unique_exit_offset_sec.notna().sum()) == marker['unique_exit_rows']
    receipt = dict(status='passed', verifier_sha256=v.sha256(__file__), source_sha256=v.sha256(source), protocol_sha256=v.sha256(joined / 'protocol.json'), manifest_sha256=v.sha256(joined / 'manifest.json'), queries=len(q), matched=int(q.public_matched.sum()), unique_exit_rows=int(features.unique_exit_offset_sec.notna().sum()), witness_rows=len(events), all_query_feature_and_witness_cells_exact=True, coverage_exact=True, private_columns_read=allowed, private_targets_read=False, private_dep_block_read=False, peak_bytes=guard(), limitation='Rebuilds join from frozen public inspection artifacts; does not independently re-extract all 8.4 million public events or certify parking exits as offblock.')
    v.write_json(OUT / 'receipt.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
