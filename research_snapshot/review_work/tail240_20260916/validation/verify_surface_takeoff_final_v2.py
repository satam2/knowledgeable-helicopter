"""Independent exact-time takeoff anchor replay; no private labels."""
from collections import Counter
import json
import verify_surface_join as previous
import numpy as np
import pandas as pd

v, ID, TIME = previous.v, previous.ID, previous.TIME
BASE = previous.BASE
OUT = v.ROOT / 'private_runs/tail240_20260916/validation/surface_takeoff_final_v2'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    prior = v.read_json(previous.OUT / 'receipt.json')
    assert prior['status'] == 'passed'
    folder = BASE / 'takeoff_join_v2'
    marker = v.read_json(folder / 'manifest.json')
    protocol = v.read_json(folder / 'protocol.json')
    assert marker['status'] == 'complete'
    assert protocol['source_sha256'] == v.sha256(v.ROOT / 'review_work/tail240_20260916/forensics/surface_takeoff_join_v2.py')
    assert marker['source_sha256'] == protocol['imported_source_sha256']
    assert protocol['imported_source_sha256'] == prior['source_sha256']
    assert marker['protocol_sha256'] == v.sha256(folder / 'protocol.json')
    assert protocol['firstseen_control_manifest_sha256'] == prior['manifest_sha256']
    for name, digest in marker['outputs'].items():
        assert v.sha256(folder / name) == digest
    original = pd.read_parquet(BASE / 'join_v2/queries.parquet')
    query = pd.read_parquet(folder / 'queries.parquet')
    fields = protocol['query_columns'] + ['match_mode', 'match_code']
    pd.testing.assert_frame_equal(original[fields], query[fields])
    public = pd.read_parquet(BASE / 'inspection_v2/public_origin_flights.parquet')
    for name in ['flt_id', 'adep', 'ades', 'icao24']:
        public[name] = previous.normalize(public[name])
    public = public.loc[public.first_seen.notna() & public.last_seen.gt(public.first_seen) & public.icao24.ne('') & public.flt_id.ne('')]
    events = pd.read_parquet(BASE / 'inspection_v2/airport_events.parquet', columns=['event_id_exact', 'flight_id_exact', 'type', 'event_time', 'event_airport', 'distance_to_airport_m', 'source', 'version', 'info'])
    anchor = events.loc[events.type.eq('take-off')].merge(public[['flight_id_exact', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen']], on='flight_id_exact', validate='many_to_one')
    anchor = anchor.loc[anchor.event_airport.eq(anchor.adep) & anchor.event_time.ge(anchor.first_seen) & anchor.event_time.le(anchor.last_seen)].sort_values(['event_time', 'event_id_exact'], kind='stable')
    groups = {k: g for k, g in anchor.groupby(['flt_id', 'adep', 'ades'])}
    rows = []
    for row in query.itertuples(index=False):
        now = pd.Timestamp(getattr(row, TIME))
        group = groups.get((row.match_code, row.ADEP_mvt, row.ADES_mvt))
        matches = [] if group is None else list(group.loc[group.event_time.between(now - pd.Timedelta(seconds=120), now + pd.Timedelta(seconds=120))].itertuples(index=False))
        flight_ids = {str(hit.flight_id_exact) for hit in matches}
        record = dict(candidate_count=len(flight_ids), flight_id_exact=next(iter(flight_ids)) if len(flight_ids) == 1 else '', anchor_event_ids_json=json.dumps([str(hit.event_id_exact) for hit in matches]), anchor_event_count=len(matches), anchor_offset_min_sec=np.nan, anchor_offset_max_sec=np.nan, anchor_firstseen_gap_min_sec=np.nan, anchor_firstseen_gap_max_sec=np.nan)
        if matches:
            offsets = [(hit.event_time-now).total_seconds() for hit in matches]
            gaps = [(hit.event_time-hit.first_seen).total_seconds() for hit in matches]
            record.update(anchor_offset_min_sec=min(offsets), anchor_offset_max_sec=max(offsets), anchor_firstseen_gap_min_sec=min(gaps), anchor_firstseen_gap_max_sec=max(gaps))
        rows.append(record)
    expected = pd.DataFrame(rows)
    reused = Counter(i for i in expected.flight_id_exact if i)
    expected['public_match_reused'] = [bool(i and reused[i] > 1) for i in expected.flight_id_exact]
    expected.loc[expected.public_match_reused, 'flight_id_exact'] = ''
    pd.testing.assert_frame_equal(expected, query[expected.columns], check_dtype=False)
    assert query.public_matched.equals(query.flight_id_exact.ne(''))
    selected = events.merge(query.loc[query.public_matched, [ID, TIME, 'ADEP_mvt', 'flight_id_exact', 'first_seen', 'last_seen']], on='flight_id_exact', validate='many_to_one')
    selected = selected.loc[selected.event_airport.eq(selected.ADEP_mvt)].copy()
    selected['within_segment'] = selected.event_time.ge(selected.first_seen) & selected.event_time.le(selected.last_seen)
    selected['no_later_than_takeoff'] = selected.event_time.le(selected[TIME])
    selected['offset_sec'] = (selected[TIME] - selected.event_time).dt.total_seconds()
    selected['eligible_parking_exit'] = selected.type.eq('exit-parking_position') & selected.within_segment & selected.no_later_than_takeoff
    witnesses = pd.read_parquet(folder / 'event_witnesses.parquet')
    pd.testing.assert_frame_equal(selected[witnesses.columns].reset_index(drop=True), witnesses.reset_index(drop=True), check_dtype=False)
    f = query[[ID, 'ADEP_mvt', 'match_mode', 'candidate_count', 'public_match_reused', 'public_matched']].set_index(ID)
    f['opdi_first_seen_offset_sec'] = (query[TIME] - query.first_seen).dt.total_seconds().to_numpy()
    for name, subset in [('matched_event_count', selected), ('parking_exit_count', selected.loc[selected.type.eq('exit-parking_position')]), ('eligible_parking_exit_count', selected.loc[selected.eligible_parking_exit])]:
        f[name] = subset.groupby(ID).size().reindex(f.index).fillna(0).astype('int32')
    f['parking_exit_chronology_rejected_count'] = f.parking_exit_count - f.eligible_parking_exit_count
    stats = selected.loc[selected.eligible_parking_exit].groupby(ID).offset_sec.agg(['min', 'max'])
    f['exit_offset_min_sec'] = stats['min'].reindex(f.index)
    f['exit_offset_max_sec'] = stats['max'].reindex(f.index)
    f['unique_exit_offset_sec'] = f.exit_offset_min_sec.where(f.eligible_parking_exit_count.eq(1))
    observed = pd.read_parquet(folder / 'features.parquet')
    pd.testing.assert_frame_equal(f.reset_index()[observed.columns], observed, check_dtype=False)
    assert len(query) == marker['queries'] and int(query.public_matched.sum()) == marker['total_matched']
    assert int(f.unique_exit_offset_sec.notna().sum()) == marker['unique_exit_rows']
    receipt = dict(status='passed', source_sha256=marker['source_sha256'], verifier_sha256=v.sha256(__file__), protocol_sha256=marker['protocol_sha256'], manifest_sha256=v.sha256(folder / 'manifest.json'), original_query_fields_exact=True, exact_nanosecond_anchor_candidates_and_fields=True, all_feature_and_witness_cells_exact=True, queries=len(query), matched=int(query.public_matched.sum()), unique_exit_rows=int(f.unique_exit_offset_sec.notna().sum()), witness_rows=len(selected), peak_bytes=previous.guard(), private_targets_read=False, limitation='Frozen public inspection dependency; distinct flight candidates, with repeated same-flight event detections retained. No certification of parking exits as offblock.')
    v.write_json(OUT / 'receipt.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
