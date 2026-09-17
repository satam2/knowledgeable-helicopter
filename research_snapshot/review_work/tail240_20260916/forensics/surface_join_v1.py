"""Frozen target-free January surface-event coverage pilot."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from surface_acquire_v1 import guard
INSPECT = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot/inspection_v2'
OUT = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot/join_v1'
ID, TIME = common.ID, common.MOVEMENT
START, END = '2025-01-05T00:00:00Z', '2025-01-15T00:00:00Z'
QUERY_COLUMNS = [ID, TIME, 'PHASE_mvt','ADEP_mvt','ADES_mvt','CALLSIGN_flt','FLIGHT_mvt']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def normalize(values):
    return values.astype('string').fillna('').str.upper().str.replace(r'\s+', '', regex=True)


def unique_matches(query, public):
    groups = {}
    for key, group in public.groupby(['match_code','adep','ades'], sort=False):
        order = group.first_seen.sort_values(kind='stable').index
        ordered = group.loc[order]
        groups[key] = (ordered.first_seen.astype('int64').to_numpy() // 10**9, ordered.flight_id_exact.astype(str).to_numpy())
    records = []
    for row in query.itertuples(index=False):
        moment = pd.Timestamp(getattr(row, TIME)).value // 10**9
        group = groups.get((row.match_code,row.ADEP_mvt,row.ADES_mvt)) if row.match_code else None
        count, match = 0, ''
        if group is not None:
            seconds, ids = group
            lo = np.searchsorted(seconds, moment-1800, side='left')
            hi = np.searchsorted(seconds, moment+1800, side='right')
            count = int(hi-lo)
            if count == 1:
                match = ids[lo]
        records.append({ID:getattr(row,ID), 'candidate_count':count, 'flight_id_exact':match})
    result = pd.DataFrame(records)
    matched = result.flight_id_exact.ne('')
    reuse = result.loc[matched,'flight_id_exact'].value_counts()
    result['public_match_reused'] = result.flight_id_exact.map(reuse).fillna(0).gt(1)
    result.loc[result.public_match_reused,'flight_id_exact'] = ''
    return result


def declare():
    marker = common.read_json(INSPECT / 'manifest.json')
    assert marker['status'] == 'complete' and marker['duplicate_event_ids'] == 0
    raw = common.RAW / 'training_2025-01-01_2025-02-01.parquet'
    expected = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes'][raw.name]
    protocol = dict(source_sha256=common.sha256(__file__), inspection_manifest_sha256=common.sha256(INSPECT/'manifest.json'), raw_sha256=expected,
        query_columns=QUERY_COLUMNS, period=[START,END], query_scope='Every supplied DEP movement timestamp in closed-open Jan05-Jan15UTC, allairports, missingorambiguousretained.',
        callsign='Nonblank suppliedNM CALLSIGN_flt preferred; else raw FLIGHT_mvt. Uppercaseandwhitespaceonly, noalias/leadingzerochanges. PreserveNMpreferred/fallbackmode.',
        match='Exactcallsignandbothairportcodes; validpublicduration/nonblankICAO24; first_seenwithin+/-1800s inclusively. Exactlyone publicIDcandidate. If onepublicIDservesmultiplequeries, abstainallthosequeries.',
        events='Exactuint64-deriveddecimalstring publicIDs. Pin actualeventversionv0.0.2 andflightversionv2.0.0. Nearestfixedairportwithin10km mustequalqueryorigin. Keepallmatchedsurfaceobservations, includingchronologicalexceptions.',
        exits='Reportallparkingexitsandcount withinpublicsegmentboundsand<=suppliedtakeoff. Onlyexactlyone eligibleexit suppliesuniqueoffset. Multipleexitsnoarbitraryselection; earliest/latestoffsetrangesareobservedsetdiagnostics. No event renamedAOBT.',
        causality='Finalsuppliedbatchretrospectivelinkage; noeventtimeavailabilityclaim. Nearmidnight/fileedgeslimitedtosingledownload.',
        stages='No target/AOBT/privateDEPblock read duringconstruction. Freezehashedfeatures+allqueries+witnesses+coverage before separatelydeclared rawfitperiod residualaudit. No model/expansion/score.',
        resources='1CPU,<2GiBOSpeak,currenthostreserve>=8GiB,start>=10GiBavailable.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == protocol
    else:
        common.write_json(path,protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    protocol=declare()
    if args.declare_only:
        print('DECLARED_TARGET_FREE_SURFACE_JOIN',flush=True)
        return
    assert psutil.virtual_memory().available>=10*1024**3
    assert not (OUT/'features.parquet').exists()
    marker=common.read_json(INSPECT/'manifest.json')
    for name,expected in marker['outputs'].items():
        assert common.sha256(INSPECT/name)==expected
    raw=common.RAW/'training_2025-01-01_2025-02-01.parquet'
    assert common.sha256(raw)==protocol['raw_sha256']
    query=pd.read_parquet(raw,columns=QUERY_COLUMNS,filters=[('PHASE_mvt','=','DEP'),(TIME,'>=',pd.Timestamp(START)),(TIME,'<',pd.Timestamp(END))])
    assert query[ID].is_unique and query[ID].notna().all()
    for name in ['ADEP_mvt','ADES_mvt','CALLSIGN_flt','FLIGHT_mvt']:
        query[name]=normalize(query[name])
    query['match_mode']=np.where(query.CALLSIGN_flt.ne(''),'nm_preferred','movement_fallback')
    query['match_code']=query.CALLSIGN_flt.where(query.CALLSIGN_flt.ne(''),query.FLIGHT_mvt)
    public=pd.read_parquet(INSPECT/'public_origin_flights.parquet')
    assert public.flight_id_exact.is_unique and public.flight_version.eq('v2.0.0').all()
    for name in ['flt_id','adep','ades','icao24']:
        public[name]=normalize(public[name])
    public['match_code']=public.flt_id
    public=public.loc[public.first_seen.notna()&public.last_seen.gt(public.first_seen)&public.icao24.ne('')&public.match_code.ne('')].copy()
    matches=unique_matches(query,public)
    query=query.merge(matches,on=ID,validate='one_to_one',sort=False)
    np.testing.assert_array_equal(query[ID],matches[ID])
    linked=query.merge(public[['flight_id_exact','first_seen','last_seen','icao24','flight_version']],on='flight_id_exact',how='left',validate='many_to_one',sort=False)
    linked['public_matched']=linked.flight_id_exact.ne('')
    events=pd.read_parquet(INSPECT/'airport_events.parquet',columns=['event_id_exact','flight_id_exact','type','event_time','event_airport','distance_to_airport_m','source','version','info'])
    assert events.event_id_exact.is_unique and events.version.eq('v0.0.2').all()
    selected=events.merge(linked.loc[linked.public_matched,[ID,TIME,'ADEP_mvt','flight_id_exact','first_seen','last_seen']],on='flight_id_exact',how='inner',validate='many_to_one')
    selected=selected.loc[selected.event_airport.eq(selected.ADEP_mvt)].copy()
    selected['within_segment']=selected.event_time.ge(selected.first_seen)&selected.event_time.le(selected.last_seen)
    selected['no_later_than_takeoff']=selected.event_time.le(selected[TIME])
    selected['offset_sec']=(selected[TIME]-selected.event_time).dt.total_seconds()
    selected['eligible_parking_exit']=selected.type.eq('exit-parking_position')&selected.within_segment&selected.no_later_than_takeoff
    features=linked[[ID,'ADEP_mvt','match_mode','candidate_count','public_match_reused','public_matched']].copy().set_index(ID)
    features['opdi_first_seen_offset_sec']=(linked[TIME]-linked.first_seen).dt.total_seconds().to_numpy()
    features['matched_event_count']=selected.groupby(ID).size().reindex(features.index).fillna(0).astype('int32')
    exits=selected.loc[selected.type.eq('exit-parking_position')]
    eligible=selected.loc[selected.eligible_parking_exit]
    features['parking_exit_count']=exits.groupby(ID).size().reindex(features.index).fillna(0).astype('int32')
    features['eligible_parking_exit_count']=eligible.groupby(ID).size().reindex(features.index).fillna(0).astype('int32')
    features['parking_exit_chronology_rejected_count']=features.parking_exit_count-features.eligible_parking_exit_count
    stats=eligible.groupby(ID).offset_sec.agg(['min','max','count'])
    features['exit_offset_min_sec']=stats['min'].reindex(features.index)
    features['exit_offset_max_sec']=stats['max'].reindex(features.index)
    features['unique_exit_offset_sec']=features.exit_offset_min_sec.where(features.eligible_parking_exit_count.eq(1))
    coverage=[]
    for (airport,mode),group in features.groupby(['ADEP_mvt','match_mode']):
        coverage.append(dict(airport=airport,mode=mode,queries=len(group),matched=int(group.public_matched.sum()),
            ambiguous_public_candidates=int(group.candidate_count.gt(1).sum()),reused_public_matches=int(group.public_match_reused.sum()),
            any_parking_exit=int(group.parking_exit_count.gt(0).sum()),eligible_exit=int(group.eligible_parking_exit_count.gt(0).sum()),
            unique_eligible_exit=int(group.eligible_parking_exit_count.eq(1).sum()),multiple_eligible_exits=int(group.eligible_parking_exit_count.gt(1).sum()),
            chronology_rejected_events=int(group.parking_exit_chronology_rejected_count.sum())))
    linked.to_parquet(OUT/'queries.parquet',index=False)
    features.reset_index().to_parquet(OUT/'features.parquet',index=False)
    selected.to_parquet(OUT/'event_witnesses.parquet',index=False)
    common.write_json(OUT/'manifest.json',dict(status='complete',source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),
        queries=len(query),query_id_hash=common.object_hash(query[ID].tolist()),coverage=coverage,
        total_matched=int(features.public_matched.sum()),unique_exit_rows=int(features.unique_exit_offset_sec.notna().sum()),
        peak_bytes=guard(),private_targets_read=False,private_dep_block_read=False,
        outputs={name:common.sha256(OUT/name) for name in ['queries.parquet','features.parquet','event_witnesses.parquet']}))
    print('SURFACE_JOIN',len(query),'matched',int(features.public_matched.sum()),'unique_exit',int(features.unique_exit_offset_sec.notna().sum()),'peak',guard(),flush=True)
    print('COVERAGE',coverage,flush=True)


if __name__=='__main__':
    main()
