"""Second fixed label-free arm: observed takeoff event within 120 seconds."""
import surface_join_v2 as base
from pathlib import Path
import numpy as np
import pandas as pd

ROOT,ID,TIME,common=base.ROOT,base.ID,base.TIME,base.common
FIRST=base.OUT
OUT=ROOT/'private_runs/tail240_20260916/forensics/surface_event_pilot/takeoff_join_v1'
base.OUT=OUT


def declare():
    first=common.read_json(FIRST/'protocol.json')
    assert first['source_sha256']==common.sha256(base.__file__)
    marker=common.read_json(FIRST/'manifest.json')
    assert marker['status']=='complete' and marker['private_targets_read'] is False
    record=dict(first)
    record.update(source_sha256=common.sha256(__file__),imported_source_sha256=common.sha256(base.__file__),
        firstseen_control_manifest_sha256=common.sha256(FIRST/'manifest.json'),
        match='Exactcallsignandbothairports; eventtype take-off nearestorigincenterwithin10km, exactEVENT.flight_idtoFLIGHT.id. Exactlyone eventcandidate within+/-120seconds inclusive of suppliedT. Multipleevents(includingsameflightID) abstain. ReusedpublicIDsacrossqueries abstainall. No first_seenwindowfilter.',
        bounds='Do not assume takeoffeventsfallwithinFLIGHTbounds: recordevent-minus-firstseen andwithinsegment flag. Parkingexit eligibility retainsoriginalwithinsegment/<=T rule. Versionlabels pinobservedv0.0.2/v2.0.0; compatibleIDmapping not assumedcompatibletimebounds.',
        event_candidate_schema='Anchor eventID and flightID exactdecimalstrings; explicitUTCtimestamp, offset, flight-firstseen gap; no private targets/DEPblocks.',
        comparison='Predeclared secondlinkagearm versus immutablefirstseen±1800 control; same49,522queryIDs. Freeze before anylabeldiagnostics. No learnedalias/model/newdownload.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record
    else:common.write_json(path,record)
    return record


def match_anchors(query,public,events):
    wanted=events.loc[events.type.eq('take-off')].merge(public[['flight_id_exact','match_code','adep','ades','first_seen','last_seen']],on='flight_id_exact',how='inner',validate='many_to_one')
    wanted=wanted.loc[wanted.event_airport.eq(wanted.adep)&wanted.event_time.notna()].copy()
    assert wanted.event_id_exact.is_unique
    groups={}
    for key,group in wanted.groupby(['match_code','adep','ades'],sort=False):
        group=group.sort_values('event_time',kind='stable')
        groups[key]=(group.event_time.dt.as_unit('ns').astype('int64').to_numpy()//10**9,group)
    records=[]
    for row in query.itertuples(index=False):
        moment=pd.Timestamp(getattr(row,TIME)).value//10**9
        item={ID:getattr(row,ID),'candidate_count':0,'flight_id_exact':'','anchor_event_id_exact':'',
            'anchor_event_time':pd.NaT,'anchor_minus_takeoff_sec':np.nan,'anchor_minus_firstseen_sec':np.nan,'anchor_within_segment':False}
        group=groups.get((row.match_code,row.ADEP_mvt,row.ADES_mvt)) if row.match_code else None
        if group is not None:
            seconds,frame=group
            lo=np.searchsorted(seconds,moment-120,side='left');hi=np.searchsorted(seconds,moment+120,side='right')
            item['candidate_count']=int(hi-lo)
            if hi-lo==1:
                hit=frame.iloc[lo]
                item.update(flight_id_exact=str(hit.flight_id_exact),anchor_event_id_exact=str(hit.event_id_exact),anchor_event_time=hit.event_time,
                    anchor_minus_takeoff_sec=float((hit.event_time-pd.Timestamp(getattr(row,TIME))).total_seconds()),
                    anchor_minus_firstseen_sec=float((hit.event_time-hit.first_seen).total_seconds()),
                    anchor_within_segment=bool(hit.first_seen<=hit.event_time<=hit.last_seen))
        records.append(item)
    result=pd.DataFrame(records)
    result['anchor_event_time']=pd.to_datetime(result.anchor_event_time,utc=True)
    reuse=result.loc[result.flight_id_exact.ne(''),'flight_id_exact'].value_counts()
    result['public_match_reused']=result.flight_id_exact.map(reuse).fillna(0).gt(1)
    result.loc[result.public_match_reused,'flight_id_exact']=''
    return result


def unique_matches(query,public):
    events=pd.read_parquet(base.INSPECT/'airport_events.parquet',columns=['event_id_exact','flight_id_exact','type','event_time','event_airport','version'])
    assert events.version.eq('v0.0.2').all()
    return match_anchors(query,public,events)


base.declare=declare
base.unique_matches=unique_matches

if __name__=='__main__':base.main()
