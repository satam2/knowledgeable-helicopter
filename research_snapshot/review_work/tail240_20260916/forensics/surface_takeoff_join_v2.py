"""Final fixed takeoff arm: unique flight identity, not unique detection."""
import surface_join_v2 as base
import argparse
import json
import numpy as np
import pandas as pd

ROOT,ID,TIME,common=base.ROOT,base.ID,base.TIME,base.common
FIRST=base.OUT
OUT=ROOT/'private_runs/tail240_20260916/forensics/surface_event_pilot/takeoff_join_v2'
base.OUT=OUT


def declare():
    first=common.read_json(FIRST/'protocol.json')
    assert first['source_sha256']==common.sha256(base.__file__)
    marker=common.read_json(FIRST/'manifest.json')
    assert marker['status']=='complete' and marker['private_targets_read'] is False
    record=dict(first)
    record.update(source_sha256=common.sha256(__file__),imported_source_sha256=common.sha256(base.__file__),
        firstseen_control_manifest_sha256=common.sha256(FIRST/'manifest.json'),
        match='Exactcallsignandbothairports; eventtype take-off nearestorigincenterwithin10km andwithinlinkedFLIGHTfirst/lastbounds; exactEVENT.flight_idtoFLIGHT.id. Atleastone eventwithin+/-120secondsinclusive ofsuppliedT, exactlyoneDISTINCTpublicFLIGHTIDcandidate. MultipleeventsfromsameIDretainedasobservedset. MultipleIDsorIDreusedacrossqueries=>abstainall.',
        bounds='Finalclarificationrequiresqualifyingtakeoffeventwithinlinkedfirst/lastbounds. Allqualifyingcounts/minmaxoffsets/IDsretained; notimechosenoraveraged. Parkingexit eligibility retainsoriginalwithinsegment/<=T rule. Expectedsourceversionlabels observedv0.0.2/v2.0.0.',
        event_candidate_schema='Allqualifyinganchor eventIDs asJSONarrayofexactstrings; eventcount, distinctflightcount, min/maxevent-minus-T andevent-minus-firstseen; noselectedtrueevent.',
        comparison='Final prelabel linkageclarification versus frozenfirstseen±1800control. Strictsingle-eventv1 preserved. No additionaljoinvariants afterthis; freeze botharms beforefitlabeldiagnostic.',
        manifest_source_roles='Inheritedbase.main manifest.source_sha256 binds importedbase; protocol.source_sha256 binds thiswrapper. source_roles.json binds bothplusmanifest/protocol. Noartifactrewriting.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record
    else:common.write_json(path,record)
    return record


def match_anchors(query,public,events):
    wanted=events.loc[events.type.eq('take-off')].merge(public[['flight_id_exact','match_code','adep','ades','first_seen','last_seen']],on='flight_id_exact',how='inner',validate='many_to_one')
    wanted=wanted.loc[wanted.event_airport.eq(wanted.adep)&wanted.event_time.ge(wanted.first_seen)&wanted.event_time.le(wanted.last_seen)].copy()
    assert wanted.event_id_exact.is_unique
    groups={}
    for key,group in wanted.groupby(['match_code','adep','ades'],sort=False):
        group=group.sort_values(['event_time','event_id_exact'],kind='stable')
        groups[key]=(group.event_time.dt.as_unit('ns').astype('int64').to_numpy()//10**9,group)
    records=[]
    for row in query.itertuples(index=False):
        timestamp=pd.Timestamp(getattr(row,TIME));moment=timestamp.value//10**9
        item={ID:getattr(row,ID),'candidate_count':0,'flight_id_exact':'','anchor_event_ids_json':'[]','anchor_event_count':0,
            'anchor_offset_min_sec':np.nan,'anchor_offset_max_sec':np.nan,'anchor_firstseen_gap_min_sec':np.nan,'anchor_firstseen_gap_max_sec':np.nan}
        group=groups.get((row.match_code,row.ADEP_mvt,row.ADES_mvt)) if row.match_code else None
        if group is not None:
            seconds,frame=group
            lo=np.searchsorted(seconds,moment-120,side='left');hi=np.searchsorted(seconds,moment+120,side='right')
            selected=frame.iloc[lo:hi]
            item['candidate_count']=int(selected.flight_id_exact.nunique())
            item['anchor_event_count']=len(selected)
            item['anchor_event_ids_json']=json.dumps(selected.event_id_exact.astype(str).tolist())
            if len(selected):
                offset=(selected.event_time-timestamp).dt.total_seconds()
                gap=(selected.event_time-selected.first_seen).dt.total_seconds()
                item.update(anchor_offset_min_sec=float(offset.min()),anchor_offset_max_sec=float(offset.max()),
                    anchor_firstseen_gap_min_sec=float(gap.min()),anchor_firstseen_gap_max_sec=float(gap.max()))
            if item['candidate_count']==1:item['flight_id_exact']=str(selected.flight_id_exact.iloc[0])
        records.append(item)
    result=pd.DataFrame(records)
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

if __name__=='__main__':
    base.main()
    if (OUT/'manifest.json').exists():
        common.write_json(OUT/'source_roles.json',dict(wrapper_sha256=common.sha256(__file__),imported_base_sha256=common.sha256(base.__file__),
            protocol_sha256=common.sha256(OUT/'protocol.json'),manifest_sha256=common.sha256(OUT/'manifest.json'),
            note='Manifest source_sha256 is inherited base.main producer; protocol source_sha256 is wrapper defining final fixed matching arm.'))
