import json
import pandas as pd
import surface_takeoff_join_v2 as subject


def fixtures():
    t=pd.Timestamp('2025-01-06T12:00:00Z')
    public=pd.DataFrame({'flight_id_exact':['18446744073709551614'],'match_code':['ABC001'],'adep':['LSZH'],'ades':['EDDF'],
        'first_seen':pd.Series([t-pd.Timedelta(hours=1)],dtype='datetime64[us, UTC]'),'last_seen':[t+pd.Timedelta(hours=2)]})
    query=pd.DataFrame({subject.ID:[1],subject.TIME:[t],'match_code':['ABC001'],'ADEP_mvt':['LSZH'],'ADES_mvt':['EDDF']})
    events=pd.DataFrame({'flight_id_exact':['18446744073709551614']*2,'event_id_exact':['18446744073709551615','2'],'type':['take-off']*2,
        'event_time':pd.Series([t+pd.Timedelta(seconds=120),t-pd.Timedelta(seconds=120)],dtype='datetime64[us, UTC]'),'event_airport':['LSZH']*2})
    return t,public,query,events


def test_multiple_events_one_exact_identity():
    t,public,query,events=fixtures()
    result=subject.match_anchors(query,public,events)
    assert result.flight_id_exact.tolist()==['18446744073709551614']
    assert result.anchor_event_count.tolist()==[2] and result.candidate_count.tolist()==[1]
    assert result.anchor_offset_min_sec.tolist()==[-120] and result.anchor_offset_max_sec.tolist()==[120]
    assert set(json.loads(result.anchor_event_ids_json.iloc[0]))=={'2','18446744073709551615'}


def test_distinct_id_ambiguity_and_public_reuse_and_bounds():
    t,public,query,events=fixtures()
    extra=public.assign(flight_id_exact='99')
    duplicate_events=events.iloc[:1].assign(flight_id_exact='99',event_id_exact='3')
    ambiguous=subject.match_anchors(query,pd.concat([public,extra]),pd.concat([events,duplicate_events]))
    assert ambiguous.candidate_count.tolist()==[2] and ambiguous.flight_id_exact.eq('').all()
    reused=subject.match_anchors(pd.concat([query,query.assign(**{subject.ID:2})]),public,events)
    assert reused.public_match_reused.all() and reused.flight_id_exact.eq('').all()
    public.first_seen=t+pd.Timedelta(seconds=121)
    assert subject.match_anchors(query,public,events).candidate_count.tolist()==[0]
