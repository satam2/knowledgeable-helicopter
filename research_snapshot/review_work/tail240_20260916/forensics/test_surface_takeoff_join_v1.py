import pandas as pd
import surface_takeoff_join_v1 as subject


def test_long_ground_match_exact_id_and_event_ambiguity():
    t=pd.Timestamp('2025-01-06T12:00:00Z')
    public=pd.DataFrame({'flight_id_exact':['18446744073709551614'],'match_code':['ABC001'],'adep':['LSZH'],'ades':['EDDF'],
        'first_seen':pd.Series([t-pd.Timedelta(hours=1)],dtype='datetime64[us, UTC]'),'last_seen':[t+pd.Timedelta(hours=2)]})
    query=pd.DataFrame({subject.ID:[1],subject.TIME:[t],'match_code':['ABC001'],'ADEP_mvt':['LSZH'],'ADES_mvt':['EDDF']})
    events=pd.DataFrame({'flight_id_exact':['18446744073709551614'],'event_id_exact':['18446744073709551615'],'type':['take-off'],
        'event_time':pd.Series([t+pd.Timedelta(seconds=120)],dtype='datetime64[us, UTC]'),'event_airport':['LSZH']})
    result=subject.match_anchors(query,public,events)
    assert result.flight_id_exact.tolist()==['18446744073709551614']
    assert result.anchor_minus_firstseen_sec.tolist()==[3720]
    assert result.anchor_event_id_exact.tolist()==['18446744073709551615']
    repeated=pd.concat([events,events.assign(event_id_exact='2',event_time=t)],ignore_index=True)
    assert subject.match_anchors(query,public,repeated).flight_id_exact.tolist()==['']
    assert subject.match_anchors(query,public,repeated).candidate_count.tolist()==[2]


def test_reused_flight_abstains_and_outside_bound_flagged():
    t=pd.Timestamp('2025-01-06T12:00:00Z')
    public=pd.DataFrame({'flight_id_exact':['1'],'match_code':['ABC1'],'adep':['LSZH'],'ades':['EDDF'],
        'first_seen':[t+pd.Timedelta(seconds=5)],'last_seen':[t+pd.Timedelta(hours=2)]})
    query=pd.DataFrame({subject.ID:[1,2],subject.TIME:[t,t],'match_code':['ABC1']*2,'ADEP_mvt':['LSZH']*2,'ADES_mvt':['EDDF']*2})
    events=pd.DataFrame({'flight_id_exact':['1'],'event_id_exact':['2'],'type':['take-off'],'event_time':[t],'event_airport':['LSZH']})
    result=subject.match_anchors(query,public,events)
    assert result.public_match_reused.all() and result.flight_id_exact.eq('').all()
    assert not result.anchor_within_segment.any()
