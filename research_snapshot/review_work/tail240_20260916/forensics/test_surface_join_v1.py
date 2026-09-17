import pandas as pd
import surface_join_v1 as subject


def test_exact_id_and_ambiguity_and_reuse():
    t=pd.Timestamp('2025-01-06T12:00:00Z')
    public=pd.DataFrame({'match_code':['ABC01','ABC2','ABC2'],'adep':['LSZH']*3,'ades':['EDDF']*3,
        'first_seen':[t,t,t+pd.Timedelta(seconds=60)],'flight_id_exact':['18446744073709551614','2','3']})
    query=pd.DataFrame({subject.ID:[1,2,3,4],subject.TIME:[t]*4,'match_code':['ABC01','ABC2','NONE','ABC01'],
        'ADEP_mvt':['LSZH']*4,'ADES_mvt':['EDDF']*4})
    first=subject.unique_matches(query.iloc[:3],public)
    assert first.flight_id_exact.tolist()==['18446744073709551614','','']
    assert first.candidate_count.tolist()==[1,2,0]
    reused=subject.unique_matches(query,public)
    assert reused.public_match_reused.tolist()==[True,False,False,True]
    assert reused.flight_id_exact.eq('').all()


def test_normalization_preserves_zero_and_exact_window():
    assert subject.normalize(pd.Series([' abc001 x ',None])).tolist()==['ABC001X','']
    t=pd.Timestamp('2025-01-06T12:00:00Z')
    public=pd.DataFrame({'match_code':['ABC01'],'adep':['LSZH'],'ades':['EDDF'],'first_seen':[t-pd.Timedelta(seconds=1800)],'flight_id_exact':['123']})
    query=pd.DataFrame({subject.ID:[1],subject.TIME:[t],'match_code':['ABC01'],'ADEP_mvt':['LSZH'],'ADES_mvt':['EDDF']})
    assert subject.unique_matches(query,public).flight_id_exact.tolist()==['123']
    query[subject.TIME]=t+pd.Timedelta(seconds=1)
    assert subject.unique_matches(query,public).candidate_count.tolist()==[0]
