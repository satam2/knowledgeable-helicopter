import numpy as np
import pandas as pd
import following_groups as subject


def frame():
    times=pd.to_datetime(['2025-01-31 23:00','2025-01-31 23:01','2025-01-31 23:02','2025-01-31 23:02','2025-02-01 00:00'],utc=True)
    x=pd.DataFrame({subject.ID:[1.,2.,3.,4.,5.],subject.FLIGHT_ID:[10.,10.,20.,30.,40.],subject.PHASE:'DEP',
        subject.MOVEMENT:times,'ADEP_mvt':'LIRF','RUNWAY_mvt':'25','STAND_mvt':'A'})
    for c in subject.CLOCKS:
        x[c]=times-pd.to_timedelta([500,600,700,900,1000],unit='s')
    return x


def test_strict_future_sameflight_tiedmean_and_month():
    result=subject.build_frame(subject.prepare(frame()))
    assert result.following_runway_window_count.iloc[0]==2
    assert result.following_runway_nearest_count.iloc[0]==2
    assert result.following_runway_nearest_nm_mean_sec.iloc[0]==800
    assert result.following_runway_nearest_gap_sec.iloc[0]==120
    assert result.following_runway_window_count.iloc[2]==0
    assert result.following_runway_window_count.iloc[4]==0


def test_hidden_fields_inert_unknown_groups_and_dedup():
    x=frame()
    baseline=subject.build_frame(subject.prepare(x))
    x['BLOCK_TIME_UTC_mvt']=pd.Timestamp('1900-01-01',tz='UTC')
    x['TAXITIME_SEC_mvt']=1e100
    pd.testing.assert_frame_equal(baseline,subject.build_frame(subject.prepare(x)))
    x.loc[0,'STAND_mvt']=None
    out=subject.build_frame(subject.prepare(x))
    assert out.following_stand_window_count.iloc[0]==0
    assert np.isnan(out.following_stand_nearest_nm_mean_sec.iloc[0])
    duplicate=x.iloc[[2]].copy()
    duplicate[subject.ID]=9.
    out=subject.build_frame(subject.prepare(pd.concat([x,duplicate],ignore_index=True)))
    assert out.following_runway_window_count.iloc[0]==2


def test_horizon_inclusive_and_raw_negative_missing_values():
    x=frame().iloc[:3].copy()
    x[subject.FLIGHT_ID]=[10.,20.,30.]
    x[subject.MOVEMENT]=pd.to_datetime(['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 01:00:01'],utc=True,format='mixed')
    for c in subject.CLOCKS:
        x[c]=x[subject.MOVEMENT]+pd.Timedelta(seconds=50)
    x.loc[1,subject.CLOCKS[0]]=pd.NaT
    out=subject.build_frame(subject.prepare(x))
    assert out.following_stand_window_count.iloc[0]==1
    assert out.following_stand_window_nm_missing_share.iloc[0]==1
    assert out.following_stand_window_est_mean_sec.iloc[0]==-50
    assert out.following_stand_nearest_gap_sec.iloc[0]==3600
