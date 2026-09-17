import numpy as np
import pandas as pd
from build_features import prior_features


def test_strict_prior_missing_and_stale():
    weather=pd.DataFrame({'station':['AAA','AAA','AAA'],'valid':pd.to_datetime(['2025-01-01 00:00Z','2025-01-01 01:00Z','2025-01-01 02:00Z']),
                          'signal':[1.,2.,999.]})
    query=pd.DataFrame({'ADEP_mvt':['AAA']*4,'query':pd.to_datetime(['2025-01-01 01:00Z','2025-01-01 01:30Z','2025-01-01 07:00Z',None])})
    result=prior_features(query,weather,['signal'],'query','wx_')
    assert result.wx_signal.iloc[0]==1.
    assert result.wx_signal.iloc[1]==2.
    assert result.wx_signal.iloc[2:].isna().all()
    assert result.wx_available.tolist()==[1,1,0,0]
    future=weather.copy()
    future.loc[2,'signal']=-1234
    changed=prior_features(query,future,['signal'],'query','wx_')
    pd.testing.assert_frame_equal(result,changed)


def test_unknown_station_retains_query_order():
    weather=pd.DataFrame({'station':['AAA'],'valid':pd.to_datetime(['2025-01-01 00:00Z']),'signal':[3.]})
    query=pd.DataFrame({'ADEP_mvt':['BBB','AAA'],'query':pd.to_datetime(['2025-01-01 00:30Z']*2)})
    result=prior_features(query,weather,['signal'],'query','wx_')
    assert np.isnan(result.wx_signal.iloc[0])
    assert result.wx_signal.iloc[1]==3.
