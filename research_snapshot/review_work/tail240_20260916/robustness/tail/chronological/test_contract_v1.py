"""Temporal and entity contracts for the inherited historical template."""
import run_v1 as run
import numpy as np
import pandas as pd
import pytest


def fixture():
    x=pd.DataFrame(dict(airport=['A']*4,schedule_bucket=['0_30m']*4,flight=['X']*4,
        stand=['1']*4,schedule_hhmm=[600]*4),index=[1,2,3,4])
    times=pd.Series(pd.to_datetime(['2025-01-01','2025-01-31','2025-02-01','2025-03-01'],utc=True),index=x.index)
    return x,times


def test_whole_month_crossfit_and_future_label_mutation():
    x,times=fixture()
    y=np.array([100.,200.,300.,400.])
    a=run.forest.missing.crossfit_templates(x,y,times)
    assert a.loc[1,'template_mean_sec']==900 and a.loc[2,'template_mean_sec']==900
    assert a.loc[3,'template_history_n']==2 and a.loc[4,'template_history_n']==3
    changed=y.copy()
    changed[2:]=1e9
    b=run.forest.missing.crossfit_templates(x,changed,times)
    pd.testing.assert_frame_equal(a.loc[[1,2,3]],b.loc[[1,2,3]])


def test_repeated_known_flight_rejected_nulls_preserved():
    ok=pd.DataFrame({'FLIGHT_ID_mvt':[1.,2.,np.nan,np.nan]})
    assert run.assert_no_known_flight_reuse(ok)['null_flight_ids']==2
    with pytest.raises(ValueError):
        run.assert_no_known_flight_reuse(pd.DataFrame({'FLIGHT_ID_mvt':[1.,1.]}))


def test_prior_rejects_queries_before_fit_cutoff():
    x,times=fixture()
    prior=run.forest.missing.HistoricalTemplate().fit(x,np.arange(4.),times)
    with pytest.raises(ValueError):
        prior.transform(x.iloc[:1],times.iloc[:1])
