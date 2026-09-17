"""No-GPU tests for fitted encoders, anchor availability and metadata serialization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from encoders import FrameEncoder,json_safe
from run_full import anchors,CLOCK_ORDER


def main():
    train=pd.DataFrame({"cat":pd.Categorical(["B","A",None],categories=["A","B","NEVER_OBSERVED"]),
        "n":[0.,2.,-999999.]})
    future=pd.DataFrame({"cat":pd.Categorical(["B","NEW",None]),"n":[1.,np.inf,np.nan]})
    encoder=FrameEncoder().fit(train)
    assert "NEVER_OBSERVED" not in encoder.categories["cat"]
    transformed=encoder.transform(future)
    assert np.isinf(future.loc[1,"n"])
    assert train.loc[2,"n"]==-999999.
    assert transformed["cat"].astype(int).tolist()==[3,1,0]
    assert transformed["n"].isna().tolist()==[False,True,True]
    neural=FrameEncoder().fit(train,neural=True)
    numbers,categories=neural.transform(future)
    assert neural.medians.tolist()==[1.]
    assert np.isfinite(numbers).all() and np.array_equal(numbers[:,1],[0,1,1])
    assert np.array_equal(categories[:,0],[3,1,0])
    assert json.dumps(json_safe({"missing":float("nan"),"array":[float("inf")]}),allow_nan=False)
    x=pd.DataFrame({"takeoff_minus_"+clock:[-999999.,-999999.,-999999.,-999999.] for clock in CLOCK_ORDER})
    x["takeoff_minus_AOBT_3_flt"]=[-20.,9000.,-999999.,-999999.]
    x["takeoff_minus_EOBT_1_flt"]=[50.,50.,100.,-999999.]
    x["takeoff_minus_SCHED_TIME_UTC_mvt"]=[200.,200.,200.,-300.]
    meta=pd.DataFrame({"proxy_sec":[-20.,9000.,np.nan,np.nan]})
    _,a,e,_=anchors(x,meta,"aobt_allfinite")
    assert np.array_equal(e,[True,True,False,False]) and a[:2].tolist()==[-20.,9000.]
    _,a,e,_=anchors(x,meta,"multi_anchor")
    assert e.all() and a.tolist()==[-20.,9000.,100.,-300.]
    _,a,e,_=anchors(x,meta,"direct")
    assert e.all() and not a.any()
    print("PASS fit-only encoder, unknown/missing, original-frame isolation, JSON, negative/long/missing multi-anchor contracts",flush=True)


if __name__=="__main__":
    main()
