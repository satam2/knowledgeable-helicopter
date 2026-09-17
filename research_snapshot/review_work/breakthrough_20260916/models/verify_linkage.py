"""Independent matched-link field checks plus perturbation test; no outcome columns."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT / "review_work/mechanism_20260916/information"))
import record_linkage as linkage
sys.path.insert(0,str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.artifacts import read_json,write_json,sha256,utc_now
from taxiout.schema import ID,TARGET,BLOCK,MOVEMENT,FLIGHT_ID,PHASE


def main():
    assert TARGET not in linkage.COLS and BLOCK not in linkage.COLS
    source=ROOT / "private_runs/mechanism_20260916/information"
    raw=ROOT / "data/09-15-2026-18-55-03_files_list"
    metadata=read_json(source / "record_linkage_results.json")
    results={}
    for fold in ("F1","F3"):
        files=metadata["cohorts"][fold]["context_files"]
        frames=[pd.read_parquet(raw / name,columns=linkage.COLS) for name in files]
        query=frames[1].loc[frames[1][PHASE].eq("DEP")]
        arrivals=pd.concat([f.loc[f[PHASE].eq("ARR")] for f in frames],ignore_index=True)
        links=pd.read_parquet(source / f"{fold}_exact_missing_links.parquet")
        query_lookup=query.set_index(ID)
        arrival_lookup=arrivals.set_index(ID)
        left=query_lookup.loc[links[ID+"_dep"]].reset_index(drop=True)
        right=arrival_lookup.loc[links[ID+"_arr"]].reset_index(drop=True)
        for key in ("FLIGHT_mvt","ADEP_mvt","ADES_mvt"):
            assert np.array_equal(left[key],right[key])
        proxy=(left[MOVEMENT]-right.AOBT_3_flt).dt.total_seconds().to_numpy()
        assert left.AOBT_3_flt.isna().all()
        assert np.array_equal(proxy,links.candidate_proxy_sec)
        assert ((proxy>=0)&(proxy<=7200)).all()
        assert np.array_equal(right[FLIGHT_ID],links[FLIGHT_ID+"_arr"])
        assert (right[MOVEMENT]>left[MOVEMENT]).all()
        perturbed=left.copy()
        perturbed[ID]=links[ID+"_dep"].to_numpy()
        perturbed[FLIGHT_ID]=-12345
        perturbed["AOBT_3_flt"]=pd.Timestamp("1999-01-01",tz="UTC")
        _,_,matched=linkage.match(perturbed,arrivals)
        assert set(matched[ID+"_dep"])==set(links[ID+"_dep"])
        again=matched.set_index(ID+"_dep").loc[links[ID+"_dep"]]
        assert np.array_equal(again.candidate_proxy_sec,links.candidate_proxy_sec)
        results[fold]={"verified_links":len(links),"all_query_nm_missing":True,"exact_route_flight_fields":True,
            "all_arrival_landings_after_query":True,"query_aobt_and_flightid_perturbation_invariant":True,
            "links_sha256":sha256(source / f"{fold}_exact_missing_links.parquet")}
    write_json(ROOT / "private_runs/breakthrough_20260916/models/linkage_validation.json",{
        "created_utc":utc_now(),"status":"verified","cohorts":results,"matching_source_sha256":sha256(Path(linkage.__file__)),
        "script_sha256":sha256(__file__),"no_hidden_columns_loaded":True,
        "limits":["Retrospective batch context, not a causal takeoff-time model.","Masked known-NM precision does not prove matching precision for missing NM rows.","Three ranking matches have arrival before query and require separate ambiguity review.","Checked match-field equality and query-hidden-source independence; no outcome labels used or accuracy evaluated."]})
    print("LINKAGE VERIFIED",results,flush=True)


if __name__=="__main__":
    main()
