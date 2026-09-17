"""Retrospective exact-journey ARR linkage; no registration/rotation inference."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_information import ROOT, RAW, OUT, ID, FLIGHT_ID, MOVEMENT, PHASE

COLS = [ID,FLIGHT_ID,PHASE,MOVEMENT,"AOBT_3_flt","FLIGHT_mvt","ADEP_mvt","ADES_mvt"]


def frame(path):
    return pq.read_table(path, columns=COLS, use_threads=False).to_pandas()


def match(departures, arrivals):
    dep = departures.dropna(subset=["FLIGHT_mvt","ADEP_mvt","ADES_mvt"]).copy()
    arr = arrivals.dropna(subset=["FLIGHT_mvt","ADEP_mvt","ADES_mvt",FLIGHT_ID,"AOBT_3_flt"]).copy()
    arr = arr.drop_duplicates([FLIGHT_ID,"FLIGHT_mvt","ADEP_mvt","ADES_mvt","AOBT_3_flt"])
    dep["query_day"] = dep[MOVEMENT].dt.floor("D")
    arr["arrival_day"] = arr[MOVEMENT].dt.floor("D")
    pieces = []
    for shift in (-1,0,1):
        a = arr.copy()
        a["query_day"] = a.arrival_day + pd.Timedelta(days=shift)
        joined = dep.merge(a,on=["FLIGHT_mvt","ADEP_mvt","ADES_mvt","query_day"],suffixes=("_dep","_arr"),how="inner")
        proxy = (joined[MOVEMENT+"_dep"] - joined.AOBT_3_flt_arr).dt.total_seconds()
        joined = joined.loc[(proxy >= 0) & (proxy <= 7200)].copy()
        joined["candidate_proxy_sec"] = proxy.loc[joined.index]
        pieces.append(joined)
    candidates = pd.concat(pieces,ignore_index=True).drop_duplicates([ID+"_dep",FLIGHT_ID+"_arr","AOBT_3_flt_arr"])
    counts = candidates.groupby(ID+"_dep").size()
    single_ids = counts.index[counts.eq(1)]
    single = candidates.loc[candidates[ID+"_dep"].isin(single_ids)].copy()
    return candidates, counts, single


def summarize(dep, candidates, counts, single):
    records = {}
    for name, missing in [("missing_nm",True),("known_nm_masked_for_matching",False)]:
        population = dep.loc[dep.AOBT_3_flt.isna().eq(missing)]
        ids = population[ID]
        eligible = counts.reindex(ids,fill_value=0)
        s = single.loc[single[ID+"_dep"].isin(ids)]
        info = {"query_rows":len(population),"at_least_one_candidate":int(eligible.gt(0).sum()),
            "unique_journey_candidate":len(s),"ambiguous_journey_candidates":int(eligible.gt(1).sum()),
            "unique_coverage_pct":float(100*len(s)/max(1,len(population))),
            "arr_landing_after_query":int(s[MOVEMENT+"_arr"].gt(s[MOVEMENT+"_dep"]).sum()),
            "by_airport":{str(k):int(v) for k,v in s.groupby("ADEP_mvt",observed=True).size().items()}}
        if not missing and len(s):
            match_id=s[FLIGHT_ID+"_dep"].eq(s[FLIGHT_ID+"_arr"])
            clock_equal=s.AOBT_3_flt_dep.eq(s.AOBT_3_flt_arr)
            gap=(s.AOBT_3_flt_dep-s.AOBT_3_flt_arr).dt.total_seconds()
            info.update({"original_flight_id_equal_rows":int(match_id.sum()),"original_flight_id_equal_pct":float(100*match_id.mean()),
                "original_aobt_equal_rows":int(clock_equal.sum()),"original_aobt_equal_pct":float(100*clock_equal.mean()),
                "original_aobt_gap_rmse_sec":float(np.sqrt(np.mean(gap**2))),
                "known_match_precision_is_not_proof_for_missing_population":True})
        records[name]=info
    return records


def main():
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),"schema":"exact-flight-route-arr-linkage-v1",
        "match_keys":["FLIGHT_mvt exact string","ADEP_mvt","ADES_mvt"],
        "time_filter":"arrival UTC date within1day of query takeoff; query takeoff - candidate NM AOBT in[0,7200]sec",
        "candidate_uniqueness":"Distinct NM FLIGHT_ID+AOBT per departure after duplicate ARR removal; ambiguous rows excluded",
        "label_independence":"No TAXITIME or BLOCK columns loaded. Known query AOBT/FLIGHT_ID excluded from candidate construction; used only afterward for masked precision audit.",
        "availability":"RETROSPECTIVE supplied-arrival batch only; not current causal model. Arrival landing may occur after query.",
        "scope":"same operational journey linkage, never aircraft registration/rotation", "normalization":"none; exact raw flightnumber first"}
    (OUT/"record_linkage_protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    results={"protocol":protocol,"cohorts":{}}
    for fold,month in [("F1",7),("F3",11),("ranking",None)]:
        if month is None:
            raw=frame(RAW/"ranking.parquet")
            dep=raw.loc[raw[PHASE].eq("DEP")].copy()
            arr=raw.loc[raw[PHASE].eq("ARR")].copy()
            filenames=["ranking.parquet"]
        else:
            paths=[next(RAW.glob(f"training_2025-{m:02d}-01_*.parquet")) for m in (month-1,month,month+1)]
            frames=[frame(path) for path in paths]
            dep=frames[1].loc[frames[1][PHASE].eq("DEP")].copy()
            arr=pd.concat([f.loc[f[PHASE].eq("ARR")] for f in frames],ignore_index=True)
            filenames=[p.name for p in paths]
        candidates,counts,single=match(dep,arr)
        results["cohorts"][fold]={"context_files":filenames,"context_arrivals":len(arr),"query_departures":len(dep),
            "counts":summarize(dep,candidates,counts,single)}
        missing_matches=single.loc[single.AOBT_3_flt_dep.isna()]
        if len(missing_matches):
            # Private row-level matches stay only in the explicitly external private campaign folder.
            missing_matches[[ID+"_dep",ID+"_arr",FLIGHT_ID+"_arr","AOBT_3_flt_arr","candidate_proxy_sec"]].to_parquet(OUT/f"{fold}_exact_missing_links.parquet",index=False)
        print(fold,json.dumps(results["cohorts"][fold]["counts"]),flush=True)
    results["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/"record_linkage_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
