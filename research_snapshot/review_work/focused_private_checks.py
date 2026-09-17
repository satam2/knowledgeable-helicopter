"""Aggregate follow-up checks of schedule regimes and flight-designator transfer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
REPO=(ROOT/"knowledgeable-helicopter").resolve()
RAW=(ROOT/"data/09-15-2026-18-55-03_files_list").resolve()
OUT=(ROOT/"private_analysis").resolve()
assert not RAW.is_relative_to(REPO) and not OUT.is_relative_to(REPO)
COLS=["PHASE_mvt","ADEP_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt","MVT_TIME_UTC_mvt",
      "SCHED_TIME_UTC_mvt","AOBT_3_flt","TAXITIME_SEC_mvt"]


def main():
    parts=[]
    for path in sorted(RAW.glob("*.parquet")):
        if path.name=="submitting.parquet":
            continue
        d=pq.read_table(path,columns=COLS,filters=[("PHASE_mvt","=","DEP")]).to_pandas(strings_to_categorical=True)
        d["kind"]="ranking" if path.name=="ranking.parquet" else "training"
        d["month"]=d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
        d["missing"]=d.AOBT_3_flt.isna()
        d["anchor"]=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
        d["y"]=d.TAXITIME_SEC_mvt
        d["prefix"]=d.FLIGHT_mvt.astype("string").str.extract(r"^([A-Z]{2,3})(?=[0-9])",expand=False)
        for c in ["ADEP_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt"]:
            d[c]=d[c].astype("string")
        parts.append(d[["kind","month","missing","anchor","y","ADEP_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt","prefix"]])
    data=pd.concat(parts,ignore_index=True)
    train=data.loc[data.kind.eq("training")]
    rank=data.loc[data.kind.eq("ranking")]
    missing=train.loc[train.missing]
    schedule={}
    for key,g in [("all_missing",missing),("l irf".replace(" ",""),missing.loc[missing.ADEP_mvt.eq("LIRF")]),("other_airports",missing.loc[~missing.ADEP_mvt.eq("LIRF")])]:
        tail=g.y.gt(7200)
        high=g.anchor.gt(7200)
        schedule[key]={"n":len(g),"tail_n":int(tail.sum()),"high_anchor_n":int(high.sum()),
                       "high_anchor_true_tail_n":int((tail&high).sum()),
                       "tail_recall_pct":float((tail&high).sum()/tail.sum()*100),
                       "tail_precision_pct":float((tail&high).sum()/high.sum()*100),
                       "schedule_within300_tail_pct":float((g.loc[tail,"y"]-g.loc[tail,"anchor"]).abs().le(300).mean()*100),
                       "schedule_within300_nontail_pct":float((g.loc[~tail,"y"]-g.loc[~tail,"anchor"]).abs().le(300).mean()*100)}
    prefix={}
    known=set(train.prefix.dropna())
    for key,g in [("train",train),("ranking",rank),("ranking_missing",rank.loc[rank.missing])]:
        valid=g.prefix.notna()
        prefix[key]={"n":len(g),"parse_pct":float(valid.mean()*100),
                     "new_among_parsed_pct":float((~g.loc[valid,"prefix"].isin(known)).mean()*100),
                     "new_or_unparsed_pct":float((~g.prefix.isin(known)).mean()*100),
                     "exact_designator_new_or_missing_pct":float((g.FLIGHT_mvt.isna()|~g.FLIGHT_mvt.isin(set(train.FLIGHT_mvt.dropna()))).mean()*100)}
    both=train.loc[train.prefix.notna()&train.AIRCRAFT_OPERATOR_flt.notna()]
    pairs=both.groupby(["prefix","AIRCRAFT_OPERATOR_flt"],observed=True).size()
    correct=pairs.groupby(level=0).max().sum()
    prefix["matched_training"]={"n":len(both),"prefix_equals_operator_pct":float(both.prefix.eq(both.AIRCRAFT_OPERATOR_flt).mean()*100),
                                 "in_sample_modal_operator_agreement_pct":float(correct/len(both)*100)}
    monthly_tail=[]
    for (month,airport),g in missing.groupby(["month","ADEP_mvt"],observed=True):
        if airport=="LIRF":
            monthly_tail.append({"month":month,"missing_n":len(g),"tail_n":int(g.y.gt(7200).sum()),
                                 "schedule_within300_pct":float((g.y-g.anchor).abs().le(300).mean()*100)})
    result={"schedule_regimes":schedule,"flight_prefix":prefix,"l irf_monthly".replace(" ",""):monthly_tail,
            "note":"Descriptive profiles only. Target-defined regimes and in-sample operator mapping are not deployable or validated classifiers."}
    (OUT/"focused_checks.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
