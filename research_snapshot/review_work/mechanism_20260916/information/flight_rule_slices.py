"""Diagnostic missing-NM source slices; no model fitted or score-derived rule."""
import json
from pathlib import Path
import sys
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_information import ROOT, OUT, RAW, ID, TARGET, PHASE, MOVEMENT


def describe(part, total_sse):
    err = part.prediction_sec.to_numpy() - part[TARGET].to_numpy()
    schedule_gap = part[TARGET].to_numpy() - part.schedule_proxy_sec.to_numpy()
    return {"rows":len(part), "reference_rmse_sec":float(np.sqrt(np.mean(err**2))),
        "reference_mae_sec":float(np.mean(np.abs(err))), "reference_bias_sec":float(np.mean(err)),
        "share_of_missing_sse_pct":float(100*np.sum(err**2)/total_sse),
        "schedule_within60_pct":float(100*np.mean(np.abs(schedule_gap)<=60)),
        "schedule_proxy_rmse_sec":float(np.sqrt(np.mean(schedule_gap**2))),
        "target_over3600_pct":float(100*np.mean(part[TARGET]>3600)),
        "target_median_sec":float(np.median(part[TARGET])),
        "rule":"label-aware diagnostic; not inference routing"}


def main():
    result = {"created_utc":datetime.now(timezone.utc).isoformat(),"diagnostic_only":True,"folds":{}}
    fields=[ID,PHASE,MOVEMENT,"FLIGHT_RULE_mvt","FLIGHT_mvt","AIRCRAFT_TYPE_mvt","ADEP_mvt","AOBT_3_flt","SCHED_TIME_UTC_mvt"]
    for fold, month in [("F1","07"),("F3","11")]:
        rawfile=next(RAW.glob(f"training_2025-{month}-01_*.parquet"))
        raw=pq.read_table(rawfile,columns=fields,use_threads=False).to_pandas()
        refdir=ROOT/"private_runs/next_230/models"/f"clock_and_rome_ensemble_{fold}_s20260910"
        manifest=json.loads((refdir/"manifest.json").read_text())
        predfile=refdir/"score_predictions.parquet"
        assert hashlib.sha256(predfile.read_bytes()).hexdigest()==manifest["outputs"][predfile.name]
        ref=pd.read_parquet(predfile,columns=[ID,TARGET,"prediction_sec"])
        data=ref.merge(raw,on=ID,how="left",validate="one_to_one")
        assert len(data)==len(ref) and data[PHASE].eq("DEP").all()
        data["schedule_proxy_sec"]=(data[MOVEMENT]-data.SCHED_TIME_UTC_mvt).dt.total_seconds()
        missing=data.loc[data.AOBT_3_flt.isna()].copy()
        missing["flight_rule"]=missing.FLIGHT_RULE_mvt.astype("string").fillna("NULL")
        sse=float(np.sum((missing.prediction_sec-missing[TARGET])**2))
        overall=describe(missing,sse)
        rules={str(k):describe(p,sse) for k,p in missing.groupby("flight_rule",observed=True)}
        pairs={str(a)+"/"+str(r):describe(p,sse) for (a,r),p in missing.groupby(["ADEP_mvt","flight_rule"],observed=True)}
        all_rules=data.FLIGHT_RULE_mvt.astype("string").fillna("NULL").value_counts().to_dict()
        result["folds"][fold]={"all_score_rows":len(data),"all_score_rule_counts":{str(k):int(v) for k,v in all_rules.items()},
            "missing_overall":overall,"missing_by_flight_rule":rules,"missing_by_airport_rule":pairs}
    (OUT/"flight_rule_slices.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({fold:{"overall":rec["missing_overall"],"rules":rec["missing_by_flight_rule"]} for fold,rec in result["folds"].items()},indent=2))


if __name__=="__main__":
    main()
