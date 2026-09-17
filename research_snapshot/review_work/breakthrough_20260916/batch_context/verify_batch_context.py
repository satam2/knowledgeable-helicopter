"""Independent brute-force checks of saved retrospective source/surface features."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,FLIGHT_ID,MOVEMENT,PHASE

OUT=external_path(ROOT/"private_runs/breakthrough_20260916/batch_context")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")
COLS=[ID,FLIGHT_ID,PHASE,MOVEMENT,"AOBT_3_flt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt","WK_TBL_CAT_flt"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def same(actual,expected,name):
    if not np.isclose(actual,expected,rtol=1e-5,atol=.001,equal_nan=True):
        raise AssertionError(f"{name} mismatch actual={actual} expected={expected}")


def main():
    manifest=json.loads((OUT/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    for name,digest in manifest["outputs"].items():assert sha(OUT/name)==digest
    actual_ranking=pd.read_parquet(OUT/"ranking_features.parquet",columns=[ID])
    raw_ranking=pq.read_table(RAW/"ranking.parquet",columns=[ID,PHASE],use_threads=False).to_pandas()
    assert np.array_equal(actual_ranking[ID],raw_ranking.loc[raw_ranking[PHASE].eq("DEP"),ID])
    del actual_ranking,raw_ranking
    feature=pd.read_parquet(OUT/"training_features.parquet").set_index(ID)
    checks=[]
    for month in (7,11):
        paths=[next(RAW.glob(f"training_2025-{m:02d}-01_*.parquet")) for m in (month-1,month,month+1)]
        frames=[pq.read_table(p,columns=COLS,use_threads=False).to_pandas() for p in paths]
        context=pd.concat(frames,ignore_index=True)
        context["airport"]=context.ADEP_mvt.where(context[PHASE].eq("DEP"),context.ADES_mvt)
        context["fkey"]=context[FLIGHT_ID].astype("string").fillna("movement:"+context[ID].astype("string"))
        context=context.sort_values(ID).drop_duplicates(["fkey","airport",PHASE,MOVEMENT])
        candidates=frames[1].loc[frames[1][PHASE].eq("DEP")]
        chosen=candidates.sample(n=35,random_state=20260916)
        missing=candidates.loc[candidates.AOBT_3_flt.isna()].head(5)
        chosen=pd.concat([chosen,missing]).drop_duplicates(ID)
        for _,row in chosen.iterrows():
            local=context.loc[context.airport.eq(row.ADEP_mvt)&context[MOVEMENT].dt.strftime("%Y-%m").eq(row[MOVEMENT].strftime("%Y-%m"))&context[ID].ne(row[ID])]
            if pd.notna(row[FLIGHT_ID]):local=local.loc[local[FLIGHT_ID].ne(row[FLIGHT_ID])]
            dep=local.loc[local[PHASE].eq("DEP")].copy();arr=local.loc[local[PHASE].eq("ARR")]
            dep["proxy"]=(dep[MOVEMENT]-dep.AOBT_3_flt).dt.total_seconds()
            actual=feature.loc[row[ID]]
            ownproxy=(row[MOVEMENT]-row.AOBT_3_flt).total_seconds() if pd.notna(row.AOBT_3_flt) else np.nan
            for stage in ("T","N"):
                qt=row.AOBT_3_flt if stage=="N" and np.isfinite(ownproxy) and 0<=ownproxy<=7200 else row[MOVEMENT]
                prefix="batch_surface_"+stage+"_"
                same(actual[prefix+"query_fallback_to_T"],int(stage=="N" and not (np.isfinite(ownproxy) and 0<=ownproxy<=7200)),"query_fallback")
                monthstart=pd.Timestamp(row[MOVEMENT].strftime("%Y-%m")+"-01",tz="UTC")
                same(actual[prefix+"month_context_age_sec"],max(0,(qt-monthstart).total_seconds()),"month_age")
                inventory=dep.proxy.ge(0)&dep.AOBT_3_flt.le(qt)&dep[MOVEMENT].gt(qt)
                same(actual[prefix+"nm_proxy_inventory"],inventory.sum(),prefix+"inventory")
                same(actual[prefix+"long_proxy_inventory"],(inventory&dep.proxy.gt(7200)).sum(),prefix+"long_inventory")
                for w in (5,15,30):
                    low=qt-pd.Timedelta(minutes=w)
                    same(actual[prefix+f"pushbacks_{w}m"],(dep.proxy.ge(0)&dep.AOBT_3_flt.ge(low)&dep.AOBT_3_flt.lt(qt)).sum(),"pushbacks")
                    same(actual[prefix+f"discharges_{w}m"],(dep[MOVEMENT].ge(low)&dep[MOVEMENT].lt(qt)).sum(),"discharges")
                    same(actual[prefix+f"landings_{w}m"],(arr[MOVEMENT].ge(low)&arr[MOVEMENT].lt(qt)).sum(),"landings")
                prior=dep.loc[dep[MOVEMENT].lt(qt)].sort_values(MOVEMENT)
                recent=prior.loc[prior[MOVEMENT].ge(qt-pd.Timedelta(minutes=15))]
                query_runway="MISSING" if pd.isna(row.RUNWAY_mvt) else row.RUNWAY_mvt
                runway_count=recent.RUNWAY_mvt.fillna("MISSING").eq(query_runway).sum()
                same(actual[prefix+"runway_discharge_share_15m"],runway_count/len(recent) if len(recent) else np.nan,"runway_share")
                same(actual[prefix+"observed_active_runways_15m"],recent.RUNWAY_mvt.fillna("MISSING").nunique(),"active_runways")
                previous=prior.loc[prior.RUNWAY_mvt.fillna("MISSING").eq(query_runway)]
                wake_map={"L":1,"M":2,"H":3,"J":4}
                wake=wake_map.get(previous.WK_TBL_CAT_flt.iloc[-1],0) if len(previous) else 0
                own_wake=wake_map.get(row.WK_TBL_CAT_flt,0)
                same(actual[prefix+"runway_previous_wake"],wake,"previous_wake")
                same(actual[prefix+"runway_previous_wake_heavier"],int(wake>own_wake and own_wake>0),"wake_heavier")
                same(actual[prefix+"runway_previous_wake_same"],int(wake==own_wake and wake>0),"wake_same")
            for scope in ("airport","stand"):
                peers=dep if scope=="airport" else dep.loc[dep.STAND_mvt.fillna("MISSING").eq("MISSING" if pd.isna(row.STAND_mvt) else row.STAND_mvt)]
                for w in (15,60):
                    for mode in ("past","twosided"):
                        low=row[MOVEMENT]-pd.Timedelta(minutes=w)
                        end=row[MOVEMENT]+pd.Timedelta(minutes=w) if mode=="twosided" else row[MOVEMENT]
                        selected=peers.loc[peers[MOVEMENT].ge(low)&(peers[MOVEMENT].le(end) if mode=="twosided" else peers[MOVEMENT].lt(end))]
                        prefix=f"batch_source_{scope}_{mode}_{w}m_"
                        same(actual[prefix+"count"],len(selected),"source_count")
                        valid=selected.proxy.between(0,7200)
                        same(actual[prefix+"valid_proxy_count"],valid.sum(),"source_valid_count")
                        same(actual[prefix+"valid_proxy_median_sec"],selected.loc[valid,"proxy"].median(),"source_median")
                        same(actual[prefix+"missing_nm_share"],selected.proxy.isna().mean(),"source_missing")
                        same(actual[prefix+"negative_proxy_share"],selected.proxy.lt(0).mean(),"source_negative")
                        same(actual[prefix+"long_proxy_share"],selected.proxy.gt(7200).mean(),"source_long")
        checks.append({"month":f"2025-{month:02d}","queries":len(chosen),"brute_force":"All84 features: T/N inventory, clocks,runway,wake;5/15/30m pushbacks/discharges/landings;airport/stand prior/twosided15/60m count,median,missing,negative,long"})
        print("VERIFIED",checks[-1],flush=True)
    report={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,
        "total_queries":sum(c["queries"] for c in checks),"manifest_sha256":sha(OUT/"manifest.json"),
        "verifier_sha256":sha(__file__),"no_hidden_columns_loaded":True,
        "ranking_id_order_verified":True,"features_checked":84,
        "scope":"Independent brute-force sampled real queries, all84 columns; not a prediction-accuracy test."}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
