"""Retrospective NM-proxy surface state and paired source-cohort contexts."""
from bisect import bisect_left, insort
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,FLIGHT_ID,MOVEMENT,PHASE

OUT=external_path(ROOT/"private_runs/breakthrough_20260916/batch_context")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")
COLS=[ID,FLIGHT_ID,PHASE,MOVEMENT,"AOBT_3_flt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt","WK_TBL_CAT_flt"]
MISSING=-9223372036854775808
POLICY={"availability":"Explicit RETROSPECTIVE final supplied movement batch; not the existing causal pipeline",
 "inventory":"Other observed departure NM offblock <=query<takeoff, nonnegative proxy only; long durations retained and separately counted; not a true surface queue",
 "query_times":"T=supplied takeoff; N=observed NM offblock only when proxy0..7200, otherwise T with fallback flag",
 "context_month":"UTC movement month of query departure; observed events from overlapping raw packs joined then month-isolated",
 "source_windows":"Airport and airport+stand; [T-width,T) strict-prior and [T-width,T+width] two-sided; widths15,60 minutes",
 "source_proxy_summary":"Exact median of finite0..7200 NM taxi proxies; valid support reported; no target used or clipped",
 "duplicate_rule":"Same nonnull FLIGHT_ID+reportingairport+phase+movementtime keeps smallestMVT_ID; missingflightIDs remain distinct",
 "exclusion":"Exclude query's own MVT_ID and every context row with same nonnull FLIGHT_ID",
 "hidden_columns_loaded":[],"thread_budget":1,"memory_limit_gib":4}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def ns(series):return pd.to_datetime(series,utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def prepare(raw):
    x=raw.copy()
    x["airport"]=x.ADEP_mvt.where(x[PHASE].eq("DEP"),x.ADES_mvt).astype("string").fillna("MISSING")
    x["stand"]=x.STAND_mvt.astype("string").fillna("MISSING")
    x["runway"]=x.RUNWAY_mvt.astype("string").fillna("MISSING")
    x["month"]=x[MOVEMENT].dt.strftime("%Y-%m")
    x["time"]=ns(x[MOVEMENT]);x["offblock"]=ns(x.AOBT_3_flt)
    x["flightkey"]=[("f",float(f)) if pd.notna(f) else ("m",float(i)) for f,i in zip(x[FLIGHT_ID],x[ID])]
    x=x.sort_values(ID).drop_duplicates(["flightkey","airport",PHASE,"time"])
    x["proxy"]=(x["time"].to_numpy().astype(float)-x.offblock.to_numpy().astype(float))/1e9
    x.loc[x.offblock.eq(MISSING),"proxy"]=np.nan
    x["wake"]=x.WK_TBL_CAT_flt.map({"L":1,"M":2,"H":3,"J":4}).fillna(0).astype(int)
    return x


def exclusion_map(event):
    groups=defaultdict(list)
    for i,key in enumerate(event.flightkey):groups[key].append(i)
    return groups


def counts(events,queries,timefield,qtime,widths):
    if events.empty:return {w:np.zeros(len(queries),dtype=float) for w in widths}
    e=events.loc[events[timefield].ne(MISSING)].sort_values(timefield).reset_index(drop=True)
    times=e[timefield].to_numpy(np.int64); q=np.asarray(qtime,np.int64)
    right=np.searchsorted(times,q,side="left")
    excluded=exclusion_map(e)
    output={}
    for w in widths:
        left=np.searchsorted(times,q-int(w*60e9),side="left")
        value=(right-left).astype(float)
        for i,key in enumerate(queries.flightkey):
            value[i]-=sum(left[i]<=j<right[i] for j in excluded.get(key,()))
        output[w]=value
    return output


def inventory(events,queries,qtime,long_only=False):
    e=events.loc[events.proxy.ge(0) & (events.proxy.gt(7200) if long_only else True)].reset_index(drop=True)
    if e.empty:return np.zeros(len(queries),dtype=float)
    starts=e.offblock.to_numpy(np.int64);ends=e.time.to_numpy(np.int64);q=np.asarray(qtime,np.int64)
    result=(np.searchsorted(np.sort(starts),q,side="right")-np.searchsorted(np.sort(ends),q,side="right")).astype(float)
    excluded=exclusion_map(e)
    for i,key in enumerate(queries.flightkey):
        result[i]-=sum(starts[j]<=q[i]<ends[j] for j in excluded.get(key,()))
    return result


def previous_wake(events,queries,qtime):
    e=events.sort_values("time").reset_index(drop=True)
    times=e.time.to_numpy(np.int64);q=np.asarray(qtime,np.int64)
    before=np.searchsorted(times,q,side="left")-1
    keys=list(e.flightkey);wakes=e.wake.to_numpy()
    result=np.zeros(len(queries),dtype=float)
    for i,key in enumerate(queries.flightkey):
        j=before[i]
        while j>=0 and keys[j]==key:j-=1
        if j>=0:result[i]=wakes[j]
    return result


def source_statistics(events,queries,width,twosided):
    e=events.sort_values("time").reset_index(drop=True)
    times=e.time.to_numpy(np.int64);q=queries.time.to_numpy(np.int64)
    lower=np.searchsorted(times,q-int(width*60e9),side="left")
    upper=np.searchsorted(times,q+int(width*60e9) if twosided else q,side="right" if twosided else "left")
    proxy=e.proxy.to_numpy(float)
    valid=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    values=np.column_stack([np.ones(len(e)),~np.isfinite(proxy),np.isfinite(proxy)&(proxy<0),proxy>7200,valid]).astype(float)
    prefix=np.vstack([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0)])
    totals=prefix[upper]-prefix[lower]
    excluded=exclusion_map(e)
    for i,key in enumerate(queries.flightkey):
        for j in excluded.get(key,()):
            if lower[i]<=j<upper[i]:totals[i]-=values[j]
    medians=np.full(len(queries),np.nan)
    ordered=np.argsort(q,kind="stable")
    active=[];lo=0;hi=0
    for i in ordered:
        while hi<upper[i]:
            if valid[hi]:insort(active,float(proxy[hi]))
            hi+=1
        while lo<lower[i]:
            if valid[lo]:active.pop(bisect_left(active,float(proxy[lo])))
            lo+=1
        removed=[]
        for j in excluded.get(queries.flightkey.iloc[i],()):
            if lower[i]<=j<upper[i] and valid[j]:
                v=float(proxy[j]);active.pop(bisect_left(active,v));removed.append(v)
        if active:
            mid=len(active)//2
            medians[i]=active[mid] if len(active)%2 else (active[mid-1]+active[mid])/2
        for v in removed:insort(active,v)
    denom=np.where(totals[:,0]>0,totals[:,0],np.nan)
    return {"count":totals[:,0],"missing_nm_share":totals[:,1]/denom,
        "negative_proxy_share":totals[:,2]/denom,"long_proxy_share":totals[:,3]/denom,
        "valid_proxy_count":totals[:,4],"valid_proxy_median_sec":medians}


def build(query_raw,context_raw):
    q=prepare(query_raw)
    # Query IDs remain complete even where a context duplicate is deduplicated.
    if len(q)!=len(query_raw):
        q=prepare(query_raw.assign(**{FLIGHT_ID:np.nan}))
        q[FLIGHT_ID]=query_raw.set_index(ID).loc[q[ID],FLIGHT_ID].to_numpy()
        q["flightkey"]=[("f",float(f)) if pd.notna(f) else ("m",float(i)) for f,i in zip(q[FLIGHT_ID],q[ID])]
    q=q.set_index(ID,drop=False).loc[query_raw[ID]].reset_index(drop=True)
    context=prepare(context_raw)
    out=pd.DataFrame(index=pd.Index(q[ID].to_numpy(),name=ID))
    for (airport,month),positions in q.groupby(["airport","month"],observed=True).indices.items():
        query=q.iloc[positions].reset_index(drop=True)
        local=context.loc[context.airport.eq(airport)&context.month.eq(month)]
        deps=local.loc[local[PHASE].eq("DEP")].copy()
        arrs=local.loc[local[PHASE].eq("ARR")].copy()
        for stage in ("T","N"):
            valid=query.proxy.between(0,7200).to_numpy()
            qt=query.time.to_numpy().copy()
            if stage=="N":qt[valid]=query.offblock.to_numpy()[valid]
            prefix="batch_surface_"+stage+"_"
            block={"nm_proxy_inventory":inventory(deps,query,qt),"long_proxy_inventory":inventory(deps,query,qt,True)}
            monthstart=pd.Timestamp(month+"-01",tz="UTC").value
            block["month_context_age_sec"]=np.maximum(0,(qt-monthstart)/1e9)
            block["query_fallback_to_T"]=np.zeros(len(query)) if stage=="T" else (~valid).astype(float)
            for label,events,field in [("pushbacks",deps.loc[deps.proxy.ge(0)],"offblock"),("discharges",deps,"time"),("landings",arrs,"time")]:
                for w,value in counts(events,query,field,qt,[5,15,30]).items():block[f"{label}_{w}m"]=value
            rwcounts=np.zeros(len(query));wake=np.zeros(len(query));active_runways=np.zeros(len(query))
            for runway,events in deps.groupby("runway",observed=True):
                c=counts(events,query,"time",qt,[15])[15]
                active_runways+=c>0
                is_query=query.runway.eq(runway).to_numpy()
                rwcounts[is_query]=c[is_query]
                if is_query.any():wake[is_query]=previous_wake(events,query.loc[is_query],qt[is_query])
            block["runway_discharge_share_15m"]=np.divide(rwcounts,block["discharges_15m"],out=np.full(len(query),np.nan),where=block["discharges_15m"]>0)
            block["observed_active_runways_15m"]=active_runways
            block["runway_previous_wake"]=wake
            block["runway_previous_wake_heavier"]=(wake>query.wake.to_numpy())&(query.wake.to_numpy()>0)
            block["runway_previous_wake_same"]=(wake==query.wake.to_numpy())&(wake>0)
            for name,value in block.items():out.loc[query[ID],prefix+name]=np.asarray(value,dtype=np.float32)
        for scope in ("airport","stand"):
            groups=[(query,deps)] if scope=="airport" else [(part,deps.loc[deps.stand.eq(stand)]) for stand,part in query.groupby("stand",observed=True)]
            for part,events in groups:
                part=part.reset_index(drop=True)
                for width in (15,60):
                    for mode in ("past","twosided"):
                        features=source_statistics(events,part,width,mode=="twosided")
                        for name,value in features.items():out.loc[part[ID],f"batch_source_{scope}_{mode}_{width}m_{name}"]=np.asarray(value,dtype=np.float32)
    return out.astype(np.float32)


def tests():
    t=pd.Timestamp("2025-07-01T12:00Z")
    rows=[]
    for i,flight,mins,off in [(1,1,0,-10),(2,2,-5,-20),(3,3,5,-5),(4,4,20,10),(5,1,1,-9),(6,2,-5,-20)]:
        rows.append({ID:float(i),FLIGHT_ID:float(flight),PHASE:"DEP",MOVEMENT:t+pd.Timedelta(minutes=mins),"AOBT_3_flt":t+pd.Timedelta(minutes=off),"ADEP_mvt":"AAA","ADES_mvt":"BBB","STAND_mvt":"S1","RUNWAY_mvt":"R1","WK_TBL_CAT_flt":"M"})
    data=pd.DataFrame(rows);query=data.iloc[:1].copy()
    x=build(query,data)
    assert x.batch_surface_T_nm_proxy_inventory.iloc[0]==1
    assert x.batch_surface_T_discharges_15m.iloc[0]==1
    assert x.batch_source_airport_past_15m_count.iloc[0]==1
    assert x.batch_source_airport_twosided_15m_count.iloc[0]==2
    assert x.batch_source_airport_twosided_15m_valid_proxy_median_sec.iloc[0]==750
    assert x.batch_surface_N_nm_proxy_inventory.iloc[0]==1
    outside=data.copy();outside.loc[outside[ID].eq(4),MOVEMENT]=t+pd.Timedelta(minutes=120)
    xx=build(query,outside)
    assert x.batch_source_airport_twosided_15m_count.iloc[0]==xx.batch_source_airport_twosided_15m_count.iloc[0]
    othermonth=data.copy();othermonth[MOVEMENT]-=pd.Timedelta(days=31);othermonth[ID]+=100;othermonth[FLIGHT_ID]+=100
    pd.testing.assert_frame_equal(x,build(query,pd.concat([data,othermonth],ignore_index=True)))
    poisoned=data.assign(TAXITIME_SEC_mvt=123456,BLOCK_TIME_UTC_mvt=t)
    pd.testing.assert_frame_equal(x,build(query,poisoned))
    print("TESTS_PASS inventory,past/twosided,median,self/sameflight,dedup,futureWindow,monthIsolation,hiddenSchema",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--test-only",action="store_true");args=parser.parse_args()
    tests()
    if args.test_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"manifest.json").exists():raise ValueError("Completed batch cache preserved")
    frozen=json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    sourcehash=sha(__file__)
    (OUT/"protocol.json").write_text(json.dumps({"created_utc":now(),"policy":POLICY,"source_sha256":sourcehash},indent=2),encoding="utf-8")
    paths=sorted(RAW.glob("training_*.parquet"));writer=None;records=[];begin=time.monotonic()
    try:
        for i,path in enumerate(paths):
            started=time.monotonic();contextpaths=paths[max(0,i-1):min(len(paths),i+2)];frames=[]
            for p in contextpaths:
                assert sha(p)==frozen["raw_hashes"][p.name]
                frames.append(pq.read_table(p,columns=COLS,use_threads=False).to_pandas())
            raw=frames[contextpaths.index(path)]
            dep=raw.loc[raw[PHASE].eq("DEP")]
            x=build(dep,pd.concat(frames,ignore_index=True))
            assert np.array_equal(x.index,dep[ID])
            table=pa.Table.from_pandas(x.reset_index(),preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(OUT/"training_features.parquet",table.schema,compression="zstd")
            writer.write_table(table)
            peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
            if peak>4*1024**3:raise MemoryError("4GiB memorylimit")
            rec={"file":path.name,"rows":len(x),"features":len(x.columns),"seconds":time.monotonic()-started,"peak_rss_bytes":peak,
                "inventory_T_mean":float(x.batch_surface_T_nm_proxy_inventory.mean()),"inventory_N_mean":float(x.batch_surface_N_nm_proxy_inventory.mean()),
                "own_N_fallback_pct":float(100*x.batch_surface_N_query_fallback_to_T.mean()),"source_airport_15m_median_support_pct":float(100*x.batch_source_airport_twosided_15m_valid_proxy_median_sec.notna().mean())}
            records.append(rec);(OUT/"progress.json").write_text(json.dumps(records,indent=2),encoding="utf-8");print(rec,flush=True)
            del frames,raw,dep,x,table;gc.collect()
    finally:
        if writer is not None:writer.close()
    p=RAW/"ranking.parquet";assert sha(p)==frozen["raw_hashes"][p.name]
    raw=pq.read_table(p,columns=COLS,use_threads=False).to_pandas();dep=raw.loc[raw[PHASE].eq("DEP")]
    x=build(dep,raw);x.reset_index().to_parquet(OUT/"ranking_features.parquet",index=False)
    actual=pd.read_parquet(OUT/"training_features.parquet",columns=[ID]);expected=pd.read_parquet(ROOT/"private_runs/screening_230/data/interim/audit/departures.parquet",columns=[ID])
    assert np.array_equal(actual[ID],expected[ID])
    manifest={"created_utc":now(),"status":"complete","source_sha256":sourcehash,"policy":POLICY,"training_rows":len(actual),"ranking_rows":len(x),"features":list(x),"records":records,
        "outputs":{p.name:sha(p) for p in OUT.glob("*.parquet")},"runtime_sec":time.monotonic()-begin,"training_id_order_verified":True,"tests_passed":True,"prediction_gain_tested":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print("COMPLETE",len(actual),len(x),flush=True)


if __name__=="__main__":main()
