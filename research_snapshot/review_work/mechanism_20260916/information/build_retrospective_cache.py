"""Observed-field retrospective linkage/features, with label-free selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_information import ROOT, RAW, OUT, ID, FLIGHT_ID, MOVEMENT, PHASE, external_path

ROOT_OUT = external_path(OUT / "retrospective_v2")
COLS = [ID, FLIGHT_ID, PHASE, MOVEMENT, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "ADEP_flt", "ADES_FILED_flt", "ADES_flt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"]
KEYS = ["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"]
PARAMS = {
    "exact_proxy_sec": [0, 7200], "broad_proxy_sec": [0, 172800],
    "arrival_query_date_gap_days": 1,
    "matching_keys": KEYS, "text_normalization": "none",
    "broad_selection": "minimum absolute candidate EOBT minus query airport schedule; if candidate EOBT missing use IOBT; if missing use candidate AOBT; exact tied minimum is ambiguous and no selected clock",
    "broad_schedule_source_priority": ["EOBT_1_flt", "IOBT_flt", "AOBT_3_flt"],
    "known_query_nm_fields_used_for_matching": False,
    "hidden_columns_loaded": [],
    "availability": "RETROSPECTIVE supplied input, destination ARR may be after query; not the strict-prior causal pipeline",
    "purpose": "candidate recovery and reliability features only, not trusted source identity or a deployment score",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, expected):
    actual = digest(path)
    if actual != expected[path.name]:
        raise ValueError("Raw source hash mismatch")
    return pq.read_table(path, columns=COLS, use_threads=False).to_pandas()


def freeze_exact():
    dest = external_path(OUT / "frozen_exact_v1")
    dest.mkdir(exist_ok=True)
    sources = [HERE/"record_linkage.py", OUT/"record_linkage_protocol.json", OUT/"record_linkage_results.json",
               *sorted(OUT.glob("*_exact_missing_links.parquet"))]
    recorded = {}
    for source in sources:
        target = dest/source.name
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError("Original exact matching artifact changed since freeze")
        else:
            shutil.copyfile(source,target)
        recorded[source.name] = digest(target)
    receipt = dest/"snapshot.json"
    if not receipt.exists():
        receipt.write_text(json.dumps({"created_utc":now(),"hashes":recorded},indent=2),encoding="utf-8")


def generate_candidates(dep, arr):
    # Keep the query's observed NM identifiers/clocks outside the matching frame.
    d = dep[[ID,MOVEMENT,"SCHED_TIME_UTC_mvt",*KEYS]].dropna(subset=KEYS).copy()
    a = arr.dropna(subset=[*KEYS,FLIGHT_ID,"AOBT_3_flt"]).copy()
    a = a.sort_values(ID).drop_duplicates([FLIGHT_ID,*KEYS,"AOBT_3_flt"])
    d["query_day"] = d[MOVEMENT].dt.floor("D")
    a["arrival_day"] = a[MOVEMENT].dt.floor("D")
    pieces = []
    for shift in (-1,0,1):
        shifted = a.copy()
        shifted["query_day"] = shifted.arrival_day + pd.Timedelta(days=shift)
        joined = d.merge(shifted,on=[*KEYS,"query_day"],suffixes=("_dep","_arr"),how="inner")
        proxy = (joined[MOVEMENT+"_dep"]-joined.AOBT_3_flt).dt.total_seconds()
        joined = joined.loc[proxy.between(0,172800)].copy()
        joined["proxy_sec"] = proxy.loc[joined.index]
        schedule_source = joined.EOBT_1_flt.combine_first(joined.IOBT_flt).combine_first(joined.AOBT_3_flt)
        joined["schedule_distance_sec"] = (schedule_source-joined.SCHED_TIME_UTC_mvt_dep).dt.total_seconds().abs()
        pieces.append(joined)
    result=pd.concat(pieces,ignore_index=True)
    return result.drop_duplicates([ID+"_dep",FLIGHT_ID,"AOBT_3_flt"])


def matched_features(dep, candidates):
    x=pd.DataFrame(index=pd.Index(dep[ID].to_numpy(),name=ID))
    selected = {}
    for name, c in [("exact",candidates.loc[candidates.proxy_sec.le(7200)]),("broad",candidates)]:
        counts=c.groupby(ID+"_dep").size()
        x[f"retro_{name}_candidate_count"]=counts.reindex(x.index,fill_value=0).to_numpy(dtype=np.float32)
        if name == "exact":
            chosen=c.loc[c[ID+"_dep"].isin(counts.index[counts.eq(1)])].copy()
            margin=pd.Series(dtype=float)
        else:
            finite=c.loc[c.schedule_distance_sec.notna()].copy()
            finite=finite.sort_values([ID+"_dep","schedule_distance_sec",ID+"_arr"])
            minimum=finite.groupby(ID+"_dep").schedule_distance_sec.transform("min")
            tied=finite.loc[finite.schedule_distance_sec.eq(minimum)].groupby(ID+"_dep").size()
            chosen=finite.drop_duplicates(ID+"_dep").loc[lambda z:z[ID+"_dep"].isin(tied.index[tied.eq(1)])].copy()
            second=finite.groupby(ID+"_dep").nth(1).set_index(ID+"_dep").schedule_distance_sec
            first=finite.drop_duplicates(ID+"_dep").set_index(ID+"_dep").schedule_distance_sec
            margin=second-first
        chosen=chosen.set_index(ID+"_dep",drop=False)
        x[f"retro_{name}_has_selected"] = x.index.isin(chosen.index).astype(np.float32)
        for feature,values in {
            "proxy_sec":chosen.proxy_sec,
            "schedule_distance_sec":chosen.schedule_distance_sec,
            "airtime_sec":(chosen[MOVEMENT+"_arr"]-chosen[MOVEMENT+"_dep"]).dt.total_seconds(),
            "arvt3_minus_landing_sec":(chosen.ARVT_3_flt-chosen[MOVEMENT+"_arr"]).dt.total_seconds(),
        }.items():
            x[f"retro_{name}_{feature}"]=values.reindex(x.index).to_numpy(dtype=np.float32)
        if name=="broad":
            x["retro_broad_second_best_margin_sec"]=margin.reindex(x.index).to_numpy(dtype=np.float32)
        selected[name]=chosen
    return x,selected


def unused_source_features(dep):
    x=pd.DataFrame(index=pd.Index(dep[ID].to_numpy(),name=ID))
    fields = {
        "retro_own_arvt3_minus_takeoff_sec":(dep.ARVT_3_flt-dep[MOVEMENT]).dt.total_seconds(),
        "retro_own_arvt3_minus_arvt1_sec":(dep.ARVT_3_flt-dep.ARVT_1_flt).dt.total_seconds(),
        "retro_own_planned_duration_sec":(dep.ARVT_1_flt-dep.EOBT_1_flt).dt.total_seconds(),
        "retro_own_flown_duration_sec":(dep.ARVT_3_flt-dep.AOBT_3_flt).dt.total_seconds(),
        "source_airport_rule_ifr":dep.FLIGHT_RULE_mvt.eq("I"),
        "source_airport_rule_vfr":dep.FLIGHT_RULE_mvt.eq("V"),
        "source_airport_rule_missing":dep.FLIGHT_RULE_mvt.isna(),
        "source_nm_rule_missing":dep.FLIGHT_RULE_flt.isna(),
        "source_filed_destination_changed":dep.ADES_FILED_flt.notna() & dep.ADES_flt.notna() & dep.ADES_FILED_flt.ne(dep.ADES_flt),
        "source_callsign_flight_exact_equal":dep.CALLSIGN_flt.notna() & dep.FLIGHT_mvt.notna() & dep.CALLSIGN_flt.eq(dep.FLIGHT_mvt),
        "source_nm_record_missing":dep.AOBT_3_flt.isna(),
    }
    for name,values in fields.items():
        x[name]=values.to_numpy(dtype=np.float32)
    x["retro_own_duration_revision_sec"]=x.retro_own_flown_duration_sec-x.retro_own_planned_duration_sec
    x["retro_own_planned_minus_airborne_anchor_sec"]=x.retro_own_planned_duration_sec-x.retro_own_arvt3_minus_takeoff_sec
    for source in ["ADEP","ADES","AIRCRAFT_TYPE"]:
        m,n=dep[source+"_mvt"],dep[source+"_flt"]
        x["source_"+source.lower()+"_agreement"]=(m.notna() & n.notna() & m.eq(n)).to_numpy(dtype=np.float32)
        x["source_"+source.lower()+"_disagreement"]=(m.notna() & n.notna() & m.ne(n)).to_numpy(dtype=np.float32)
    return x


def precision(dep, selected):
    out={}
    query=dep.set_index(ID)
    for name, chosen in selected.items():
        actual=query.reindex(chosen.index)
        known=actual.AOBT_3_flt.notna()
        equal=actual.loc[known,"AOBT_3_flt"].eq(chosen.loc[known,"AOBT_3_flt"])
        same_id=actual.loc[known,FLIGHT_ID].eq(chosen.loc[known,FLIGHT_ID])
        missing=~known
        out[name]={"selected_rows":len(chosen),"selected_missing_nm":int(missing.sum()),"selected_known_nm":int(known.sum()),
            "known_aobt_exact":int(equal.sum()),"known_flightid_exact":int(same_id.sum()),
            "known_aobt_precision_pct":float(100*equal.mean()) if len(equal) else None,
            "missing_arrival_after_query":int((chosen.loc[missing,MOVEMENT+"_arr"]>chosen.loc[missing,MOVEMENT+"_dep"]).sum()),
            "diagnostic_known_precision_not_missing_population_proof":True}
    return out


def build_block(dep, arr):
    c=generate_candidates(dep,arr)
    x,selected=matched_features(dep,c)
    x=x.join(unused_source_features(dep),validate="one_to_one")
    return x,selected,precision(dep,selected)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    if args.test_only:
        tests()
        return
    freeze_exact()
    ROOT_OUT.mkdir(exist_ok=True)
    if (ROOT_OUT/"manifest.json").exists():
        raise ValueError("Completed retrospective cache already exists")
    raw_protocol=json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    protocol={"created_utc":now(),"parameters":PARAMS,"source_sha256":digest(__file__),"raw_hashes":raw_protocol["raw_hashes"]}
    (ROOT_OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    tests()
    records=[]
    files=sorted(RAW.glob("training_*.parquet"))
    writer=None
    begin=time.monotonic()
    try:
        for i,path in enumerate(files):
            started=time.monotonic()
            context=files[max(0,i-1):min(len(files),i+2)]
            data=[read(p,raw_protocol["raw_hashes"]) for p in context]
            raw=data[context.index(path)]
            dep=raw.loc[raw[PHASE].eq("DEP")].copy()
            arr=pd.concat([d.loc[d[PHASE].eq("ARR")] for d in data],ignore_index=True)
            x,selected,info=build_block(dep,arr)
            assert np.array_equal(x.index,dep[ID])
            table=pa.Table.from_pandas(x.reset_index(),preserve_index=False)
            if writer is None:
                writer=pq.ParquetWriter(ROOT_OUT/"training_features.parquet",table.schema,compression="zstd")
            writer.write_table(table)
            peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
            if peak>4*1024**3:
                raise MemoryError("Retrospective build exceeded4GiB")
            records.append({"file":path.name,"context_files":[p.name for p in context],"rows":len(x),"columns":list(x),"precision":info,"seconds":time.monotonic()-started,"peak_rss_bytes":peak})
            (ROOT_OUT/"progress.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
            print(path.name,info,"seconds",round(time.monotonic()-started,2),flush=True)
    finally:
        if writer is not None:
            writer.close()
    raw=read(RAW/"ranking.parquet",raw_protocol["raw_hashes"])
    dep=raw.loc[raw[PHASE].eq("DEP")].copy()
    x,selected,rankinfo=build_block(dep,raw.loc[raw[PHASE].eq("ARR")].copy())
    x.reset_index().to_parquet(ROOT_OUT/"ranking_features.parquet",index=False)
    meta=pd.read_parquet(ROOT/"private_runs/screening_230/data/interim/audit/departures.parquet",columns=[ID])
    ids=pd.read_parquet(ROOT_OUT/"training_features.parquet",columns=[ID])
    assert np.array_equal(ids[ID],meta[ID])
    manifest={"status":"complete","created_utc":now(),"parameters":PARAMS,"training_rows":len(ids),"ranking_rows":len(x),
        "training_id_order_matches_audit":True,"records":records,"ranking_precision":rankinfo,"runtime_sec":time.monotonic()-begin,
        "source_sha256":digest(__file__),"protocol_sha256":digest(ROOT_OUT/"protocol.json"),
        "outputs":{p.name:digest(p) for p in ROOT_OUT.glob("*.parquet")},
        "no_training_labels_loaded":True,"no_prediction_gain_claimed":True,"synthetic_tests_passed":True}
    (ROOT_OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("COMPLETE",manifest["training_rows"],manifest["ranking_rows"],flush=True)


def tests():
    q=pd.DataFrame({ID:[1.,2.,3.],FLIGHT_ID:[100.,np.nan,101.],PHASE:["DEP"]*3,
        MOVEMENT:pd.to_datetime(["2025-07-01T12:00Z"]*3),"SCHED_TIME_UTC_mvt":pd.to_datetime(["2025-07-01T11:00Z"]*3),
        "FLIGHT_mvt":["AA1","AA1","ZZ9"],"ADEP_mvt":["AAA"]*3,"ADES_mvt":["BBB"]*3})
    for col in COLS:
        if col not in q:
            q[col]=pd.NaT if col in ["AOBT_3_flt","EOBT_1_flt","IOBT_flt","ARVT_1_flt","ARVT_3_flt"] else None
    a=q.iloc[:1].copy()
    a[ID]=10.;a[FLIGHT_ID]=100.;a[PHASE]="ARR"
    a[MOVEMENT]=pd.to_datetime(["2025-07-01T14:00Z"])
    a["AOBT_3_flt"]=pd.to_datetime(["2025-07-01T11:40Z"])
    a["EOBT_1_flt"]=pd.to_datetime(["2025-07-01T11:00Z"])
    a["IOBT_flt"]=a.EOBT_1_flt;a["ARVT_1_flt"]=a[MOVEMENT];a["ARVT_3_flt"]=a[MOVEMENT]
    c=generate_candidates(q,a)
    x,_=matched_features(q,c)
    assert x.retro_exact_has_selected.tolist()==[1,1,0]
    assert x.retro_exact_proxy_sec.iloc[0]==1200
    masked=q.copy();masked[FLIGHT_ID]=99999.;masked["AOBT_3_flt"]=pd.to_datetime(["2000-01-01T00:00Z"]*3)
    xx,_=matched_features(masked,generate_candidates(masked,a))
    pd.testing.assert_frame_equal(x,xx)
    duplicate=pd.concat([a,a],ignore_index=True)
    xd,_=matched_features(q,generate_candidates(q,duplicate))
    pd.testing.assert_frame_equal(x,xd)
    other=a.copy();other[ID]=11.;other[FLIGHT_ID]=101.;other["AOBT_3_flt"]=pd.to_datetime(["2025-07-01T11:30Z"])
    ambiguous,_=matched_features(q,generate_candidates(q,pd.concat([a,other],ignore_index=True)))
    assert ambiguous.retro_exact_has_selected.iloc[0]==0
    assert ambiguous.retro_broad_has_selected.iloc[0]==0
    future=a.copy();future["AOBT_3_flt"]=pd.to_datetime(["2025-07-01T12:01Z"])
    assert len(generate_candidates(q,future))==0
    print("SYNTHETIC_PASS exact,unknown,maskedNM,duplicates,ambiguousTie,futureOffblock",flush=True)


if __name__=="__main__":
    main()
