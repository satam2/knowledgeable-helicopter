"""Read the external PRC pack and emit aggregates outside the repository only."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = (ROOT / "knowledgeable-helicopter").resolve()
RAW = (ROOT / "data/09-15-2026-18-55-03_files_list").resolve()
OUT = (ROOT / "private_analysis").resolve()
assert not RAW.is_relative_to(REPO) and not OUT.is_relative_to(REPO)

ID = "MVT_ID_mvt"
Y = "TAXITIME_SEC_mvt"
T = "MVT_TIME_UTC_mvt"
B = "BLOCK_TIME_UTC_mvt"
S = "SCHED_TIME_UTC_mvt"
N = "AOBT_3_flt"
NM = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "CALLSIGN_flt"]
CATEGORIES = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "AIRCRAFT_OPERATOR_flt"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numeric(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)),
            "p01": float(np.quantile(v,.01)), "p05": float(np.quantile(v,.05)), "p50": float(np.median(v)),
            "p95": float(np.quantile(v,.95)), "p99": float(np.quantile(v,.99)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def group_record(g):
    missing = g.proxy.isna()
    r = {"n": len(g), "missing": int(missing.sum()), "missing_pct": float(missing.mean()*100),
         "negative_proxy": int(g.proxy.lt(0).sum()), "long_proxy": int(g.proxy.gt(7200).sum()),
         "schedule": numeric(g.schedule), "schedule_missing": numeric(g.loc[missing,"schedule"])}
    if g[Y].notna().any():
        tail = g[Y].gt(7200)
        r.update(target=numeric(g[Y]), tail=int(tail.sum()), missing_tail=int((tail&missing).sum()),
                 target_missing=numeric(g.loc[missing,Y]))
    return r


def anchor_diagnostics(g):
    g = g.loc[g[Y].notna() & g.schedule.notna()]
    r = {"n": len(g), "target": numeric(g[Y]), "anchor": numeric(g.schedule), "correction": numeric(g[Y]-g.schedule),
         "pearson": float(g[[Y,"schedule"]].corr().iloc[0,1]),
         "spearman": float(g[[Y,"schedule"]].corr(method="spearman").iloc[0,1]),
         "raw_anchor_rmse": float(np.sqrt(np.mean((g[Y]-g.schedule)**2))),
         "anchor_negative_pct": float(g.schedule.lt(0).mean()*100),
         "anchor_within_300s_target_pct": float((g[Y]-g.schedule).abs().le(300).mean()*100),
         "anchor_exact_target_pct": float(g[Y].eq(g.schedule).mean()*100)}
    return r


def coverage(reference, query):
    r = {}
    for cols in [["FLIGHT_mvt"], ["ADEP_mvt","FLIGHT_mvt"], ["ADEP_mvt","STAND_mvt"], ["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]]:
        ref = pd.MultiIndex.from_frame(reference[cols].fillna("<MISSING>"))
        q = pd.MultiIndex.from_frame(query[cols].fillna("<MISSING>"))
        counts = pd.Series(1, index=ref).groupby(level=list(range(len(cols)))).sum()
        if len(cols)==1:
            support = counts.reindex(q.get_level_values(0),fill_value=0).to_numpy()
        else:
            support = counts.reindex(q,fill_value=0).to_numpy()
        r["+".join(cols)] = {"query_n":len(q),"unseen_pct":float(np.mean(support==0)*100),
                              "support_lt5_pct":float(np.mean(support<5)*100),"support_lt20_pct":float(np.mean(support<20)*100)}
    return r


def main():
    manifest=json.loads((REPO/"reports/input_manifest.json").read_text(encoding="utf-8"))
    expected={r["file"]:r for r in manifest["files"]}
    actual={p.name:p for p in RAW.glob("*.parquet")}
    assert set(actual)==set(expected), "Raw pack differs in filenames"
    files=[]
    for name,path in sorted(actual.items()):
        sha=digest(path)
        assert sha==expected[name]["sha256"], f"Input hash mismatch: {name}"
        files.append({"name":name,"sha256":sha,"rows":pq.read_metadata(path).num_rows,"bytes":path.stat().st_size})
    print("Verified all 14 input hashes; profiling records in memory.",flush=True)
    frames=[]
    joint=[]
    shape=[]
    total_movements=0
    checks={"departure_identity":True,"unique_training_ids":True,"schedule_complete":True,"ranking_labels_hidden":True}
    clock_precision=[]
    for name,path in sorted(actual.items()):
        if name=="submitting.parquet":
            continue
        raw=pq.read_table(path).to_pandas(strings_to_categorical=True)
        if name.startswith("training"):
            total_movements+=len(raw)
        dep=raw.loc[raw.PHASE_mvt.eq("DEP")].copy()
        month=dep[T].dt.strftime("%Y-%m")
        nm_missing=dep[N].isna()
        patterns=dep[NM].isna()
        joint.append({"file":name,"dep_n":len(dep),"missing_nm":int(nm_missing.sum()),
                      "all_eight_missing":int(patterns.all(axis=1).sum()),
                      "aobt_missing_but_another_nm_present":int((nm_missing&(~patterns).any(axis=1)).sum()),
                      "pattern_count":int(patterns.drop_duplicates().shape[0])})
        train=name.startswith("training")
        if train:
            identity=(dep[T]-dep[B]).dt.total_seconds()
            assert np.array_equal(identity.to_numpy(),dep[Y].to_numpy()), "Departure identity mismatch"
        else:
            assert dep[[Y,B]].isna().all().all(), "Ranking labels visible unexpectedly"
        assert dep[S].notna().all()
        frame=dep[[ID,"FLIGHT_ID_mvt",*CATEGORIES,Y]].copy()
        for c in CATEGORIES:
            frame[c]=frame[c].astype("string")
        frame["month"]=month
        frame["day"]=dep[T].dt.strftime("%Y-%m-%d")
        frame["proxy"]=(dep[T]-dep[N]).dt.total_seconds()
        frame["schedule"]=(dep[T]-dep[S]).dt.total_seconds()
        frame["nm_eobt_delta"]=(dep[N]-dep.EOBT_1_flt).dt.total_seconds()
        frame["nm_lobt_delta"]=(dep[N]-dep.LOBT_flt).dt.total_seconds()
        frame["source_disagree"]=(dep.ADEP_mvt.astype("string")!=dep.ADEP_flt.astype("string")).fillna(False)
        frame["kind"]="training" if train else "ranking"
        frame["utc_hour"]=dep[T].dt.hour
        frames.append(frame)
        flight=dep.FLIGHT_mvt.astype("string")
        shape.append({"file":name,"flight_available_pct":float(flight.notna().mean()*100),
                      "flight_unique":int(flight.nunique()),"three_letters_then_suffix_pct":float(flight.str.match(r"^[A-Z]{3}[A-Z0-9]+$",na=False).mean()*100),
                      "flight_equals_callsign_pct":float(flight.eq(dep.CALLSIGN_flt.astype("string")).fillna(False).mean()*100)})
        clock_precision.append({"file":name,"schedule_second_zero_pct":float(dep[S].dt.second.eq(0).mean()*100),
                                "nm_second_zero_pct":float(dep.loc[~nm_missing,N].dt.second.eq(0).mean()*100)})
        print(f"Profiled {name}: {len(dep):,} departures",flush=True)
    all_dep=pd.concat(frames,ignore_index=True)
    del frames
    train=all_dep.loc[all_dep.kind.eq("training")].copy()
    rank=all_dep.loc[all_dep.kind.eq("ranking")].copy()
    assert train[ID].is_unique and rank[ID].is_unique and not train[ID].isin(rank[ID]).any()
    template=pq.read_table(actual["submitting.parquet"],columns=[ID]).to_pandas()
    assert template[ID].is_unique and len(template)==len(rank) and template[ID].isin(rank[ID]).all()
    missing=train.proxy.isna()
    normal=train.proxy.between(0,7200)
    train["residual"]=train[Y]-train.proxy
    train["airport"] = train.ADEP_mvt
    monthly={str(month):group_record(g) for month,g in all_dep.groupby("month",observed=True)}
    airports={str(a):group_record(g) for a,g in train.groupby("ADEP_mvt",observed=True)}
    ranking_airports={str(a):group_record(g) for a,g in rank.groupby("ADEP_mvt",observed=True)}
    airport_month=[]
    for (month,airport),g in all_dep.groupby(["month","ADEP_mvt"],observed=True):
        airport_month.append({"month":str(month),"airport":str(airport),**group_record(g)})
    anchors={"missing_all":anchor_diagnostics(train.loc[missing]),
             "missing_0_2h":anchor_diagnostics(train.loc[missing&train[Y].between(0,7200)]),
             "missing_over_2h":anchor_diagnostics(train.loc[missing&train[Y].gt(7200)]),
             "nm_present":anchor_diagnostics(train.loc[~missing])}
    anchors["missing_by_airport"]={str(a):anchor_diagnostics(g) for a,g in train.loc[missing].groupby("ADEP_mvt",observed=True) if len(g)>=30}
    diagnostics={}
    diagnostics["proxy_residual"] = numeric(train.loc[normal,"residual"])
    diagnostics["proxy_residual_by_airport"]={str(a):numeric(g.residual) for a,g in train.loc[normal].groupby("ADEP_mvt",observed=True)}
    diagnostics["residual_bins"]={}
    for name,mask in {"<=60":train.residual.abs().le(60),"61-300":train.residual.abs().gt(60)&train.residual.abs().le(300),
                      "301-900":train.residual.abs().gt(300)&train.residual.abs().le(900),">900":train.residual.abs().gt(900)}.items():
        g=train.loc[normal&mask]
        diagnostics["residual_bins"][name]={"n":len(g),"raw_proxy_sse":float(np.square(g.residual).sum())}
    bins=[-np.inf,-7200,-1800,0,1800,7200,np.inf]
    train["schedule_bin"]=pd.cut(train.schedule,bins=bins).astype(str)
    diagnostics["missing_schedule_bins"]=[{"bin":str(b),"n":len(g),"tail_n":int(g[Y].gt(7200).sum()),"target_median":float(g[Y].median())}
                                           for b,g in train.loc[missing].groupby("schedule_bin",observed=True)]
    diagnostics["missing_tail_concentration"]={"total":int((missing&train[Y].gt(7200)).sum()),
        "by_airport":{str(a):int(g[Y].gt(7200).sum()) for a,g in train.loc[missing].groupby("ADEP_mvt",observed=True)}}
    # Counts are computed from the same ID-stable random-sample procedure as the release.
    sorted_positions=np.argsort(train[ID].to_numpy(),kind="stable")
    selected=np.sort(sorted_positions[np.random.default_rng(20260910).choice(len(train),250000,replace=False)])
    sampled=train.iloc[selected]
    support={"ranking_all_training":coverage(train,rank),"ranking_250k":coverage(sampled,rank),
             "ranking_missing_all_training":coverage(train,rank.loc[rank.proxy.isna()]),
             "ranking_missing_specialist":coverage(train.loc[missing],rank.loc[rank.proxy.isna()])}
    for fold,end,month in [("F1","2025-07","2025-07"),("F3","2025-11","2025-11")]:
        ref=train.loc[train.month.lt(end)]
        q=train.loc[train.month.eq(month)]
        # Related-flight purging mirrors refit-against-score in the project.
        related=q.FLIGHT_ID_mvt.dropna().unique()
        ref=ref.loc[~(ref.FLIGHT_ID_mvt.notna()&ref.FLIGHT_ID_mvt.isin(related))]
        support[fold]=coverage(ref,q)
        support[fold+"_missing_specialist"]=coverage(ref.loc[ref.proxy.isna()],q.loc[q.proxy.isna()])
    hist_edges=np.array([-100000,0,300,600,900,1200,1800,3600,7200,14400,172800],float)
    hist={"edges":hist_edges.tolist(),"observed":np.histogram(train.loc[~missing,Y],hist_edges)[0].tolist(),
          "missing":np.histogram(train.loc[missing,Y],hist_edges)[0].tolist()}
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"source_root":str(RAW),"output_root":str(OUT),
            "privacy":{"raw_outside_repo":True,"no_raw_copies_or_links":True,"records_written":False,"aggregates_only":True},
            "input_verification":{"files":files,"all_manifest_hashes_match":True,"total_bytes":sum(f["bytes"] for f in files)},
            "checks":checks,"training_movements":total_movements,"training_departures":len(train),"ranking_departures":len(rank),
            "monthly":monthly,"airports":airports,"ranking_airports":ranking_airports,"airport_month":airport_month,
            "joint_missingness":joint,"flight_field":shape,"precision":clock_precision,"schedule_anchor":anchors,
            "diagnostics":diagnostics,"category_support":support,"target_histogram":hist}
    OUT.mkdir(exist_ok=True)
    path=OUT/"profile.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"output":str(path),"training_departures":len(train),"ranking_departures":len(rank),
                      "all_hashes_match":True,"anchors":anchors,"support":support,
                      "missing_tail":diagnostics["missing_tail_concentration"]},indent=2),flush=True)


if __name__=="__main__":
    main()
