"""Target-free official airport-day ATFM covariates for retrospective analysis."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from acquire import ROOT,OUT as ACQUISITION,digest,write,now
from inspect_workbooks import OUT as INSPECTION,AIRPORTS

sys.path.insert(0,str(ROOT / "review_work/campaign_20260916"))
import common
ID,TIME=common.ID,common.MOVEMENT
OUT=ROOT / "private_runs/tail240_20260916/forensics/atfm/features_v1"
TABLES=INSPECTION / "tables"
CAUSES=["A","C","D","E","G","I","M","N","O","P","R","S","T","V","W","NA"]
FEATURES=["atfm_slot_row_present","atfm_arrival_row_present","atfm_departures","atfm_regulated_fraction",
    "atfm_early_fraction","atfm_within_fraction","atfm_late_fraction","atfm_arrivals",
    "atfm_arrival_delay_minutes","atfm_arrival_delay_minutes_per_arrival","atfm_arrival_delay_populated",
    "atfm_arrival_delayed_fraction","atfm_arrival_delayed_over15_fraction"]+["atfm_arrival_cause_"+c+"_minutes_per_arrival" for c in CAUSES]
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def ratio(numerator,denominator):
    numerator=pd.to_numeric(numerator,errors="raise")
    denominator=pd.to_numeric(denominator,errors="raise")
    return numerator/denominator.where(denominator.gt(0))


def daily_features(slot,arrival):
    keys=["APT_ICAO","date_utc"]
    for source in [slot,arrival]:
        if source.duplicated(keys).any() or source[keys].isna().any().any():
            raise ValueError("Daily source keys must be nonnull and unique")
    s=slot.set_index(keys);a=arrival.set_index(keys)
    days=pd.DataFrame(index=s.index.union(a.index,sort=False))
    days["atfm_slot_row_present"]=days.index.isin(s.index).astype("float32")
    days["atfm_arrival_row_present"]=days.index.isin(a.index).astype("float32")
    days["atfm_departures"]=s.FLT_DEP_1
    days["atfm_regulated_fraction"]=ratio(s.FLT_DEP_REG_1,s.FLT_DEP_1)
    for label,column in [("early","FLT_DEP_OUT_EARLY_1"),("within","FLT_DEP_IN_1"),("late","FLT_DEP_OUT_LATE_1")]:
        days[f"atfm_{label}_fraction"]=ratio(s[column],s.FLT_DEP_REG_1)
    days["atfm_arrivals"]=a.FLT_ARR_1
    days["atfm_arrival_delay_minutes"]=a.DLY_APT_ARR_1
    days["atfm_arrival_delay_minutes_per_arrival"]=ratio(a.DLY_APT_ARR_1,a.FLT_ARR_1)
    days["atfm_arrival_delay_populated"]=a.DLY_APT_ARR_1.notna().astype("float32")
    days["atfm_arrival_delayed_fraction"]=ratio(a.FLT_ARR_1_DLY,a.FLT_ARR_1)
    days["atfm_arrival_delayed_over15_fraction"]=ratio(a.FLT_ARR_1_DLY_15,a.FLT_ARR_1)
    for cause in CAUSES:
        days[f"atfm_arrival_cause_{cause}_minutes_per_arrival"]=ratio(a[f"DLY_APT_ARR_{cause}_1"],a.FLT_ARR_1)
    assert list(days)==FEATURES and not np.isinf(days.to_numpy()).any()
    if (days.select_dtypes("number")<0).any().any():raise ValueError("Unexpected negative aggregate source value")
    return days.astype("float32")


def attach(query,days):
    if query[ID].isna().any() or not query[ID].is_unique:
        raise ValueError("Query IDs must be nonnull and unique")
    if not days.index.is_unique:raise ValueError("Daily feature keys duplicated")
    timestamp=pd.to_datetime(query[TIME],utc=True,errors="raise")
    if timestamp.isna().any():raise ValueError("Query movement times required")
    keys=pd.MultiIndex.from_arrays([query.ADEP_mvt.astype("string"),timestamp.dt.strftime("%Y-%m-%d")],names=["APT_ICAO","date_utc"])
    result=days.reindex(keys).copy()
    result.index=pd.Index(query[ID].to_numpy(),name=ID)
    for column in ["atfm_slot_row_present","atfm_arrival_row_present"]:
        result[column]=result[column].fillna(0)
    return result[FEATURES].astype("float32")


def declaration():
    value={"source_sha256":digest(__file__),"features":FEATURES,"cause_codes":CAUSES,
        "scope":"Retrospective aggregate airport-day covariates only, target-free construction and joins; no taxiout indicator or private block/time labels.",
        "source_manifest_sha256":digest(TABLES / "manifest.json"),"acquisition_manifest_sha256":digest(ACQUISITION / "manifest.json"),
        "join":"Exact airport+UTC movement calendar date. WorkbookFLT_DATE only saysDateofflight, timezone not explicit; matchingtoUTCdate is a declared modelingassumption, not verified source-day equivalence.",
        "availability":"Whole-day totals may include laterflights and are publishedin arrears. Noncausal retrospective context; not realtime or event-time availability.",
        "rates":"Regulated/IFRdepartures; early,within,late/regulateddepartures; totalATFMminutes/IFRarrivals; delayedcounts/IFRarrivals; each16causecodes totalminutes/IFRarrivals. Nonpositive/missingdenominator=>missing. Originalfinitezero retained; blank remainsmissing.",
        "blank_semantics":"No statementthat blankdelaymeanszero. PreserveNaN and explicit totaldelaypopulatedflag. Ddeicing observedzero onlyonpopulateddays; no inferreddeicingabsenceonblankdays.",
        "inputs":"Two independentlyextracted publictables; trainingaudit metadata ID,airport,movement only; rawranking DEP ID,airport,movement only. No target/block read.",
        "scope_dates":"All2025training and suppliedrankingJan/July2026; everyrow retained. No labels or privatequeries sent externally.",
        "resources":"2CPU6GiB process ceiling8GiBhostavailable; preserveoriginalworkbooks and sources.",
        "licensing":"EUROCONTROL PRU attribution; localnoncommercial research only; originalinformationunmodified; prizeeligibilityunresolved."}
    path=OUT / "protocol.json"
    if path.exists():assert json.loads(path.read_text())["declaration"]==value
    else:
        OUT.mkdir(parents=True,exist_ok=False);write(path,{"created_utc":now(),"declaration":value})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--declare-only",action="store_true");args=parser.parse_args()
    declaration()
    if args.declare_only:
        print("Declared29 publicdailyATFM features; no private target inputs",flush=True);return
    manifest=json.loads((TABLES / "manifest.json").read_text())
    filenames=["ATFM_Slot_Adherence_airports.parquet","Airport_Arrival_ATFM_Delay_airports.parquet"]
    for name in filenames:assert digest(TABLES / name)==manifest["outputs"][name]
    days=daily_features(*[pd.read_parquet(TABLES / name) for name in filenames])
    days.reset_index().to_parquet(OUT / "airport_days.parquet",index=False)
    metadata=ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert digest(metadata)==common.read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][metadata.name]
    training=pd.read_parquet(metadata,columns=[ID,"ADEP_mvt",TIME])
    ranking_path=common.RAW / "ranking.parquet"
    assert digest(ranking_path)==common.read_json(ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"][ranking_path.name]
    ranking=pd.read_parquet(ranking_path,columns=[ID,"ADEP_mvt",TIME],filters=[("PHASE_mvt","=","DEP")])
    receipts={}
    for split,query in [("training",training),("ranking",ranking)]:
        frame=attach(query,days)
        assert np.array_equal(frame.index,query[ID]) and not np.isinf(frame.to_numpy()).any()
        frame.reset_index().to_parquet(OUT / f"{split}_features.parquet",index=False)
        coverage=[]
        source=query.assign(month=pd.to_datetime(query[TIME],utc=True).dt.strftime("%Y-%m")).set_index(ID)
        for (airport,month),group in source.groupby(["ADEP_mvt","month"],observed=True):
            selected=frame.loc[group.index]
            coverage.append({"airport":str(airport),"month":month,"rows":len(group),
                "slot_matched":int(selected.atfm_slot_row_present.sum()),"arrival_matched":int(selected.atfm_arrival_row_present.sum()),
                "arrival_delay_populated":int(selected.atfm_arrival_delay_populated.fillna(0).sum())})
        receipts[split]={"rows":len(query),"id_hash":common.object_hash(query[ID].tolist()),"coverage":coverage,
            "missing_cells":{c:int(frame[c].isna().sum()) for c in frame}}
        print("ATFM_FEATURES",split,len(frame),len(frame.columns),"all_slot",frame.atfm_slot_row_present.eq(1).all(),"all_arrival",frame.atfm_arrival_row_present.eq(1).all(),flush=True)
    peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
    assert peak<6*1024**3 and psutil.virtual_memory().available>8*1024**3
    write(OUT / "manifest.json",{"status":"complete","source_sha256":digest(__file__),"protocol_sha256":digest(OUT / "protocol.json"),
        "features":FEATURES,"splits":receipts,"peak_bytes":peak,"private_targets_read":False,
        "source_table_hashes":{name:digest(TABLES / name) for name in filenames},"metadata_sha256":digest(metadata),"ranking_sha256":digest(ranking_path),
        "outputs":{p.name:digest(p) for p in OUT.iterdir() if p.is_file()}})


if __name__=="__main__":main()
