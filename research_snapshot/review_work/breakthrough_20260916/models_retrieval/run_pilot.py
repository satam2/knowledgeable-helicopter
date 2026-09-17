"""Centrally scheduled subset-fit/full-score neural and retrieval pilots."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch
import argparse
import gc
import importlib
from importlib.metadata import version
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json,write_json,sha256,object_hash,utc_now
from taxiout.metrics import evaluate,paired_stability,season_score
from taxiout.paths import external_path
from taxiout.schema import ID,TARGET

OUT=external_path(ROOT/"private_runs/breakthrough_20260916/models_retrieval")


def hashes():return {name:sha256(HERE/name) for name in ("run_pilot.py","preprocessing.py","realmlp_adapter.py","tabdpt_adapter.py")}


def numeric_json(value):
    if isinstance(value,dict):return {str(k):numeric_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [numeric_json(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    return value


def module(family):return importlib.import_module(family+"_adapter")


def sample_rows(idx,limit,seed):
    result={k:v.copy() for k,v in idx.items()}
    for stage in ("fit","refit"):
        if len(result[stage])>limit:result[stage]=np.sort(np.random.default_rng(seed).choice(result[stage],limit,replace=False))
    return result


def canary(args):
    directory=OUT/"canaries"/f"{args.family}_{args.device}_s{args.seed}"
    if directory.exists():raise ValueError("Preserve existing canary; use a new seed/version")
    directory.mkdir(parents=True)
    source=hashes();started=time.monotonic()
    record={"status":"running","created_utc":utc_now(),"source_hashes":source,"private_data_loaded":False,"family":args.family,"device":args.device}
    write_json(directory/"manifest.json",record)
    try:
        rng=np.random.default_rng(args.seed)
        x=pd.DataFrame({"signal":rng.normal(size=640),"airport":pd.Categorical(np.where(np.arange(640)%2,"A","B"))})
        y=30+2*x.signal.to_numpy()+rng.normal(size=640)
        api=module(args.family)
        model,evidence=api.fit(x.iloc[:512],y[:512],(x.iloc[512:576],y[512:576]),seed=args.seed,threads=args.threads,device=args.device,max_epochs=1 if args.family=="realmlp" else None)
        predictions=api.predict(model,x.iloc[576:])
        assert len(predictions)==64 and np.isfinite(predictions).all()
        joblib.dump(model,directory/"model.joblib",compress=0)
        replay=api.predict(joblib.load(directory/"model.joblib"),x.iloc[576:])
        delta=float(np.max(np.abs(predictions-replay)));assert delta<=1e-4
        record.update(status="passed",evidence=numeric_json(evidence),prediction_rows=64,reload_max_abs_delta=delta,
            runtime_sec=time.monotonic()-started,peak_rss_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0)
    except Exception as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        write_json(directory/"manifest.json",record);raise
    assert source==hashes()
    record["outputs"]={p.name:sha256(p) for p in directory.iterdir() if p.name!="manifest.json"}
    write_json(directory/"manifest.json",record);print("CANARY",numeric_json(record),flush=True)


def run(args,fold,x,meta):
    directory=OUT/f"{args.family}_direct_{fold}_s{args.seed}"
    if directory.exists():raise ValueError("Preserve prior attempt; use newseed/version")
    idx,split,_=common.fold_data(meta,fold,full=True)
    rows=sample_rows(idx,200000 if args.family=="realmlp" else 32000,args.seed)
    reference,refrec=common.reference(fold)
    assert object_hash(split)==object_hash(refrec["split"])
    assert np.array_equal(reference[ID],meta.iloc[idx["score"]][ID])
    assert np.array_equal(reference[TARGET],meta.iloc[idx["score"]][TARGET])
    api=module(args.family)
    directory.mkdir(parents=True)
    snapshot=directory/"source";snapshot.mkdir()
    for name in hashes():shutil.copyfile(HERE/name,snapshot/name)
    record={"status":"running","created_utc":utc_now(),"family":args.family,"formulation":"direct","fold":fold,"seed":args.seed,"source_hashes":hashes(),"split":split,
        "scope":"200K fit/refit independently sampled from original fold indices" if args.family=="realmlp" else "32K earlier labeled reference subset; no task gradient training",
        "row_counts":{stage:len(v) for stage,v in rows.items()},"row_id_hashes":{stage:object_hash(meta.iloc[v][ID].tolist()) for stage,v in rows.items()},
        "complete_score_rows":len(reference),"score_subset":False,"columns":list(x),"libraries":{n:version(n) for n in ("pytabkit","tabdpt","faiss-cpu","torch","numpy")},"reference_manifest_hash":object_hash(refrec),"raw_hashes":read_json(ROOT/"private_runs/submission_v2/protocol.json")["raw_hashes"]}
    write_json(directory/"manifest.json",record)
    started=time.monotonic()
    try:
        labels=meta[TARGET].to_numpy(float)
        fitted,fit_evidence=api.fit(x.iloc[rows["fit"]],labels[rows["fit"]],(x.iloc[rows["tune"]],labels[rows["tune"]]),seed=args.seed,threads=args.threads,device=args.device)
        write_json(directory/"fit_evidence.json",numeric_json(fit_evidence))
        # No tune evaluation is needed for a frozen pretrained ICL configuration.
        if args.family=="realmlp":
            tune=api.predict(fitted,x.iloc[rows["tune"]])
            pd.DataFrame({ID:meta.iloc[rows["tune"]][ID].to_numpy(),"prediction_sec":tune}).to_parquet(directory/"tune_predictions.parquet",index=False)
        del fitted;gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
        model,refit_evidence=api.fit(x.iloc[rows["refit"]],labels[rows["refit"]],steps=int(fit_evidence["steps"]),seed=args.seed,threads=args.threads,device=args.device)
        write_json(directory/"refit_evidence.json",numeric_json(refit_evidence))
        query=x.iloc[rows["score"]]
        probe_start=time.monotonic();probe=api.predict(model,query.iloc[:min(2048,len(query))]);probe_seconds=time.monotonic()-probe_start
        forecast=probe_seconds*len(query)/len(probe)
        write_json(directory/"throughput_canary.json",{"rows":len(probe),"seconds":probe_seconds,"projected_complete_score_sec":forecast,"budget_sec":args.max_score_seconds})
        if forecast>args.max_score_seconds:raise RuntimeError(f"Full score forecast{forecast:.1f}s exceedsdeclaredbudget{args.max_score_seconds}s")
        joblib.dump(model,directory/"model.joblib",compress=0)
        predictions=api.predict(model,query)
        repeated=api.predict(joblib.load(directory/"model.joblib"),query.iloc[:min(2048,len(query))])
        delta=float(np.max(np.abs(predictions[:len(repeated)]-repeated)));assert delta<=1e-4
        reports={}
        for name,values in {"candidate":predictions,"blend25":reference.prediction_sec.to_numpy()+.25*(predictions-reference.prediction_sec.to_numpy())}.items():
            frame=reference.drop(columns=[TARGET,"error_sec","squared_error","label_bin","month","day"],errors="ignore").copy();frame["prediction_sec"]=values
            metrics,errors=evaluate(frame,meta.iloc[rows["score"]][[ID,TARGET]])
            errors.to_parquet(directory/(name+".parquet"),index=False)
            reports[name]={"metrics":metrics,"stability":paired_stability(reference,errors,repetitions=1000)}
        assert record["source_hashes"]==hashes()
        record.update(status="complete",completed_utc=utc_now(),fit=numeric_json(fit_evidence),refit=numeric_json(refit_evidence),reports=reports,runtime_sec=time.monotonic()-started,
            replay_scope="first2048scorequeries",reload_max_abs_delta=delta,peak_rss_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0)
        record["outputs"]={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_json(directory/"manifest.json",record)
        print("RESULT",fold,{k:v["metrics"]["overall"]["rmse_sec"] for k,v in reports.items()},flush=True)
        return record
    except Exception as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        write_json(directory/"manifest.json",record);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--family",choices=["realmlp","tabdpt"],required=True)
    parser.add_argument("--canary",action="store_true");parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--folds",nargs="+",choices=["F1","F3"],default=["F1","F3"])
    parser.add_argument("--threads",type=int,default=4);parser.add_argument("--seed",type=int,default=20260916)
    parser.add_argument("--max-score-seconds",type=float,default=1800)
    args=parser.parse_args()
    if args.canary:return canary(args)
    receipt=read_json(OUT/"canaries"/f"{args.family}_{args.device}_s{args.seed}"/"manifest.json")
    assert receipt["status"]=="passed" and receipt["source_hashes"]==hashes(),"Matching syntheticcanary required"
    x,meta=common.load_data()
    records={}
    for fold in args.folds:
        records[fold]=run(args,fold,x,meta);gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
    if set(records)=={"F1","F3"}:
        summary={variant:season_score(records["F1"]["reports"][variant]["metrics"]["overall"],records["F3"]["reports"][variant]["metrics"]["overall"]) for variant in ("candidate","blend25")}
        write_json(OUT/f"{args.family}_direct_s{args.seed}_summary.json",summary);print("SEASONAL",summary,flush=True)


if __name__=="__main__":main()
