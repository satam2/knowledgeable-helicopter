"""Coordinator-launched full-data model/formulation comparisons outside checkouts."""
import argparse
import gc
import importlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from importlib.metadata import version

if "tabm" in sys.argv:
    import torch
import xgboost
import numpy as np
import pandas as pd
import joblib
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json,write_json,sha256,object_hash,utc_now
from taxiout.paths import external_path
from taxiout.schema import ID,TARGET
from taxiout.metrics import evaluate,paired_stability,season_score
from encoders import json_safe

OUT=external_path(ROOT / "private_runs/breakthrough_20260916/models")
CLOCK_ORDER=["AOBT_3_flt","EOBT_1_flt","IOBT_flt","LOBT_flt","SCHED_TIME_UTC_mvt"]


def anchors(x,meta,formulation):
    if formulation=="direct":
        return x,np.zeros(len(x)),np.ones(len(x),bool),{"rule":"raw taxi time; all original labels"}
    if formulation=="aobt_allfinite":
        offset=meta.proxy_sec.to_numpy(float)
        return x,offset,np.isfinite(offset),{"rule":"observed actual-NM proxy; every finite value including negative and >7200; missing rows retain submitted reference"}
    values=np.column_stack([x["takeoff_minus_"+clock].to_numpy(float) for clock in CLOCK_ORDER])
    valid=np.isfinite(values)&(values!=-999999)
    available=valid.any(axis=1)
    selected=valid.argmax(axis=1)
    offset=values[np.arange(len(x)),selected]
    offset[~available]=np.nan
    frame=x.copy()
    frame["anchor_source"]=pd.Categorical(np.where(available,np.array(CLOCK_ORDER)[selected],"none"))
    frame["anchor_seconds"]=np.where(available,offset,-999999).astype(np.float32)
    return frame,offset,available,{"rule":"first finite observed clock in predeclared priority; no range or sign clipping","clock_priority":CLOCK_ORDER,
                                  "no_anchor_route":"keep submitted reference"}


def source_hashes():
    files=[HERE / f for f in ("run_full.py","encoders.py","xgb_gpu.py","tabm_gpu.py")]
    return {**{p.name:sha256(p) for p in files},
            "prior_common.py":sha256(ROOT / "review_work/campaign_20260916/common.py")}


def declare(args):
    OUT.mkdir(parents=True,exist_ok=True)
    key=f"{args.family}_{args.formulation}_s{args.seed}"
    path=OUT / "protocols" / f"{key}.json"
    payload={"created_utc":utc_now(),"family":args.family,"formulation":args.formulation,"seed":args.seed,
        "folds":args.folds,"threads":args.threads,"source_hashes":source_hashes(),
        "training":"All original eligible fit rows, original tune month, independent all-eligible refit; original flight-ID purges.",
        "objective":"Conditional mean raw squared loss, affine target scaling only for TabM; original labels unmodified.",
        "variants":["candidate","blend25"],"blend_weight":.25,
        "reference":"unchanged submitted clock_and_rome_ensemble fold predictions",
        "selection":"Broader five-hour authorized development search; exposed score folds; no new holdout or official score.",
        "resources":{"gpu":"single centrally scheduled GPU; no concurrent CUDA fit","threads":args.threads,"minimum_host_ram_gib":8},
        "library_versions":{name:version(name) for name in ("numpy","pandas","pyarrow","xgboost","catboost","torch","tabm")},
        "raw_hashes":read_json(ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]}
    if path.exists():
        old=read_json(path)
        for field in ("family","formulation","seed","source_hashes","threads","library_versions"):
            if old[field]!=payload[field]:
                raise ValueError(f"Protocol changed: {field}")
    else:
        write_json(path,payload)
        snapshot=OUT / "source_snapshots" / key
        snapshot.mkdir(parents=True,exist_ok=True)
        for filename in ("run_full.py","encoders.py","xgb_gpu.py","tabm_gpu.py"):
            shutil.copyfile(HERE / filename,snapshot / filename)
        shutil.copyfile(ROOT / "review_work/campaign_20260916/common.py",snapshot / "prior_common.py")
    return path


def run(args,fold,x,meta,offset,eligible,anchor_info,protocol):
    name=f"{args.family}_{args.formulation}_{fold}_s{args.seed}"
    directory=external_path(OUT / name)
    if directory.exists():
        record=read_json(directory / "manifest.json")
        if record["status"]!="complete":
            raise ValueError("Prior incomplete attempt preserved; inspect before resuming")
        if record["source_hashes"]!=source_hashes():
            raise ValueError("Completed source differs")
        for filename,digest in record["outputs"].items():
            if sha256(directory / filename)!=digest:
                raise ValueError("Completed output changed")
        print("REUSED",name,flush=True)
        return record
    module=importlib.import_module("xgb_gpu" if args.family=="xgb" else "tabm_gpu")
    idx,split,_=common.fold_data(meta,fold,full=True)
    reference,reference_record=common.reference(fold)
    assert object_hash(split)==object_hash(reference_record["split"])
    assert np.array_equal(reference[ID],meta.iloc[idx["score"]][ID])
    assert np.array_equal(reference[TARGET],meta.iloc[idx["score"]][TARGET])
    rows={stage:positions[eligible[positions]] for stage,positions in idx.items()}
    directory.mkdir(parents=True)
    record={"status":"running","name":name,"fold":fold,"family":args.family,"formulation":args.formulation,
        "created_utc":utc_now(),"protocol_sha256":sha256(protocol),"source_hashes":source_hashes(),
        "split":split,"anchor":anchor_info,"seed":args.seed,"threads":args.threads,"training_scope":"full eligible",
        "fit_ids":{stage:{"n":len(positions),"hash":object_hash(meta.iloc[positions][ID].tolist())} for stage,positions in rows.items()},
        "complete_score_rows":len(reference),"feature_columns":list(x),"reference_manifest_hash":object_hash(reference_record)}
    write_json(directory / "manifest.json",record)
    started=time.monotonic()
    try:
        if psutil.virtual_memory().available<8*1024**3:
            raise MemoryError("At least8GiB host memory must remain available before fit")
        target=meta[TARGET].to_numpy(float)-offset
        fit_model,fit_evidence=module.fit(x.iloc[rows["fit"]],target[rows["fit"]],
            (x.iloc[rows["tune"]],target[rows["tune"]]),seed=args.seed,threads=args.threads)
        tune=module.predict(fit_model,x.iloc[rows["tune"]])+offset[rows["tune"]]
        pd.DataFrame({ID:meta.iloc[rows["tune"]][ID].to_numpy(),"prediction_sec":tune}).to_parquet(directory / "tune_predictions.parquet",index=False)
        joblib.dump(fit_model,directory / "fit_model.joblib",compress=0)
        write_json(directory / "fit_evidence.json",json_safe(fit_evidence))
        del fit_model
        gc.collect()
        model,refit_evidence=module.fit(x.iloc[rows["refit"]],target[rows["refit"]],
            steps=int(fit_evidence["steps"]),seed=args.seed,threads=args.threads)
        joblib.dump(model,directory / "model.joblib",compress=0)
        inference_started=time.monotonic()
        predictions=module.predict(model,x.iloc[rows["score"]])+offset[rows["score"]]
        inference_seconds=time.monotonic()-inference_started
        repeated=module.predict(joblib.load(directory / "model.joblib"),x.iloc[rows["score"]])+offset[rows["score"]]
        delta=float(np.max(np.abs(predictions-repeated)))
        if delta>1e-4 or not np.isfinite(predictions).all():
            raise AssertionError("Saved model replay or prediction finiteness failed")
        changed=eligible[idx["score"]]
        candidate=reference.prediction_sec.to_numpy().copy()
        candidate[changed]=predictions
        reports={}
        for variant,values in {"candidate":candidate,"blend25":reference.prediction_sec.to_numpy()+.25*(candidate-reference.prediction_sec.to_numpy())}.items():
            if not np.array_equal(values[~changed],reference.loc[~changed,"prediction_sec"]):
                raise AssertionError("Unsupported reference route changed")
            frame=reference.drop(columns=[TARGET,"error_sec","squared_error","label_bin","month","day"],errors="ignore").copy()
            frame["prediction_sec"]=values
            metric,errors=evaluate(frame,meta.iloc[idx["score"]][[ID,TARGET]])
            errors.to_parquet(directory / f"{variant}.parquet",index=False)
            reports[variant]={"metrics":metric,"stability":paired_stability(reference,errors,repetitions=1000)}
        record.update(status="complete",completed_utc=utc_now(),fit=json_safe(fit_evidence),refit=json_safe(refit_evidence),
            runtime_sec=time.monotonic()-started,inference_runtime_sec=inference_seconds,reload_max_abs_delta=delta,
            peak_rss_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss),reports=reports)
        if record["source_hashes"]!=source_hashes():
            raise AssertionError("Source changed during execution")
        record["outputs"]={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_json(directory / "manifest.json",record)
        print("RESULT",name,{variant:r["metrics"]["overall"]["rmse_sec"] for variant,r in reports.items()},flush=True)
        return record
    except Exception as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        write_json(directory / "manifest.json",record)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",required=True,choices=["xgb","tabm"])
    parser.add_argument("--formulation",required=True,choices=["direct","aobt_allfinite","multi_anchor"])
    parser.add_argument("--folds",nargs="+",default=["F1","F3"])
    parser.add_argument("--seed",type=int,default=20260916)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--declare-only",action="store_true")
    args=parser.parse_args()
    protocol=declare(args)
    if args.declare_only:
        print("DECLARED",protocol,flush=True)
        return
    x,meta=common.load_data()
    x,offset,eligible,info=anchors(x,meta,args.formulation)
    records={}
    for fold in args.folds:
        records[fold]=run(args,fold,x,meta,offset,eligible,info,protocol)
        gc.collect()
    if set(records)=={"F1","F3"}:
        report={variant:season_score(records["F1"]["reports"][variant]["metrics"]["overall"],records["F3"]["reports"][variant]["metrics"]["overall"])
                for variant in ("candidate","blend25")}
        write_json(OUT / f"{args.family}_{args.formulation}_s{args.seed}_summary.json",{"created_utc":utc_now(),"seasonal_rmse":report})
        print("SEASONAL",report,flush=True)


if __name__=="__main__":
    main()
