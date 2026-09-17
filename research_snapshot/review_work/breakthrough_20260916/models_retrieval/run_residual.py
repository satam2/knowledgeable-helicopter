"""Centrally scheduled finite-NM residual pilots; every original score row retained."""
import run_pilot as core
import argparse
import gc
import shutil
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch


def source_paths():
    paths={name:core.HERE/name for name in core.hashes()}
    paths["run_residual.py"]=Path(__file__)
    paths["prior_common.py"]=Path(core.common.__file__)
    return paths


def hashes():return {name:core.sha256(path) for name,path in source_paths().items()}


def eligible_rows(index,proxy,limit,seed):
    finite=np.isfinite(np.asarray(proxy,dtype=float))
    eligible={stage:positions[finite[positions]] for stage,positions in index.items()}
    sampled=core.sample_rows(eligible,limit,seed)
    return sampled,eligible,finite


def assemble_predictions(reference,score_positions,proxy,residual):
    base=np.asarray(reference,dtype=float)
    local_proxy=np.asarray(proxy,dtype=float)[score_positions]
    available=np.isfinite(local_proxy)
    residual=np.asarray(residual,dtype=float).reshape(-1)
    if len(base)!=len(score_positions) or len(residual)!=int(available.sum()):raise ValueError("Residual/score alignment mismatch")
    if not np.isfinite(base).all() or not np.isfinite(residual).all():raise ValueError("Nonfinite model/reference predictions")
    candidate=base.copy()
    candidate[available]=local_proxy[available]+residual
    if not np.isfinite(candidate).all():raise ValueError("Nonfinite anchored prediction")
    if not np.array_equal(candidate[~available],base[~available]):raise AssertionError("Missing NM reference changed")
    return candidate,available


def verify_adapter_canary(args):
    directory=core.OUT/"canaries"/f"{args.family}_{args.device}_s{args.seed}"
    path=directory/"manifest.json"
    receipt=core.read_json(path)
    if receipt.get("status")!="passed" or receipt.get("source_hashes")!=core.hashes():raise ValueError("Matching frozen adapter canary required")
    for name,digest in receipt.get("outputs",{}).items():
        if core.sha256(directory/name)!=digest:raise ValueError("Adapter canary artifact changed")
    return {"path":str(path),"sha256":core.sha256(path),"coverage":"Synthetic adapter API/serialization; residual wrapper covered by separate contract tests and source hashes"}


def contract_tests():
    proxy=np.array([0.,-20.,np.nan,9000.,30.,np.inf,42.,70.,np.nan,80.,90.,100.])
    index={"fit":np.array([0,1,2,3,4,5]),"tune":np.array([6,7]),"refit":np.array([1,3,4,6,7]),"score":np.array([8,9,10,11])}
    rows,eligible,finite=eligible_rows(index,proxy,3,20260916)
    np.testing.assert_array_equal(eligible["fit"],np.array([0,1,3,4]))
    assert len(rows["fit"])==3 and len(rows["refit"])==3
    assert set(rows["refit"]).issubset(set(index["refit"]))
    np.testing.assert_array_equal(rows["tune"],index["tune"])
    np.testing.assert_array_equal(rows["score"],np.array([9,10,11]))
    np.testing.assert_array_equal(index["score"],np.array([8,9,10,11]))
    base=np.array([1000.,1100.,1200.,1300.])
    prediction,changed=assemble_predictions(base,index["score"],proxy,np.array([-2.,3.,4.]))
    np.testing.assert_array_equal(prediction,np.array([1000.,78.,93.,104.]))
    np.testing.assert_array_equal(changed,np.array([False,True,True,True]))
    np.testing.assert_array_equal(base,np.array([1000.,1100.,1200.,1300.]))
    # Negative and long proxies remain eligible and reconstruct unbounded raw targets.
    source=np.array([-100.,9001.,np.nan])
    value,_=assemble_predictions(np.array([1.,2.,3.]),np.arange(3),source,np.array([80.,-10000.]))
    np.testing.assert_array_equal(value,np.array([-20.,-999.,3.]))
    try:assemble_predictions(base,index["score"],proxy,np.zeros(4))
    except ValueError:pass
    else:raise AssertionError("Wrong residual count accepted")
    print("RESIDUAL_CONTRACT_PASS finiteEligibility,negativeLongIncluded,originalRefitSampling,fullScoreRetention,missingReferenceExact,noClamp,alignment",flush=True)


def run(args,fold,x,meta,canary_receipt):
    directory=core.OUT/f"{args.family}_aobt_allfinite_{fold}_s{args.seed}"
    if directory.exists():raise ValueError("Existing attempt preserved; select a new seed/version")
    index,split,_=core.common.fold_data(meta,fold,full=True)
    proxy=meta.proxy_sec.to_numpy(float)
    rows,eligible,finite=eligible_rows(index,proxy,200000 if args.family=="realmlp" else 32000,args.seed)
    reference,refrec=core.common.reference(fold)
    assert core.object_hash(split)==core.object_hash(refrec["split"])
    assert np.array_equal(reference[core.ID],meta.iloc[index["score"]][core.ID])
    assert np.array_equal(reference[core.TARGET],meta.iloc[index["score"]][core.TARGET])
    api=core.module(args.family)
    directory.mkdir(parents=True)
    snapshot=directory/"source";snapshot.mkdir()
    for name,path in source_paths().items():shutil.copyfile(path,snapshot/name)
    record={"status":"running","created_utc":core.utc_now(),"family":args.family,"formulation":"aobt_allfinite","fold":fold,"seed":args.seed,
        "source_hashes":hashes(),"adapter_canary":canary_receipt,"split":split,"objective":"Raw squared residual loss on Y-P; P=T-NM AOBT; no label/proxy/output clamps",
        "eligibility":"Every finite NM proxy including negative and >7200 seconds; observed input eligibility only",
        "missing_route":"Original V2 reference preserved exactly on every score row whose NM proxy is nonfinite",
        "scope":"200K eligible fit/refit independently sampled from original purged fold indices" if args.family=="realmlp" else "32K earlier eligible labeled reference subset, no task gradient training",
        "original_stage_rows":{stage:len(v) for stage,v in index.items()},"eligible_stage_rows":{stage:len(v) for stage,v in eligible.items()},
        "sampled_stage_rows":{stage:len(v) for stage,v in rows.items()},"sampled_stage_id_hashes":{stage:core.object_hash(meta.iloc[v][core.ID].tolist()) for stage,v in rows.items()},
        "complete_score_rows":len(reference),"complete_score_id_hash":core.object_hash(reference[core.ID].tolist()),"evaluated_score_subset":False,
        "columns":list(x),"device":args.device,"threads":args.threads,"libraries":{name:core.version(name) for name in ("pytabkit","tabdpt","faiss-cpu","torch","numpy")},
        "reference_manifest_hash":core.object_hash(refrec),"raw_hashes":core.read_json(core.ROOT/"private_runs/submission_v2/protocol.json")["raw_hashes"]}
    core.write_json(directory/"manifest.json",record)
    started=time.monotonic()
    try:
        if not len(rows["fit"]) or not len(rows["tune"]) or not len(rows["refit"]):raise ValueError("No eligible fit/tune/refit cohort")
        target=meta[core.TARGET].to_numpy(float)-proxy
        fitted,fit_evidence=api.fit(x.iloc[rows["fit"]],target[rows["fit"]],(x.iloc[rows["tune"]],target[rows["tune"]]),seed=args.seed,threads=args.threads,device=args.device)
        core.write_json(directory/"fit_evidence.json",core.numeric_json(fit_evidence))
        if args.family=="realmlp":
            tune=api.predict(fitted,x.iloc[rows["tune"]])+proxy[rows["tune"]]
            pd.DataFrame({core.ID:meta.iloc[rows["tune"]][core.ID].to_numpy(),"prediction_sec":tune}).to_parquet(directory/"tune_predictions.parquet",index=False)
        del fitted;gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
        model,refit_evidence=api.fit(x.iloc[rows["refit"]],target[rows["refit"]],steps=int(fit_evidence["steps"]),seed=args.seed,threads=args.threads,device=args.device)
        core.write_json(directory/"refit_evidence.json",core.numeric_json(refit_evidence))
        query=x.iloc[rows["score"]]
        if not len(query):raise ValueError("No finite proxy score queries for this pilot")
        probe_start=time.monotonic();probe=api.predict(model,query.iloc[:min(2048,len(query))]);probe_seconds=time.monotonic()-probe_start
        forecast=probe_seconds*len(query)/len(probe)
        core.write_json(directory/"throughput_canary.json",{"rows":len(probe),"finite_score_queries":len(query),"complete_score_rows":len(reference),"seconds":probe_seconds,"projected_all_finite_score_sec":forecast,"budget_sec":args.max_score_seconds})
        if forecast>args.max_score_seconds:raise RuntimeError(f"Full finite-score forecast {forecast:.1f}s exceeds budget {args.max_score_seconds}s")
        joblib.dump(model,directory/"model.joblib",compress=0)
        residual=api.predict(model,query)
        repeated=api.predict(joblib.load(directory/"model.joblib"),query.iloc[:min(2048,len(query))])
        delta=float(np.max(np.abs(residual[:len(repeated)]-repeated)));assert delta<=1e-4
        candidate,changed=assemble_predictions(reference.prediction_sec.to_numpy(),index["score"],proxy,residual)
        pd.DataFrame({core.ID:meta.iloc[rows["score"]][core.ID].to_numpy(),"proxy_sec":proxy[rows["score"]],"residual_prediction_sec":residual}).to_parquet(directory/"eligible_score_predictions.parquet",index=False)
        reports={}
        baseline=reference.prediction_sec.to_numpy()
        for name,values in {"candidate":candidate,"blend25":baseline+.25*(candidate-baseline)}.items():
            assert np.array_equal(values[~changed],baseline[~changed])
            frame=reference.drop(columns=[core.TARGET,"error_sec","squared_error","label_bin","month","day"],errors="ignore").copy();frame["prediction_sec"]=values
            metrics,errors=core.evaluate(frame,meta.iloc[index["score"]][[core.ID,core.TARGET]])
            assert len(errors)==len(reference)
            errors.to_parquet(directory/(name+".parquet"),index=False)
            reports[name]={"metrics":metrics,"stability":core.paired_stability(reference,errors,repetitions=1000)}
        if record["source_hashes"]!=hashes():raise AssertionError("Source changed during execution")
        record.update(status="complete",completed_utc=core.utc_now(),fit=core.numeric_json(fit_evidence),refit=core.numeric_json(refit_evidence),reports=reports,runtime_sec=time.monotonic()-started,
            replay_scope="first2048eligible score residual predictions",reload_max_abs_delta=delta,missing_score_rows_preserved=int((~changed).sum()),
            peak_rss_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss),peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0)
        record["outputs"]={path.name:core.sha256(path) for path in directory.iterdir() if path.is_file() and path.name!="manifest.json"}
        core.write_json(directory/"manifest.json",record)
        print("RESULT",fold,{name:r["metrics"]["overall"]["rmse_sec"] for name,r in reports.items()},flush=True)
        return record
    except Exception as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        core.write_json(directory/"manifest.json",record);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",choices=["realmlp","tabdpt"])
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--folds",nargs="+",choices=["F1","F3"],default=["F1","F3"])
    parser.add_argument("--threads",type=int,default=4);parser.add_argument("--seed",type=int,default=20260916)
    parser.add_argument("--max-score-seconds",type=float,default=1800)
    parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    contract_tests()
    if args.test_only:return
    if not args.family:parser.error("--family required unless --test-only")
    receipt=verify_adapter_canary(args)
    x,meta=core.common.load_data()
    records={}
    for fold in args.folds:
        records[fold]=run(args,fold,x,meta,receipt);gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
    if set(records)=={"F1","F3"}:
        summary={variant:core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],records["F3"]["reports"][variant]["metrics"]["overall"]) for variant in ("candidate","blend25")}
        core.write_json(core.OUT/f"{args.family}_aobt_allfinite_s{args.seed}_summary.json",{"source_hashes":hashes(),"seasonal_rmse":summary});print("SEASONAL",summary,flush=True)


if __name__=="__main__":main()
