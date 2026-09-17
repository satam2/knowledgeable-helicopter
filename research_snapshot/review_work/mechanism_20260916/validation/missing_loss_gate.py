"""One prespecified MSE-aligned missing-source gate on saved OOS tune experts."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.metrics import evaluate, paired_stability, season_score
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET

OUT = external_path(ROOT / "private_runs/mechanism_20260916/missing_loss_gate")
AUDIT = external_path(ROOT / "private_runs/mechanism_20260916/validation")
PARAMS = dict(iterations=300, depth=3, learning_rate=.03, l2_leaf_reg=30,
              loss_function="RMSE", random_seed=20260916, thread_count=2,
              bootstrap_type="No", allow_writing_files=False, verbose=100)
SCRIPT_HASH = sha256(__file__)


def features(x, endpoints):
    frame = x.loc[endpoints[ID]].copy()
    for column in ("good_expert_sec", "bad_expert_sec", "probability"):
        frame["gate_" + column] = endpoints[column].to_numpy(dtype=np.float32)
    frame["gate_expert_gap_sec"] = (endpoints.good_expert_sec - endpoints.bad_expert_sec).to_numpy(dtype=np.float32)
    if TARGET in frame or "BLOCK_TIME_UTC_mvt" in frame:
        raise AssertionError("Hidden departure fields reached gate")
    return frame


def load_inputs():
    x, meta = common.load_data()
    eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float)) & np.isfinite(meta.schedule_sec.to_numpy(float))
    x = x.loc[meta.loc[eligible, ID]].copy()
    folder = ROOT / "private_runs/next_230/features"
    marker = read_json(folder / "manifest.json")
    assert sha256(folder / "extensions.parquet") == marker["sha256"]
    columns = [c for c in pq.read_schema(folder / "extensions.parquet").names if not c.startswith("rw_")]
    ext = pd.read_parquet(folder / "extensions.parquet", columns=columns).set_index(ID).loc[x.index]
    return pd.concat([x, ext], axis=1), meta, eligible


def declare():
    declaration = {"created_utc": utc_now(), "runner_sha256": SCRIPT_HASH, "params": PARAMS,
        "hypothesis": "For fixed imperfect source experts, minimizing unweighted logloss of a 60-second label may not minimize final MSE; learn feature-dependent convex weight with exact squared-error weighting.",
        "training": "Existing saved fit-only expert predictions on June/October tune cohorts. Fixed gate capacity; no gate validation/early stopping/search.",
        "expert_scope": "Existing LightGBM missing mixture fit-model for tune; refit-model for score; original archived execution protocol retained.",
        "target": "t=(y-b)/(g-b), unclipped; sample weight=(g-b)^2 divided by its tune mean; exclude abs(g-b)<1 from gate fit only.",
        "inference": "p=clip(gate(observed features,g,b,original_probability),0,1); prediction=b+p*(g-b).",
        "score_route": "Missing/nonfinite NM proxy and finite schedule only; preserve every other submitted-reference prediction and all labels/score rows.",
        "variants": {"candidate": "replace eligible missing cases", "blend25": "reference +0.25*(candidate-reference), fixed before run"},
        "selection_status": "Outcome unknown at declaration; one bounded hypothesis pilot authorized after mechanism diagnostics.",
        "limits": "Tune labels already selected expert iterations; gate uses OOS parameter-fitted predictions but reused tuning period. Score folds previously exposed. Endpoint label oracle is diagnostic only, not a model result.",
        "script_sources": {"runner": SCRIPT_HASH,
            "common": sha256(ROOT / "review_work/campaign_20260916/common.py"),
            "original_missing_experts": sha256(ROOT / "review_work/campaign_20260916/aviation/run_missing_mixture_original_executed.py")},
        "endpoint_audit_sha256": sha256(AUDIT / "missing_gate_audit.json"), "cpu_threads": 2, "max_peak_rss_gib": 3}
    path = OUT / "protocol.json"
    if path.exists():
        old = read_json(path)
        if old["runner_sha256"] != SCRIPT_HASH or old["params"] != PARAMS:
            raise AssertionError("Declared gate source/config changed")
        return
    write_json(path, declaration)


def run_fold(fold, x, meta, eligible):
    directory = OUT / "models" / f"missing_loss_gate_{fold}_s20260916"
    if directory.exists():
        raise ValueError("Preserve previous gate attempt; never overwrite")
    directory.mkdir(parents=True)
    audit = read_json(AUDIT / "missing_gate_audit.json")
    endpoints = {}
    for stage in ("tune", "score"):
        path = AUDIT / f"missing_endpoints_{fold}_{stage}.parquet"
        if sha256(path) != audit["folds"][fold][stage]["private_endpoint_cache_sha256"]:
            raise AssertionError("Endpoint diagnostic cache changed")
        endpoints[stage] = pd.read_parquet(path)
    idx, split, _ = common.fold_data(meta, fold, full=True)
    for stage, frame in endpoints.items():
        positions = idx[stage][eligible[idx[stage]]]
        if not np.array_equal(frame[ID], meta.iloc[positions][ID]) or not np.array_equal(frame[TARGET],meta.iloc[positions][TARGET]):
            raise AssertionError("Endpoint cohort or original labels differ")
    reference, reference_manifest = common.reference(fold)
    rec = {"status":"running", "created_utc":utc_now(), "fold":fold, "split":split,
           "runner_sha256":SCRIPT_HASH,"protocol_sha256":sha256(OUT / "protocol.json"),
           "reference_manifest_sha256":object_hash(reference_manifest),
           "endpoint_cache_hashes":{stage:sha256(AUDIT / f"missing_endpoints_{fold}_{stage}.parquet") for stage in endpoints}}
    write_json(directory / "manifest.json",rec)
    started=time.monotonic()
    try:
        tune = endpoints["tune"]
        d=(tune.good_expert_sec-tune.bad_expert_sec).to_numpy(float)
        selected=np.abs(d)>=1
        target=(tune[TARGET].to_numpy(float)[selected]-tune.bad_expert_sec.to_numpy(float)[selected])/d[selected]
        raw_weight=d[selected]**2
        normalized=raw_weight/raw_weight.mean()
        sx=features(x,tune)
        model=CatBoostRegressor(**PARAMS)
        model.fit(sx.iloc[np.flatnonzero(selected)],target,sample_weight=normalized,
                  cat_features=list(sx.select_dtypes("category").columns))
        training_seconds=time.monotonic()-started
        model.save_model(str(directory / "gate.cbm"))
        test=endpoints["score"]
        tx=features(x,test)
        inference_start=time.monotonic()
        p=np.clip(model.predict(tx,thread_count=2),0,1)
        predicted=test.bad_expert_sec.to_numpy()+p*(test.good_expert_sec-test.bad_expert_sec).to_numpy()
        inference_seconds=time.monotonic()-inference_start
        loaded=CatBoostRegressor().load_model(str(directory / "gate.cbm"))
        repeated=np.clip(loaded.predict(tx,thread_count=2),0,1)
        replay=float(np.max(np.abs(p-repeated)))
        assert replay==0 and np.isfinite(predicted).all()
        changed=eligible[idx["score"]]
        values=reference.prediction_sec.to_numpy().copy()
        values[changed]=predicted
        variants={"candidate":values,"blend25":reference.prediction_sec.to_numpy()+.25*(values-reference.prediction_sec.to_numpy())}
        reports={}
        for variant,pred in variants.items():
            assert np.array_equal(pred[~changed],reference.loc[~changed,"prediction_sec"])
            frame=reference.drop(columns=[TARGET,"error_sec","squared_error","label_bin","month","day"],errors="ignore").copy()
            frame["prediction_sec"]=pred
            metric,errors=evaluate(frame,meta.iloc[idx["score"]][[ID,TARGET]])
            errors.to_parquet(directory / f"{variant}.parquet",index=False)
            reports[variant]={"metrics":metric,"stability":paired_stability(reference,errors,repetitions=1000)}
        pd.DataFrame({ID:test[ID],"gate_weight":p,"good_sec":test.good_expert_sec,"bad_sec":test.bad_expert_sec}).to_parquet(directory / "score_gate_weights.parquet",index=False)
        peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
        rec.update(status="complete",completed_utc=utc_now(),reports=reports,gate_training_rows=int(selected.sum()),
            gate_training_id_hash=object_hash(tune.loc[selected,ID].tolist()),gate_tune_rows=len(tune),
            score_missing_rows=int(changed.sum()),score_rows=len(reference),feature_columns=list(sx),
            normalized_weight_sum=float(normalized.sum()),weight_effective_sample_size=float(raw_weight.sum()**2/np.dot(raw_weight,raw_weight)),
            top10_training_weight_fraction=float(np.sort(raw_weight)[-10:].sum()/raw_weight.sum()),
            training_runtime_sec=training_seconds,inference_runtime_sec=inference_seconds,runtime_sec=time.monotonic()-started,
            peak_rss_bytes=peak,reload_max_abs_delta=replay,source_unchanged=sha256(__file__)==SCRIPT_HASH)
        if peak>3*1024**3:
            raise MemoryError("Pilot exceeded3GiB RSS budget")
        rec["outputs"]={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_json(directory / "manifest.json",rec)
        print("RESULT",fold,{v:r["metrics"]["overall"]["rmse_sec"] for v,r in reports.items()},flush=True)
        return rec
    except Exception as error:
        rec.update(status="failed",error=repr(error),runtime_sec=time.monotonic()-started)
        write_json(directory / "manifest.json",rec)
        raise


def main():
    declare()
    x,meta,eligible=load_inputs()
    records={}
    for fold in ("F1","F3"):
        records[fold]=run_fold(fold,x,meta,eligible)
        gc.collect()
    summary={variant:{"seasonal_rmse":season_score(records["F1"]["reports"][variant]["metrics"]["overall"],records["F3"]["reports"][variant]["metrics"]["overall"]),
                      "folds":{f:records[f]["reports"][variant]["metrics"]["overall"]["rmse_sec"] for f in records}}
             for variant in ("candidate","blend25")}
    write_json(OUT / "summary.json",{"created_utc":utc_now(),"results":summary,
        "gate_weight_effective_sample_sizes":{f:records[f]["weight_effective_sample_size"] for f in records},
        "runner_sha256":SCRIPT_HASH,"interpretation":"One fixed pilot; reused development score folds; no new official score."})
    print("SEASONAL",summary,flush=True)


if __name__=="__main__":
    main()
