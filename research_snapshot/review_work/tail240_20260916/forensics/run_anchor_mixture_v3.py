"""Tune-only exact source-anchor expectation experiment."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import joblib
from catboost import CatBoostClassifier
from run_lexical import ROOT, prior, resource_check
from taxiout.artifacts import read_json, write_json, sha256, utc_now, object_hash

OUT = ROOT / "private_runs/tail240_20260916/forensics/models/anchor_tune_v3"
LEXICAL = ROOT / "private_runs/tail240_20260916/forensics/models/lexical_v1"
STEPS = [50,100,200,400]
SEED = 20260916
ID,TARGET,TIME = prior.ID,prior.TARGET,prior.MOVEMENT
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def collapse_anchors(anchors):
    anchors = np.asarray(anchors,float)
    if anchors.ndim != 2 or not np.isfinite(anchors[:,0]).all():
        raise ValueError("Finite ordinary anchor is required")
    active = np.isfinite(anchors)
    for j in range(1,anchors.shape[1]):
        active[:,j] &= ~np.any((anchors[:,:j] == anchors[:,j,None]) & active[:,:j],axis=1)
    return active


def encode(y, anchors):
    y = np.asarray(y,float)
    anchors = np.asarray(anchors,float)
    if len(y) != len(anchors) or not np.isfinite(y).all():
        raise ValueError("Finite aligned raw labels required")
    active = collapse_anchors(anchors)
    q = np.zeros_like(anchors)
    for row in range(len(y)):
        slots = np.flatnonzero(active[row])
        slots = slots[np.argsort(anchors[row,slots],kind="stable")]
        values = anchors[row,slots]
        high = np.searchsorted(values,y[row],side="right")
        if high == 0:
            q[row,slots[0]] = 1.
        elif high == len(slots):
            q[row,slots[-1]] = 1.
        else:
            lower,upper = slots[high-1],slots[high]
            weight = (y[row]-anchors[row,lower])/(anchors[row,upper]-anchors[row,lower])
            q[row,lower],q[row,upper] = 1-weight,weight
    remainder = y-np.sum(q*np.where(np.isfinite(anchors),anchors,0.),axis=1)
    np.testing.assert_allclose(q.sum(axis=1),1.,rtol=0,atol=1e-12)
    assert (q>=0).all() and not (q[~active]>0).any()
    return q,remainder,active


def replicas(x,q):
    rows,classes = np.nonzero(q>0)
    weights = q[rows,classes]
    np.testing.assert_allclose(np.bincount(rows,weights=weights,minlength=len(x)),1.,rtol=0,atol=1e-12)
    return x.iloc[rows].copy(),classes,weights,x.index.to_numpy()[rows]


def expectation(probabilities,classes,anchors):
    probabilities = np.asarray(probabilities,float)
    if not np.isfinite(probabilities).all() or (probabilities<0).any():
        raise ValueError("Finite nonnegative probabilities required")
    if probabilities.shape != (len(anchors),len(classes)):
        raise ValueError("Probability shape mismatch")
    active = collapse_anchors(anchors)
    p = np.zeros_like(anchors,dtype=float)
    p[:,np.asarray(classes,dtype=int)] = probabilities
    p *= active
    total = p.sum(axis=1)
    empty = total<=0
    p[empty,0] = 1.
    p /= p.sum(axis=1,keepdims=True)
    return np.sum(p*np.where(np.isfinite(anchors),anchors,0.),axis=1)


def physical_reference(history, query):
    if len(history) and history[TIME].max() >= query[TIME].min():
        raise ValueError("Prior labels must strictly precede query")
    result = pd.DataFrame({"physical_prior_sec":900.,"physical_prior_n":0.},index=query.index)
    for keys,minimum in [(["ADEP_mvt"],1),(["ADEP_mvt","RUNWAY_mvt"],100),(["ADEP_mvt","STAND_mvt","RUNWAY_mvt"],20)]:
        table = history.groupby(keys,dropna=False)[TARGET].agg(["median","size"]).reset_index()
        joined = query[keys].reset_index().merge(table,on=keys,how="left",validate="many_to_one").set_index(ID).loc[query.index]
        enough = joined["size"].ge(minimum)
        result.loc[enough,"physical_prior_sec"] = joined.loc[enough,"median"]
        result.loc[enough,"physical_prior_n"] = joined.loc[enough,"size"]
    return result.astype("float32")


def build_anchors(rows, reference, use_dst):
    movement = pd.to_datetime(rows[TIME],utc=True)
    local = movement.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    offset = (local-movement.dt.tz_localize(None)).dt.total_seconds().to_numpy()
    offset = np.where(rows.ADEP_mvt.eq("LIRF"),offset,0.)
    ordinary = reference.physical_prior_sec.to_numpy(float)
    anchors = np.column_stack([ordinary,ordinary+offset,rows.schedule_sec.to_numpy(float),ordinary+86400])
    if not use_dst:
        anchors[:,1] = np.nan
    return anchors,offset


def declare():
    definition = {"arms":["three_anchor","dst_four_anchor"],"folds":["F1","F3"],"stage":"tune only",
        "anchors":"Earlier-knownNM physical rawY median; sameprior+Europe/Rome observed local offset forRome only; observed schedule gap; sameprior+86400. Missing and duplicate anchors masked, lowest original index retained.",
        "ordinary_prior":"Originalpurged knownNM fit cohort only, strictly earlier UTCmonths. airportmedian n>=1, airport/runway n>=100, airport/stand/runway n>=20; no labelduration filter. Empty airport history900.",
        "identity":"Y=a(X) dot q(Y,X)+r exactly. q linearlyinterpolates adjacent sorted availableanchors; outofhull residual retained and regressed. qtraining uses upto2weighted replicas/originalrow with weightsum1.",
        "classifier":"CatBoost MultiClass400 depth4 lr.05 l2=30 bootstrapNo seed20260916 CPU2. Report stages50,100,200,400; tune rawMSE chooses one perarm/fold. Mask unavailable/duplicateanchor probabilities and renormalize at inference.",
        "residual":"Shared exactoutsidehull remainder target, same botharms. Existing CatBoostRMSE600 depth5 withtuneearlystop80. No fulltarget residualmodel.",
        "features":"Same frozen lexical85-ishfeatures for botharms plus earlierphysicalprior, support, Romeoffset, observed anchorvalues/missingflags. Threeanchorcontrol has sameobservedoffsetfeature; isolates outputrepresentation.",
        "gate":"No score prediction or refit unless DSTarm beats threeanchorcontrol AND previouslexicalmodel in both June andOctober rawmissingMSE. Parent approval required for later separatelydeclared scoreexperiment.",
        "adaptive_provenance":"Exposedscore motivated lexical/DST diagnostics; this is development search, not pristineholdout.",
        "source_sha256":sha256(__file__),"dependency_source_hashes":{str(Path(p).relative_to(ROOT)):sha256(p) for p in [prior.__file__,prior.common.__file__,ROOT / "review_work/tail240_20260916/forensics/run_lexical.py"]},"lexical_completion_sha256":sha256(LEXICAL / "completion.json"),"resources":"2CPU6GiB8GiBhostreserve"}
    path = OUT / "protocol.json"
    if path.exists():
        assert read_json(path)["declaration"] == definition
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        write_json(path,{"created_utc":utc_now(),"declaration":definition})


def load_data():
    meta_path = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(meta_path) == read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][meta_path.name]
    meta = pd.read_parquet(meta_path)
    assert sha256(LEXICAL / "all_missing_features.parquet") == read_json(LEXICAL / "completion.json")["feature_sha256"]
    x = pd.read_parquet(LEXICAL / "all_missing_features.parquet").set_index(ID)
    assert x.index.equals(pd.Index(meta.loc[~np.isfinite(meta.proxy_sec),ID],name=ID))
    raw_hashes = read_json(ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]
    parts = []
    for path in sorted(prior.RAW.glob("training_*.parquet")):
        assert sha256(path) == raw_hashes[path.name]
        parts.append(pd.read_parquet(path,columns=[ID,"STAND_mvt","RUNWAY_mvt"],filters=[("PHASE_mvt","=","DEP")],dtype_backend="pyarrow"))
    details = pd.concat(parts,ignore_index=True).set_index(ID)
    data = meta.set_index(ID).join(details,validate="one_to_one")
    data[TIME] = pd.to_datetime(data[TIME],utc=True)
    for column in ["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]:
        data[column] = data[column].astype("string").fillna("<MISSING>")
    resource_check()
    return x,meta,data


def run_fold(fold,x,meta,data):
    folder = OUT / fold
    folder.mkdir(exist_ok=False)
    indices,split,_ = prior.common.fold_data(meta,fold,full=True)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    selected = {stage:meta.iloc[indices[stage][missing[indices[stage]]]][ID] for stage in ["fit","tune"]}
    assert not set(selected["fit"]).intersection(selected["tune"])
    known = data.loc[meta.iloc[indices["fit"]][ID]]
    known = known.loc[np.isfinite(known.proxy_sec)]
    frames,anchors,targets = {},{},{}
    for stage,ids in selected.items():
        query = data.loc[ids].copy()
        refs = []
        for month,rows in query.groupby(query[TIME].dt.strftime("%Y-%m")):
            refs.append(physical_reference(known.loc[known[TIME].lt(pd.Timestamp(month+"-01",tz="UTC"))],rows))
        ref = pd.concat(refs).loc[query.index]
        full,offset = build_anchors(query,ref,True)
        anchors[stage] = full
        frames[stage] = pd.concat([x.loc[ids],ref],axis=1)
        frames[stage]["rome_local_offset_sec"] = offset.astype("float32")
        for j in range(4):
            frames[stage][f"source_anchor_{j}_sec"] = np.nan_to_num(full[:,j],nan=-999999).astype("float32")
            frames[stage][f"source_anchor_{j}_missing"] = (~np.isfinite(full[:,j])).astype("float32")
        targets[stage] = query[TARGET].to_numpy(float)
        frames[stage].reset_index().to_parquet(folder / f"{stage}_features.parquet",index=False)
        pd.DataFrame({ID:ids.to_numpy(),TARGET:targets[stage],**{f"anchor_{j}":full[:,j] for j in range(4)}}).to_parquet(folder / f"{stage}_anchors.parquet",index=False)
    qfit,residual_fit,_ = encode(targets["fit"],anchors["fit"])
    _,residual_tune,_ = encode(targets["tune"],anchors["tune"])
    residual_model,residual_evidence = prior.fit_regressor(frames["fit"],residual_fit,tune=(frames["tune"],residual_tune),seed=SEED,threads=2)
    residual_prediction = residual_model.predict(frames["tune"],thread_count=2)
    joblib.dump(residual_model,folder / "residual_model.joblib")
    reports = {}
    previous_path = LEXICAL / "lexical/models" / f"historical_template_{fold}_s20260916" / "tune_predictions.parquet"
    previous_record = read_json(previous_path.parent / "manifest.json")
    assert sha256(previous_path) == previous_record["outputs"][previous_path.name]
    previous = pd.read_parquet(previous_path)
    np.testing.assert_array_equal(previous[ID],selected["tune"])
    previous_mse = float(np.mean((previous.prediction_sec.to_numpy()-targets["tune"])**2))
    for arm in ["three_anchor","dst_four_anchor"]:
        afit,atune = anchors["fit"].copy(),anchors["tune"].copy()
        if arm == "three_anchor":
            afit[:,1],atune[:,1] = np.nan,np.nan
        qfit,rfit,_ = encode(targets["fit"],afit)
        _,rtune,_ = encode(targets["tune"],atune)
        np.testing.assert_allclose(rfit,residual_fit,rtol=0,atol=1e-9)
        np.testing.assert_allclose(rtune,residual_tune,rtol=0,atol=1e-9)
        replica_x,classes,weights,replica_ids = replicas(frames["fit"],qfit)
        assert set(replica_ids) == set(selected["fit"]) and not set(replica_ids).intersection(selected["tune"])
        model = CatBoostClassifier(iterations=400,depth=4,learning_rate=.05,l2_leaf_reg=30,loss_function="MultiClass",task_type="CPU",thread_count=2,random_seed=SEED,bootstrap_type="No",verbose=False,allow_writing_files=False)
        model.fit(replica_x,classes,sample_weight=weights,cat_features=prior.cats(replica_x))
        joblib.dump(model,folder / f"{arm}.joblib")
        report = []
        for count in STEPS:
            p = expectation(model.predict_proba(frames["tune"],ntree_end=count,thread_count=2),model.classes_,atune)+residual_prediction
            assert np.isfinite(p).all()
            pd.DataFrame({ID:selected["tune"].to_numpy(),TARGET:targets["tune"],"prediction_sec":p}).to_parquet(folder / f"{arm}_{count}_tune.parquet",index=False)
            report.append({"trees":count,"mse":float(np.mean((p-targets["tune"])**2)),"rmse":float(np.sqrt(np.mean((p-targets["tune"])**2)))})
        best = min(report,key=lambda r:(r["mse"],r["trees"]))
        reports[arm] = {"stages":report,"best":best,"replica_rows":len(replica_x),"original_rows":len(frames["fit"]),"replica_weight_sum":float(weights.sum()),"classes":model.classes_.tolist()}
        print("ANCHOR",fold,arm,best,flush=True)
        del model,replica_x
        gc.collect()
        resource_check()
    passed = reports["dst_four_anchor"]["best"]["mse"] < min(reports["three_anchor"]["best"]["mse"],previous_mse)
    record = {"status":"complete","fold":fold,"split":split,"protocol_sha256":sha256(OUT / "protocol.json"),"previous_lexical_mse":previous_mse,
        "residual":residual_evidence,"reports":reports,"gate_passed":bool(passed),"score_predictions_read_or_computed":False,"score_labels_used_in_training_tuning_or_prior":False,"full_metadata_including_score_rows_loaded":True,
        "ids":{s:{"n":len(ids),"hash":object_hash(ids.tolist())} for s,ids in selected.items()},
        "outputs":{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}}
    write_json(folder / "manifest.json",record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--declare-only",action="store_true")
    args=parser.parse_args()
    declare()
    if args.declare_only:
        print("Declared exact anchor mixture, tune-only",flush=True)
        return
    x,meta,data=load_data()
    results={fold:run_fold(fold,x,meta,data) for fold in ["F1","F3"]}
    write_json(OUT / "summary.json",{"status":"complete","created_utc":utc_now(),"source_sha256":sha256(__file__),
        "gate_passed":all(r["gate_passed"] for r in results.values()),"folds":results,"peak_bytes":resource_check()})


if __name__ == "__main__":
    main()
