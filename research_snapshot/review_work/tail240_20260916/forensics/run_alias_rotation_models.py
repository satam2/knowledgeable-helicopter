"""Three-arm tune-only alias-rotation ablation with exact historical control."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="1"
import argparse
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from alias_rotation_v2 import ROOT,OUT as PILOT,BASE,SUFFIXES
sys.path.insert(0,str(ROOT / "review_work/breakthrough_20260916/missing"))
import run_id_context as context
prior=context.base
from run_lexical import resource_check
from taxiout.artifacts import read_json,write_json,sha256,utc_now,object_hash
ID,TARGET,TIME=prior.ID,prior.TARGET,prior.MOVEMENT
OUT=ROOT / "private_runs/tail240_20260916/forensics/alias_rotation/models_tune_v1"
LEXICAL=ROOT / "private_runs/tail240_20260916/forensics/models/lexical_v1"
ARMS=["control76","raw84","alias100"]


def extensions(index,airports,raw,pilot):
    if not index.is_unique or not airports.index.equals(index):
        raise ValueError("Expected exact unique query ID order")
    if not index.isin(raw.index).all() or not raw.index.is_unique or not pilot.index.is_unique:
        raise ValueError("Cache IDs are missing or duplicated")
    raw=raw.loc[index,["opdi_"+s for s in SUFFIXES]].copy()
    raw.columns=["alias_raw_"+s for s in SUFFIXES]
    blocks=[raw]
    rome=index[airports.eq("LIRF")]
    if set(rome)!=set(pilot.index):
        raise ValueError("Pilot must cover every Rome missing query exactly")
    np.testing.assert_array_equal(raw.loc[rome].to_numpy(),pilot.loc[rome,raw.columns].to_numpy())
    for mode in ["learned","lexical"]:
        block=raw.copy()
        block.columns=[f"alias_{mode}_"+s for s in SUFFIXES]
        block.loc[rome]=pilot.loc[rome,block.columns].to_numpy()
        blocks.append(block)
    result=pd.concat(blocks,axis=1).replace([np.inf,-np.inf],np.nan).fillna(-999999).astype("float32")
    assert result.index.equals(index)
    return result


def stability(y,new,control,dates):
    y,new,control=np.asarray(y,float),np.asarray(new,float),np.asarray(control,float)
    dates=pd.to_datetime(dates,utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    rows=[]
    for day in sorted(set(dates)):
        keep=dates!=day
        rows.append({"removed_day":day,"remaining_n":int(keep.sum()),
            "delta_rmse":float(np.sqrt(np.mean((new[keep]-y[keep])**2))-np.sqrt(np.mean((control[keep]-y[keep])**2)))})
    return {"day_removals":rows,"all_day_removals_improve":all(r["delta_rmse"]<0 for r in rows)}


def declare():
    definition={"arms":ARMS,"scope":"Original fullmissing fit and tune only F1/F3; no refit or scoreprediction",
        "control":"Exact76airport+IDcontext inputs and historical_template original600depth5 params/earlystop80/seed20260916/CPU2; control tune bitparity required.",
        "raw84":"Control76 +frozenoldOPDIraw8, renamed alias_raw fields; allmissing.",
        "alias100":"Control76 +raw8 +separatelearned8/lexical8. Rome uses correctedpilotv2; outsideRome learned/lexicalblocks duplicateoldraw8 sinceextension coverageRomeonly. No rawICAO/publicIDmodelinputs.",
        "routing":"Tune gate and anylatercandidate application restrictedtoRome missing; allnonRome/finitelaterpredictions mustremainbaseline. Allmissing rows retainedforfitting.",
        "gate":"BothJune/OctoberRome rawMSE alias100 improvescontrol76 ANDraw84; eachcomparison allleaveoneUTCdayout RMSEimprove. No scoreworkuntilgate+separatelydeclaredprotocol.",
        "availability":"Observedearliermonthprefixmapping/publicflightcontexts. Retrospective samecalendar-month completedlegcontext, no target/blockreadbybuilder. Historicaltemplatelabels strictlyearliermonths afteroriginalpurges.",
        "licensing":"Local attributed noncommercial research only; prizeeligibilityunresolved; no downloads.",
        "params":prior.PARAMS,"source_sha256":sha256(__file__),"dependency_sources":{str(Path(p).relative_to(ROOT)):sha256(p) for p in [context.__file__,prior.__file__,prior.common.__file__]},
        "pilot_manifest_sha256":sha256(PILOT / "manifest.json"),"old_opdi_manifest_sha256":sha256(BASE / "manifest.json"),
        "resource_limit":"2CPU6GiB minimum8GiBavailable"}
    path=OUT / "protocol.json"
    if path.exists():assert read_json(path)["declaration"]==definition
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        write_json(path,{"created_utc":utc_now(),"declaration":definition})


def load_features():
    meta_path=ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(meta_path)==read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][meta_path.name]
    meta=pd.read_parquet(meta_path)
    source=LEXICAL / "all_missing_features.parquet"
    assert sha256(source)==read_json(LEXICAL / "completion.json")["feature_sha256"]
    x=pd.read_parquet(source).set_index(ID)
    x=x[[c for c in x if not c.startswith("lex_")]]
    assert list(x)==read_json(ROOT / "private_runs/breakthrough_20260916/missing/models/historical_template_F1_s20260916/manifest.json")["features_used"]
    assert np.array_equal(x.index,meta.loc[~np.isfinite(meta.proxy_sec),ID])
    peerpath=context.CACHE / "features.parquet"
    assert sha256(peerpath)==read_json(context.CACHE / "audit.json")["feature_sha256"]
    peer=pd.read_parquet(peerpath).set_index(ID)
    assert np.array_equal(peer.index,meta[ID])
    t=meta.set_index(ID).loc[x.index,TIME]
    x=pd.concat([x,context.id_context_features(x,peer.loc[x.index],t)],axis=1)
    assert list(x)==read_json(context.OUT / "feature_manifest.json")["columns"] and len(x.columns)==76
    for folder,name in [(BASE,"training_features.parquet"),(PILOT,"features.parquet")]:
        assert sha256(folder / name)==read_json(folder / "manifest.json")["outputs"][name]
    ext=extensions(x.index,meta.set_index(ID).loc[x.index,"ADEP_mvt"],pd.read_parquet(BASE / "training_features.parquet").set_index(ID),pd.read_parquet(PILOT / "features.parquet").set_index(ID))
    pd.concat([x,ext],axis=1).reset_index().to_parquet(OUT / "features100.parquet",index=False)
    resource_check()
    return x,ext,meta


def run_fold(fold,x,ext,meta):
    folder=OUT / fold
    folder.mkdir(exist_ok=False)
    idx,split,_=prior.common.fold_data(meta,fold,full=True)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    selected={s:meta.iloc[idx[s][missing[idx[s]]]] for s in ["fit","tune"]}
    assert not set(selected["fit"][ID]).intersection(selected["tune"][ID])
    y={s:d[TARGET].to_numpy(float) for s,d in selected.items()}
    times={s:pd.Series(pd.to_datetime(d[TIME],utc=True).to_numpy(),index=pd.Index(d[ID],name=ID)) for s,d in selected.items()}
    predictors={"control76":x,"raw84":pd.concat([x,ext.iloc[:,:8]],axis=1),"alias100":pd.concat([x,ext],axis=1)}
    results,predictions={},{}
    rome=selected["tune"].ADEP_mvt.eq("LIRF").to_numpy()
    for arm,features in predictors.items():
        frames={s:features.loc[d[ID]] for s,d in selected.items()}
        model,tuning=prior.fit_arm("historical_template",frames["fit"],y["fit"],times["fit"],
            tune=(frames["tune"],y["tune"],times["tune"]),seed=20260916,threads=2)
        pred=prior.predict_arm(model,frames["tune"],times["tune"])
        assert np.isfinite(pred).all()
        joblib.dump(model,folder / f"{arm}.joblib")
        replay=prior.predict_arm(joblib.load(folder / f"{arm}.joblib"),frames["tune"],times["tune"])
        np.testing.assert_allclose(pred,replay,atol=1e-9,rtol=0)
        if arm=="control76":
            old=context.OUT / "models" / f"historical_template_{fold}_s20260916/tune_predictions.parquet"
            assert sha256(old)==read_json(old.parent / "manifest.json")["outputs"][old.name]
            previous=pd.read_parquet(old)
            np.testing.assert_array_equal(previous[ID],selected["tune"][ID])
            np.testing.assert_array_equal(previous.prediction_sec,pred)
        predictions[arm]=pred
        pd.DataFrame({ID:selected["tune"][ID].to_numpy(),TARGET:y["tune"],"prediction_sec":pred,"rome":rome}).to_parquet(folder / f"{arm}_tune.parquet",index=False)
        results[arm]={"fit":tuning,"all_missing_rmse":float(np.sqrt(np.mean((pred-y["tune"])**2))),
            "rome_rmse":float(np.sqrt(np.mean((pred[rome]-y["tune"][rome])**2))),"rome_n":int(rome.sum()),"features":list(features)}
        print("ROTATION_MODEL",fold,arm,results[arm]["all_missing_rmse"],results[arm]["rome_rmse"],flush=True)
        del model,frames
        gc.collect()
        resource_check()
    comparisons={arm:stability(y["tune"][rome],predictions["alias100"][rome],predictions[arm][rome],times["tune"].iloc[rome]) for arm in ["control76","raw84"]}
    gate=all(results["alias100"]["rome_rmse"]<results[arm]["rome_rmse"] and comparisons[arm]["all_day_removals_improve"] for arm in comparisons)
    record={"status":"complete","split":split,"fold":fold,"results":results,"comparisons":comparisons,"gate_passed":gate,
        "original_control_tune_exact":True,"score_predictions_computed":False,"metadata_fulltable_loaded_scorelabels_notused":True,
        "ids":{s:{"n":len(d),"hash":object_hash(d[ID].tolist())} for s,d in selected.items()},
        "outputs":{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}}
    write_json(folder / "manifest.json",record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--declare-only",action="store_true");args=parser.parse_args()
    declare()
    if args.declare_only:
        print("Declared three-arm rotation tune-only",flush=True);return
    x,ext,meta=load_features()
    results={f:run_fold(f,x,ext,meta) for f in ["F1","F3"]}
    write_json(OUT / "summary.json",{"status":"complete","source_sha256":sha256(__file__),"protocol_sha256":sha256(OUT / "protocol.json"),
        "feature_sha256":sha256(OUT / "features100.parquet"),"folds":results,"gate_passed":all(r["gate_passed"] for r in results.values())})


if __name__=="__main__":main()
