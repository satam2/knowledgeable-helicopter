"""Controlled family/formulation screens; no competition submission path."""
import argparse
import gc
import importlib
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import psutil
from importlib.metadata import version

from common import HERE, OUT, WORKSPACE, SEED, ID, TARGET, MOVEMENT, load_data, fold_data, reference, freeze
from common import read_json, write_json, sha256, object_hash, utc_now, external_path
from taxiout.metrics import evaluate, paired_stability


def adapter(family):
    if family in ['catboost','ridge']:
        mod=importlib.import_module('baseline_adapters')
        return getattr(mod,'fit_'+family),mod.predict,mod
    mod=importlib.import_module({'lightgbm':'lgbm_adapter','xgboost':'xgb_adapter','tabm':'tabm_adapter'}[family])
    return mod.fit,mod.predict,mod


def ordinary(meta):
    p=meta.proxy_sec.to_numpy(float)
    return np.isfinite(p)&(p>=0)&(p<=7200)


def run_one(args,x,meta,fold):
    name=f'{args.family}_{args.target}_{args.features}_{"full" if args.full else "s200k"}_{fold}_s{args.seed}'
    dest=external_path(OUT/'models'/name)
    if dest.exists():
        old=read_json(dest/'manifest.json')
        if old['status']!='complete':
            raise ValueError(f'Incomplete run preserved: {dest}')
        for f,h in old['outputs'].items():
            assert sha256(dest/f)==h
        print('REUSED',name,flush=True)
        return
    dest.mkdir(parents=True)
    fit,predict,mod=adapter(args.family)
    idx,split,sampled=fold_data(meta,fold,args.full)
    ref,refrec=reference(fold)
    assert object_hash(split)==object_hash(refrec['split'])
    assert np.array_equal(meta.iloc[idx['score']][ID],ref[ID])
    assert np.array_equal(meta.iloc[idx['score']][TARGET],ref[TARGET])
    rec=dict(status='running',name=name,created_utc=utc_now(),family=args.family,target=args.target,
        features=args.features,full=args.full,seed=args.seed,split=split,sampled=sampled,
        code_hashes={str(p.relative_to(HERE)):sha256(p) for p in [HERE/'common.py',HERE/'run.py',HERE/'baseline_adapters.py',__import__('pathlib').Path(mod.__file__)]},
        protocol_sha256=sha256(OUT/'protocol.json'),threads=args.threads,
        libraries={p:version(p) for p in ['numpy','pandas','catboost','lightgbm','xgboost','scikit-learn','torch','tabm']})
    write_json(dest/'manifest.json',rec)
    begin=time.monotonic()
    try:
        if psutil.virtual_memory().available < 8*1024**3:
            raise MemoryError('8GB host RAM reserve required')
        rows={k:v.copy() for k,v in idx.items()}
        if args.target=='correction':
            for stage in ['fit','tune','refit']:
                rows[stage]=rows[stage][ordinary(meta.iloc[rows[stage]])]
        y=meta[TARGET].to_numpy(float)
        target=y.copy()
        if args.target=='correction':
            target=target-meta.proxy_sec.to_numpy(float)
        frames={k:x.iloc[v].copy() for k,v in rows.items()}
        offsets={k:np.zeros(len(v)) for k,v in rows.items()}
        prior_evidence={}
        if args.target=='physical':
            from aviation.arrival_features import ChronologicalStandRunwayReference
            # Filled by the aviation adapter contract; q10 is a target origin, not a target clip.
            raise NotImplementedError('Physical prior interface awaits verified aviation module')
        start=time.monotonic()
        model,tune_info=fit(frames['fit'],target[rows['fit']],
            (frames['tune'],target[rows['tune']]),seed=args.seed,threads=args.threads)
        tuning_runtime=time.monotonic()-start
        tune_pred=np.asarray(predict(model,frames['tune']),float)
        if args.target=='correction':
            tune_pred+=meta.iloc[rows['tune']].proxy_sec.to_numpy(float)
        pd.DataFrame({ID:meta.iloc[rows['tune']][ID].to_numpy(),'prediction_sec':tune_pred}).to_parquet(dest/'tune_predictions.parquet',index=False)
        joblib.dump(model,dest/'fit_model.joblib',compress=0)
        del model
        gc.collect()
        start=time.monotonic()
        model,refit_info=fit(frames['refit'],target[rows['refit']],steps=int(tune_info['steps']),seed=args.seed,threads=args.threads)
        refit_runtime=time.monotonic()-start
        joblib.dump(model,dest/'model.joblib',compress=0)
        start=time.monotonic()
        pred=np.asarray(predict(model,frames['score']),float)
        infer_runtime=time.monotonic()-start
        repeated=np.asarray(predict(joblib.load(dest/'model.joblib'),frames['score']),float)
        replay=float(np.max(np.abs(pred-repeated)))
        assert replay<=1e-5 and np.isfinite(pred).all()
        assert pred.shape==(len(ref),)
        score_meta=meta.iloc[idx['score']]
        if args.target=='correction':
            changed=ordinary(score_meta)
            combined=ref.prediction_sec.to_numpy().copy()
            combined[changed]=score_meta.proxy_sec.to_numpy()[changed]+pred[changed]
        else:
            combined=pred
        variants={'candidate':combined,'blend25':.75*ref.prediction_sec.to_numpy()+.25*combined}
        if args.target=='direct':
            missing=~np.isfinite(score_meta.proxy_sec.to_numpy())
            special=ref.prediction_sec.to_numpy().copy()
            special[missing]=pred[missing]
            variants['missing_only']=special
            variants['missing_blend25']=.75*ref.prediction_sec.to_numpy()+.25*special
        reports={}
        for variant,prediction in variants.items():
            f=ref.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
            f['prediction_sec']=prediction
            metrics,errors=evaluate(f,meta.iloc[idx['score']][[ID,TARGET]])
            errors.to_parquet(dest/f'{variant}.parquet',index=False)
            reports[variant]={'metrics':metrics,'stability':paired_stability(ref,errors,repetitions=500)}
        rec.update(status='complete',completed_utc=utc_now(),tune=tune_info,refit=refit_info,
            tuning_runtime_sec=tuning_runtime,refit_runtime_sec=refit_runtime,inference_runtime_sec=infer_runtime,
            runtime_sec=time.monotonic()-begin,peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
            reload_max_abs_delta=replay,features_used=list(x),reports=reports,
            fit_ids={stage:{'n':len(rows[stage]),'hash':object_hash(meta.iloc[rows[stage]][ID].tolist())} for stage in rows})
        rec['outputs']={p.name:sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(dest/'manifest.json',rec)
        print('RESULT',name,{v:round(r['metrics']['overall']['rmse_sec'],6) for v,r in reports.items()},flush=True)
    except Exception as exc:
        rec.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-begin)
        write_json(dest/'manifest.json',rec)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--family',required=True,choices=['catboost','lightgbm','xgboost','ridge','tabm'])
    p.add_argument('--target',required=True,choices=['direct','correction','physical'])
    p.add_argument('--features',default='base',choices=['base','arrival','aviation'])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--full',action='store_true')
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    freeze()
    x,meta=load_data()
    if args.features!='base':
        raise NotImplementedError('Feature cache integration pending aviation validation')
    for fold in args.folds:
        run_one(args,x,meta,fold)
        gc.collect()
