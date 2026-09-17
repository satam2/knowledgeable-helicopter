"""Tune-selected propensity adaptation for missing-NM raw-label prediction."""
import lightgbm
import argparse
import gc
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import prepare
from prepare import common,ID,TARGET,MOVEMENT,OUT,ROOT,HERE,read_json,write_json,sha256,object_hash,utc_now
import adapter
from taxiout.metrics import evaluate,paired_stability


def declare(args,features):
    files=[Path(__file__),HERE/'adapter.py',HERE/'prepare.py',Path(prepare.original.__file__),Path(common.__file__),
        ROOT/'review_work/campaign_20260916/lgbm_adapter.py']
    declaration=dict(seed=args.seed,threads=args.threads,folds=args.folds,ratios=adapter.RATIOS,
        propensity=adapter.propensity_params(args.seed,args.threads),regression=adapter.regression_params(args.seed,args.threads),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
        feature_manifest_sha256=sha256(OUT/'features_manifest.json'),feature_sha256=features['feature_sha256'],
        features=features['features'],cold_prior=adapter.COLD_PRIOR,
        weights='Missingrowsunitweight. Known odds=p/(1-p) capped20; deterministic boundedrescaling to knownmass ratio*missingcount, finalknownweights also<=20. p numericalclip1e-6.',
        propensity_crossfit='Trainingweights for each UTCmonth use onlyearlierallowedmonth availability labels. No Y everpassed. Firstmonth fixedp.01. Cachekey hashes earlierIDs/availability/params/features.',
        fresh_refit='Recomputechronological weights on originalfit+tune; fullfit+tune propensity freshlyfitted andsaved forscore availabilitydiagnostics.',
        selection='Tuneallratios on originalmissing-NM tune RMSE only; tie chooseslower ratio. Refit ratio0 andbestpositive ratio. Candidate usesbestoverall ofall4; controlandbestpositive alsoexplicit.',
        target='RawY squaredloss withoutlabelclipping. Everymissingtraininglabel unitweight; knownlabels havepositive propensityweights forpositive ratios.',
        causality='Ownairport observations and existingstrictpriortraffic only. Covariateshift hypothesis; notmissingatrandom assumption or guarantee.',
        diagnostics='Propensityoverlap/AUC/Brier, knownweightESS andcapcounts overall/byairport. No scorelabels inselection orweightconstruction.',
        protected='All finite-NM scorepredictions remain V2exact; onlymissing routes replace, fixed25percentblend.',
        resources='CPUonly2threads, minimum8GiB availableRAM; centrallyscheduled; no realtrainingduringpreparation.')
    path=OUT/'protocol.json'
    if path.exists():
        if read_json(path)['declaration']!=declaration:
            raise ValueError('Frozenadaptation protocolchanged')
    else:
        write_json(path,dict(created_utc=utc_now(),declaration=declaration))
    return path


class PropensityStore:
    def __init__(self,args,features):
        self.args=args
        self.feature_sha256=features['feature_sha256']
        self.root=OUT/'propensity_models'
        self.root.mkdir(parents=True,exist_ok=True)

    def fit(self,x,missing):
        identity=dict(ids=object_hash(x.index.tolist()),missing=object_hash(np.asarray(missing,int).tolist()),
            feature_sha256=self.feature_sha256,params=adapter.propensity_params(self.args.seed,self.args.threads),
            source_sha256=sha256(HERE/'adapter.py'))
        name=object_hash(identity)
        path=self.root/name
        if path.exists():
            receipt=read_json(path/'manifest.json')
            if receipt['status']!='complete' or receipt['identity']!=identity or sha256(path/'model.joblib')!=receipt['model_sha256']:
                raise ValueError('Incomplete/differentpropensitycache retained')
            return joblib.load(path/'model.joblib')
        path.mkdir()
        write_json(path/'manifest.json',dict(status='running',identity=identity,created_utc=utc_now()))
        model=adapter.fit_propensity(x,missing,self.args.seed,self.args.threads)
        joblib.dump(model,path/'model.joblib',compress=0)
        write_json(path/'manifest.json',dict(status='complete',identity=identity,created_utc=utc_now(),model_sha256=sha256(path/'model.joblib')))
        return model


def weight_summary(x,missing,weights,evidence):
    rows=[]
    for airport,positions in x.groupby('airport',observed=True).indices.items():
        selected=positions[~missing[positions]]
        w=weights[selected]
        rows.append(dict(airport=str(airport),missing_rows=int(missing[positions].sum()),known_rows=len(selected),
            known_weight=float(w.sum()),known_ess=float(w.sum()**2/(w@w)) if np.any(w>0) else 0.,known_capped=int((w>=20-1e-9).sum())))
    return dict(evidence,airports=rows)


def run_fold(args,fold,x,meta,protocol,store):
    dest=OUT/'models'/f'{fold}_s{args.seed}'
    if dest.exists():
        old=read_json(dest/'manifest.json')
        if old['status']!='complete' or old['protocol_sha256']!=sha256(protocol):
            raise ValueError('Existingincomplete/differentadaptationrun retained')
        for file,digest in old['outputs'].items():
            assert sha256(dest/file)==digest
        print('REUSED',fold,flush=True)
        return
    if psutil.virtual_memory().available<8*1024**3:
        raise MemoryError('Need8GiBavailable RAM')
    idx,split,ids=common.fold_data(meta,fold,full=True)
    reference,original=common.reference(fold)
    assert object_hash(split)==object_hash(original['split'])
    np.testing.assert_array_equal(reference[ID],meta.iloc[idx['score']][ID])
    np.testing.assert_array_equal(reference[TARGET],meta.iloc[idx['score']][TARGET])
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    scoremask=missing[idx['score']]
    dest.mkdir(parents=True)
    record=dict(status='running',created_utc=utc_now(),fold=fold,split=split,fit_ids=ids,
        protocol_sha256=sha256(protocol),source_hashes=read_json(protocol)['declaration']['source_hashes'])
    write_json(dest/'manifest.json',record)
    started=time.monotonic()
    try:
        frames={stage:x.iloc[rows] for stage,rows in idx.items()}
        y=meta[TARGET].to_numpy(float)
        fm=missing[idx['fit']]
        tm=missing[idx['tune']]
        probabilities,history=adapter.crossfit(frames['fit'],fm,meta.iloc[idx['fit']][MOVEMENT],fit_function=store.fit)
        fullprop=store.fit(frames['fit'],fm)
        tuneprob=adapter.predict_propensity(fullprop,frames['tune'])
        record['propensity_fit']=dict(crossfit_history=history,fit_oof=adapter.overlap(probabilities,fm),tune=adapter.overlap(tuneprob,tm))
        pd.DataFrame({ID:frames['fit'].index,'p_missing':probabilities,'missing_source':fm}).to_parquet(dest/'fit_propensity_oof.parquet',index=False)
        joblib.dump(fullprop,dest/'fit_propensity.joblib',compress=0)
        tuned={}
        for ratio in adapter.RATIOS:
            w,evidence=adapter.weights(probabilities,fm,ratio)
            model,fitinfo=adapter.fit_regression(frames['fit'],y[idx['fit']],w,
                (frames['tune'].loc[tm],y[idx['tune'][tm]]),seed=args.seed,threads=args.threads)
            pred=adapter.predict_regression(model,frames['tune'].loc[tm])
            rmse=float(np.sqrt(np.mean((pred-y[idx['tune'][tm]])**2)))
            name=str(ratio).replace('.','p')
            joblib.dump(model,dest/f'fit_regression_ratio{name}.joblib',compress=0)
            pd.DataFrame({ID:frames['tune'].loc[tm].index,'prediction_sec':pred}).to_parquet(dest/f'tune_ratio{name}.parquet',index=False)
            tuned[ratio]=dict(ratio=ratio,tune_missing_rmse=rmse,fit=fitinfo,weights=weight_summary(frames['fit'],fm,w,evidence))
            print('ADAPT_TUNE',fold,ratio,rmse,evidence,flush=True)
            del model
            gc.collect()
        best=min(adapter.RATIOS,key=lambda ratio:(tuned[ratio]['tune_missing_rmse'],ratio))
        positive=min([r for r in adapter.RATIOS if r>0],key=lambda ratio:(tuned[ratio]['tune_missing_rmse'],ratio))
        record.update(tuning={str(k):v for k,v in tuned.items()},selected_ratio=best,best_positive_ratio=positive)
        write_json(dest/'manifest.json',record)
        rm=missing[idx['refit']]
        rp,rhistory=adapter.crossfit(frames['refit'],rm,meta.iloc[idx['refit']][MOVEMENT],fit_function=store.fit)
        fullrefit=store.fit(frames['refit'],rm)
        scoreprob=adapter.predict_propensity(fullrefit,frames['score'])
        joblib.dump(fullrefit,dest/'refit_propensity.joblib',compress=0)
        pd.DataFrame({ID:frames['refit'].index,'p_missing':rp,'missing_source':rm}).to_parquet(dest/'refit_propensity_oof.parquet',index=False)
        record['propensity_refit']=dict(crossfit_history=rhistory,refit_oof=adapter.overlap(rp,rm),score_observation_only=adapter.overlap(scoreprob,scoremask))
        predictions={}
        refits={}
        for ratio in [0.,positive]:
            w,evidence=adapter.weights(rp,rm,ratio)
            model,fitinfo=adapter.fit_regression(frames['refit'],y[idx['refit']],w,steps=tuned[ratio]['fit']['steps'],seed=args.seed,threads=args.threads)
            name=str(ratio).replace('.','p')
            path=dest/f'model_ratio{name}.joblib'
            joblib.dump(model,path,compress=0)
            prediction=adapter.predict_regression(model,frames['score'].loc[scoremask])
            replay=float(np.max(np.abs(prediction-adapter.predict_regression(joblib.load(path),frames['score'].loc[scoremask]))))
            assert replay<=1e-9 and np.isfinite(prediction).all()
            predictions[ratio]=prediction
            refits[str(ratio)]=dict(fit=fitinfo,weights=weight_summary(frames['refit'],rm,w,evidence),reload_max_abs_delta=replay)
        ref=reference.prediction_sec.to_numpy(float)
        reports={}
        for role,ratio in [('control',0.),('adapted',positive),('selected',best)]:
            candidate=ref.copy()
            candidate[scoremask]=predictions[ratio]
            for suffix,values in [('candidate',candidate),('blend25',ref+.25*(candidate-ref))]:
                assert np.array_equal(values[~scoremask],ref[~scoremask])
                frame=reference.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
                frame['prediction_sec']=values
                metrics,errors=evaluate(frame,meta.iloc[idx['score']][[ID,TARGET]])
                variant=role+'_'+suffix
                errors.to_parquet(dest/f'{variant}.parquet',index=False)
                reports[variant]=dict(ratio=ratio,metrics=metrics,stability=paired_stability(reference,errors,repetitions=500))
        record.update(status='complete',completed_utc=utc_now(),refits=refits,reports=reports,
            runtime_sec=time.monotonic()-started,protected_finite_routes_exact=True,
            peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss))
        for source,digest in record['source_hashes'].items():
            assert sha256(ROOT/source)==digest
        record['outputs']={p.name:sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(dest/'manifest.json',record)
        print('ADAPT_RESULT',fold,best,{k:v['metrics']['overall']['rmse_sec'] for k,v in reports.items()},flush=True)
    except Exception as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        write_json(dest/'manifest.json',record)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    x,meta,features=prepare.load()
    protocol=declare(args,features)
    if args.declare_only:
        print('DECLARED',x.shape,protocol,flush=True)
        return
    store=PropensityStore(args,features)
    for fold in args.folds:
        run_fold(args,fold,x,meta,protocol,store)
        gc.collect()


if __name__=='__main__':
    main()
