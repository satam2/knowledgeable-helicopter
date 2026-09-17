"""Fixed missing-route chronological leaf sensitivity and normalized diagnostic."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import lightgbm as lgb
import argparse
import gc
import importlib.util
from pathlib import Path
import sys
import threading
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/robustness'))
import train as maintrain
common=maintrain.common
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as normalized
FOREST_SOURCE=ROOT/'review_work/breakthrough_20260916/models/missing_forest/adapter.py'
spec=importlib.util.spec_from_file_location('chronological_missing_forest',FOREST_SOURCE)
forest=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=forest
spec.loader.exec_module(forest)
BASE=ROOT/'private_runs/tail240_20260916/forensics/final_missing/v1'
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/robustness/tail/chronological/v1')
ARMS={'et_leaf1':1,'et_leaf20':20,'normalized_diagnostic':None}
PANELS={'F1':['2025-06','2025-07','2025-12'],'F3':['2025-10','2025-11','2025-12']}
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def guard():
    m=psutil.Process().memory_info()
    peak=max(m.rss,getattr(m,'peak_wset',m.rss))
    if peak>=4*1024**3 or psutil.virtual_memory().available<8*1024**3:
        raise MemoryError('Missing chronological process4GiB / hostreserve8GiB exceeded')
    return peak


def monitor(event,folder):
    while not event.wait(.1):
        try:
            guard()
        except Exception as exc:
            common.write_json(folder/'resource_failure.json',dict(error=repr(exc),utc=common.utc_now()))
            os._exit(87)


def assert_no_known_flight_reuse(frame):
    known=frame.FLIGHT_ID_mvt.dropna()
    if known.duplicated().any():
        raise ValueError('Known flight reuse would invalidate the inherited crossfit prior; no fit allowed')
    return dict(rows=len(frame),known_flight_ids=len(known),null_flight_ids=int(frame.FLIGHT_ID_mvt.isna().sum()),
                known_flight_ids_unique=True,null_policy='Unknown entities retained, never pooled as one flight')


def read_stage(fold,stage):
    frame=maintrain.read_stage(fold,stage)
    frame=frame.loc[~np.isfinite(frame.proxy_sec.to_numpy(float))].copy()
    assert TARGET not in frame and len(frame) and frame[ID].is_unique
    return frame


def declare():
    prep=common.read_json(BASE/'preparation.json')
    assert prep['status']=='passed' and prep['training_rows']==22470 and len(prep['feature_columns'])==76
    assert common.sha256(BASE/'training_features.parquet')==prep['outputs']['training_features.parquet']
    cohorts=common.read_json(maintrain.COHORTS/'manifest.json')
    checks={f:assert_no_known_flight_reuse(read_stage(f,'refit')) for f in PANELS}
    source_paths=[Path(__file__),Path(__file__).with_name('test_contract_v1.py'),Path(maintrain.__file__),
        FOREST_SOURCE,Path(forest.missing.__file__),Path(normalized.__file__),
        ROOT/'review_work/breakthrough_20260916/models/encoders.py']
    record=dict(arms=ARMS,panels=PANELS,seed=20260916,source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in source_paths},
        cohort_manifest_sha256=common.sha256(maintrain.COHORTS/'manifest.json'),
        ordinary_training_protocol_sha256=common.sha256(maintrain.OUT/'protocol.json'),
        preparation_sha256=common.sha256(BASE/'preparation.json'),
        feature_sha256=prep['outputs']['training_features.parquet'],feature_columns=prep['feature_columns'],
        known_flight_uniqueness=checks,chronological_boundaries=cohorts['chronological_boundaries'],
        forest=dict(n_estimators=300,max_features=.7,criterion='squared_error',bootstrap=False,n_jobs=2,
            min_samples_leaf=[1,20],selection='None; both fixed arms fit directly on new refit interval',
            prior='Whole earlier-month crossfit in refit; known flight uniqueness enforced; coldstart900; future query prior frozen at refit cutoff'),
        normalized=dict(params=normalized.PARAMS,cap=600,patience=60,
            selection='New train-only encoder and train labels; stop-only weighted scaled RMSE, equivalent raw MSE ordering',
            refit='Fresh encoder, scale normalizer, model on new refit cohort at stop-selected rounds',
            target='Z=(Y-900)/s; s=sqrt(3600^2+(schedule-900)^2); weight=s^2/refit_mean_s^2; raw prediction900+sZ'),
        label_access='Feature cache only, never full-year training_meta labels. maintrain.read_labels called only for authorized stageframes. No calibration labels used.',
        freezing='Bothorigins x all3arms calibration+all3evaluation predictions and reload receipts must complete before any evaluation labels open',
        comparison='Primary ETleaf20 versus ETleaf1 on all missing rows per panel. Normalized separate diagnostic; no learned or fixed new mixture.',
        metrics='Raw perpanel RMSE/MSE/MAE/bias and airport slices; paired daybootstrap2000seed20260916; every day deletion; top2/5/10 ET20beneficial deletions; no row deletion in headline metrics',
        resources=dict(threads=2,process_gib=4,start_available_gib=12,host_reserve_gib=8,gpu=False),
        exposure='All2025months previously exposed; new within-run chronology only, no freshholdout or ranking guarantee',
        release='Diagnostic only; no automatic blend/pipeline promotion, ranking inference, or upload')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record,'Frozen source or contract changed'
    else:
        common.write_json(path,record)
    return record


def features(stage,all_features):
    x=all_features.loc[stage[ID]].copy()
    assert x.index.is_unique and np.array_equal(x.index.to_numpy(),stage[ID].to_numpy())
    return x


def forest_frame(stage,all_features):
    x=features(stage,all_features)
    x[forest.TIME_COLUMN]=stage[TIME].to_numpy()
    return x


def fit(fold,arm):
    protocol=declare()
    assert psutil.virtual_memory().available>=12*1024**3,'Need12GiB free before job'
    dest=common.external_path(OUT/fold/arm)
    dest.mkdir(parents=True,exist_ok=False)
    common.write_json(dest/'launch.json',dict(pid=os.getpid(),utc=common.utc_now(),protocol_sha256=common.sha256(OUT/'protocol.json')))
    stop=threading.Event()
    watcher=threading.Thread(target=monitor,args=(stop,dest),daemon=True)
    watcher.start()
    started=time.monotonic()
    events=[]
    try:
        all_features=pd.read_parquet(BASE/'training_features.parquet')
        assert list(all_features)==protocol['feature_columns'] and len(all_features)==22470
        stages=['refit','calibration']+['evaluation_'+m for m in PANELS[fold]]
        if arm=='normalized_diagnostic':
            stages=['train','stop']+stages
        parts={s:read_stage(fold,s) for s in stages}
        refit=parts['refit']
        uniqueness=assert_no_known_flight_reuse(refit)
        for name,part in parts.items():
            if name in ['calibration',*['evaluation_'+m for m in PANELS[fold]]]:
                assert part[TIME].min()>refit[TIME].max()
                assert not part.FLIGHT_ID_mvt.dropna().isin(refit.FLIGHT_ID_mvt.dropna()).any()
        def labels(stage):
            assert stage in ['train','stop','refit']
            y=maintrain.read_labels(parts[stage])
            events.append(dict(kind='label_read',stage=stage,rows=len(y),id_hash=common.object_hash(parts[stage][ID].tolist()),
                               target_hash=common.object_hash(y.tolist()),utc=common.utc_now()))
            return y
        if arm in ['et_leaf1','et_leaf20']:
            x=forest_frame(refit,all_features)
            y=labels('refit')
            forest.FAMILY='extratrees'
            model,evidence=forest.fit(x,y,tuning=None,steps=ARMS[arm],seed=20260916,threads=2)
            assert evidence['params']==dict(n_estimators=300,max_features=.7,min_samples_leaf=ARMS[arm],criterion='squared_error',n_jobs=2,bootstrap=False)
            common.write_json(dest/'fit_evidence.json',evidence)
        else:
            xf={s:features(parts[s],all_features) for s in ['train','stop','refit']}
            encoder=normalized.FrameEncoder().fit(xf['train'])
            scales={s:normalized.scale_of(xf[s]) for s in ['train','stop','refit']}
            normalizer=float(np.mean(scales['train']**2))
            ytrain,ystop=labels('train'),labels('stop')
            data=lgb.Dataset(encoder.transform(xf['train']),label=normalized.transformed(ytrain,scales['train']),
                weight=scales['train']**2/normalizer,categorical_feature=list(encoder.categories))
            valid=lgb.Dataset(encoder.transform(xf['stop']),label=normalized.transformed(ystop,scales['stop']),
                weight=scales['stop']**2/normalizer,reference=data,categorical_feature=list(encoder.categories))
            selected=lgb.train(normalized.PARAMS,data,num_boost_round=600,valid_sets=[valid],
                callbacks=[lgb.early_stopping(60,verbose=False)])
            steps=int(selected.best_iteration)
            stopz=selected.predict(encoder.transform(xf['stop']),num_iteration=steps)
            rawstop=normalized.reconstruct(stopz,scales['stop'])
            np.testing.assert_allclose(scales['stop']**2*(stopz-normalized.transformed(ystop,scales['stop']))**2,
                                       (rawstop-ystop)**2,rtol=1e-9,atol=1e-5)
            joblib.dump(dict(model=selected,encoder=encoder),dest/'selection.joblib')
            common.write_json(dest/'selection.json',dict(steps=steps,fit_rows=len(ytrain),stop_rows=len(ystop),
                model_sha256=common.sha256(dest/'selection.joblib'),objective_equivalence_verified=True))
            del selected,data,valid,encoder
            gc.collect()
            encoder=normalized.FrameEncoder().fit(xf['refit'])
            normalizer=float(np.mean(scales['refit']**2))
            yr=labels('refit')
            data=lgb.Dataset(encoder.transform(xf['refit']),label=normalized.transformed(yr,scales['refit']),
                weight=scales['refit']**2/normalizer,categorical_feature=list(encoder.categories))
            regressor=lgb.train(normalized.PARAMS,data,num_boost_round=steps)
            model=dict(model=regressor,encoder=encoder,steps=steps)
            common.write_json(dest/'fit_evidence.json',dict(steps=steps,rows=len(yr),normalizer=normalizer,params=normalized.PARAMS))
            del data,xf
        joblib.dump(model,dest/'model.joblib')
        events.append(dict(kind='refit_frozen',utc=common.utc_now(),model_sha256=common.sha256(dest/'model.joblib')))
        def predict(bundle,part):
            if arm in ['et_leaf1','et_leaf20']:
                return forest.predict(bundle,forest_frame(part,all_features))
            x=features(part,all_features)
            return normalized.reconstruct(bundle['model'].predict(bundle['encoder'].transform(x)),normalized.scale_of(x))
        stage_predictions={}
        for stage in ['calibration']+['evaluation_'+m for m in PANELS[fold]]:
            p=predict(model,parts[stage])
            assert np.isfinite(p).all()
            result=parts[stage].copy()
            result['prediction_sec']=p
            result.to_parquet(dest/(stage+'.parquet'),index=False)
            stage_predictions[stage]=dict(rows=len(result),id_hash=common.object_hash(result[ID].tolist()),sha256=common.sha256(dest/(stage+'.parquet')))
        del model
        if arm=='normalized_diagnostic':
            del regressor,encoder
        gc.collect()
        replay=joblib.load(dest/'model.joblib')
        for stage in stage_predictions:
            p=predict(replay,parts[stage])
            expected=pd.read_parquet(dest/(stage+'.parquet'),columns=['prediction_sec']).prediction_sec.to_numpy(float)
            delta=float(np.max(np.abs(p-expected)))
            assert delta<=1e-9
            stage_predictions[stage]['native_reload_max_abs_delta']=delta
        del replay
        gc.collect()
        events.append(dict(kind='all_predictions_frozen',utc=common.utc_now()))
        common.write_json(dest/'access.json',events)
        common.write_json(dest/'manifest.json',dict(status='complete_predictions_frozen',fold=fold,arm=arm,
            protocol_sha256=common.sha256(OUT/'protocol.json'),predictions=stage_predictions,
            model_sha256=common.sha256(dest/'model.joblib'),refit_id_hash=common.object_hash(refit[ID].tolist()),
            refit_rows=len(refit),known_flight_uniqueness=uniqueness,source_hash=common.sha256(__file__),
            target_access_events=events,evaluation_labels_read=False,calibration_labels_read=False,
            runtime_sec=time.monotonic()-started,peak_bytes=guard(),
            outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file()}))
        print('MISSING_FROZEN',fold,arm,len(refit),guard(),flush=True)
    finally:
        stop.set()
        watcher.join()


def metric(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    return dict(rows=len(y),mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.abs(e).mean()),bias=float(e.mean()))


def pair(y,p,c,days):
    a,b=(p-y)**2,(c-y)**2
    gain=b-a
    codes,unique=pd.factorize(days,sort=True)
    n=np.bincount(codes)
    s=np.bincount(codes,weights=gain)
    weights=np.random.default_rng(20260916).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    ci=np.quantile((weights@s)/(weights@n),[.025,.975]).tolist()
    byday=[]
    for day in unique:
        keep=days!=day
        byday.append(dict(day=str(day),rows_removed=int((~keep).sum()),mse_gain=float(gain[keep].mean()),
            rmse_gain=float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))))
    order=np.argsort(-gain,kind='stable')
    deleted={}
    for k in [2,5,10]:
        keep=np.ones(len(y),bool)
        keep[order[:k]]=False
        deleted[str(k)]=dict(mse_gain=float(gain[keep].mean()),rmse_gain=float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean())),
                            removed_positions=order[:k].tolist(),removed_sse_gain=float(gain[order[:k]].sum()))
    return dict(mse_gain=float(gain.mean()),rmse_gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())),
        mse_gain_ci95=ci,day_removals=byday,top_beneficial_deletions=deleted,
        sign='Positive favors ETleaf20; exposed-development conditional sensitivity only')


def score():
    declare()
    manifests={}
    for fold in PANELS:
        for arm in ARMS:
            folder=OUT/fold/arm
            marker=common.read_json(folder/'manifest.json')
            assert marker['status']=='complete_predictions_frozen'
            assert marker['protocol_sha256']==common.sha256(OUT/'protocol.json')
            for stage,record in marker['predictions'].items():
                assert common.sha256(folder/(stage+'.parquet'))==record['sha256']
            manifests[fold+'/'+arm]=common.sha256(folder/'manifest.json')
    dest=common.external_path(OUT/'evaluation')
    dest.mkdir(exist_ok=False)
    common.write_json(dest/'evaluation_opened.json',dict(utc=common.utc_now(),manifests=manifests,
        protocol_sha256=common.sha256(OUT/'protocol.json'),all_six_models_and_predictions_previously_frozen=True))
    reports={}
    for fold,months in PANELS.items():
        reports[fold]={}
        for month in months:
            stage='evaluation_'+month
            reference=None
            predictions={}
            for arm in ARMS:
                frame=pd.read_parquet(OUT/fold/arm/(stage+'.parquet'))
                assert TARGET not in frame
                if reference is None:
                    reference=frame.drop(columns='prediction_sec')
                else:
                    pd.testing.assert_frame_equal(reference,frame.drop(columns='prediction_sec'))
                predictions[arm]=frame.prediction_sec.to_numpy(float)
            expected=read_stage(fold,stage)
            pd.testing.assert_frame_equal(reference,expected)
            y=maintrain.read_labels(reference)
            days=reference[TIME].dt.strftime('%Y-%m-%d').to_numpy()
            record=dict(rows=len(y),id_hash=common.object_hash(reference[ID].tolist()),label_hash=common.object_hash(y.tolist()),
                metrics={arm:metric(y,p) for arm,p in predictions.items()},
                et20_vs_et1=pair(y,predictions['et_leaf20'],predictions['et_leaf1'],days),airports={})
            for airport in sorted(reference.ADEP_mvt.astype(str).unique()):
                keep=reference.ADEP_mvt.astype(str).eq(airport).to_numpy()
                record['airports'][airport]={arm:metric(y[keep],p[keep]) for arm,p in predictions.items()}
            replay=reference.copy()
            replay[TARGET]=y
            for arm,p in predictions.items():
                replay[arm]=p
            path=dest/(fold+'_'+month+'.parquet')
            replay.to_parquet(path,index=False)
            record['replay_sha256']=common.sha256(path)
            reports[fold][month]=record
    common.write_json(dest/'summary.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),
        model_manifest_hashes=manifests,folds=reports,primary='ETleaf20vsETleaf1 on missing-only raw outcomes; normalized diagnostic only',
        no_blend_or_complete_pipeline_claim=True,calibration_labels_used=False,
        limitation='Allmonths exposed,December shared acrossorigins; no unseen2026claim or automatic promotion'))
    print('MISSING_EVALUATED', {f:{m:r['et20_vs_et1']['rmse_gain'] for m,r in v.items()} for f,v in reports.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['declare','fit','score'])
    p.add_argument('--fold',choices=list(PANELS))
    p.add_argument('--arm',choices=list(ARMS))
    args=p.parse_args()
    with threadpool_limits(2):
        if args.action=='declare':
            declare()
            print('MISSING_CHRONOLOGY_FIXED_PROTOCOL',flush=True)
        elif args.action=='score':
            score()
        else:
            assert args.fold and args.arm
            fit(args.fold,args.arm)
