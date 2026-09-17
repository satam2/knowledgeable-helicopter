"""All-year finalization of the selected recipe. No new feature or recipe search."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from next230_common import (WORKSPACE, OUT as PRIOR, OLD, load_data, load_reference,
                            load_extension, fit, config_for)
from next230_clock_gate import gate_features, gate_targets, PARAMS, checked
from next230_features import schedule_features
from next230_schedule_mixture import predict_mixture, CONFIG as ROME_CONFIG
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now, source_hashes
from taxiout.availability import make_observations
from taxiout.features.pipeline import FeaturePipeline
from taxiout.io import read_raw, concat_frames
from taxiout.models.residual import proxy_status
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, MOVEMENT, PHASE, FLIGHT_ID, duration, FEATURE_INPUTS
from taxiout.submission import build_submission, validate_submission
from taxiout.train import sample_positions
from run_screening import RAW

OUT = external_path(WORKSPACE/'private_runs/submission_v2')
SEEDS = [20260910,20260911,20260912]
NAME = 'knowledgeable-helicopter_v2.parquet'
FOLDS = ['F1','F2','F3']


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    destination = OUT/'protocol.json'
    if destination.exists():
        verify()
        print('FINAL PROTOCOL ALREADY FROZEN',flush=True)
        return
    data = load_data()
    x, meta, labels = data
    declarations = {'residual':[],'long':[],'direct':[],'fallback':[],'missing':[]}
    declarations.update({f'rome_{s}_{part}':[] for s in SEEDS for part in ['classifier','regressor']})
    refs={}
    for fold in FOLDS:
        h,_,_,_=load_reference(fold,meta)
        p=PRIOR/'models'/f'capacity_d8_5000_{fold}_s20260910'
        rec=checked(p,'manifest.json')
        direct=checked(PRIOR/'clock_cpu_experts'/fold,'expert.json')
        declarations['residual'].append(rec['trees'])
        declarations['long'].append(h['iterations']['residual'])
        declarations['direct'].append(direct['trees'])
        for route in ['fallback','missing']:
            declarations[route].append(h['iterations']['direct' if route=='fallback' else route])
        for s in SEEDS:
            m=checked(PRIOR/'models'/f'rome_schedule_mixture_{fold}_s{s}','manifest.json')
            declarations[f'rome_{s}_classifier'].append(m['classifier_trees'])
            declarations[f'rome_{s}_regressor'].append(m['trees'])
        refs[fold]={'reference_run':h['run_id'],'residual_manifest':sha256(p/'manifest.json')}
    protocol={'created_utc':utc_now(),'recipe':'clock_and_rome_ensemble','local_seasonal_rmse_sec':292.8133464057956,
        'runner_sha256':sha256(__file__),'source_hashes':source_hashes(),
        'raw_hashes':read_json(PRIOR/'protocol.json')['raw_hashes'],
        'base_config':config_for('baseline'),'rome_config':ROME_CONFIG,'gate_params':PARAMS,
        'final_trees':{k:int(np.median(v)) for k,v in declarations.items()},'fold_trees':declarations,
        'tree_rule':'Per-component median F1/F2/F3; same existing release rule. G1 excluded.',
        'gate_rule':'Fixed 300-tree gate refitted on pooled fit-only expert predictions for June, September, October. No in-sample expert predictions.',
        'final_training':'All 2025 eligible rows; fallback retains original 250000-row deterministic sample. No target clipping.',
        'routes':{'present':'capacity residual + learned convex blend with direct expert',
                  'long_proxy':'2500-tree inherited H_gpu residual, not the capacity residual',
                  'negative_proxy':'sampled direct fallback','missing_non_Rome':'global missing specialist',
                  'Rome_missing_schedule':'equal mixture across the three fixed seeds'},
        'refs':refs,'seeds':SEEDS,'no_new_hyperparameter_search':True,
        'training_rows':len(x),'training_id_hash':object_hash(x.index.tolist()),
        'selection_decision_sha256':sha256(PRIOR/'decision.json')}
    write_json(destination,protocol)
    print('FINAL TREES',protocol['final_trees'],flush=True)
    # Prepare ranking inputs with the same observation policy and feature implementation.
    columns=list(dict.fromkeys([*FEATURE_INPUTS,'FLIGHT_mvt']))
    obs=make_observations(read_raw(RAW/'ranking.parquet',columns))[0]
    pipeline=FeaturePipeline(config_for('baseline'))
    rx=pipeline.transform(obs)
    dep=obs.loc[obs[PHASE].eq('DEP')].set_index(ID,drop=False)
    ext=schedule_features(dep)
    assert np.array_equal(rx.index,ext.index)
    rs=pd.concat([rx,ext],axis=1)
    rm=dep[[ID,FLIGHT_ID,'ADEP_mvt',MOVEMENT]].reset_index(drop=True)
    rm['proxy_sec']=duration(dep[MOVEMENT],dep.AOBT_3_flt).to_numpy()
    rm['schedule_sec']=duration(dep[MOVEMENT],dep.SCHED_TIME_UTC_mvt).to_numpy()
    assert list(rx)==list(x) and {k:str(v) for k,v in rx.dtypes.items()}=={k:str(v) for k,v in x.dtypes.items()}
    rx.to_parquet(OUT/'ranking_base.parquet')
    rs.to_parquet(OUT/'ranking_schedule.parquet')
    rm.to_parquet(OUT/'ranking_meta.parquet',index=False)
    write_json(OUT/'ranking_inputs.json',{'rows':len(rx),'id_hash':object_hash(rx.index.tolist()),
        'files':{p.name:sha256(p) for p in [OUT/'ranking_base.parquet',OUT/'ranking_schedule.parquet',OUT/'ranking_meta.parquet']}})
    verify_fold_replay(data)


def verify():
    protocol=read_json(OUT/'protocol.json')
    if protocol['runner_sha256']!=sha256(__file__) or protocol['source_hashes']!=source_hashes():
        raise ValueError('Final source drift')
    return protocol


def load_model(path, classifier=False):
    return (CatBoostClassifier() if classifier else CatBoostRegressor()).load_model(str(path))


def predict(x,schedule,meta,paths,means):
    config=config_for('baseline')
    status=proxy_status(meta.proxy_sec,config)
    offset=meta.proxy_sec.to_numpy(float)
    values=load_model(paths['fallback']).predict(x,thread_count=4)
    route=np.full(len(x),'direct_invalid',dtype='U32')
    present=status=='present'
    residual=offset+load_model(paths['residual']).predict(x,thread_count=4)
    direct=load_model(paths['direct']).predict(x,thread_count=4)
    gate=np.clip(load_model(paths['gate']).predict(gate_features(x,residual,direct),thread_count=4),0,1)
    values[present]=(residual+gate*(direct-residual))[present]
    route[present]='residual'
    long=np.isfinite(offset)&(offset>config['proxy_max'])
    if long.any():
        correction=load_model(paths['long']).predict(x.iloc[np.flatnonzero(long)],thread_count=4)
        values[long]=offset[long]+correction
        route[long]='residual_long_proxy'
    missing=status=='missing'
    if missing.any():
        values[missing]=load_model(paths['missing']).predict(x.iloc[np.flatnonzero(missing)],thread_count=4)
        route[missing]='specialist_missing'
    rome=missing & meta.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(meta.schedule_sec)
    if rome.any():
        sx=schedule.iloc[np.flatnonzero(rome)]
        preds=[]
        for seed in SEEDS:
            probability=load_model(paths[f'rome_{seed}_classifier'],True).predict_proba(sx,thread_count=4)[:,1]
            correction=load_model(paths[f'rome_{seed}_regressor']).predict(sx,thread_count=4)
            preds.append(predict_mixture(meta.schedule_sec.to_numpy()[rome],probability,means[str(seed)],correction))
        values[rome]=np.mean(preds,axis=0)
        route[rome]='rome_schedule_residual'
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite final predictions')
    result=meta.copy()
    result['prediction_sec'],result['route']=values,route
    return result


def verify_fold_replay(data):
    x,meta,_=data
    sx=load_extension(x,'schedule')
    report={}
    for fold in ['F1','F3']:
        reference,_,idx,_=load_reference(fold,meta)
        base=OLD/'models'/reference['run_id']
        paths={'fallback':base/'direct.cbm','missing':base/'missing.cbm','long':base/'residual.cbm',
            'residual':PRIOR/'models'/f'capacity_d8_5000_{fold}_s20260910'/'component.cbm',
            'direct':PRIOR/'clock_cpu_experts'/fold/'component.cbm',
            'gate':PRIOR/'models'/f'clock_learned_gate_{fold}_s20260910'/'gate.cbm'}
        means={}
        for seed in SEEDS:
            path=PRIOR/'models'/f'rome_schedule_mixture_{fold}_s{seed}'
            paths[f'rome_{seed}_classifier']=path/'reliability.cbm'
            paths[f'rome_{seed}_regressor']=path/'inconsistent.cbm'
            means[str(seed)]=read_json(path/'mixture.json')['consistent_mean']
        rows=idx['score']
        actual=predict(x.iloc[rows],sx.iloc[rows],meta.iloc[rows].reset_index(drop=True),paths,means)
        expected=pd.read_parquet(PRIOR/'models'/f'clock_and_rome_ensemble_{fold}_s20260910'/'score_predictions.parquet')
        assert np.array_equal(actual[ID],expected[ID])
        assert np.array_equal(actual.route,expected.route)
        delta=float(np.max(np.abs(actual.prediction_sec-expected.prediction_sec)))
        if delta>1e-9:
            raise ValueError(f'Final inference route replay failed {fold}: {delta}')
        report[fold]={'max_abs_delta_sec':delta,'rows':len(actual),'routes_exact':True}
        print('INFERENCE REPLAY',fold,report[fold],flush=True)
    write_json(OUT/'fold_replay.json',report)


def train_model(name,x,target,rows,config,trees,classifier=False):
    folder=OUT/'components'/name
    record_file=folder/'record.json'
    if record_file.exists():
        record=read_json(record_file)
        assert record['status']=='complete' and sha256(folder/'model.cbm')==record['model_sha256']
        print('REUSED',name,flush=True)
        return
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'started.json',{'utc':utc_now(),'name':name,'rows':len(rows),'trees':trees})
    start=time.monotonic()
    print('TRAIN',name,'rows',len(rows),'trees',trees,flush=True)
    if classifier:
        model=CatBoostClassifier(iterations=trees,depth=4,learning_rate=.05,l2_leaf_reg=20,
            random_seed=config['seed'],thread_count=4,loss_function='Logloss',verbose=200,allow_writing_files=False)
        model.fit(x.iloc[rows],target[rows],cat_features=list(x.select_dtypes('category').columns))
    else:
        model,_=fit(x.iloc[rows],target[rows],config,trees=trees)
    model.save_model(str(folder/'model.cbm'))
    probe=x.iloc[rows[:min(5000,len(rows))]]
    reloaded=load_model(folder/'model.cbm',classifier)
    original=model.predict_proba(probe,thread_count=4) if classifier else model.predict(probe,thread_count=4)
    repeated=reloaded.predict_proba(probe,thread_count=4) if classifier else reloaded.predict(probe,thread_count=4)
    assert np.array_equal(original,repeated)
    write_json(record_file,{'status':'complete','name':name,'rows':len(rows),'trees':model.tree_count_,
        'elapsed_sec':time.monotonic()-start,'training_id_hash':object_hash(x.index[rows].tolist()),
        'feature_names':list(x),'config':config,'model_sha256':sha256(folder/'model.cbm'),
        'reload_exact':True,'protocol_sha256':sha256(OUT/'protocol.json')})
    print('COMPLETE',name,round(time.monotonic()-start,1),'seconds',flush=True)
    del model,reloaded
    gc.collect()


def train(group):
    protocol=verify()
    x,meta,labels=load_data()
    y=labels[TARGET].to_numpy(float)
    base={**config_for('baseline'),'seed':20260910,'task_type':'CPU'}
    status=proxy_status(meta.proxy_sec,base)
    present=np.flatnonzero(status=='present')
    missing=np.flatnonzero(status=='missing')
    trees=protocol['final_trees']
    if group=='gpu':
        config={**base,'depth':8,'learning_rate':.05,'task_type':'GPU'}
        target=y-meta.proxy_sec.to_numpy()
        for name in ['residual','long']:
            train_model(name,x,target,present,config,trees[name])
    elif group=='direct':
        config={**base,'depth':6,'learning_rate':.05}
        train_model('direct',x,y,present,config,trees['direct'])
    elif group=='fallback':
        rows=sample_positions(x,np.arange(len(x)),250000,base['seed'])
        train_model('fallback',x,y,rows,base,trees['fallback'])
        train_model('missing',x,y,missing,base,trees['missing'])
    elif group=='rome':
        x=load_extension(x,'schedule')
        eligible=np.flatnonzero((status=='missing') & np.isfinite(meta.schedule_sec))
        target=y-meta.schedule_sec.to_numpy()
        consistent=np.abs(target)<=60
        bad=eligible[~consistent[eligible]]
        mean=float(target[eligible][consistent[eligible]].mean())
        means={}
        for seed in SEEDS:
            config={**ROME_CONFIG,'seed':seed}
            train_model(f'rome_{seed}_classifier',x,consistent.astype(int),eligible,config,trees[f'rome_{seed}_classifier'],True)
            train_model(f'rome_{seed}_regressor',x,target,bad,config,trees[f'rome_{seed}_regressor'])
            means[str(seed)]=mean
        write_json(OUT/'rome_means.json',means)
    elif group=='gate':
        features=[]
        targets=[]
        direct_predictions=[]
        residual_predictions=[]
        calibration=[]
        for fold in FOLDS:
            _,_,idx,split=load_reference(fold,meta)
            rp=PRIOR/'models'/f'capacity_d8_5000_{fold}_s20260910'
            dp=PRIOR/'clock_cpu_experts'/fold
            checked(rp,'manifest.json')
            checked(dp,'expert.json')
            r=pd.read_parquet(rp/'tune_predictions.parquet')
            d=pd.read_parquet(dp/'tune_direct.parquet')
            assert np.array_equal(r[ID],d[ID])
            loc=pd.Index(meta[ID]).get_indexer(r[ID])
            assert (loc>=0).all() and np.isin(loc,idx['tune']).all()
            assert np.array_equal(y[loc],d[TARGET])
            pred=meta.iloc[loc].proxy_sec.to_numpy()+r.prediction.to_numpy()
            features.append(gate_features(x.iloc[loc],pred,d.direct).reset_index())
            targets.append(y[loc])
            residual_predictions.append(pred)
            direct_predictions.append(d.direct.to_numpy())
            calibration.append({'fold':fold,'split_hash':split['split_hash'],'rows':len(loc),
                'id_hash':object_hash(r[ID].tolist()),'residual_manifest_sha256':sha256(rp/'manifest.json'),
                'direct_manifest_sha256':sha256(dp/'expert.json')})
        features=concat_frames(features).set_index(ID)
        assert features.index.is_unique
        eligible,target,weights,scale=gate_targets(np.concatenate(targets),np.concatenate(residual_predictions),np.concatenate(direct_predictions))
        folder=OUT/'components/gate'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'record.json').exists():
            raise ValueError('Gate already exists; inspect saved artifact')
        start=time.monotonic()
        print('TRAIN gate',len(features),'pooled chronological tune rows',flush=True)
        model=CatBoostRegressor(**PARAMS)
        model.fit(features.iloc[np.flatnonzero(eligible)],target,sample_weight=weights,
                  cat_features=list(features.select_dtypes('category').columns))
        model.save_model(str(folder/'model.cbm'))
        saved=load_model(folder/'model.cbm')
        assert np.array_equal(model.predict(features,thread_count=4),saved.predict(features,thread_count=4))
        write_json(folder/'record.json',{'status':'complete','name':'gate','rows':int(eligible.sum()),
            'calibration':calibration,'config':PARAMS,'weight_scale':scale,'feature_names':list(features),
            'training_id_hash':object_hash(features.index[eligible].tolist()),'reload_exact':True,
            'model_sha256':sha256(folder/'model.cbm'),'protocol_sha256':sha256(OUT/'protocol.json'),
            'elapsed_sec':time.monotonic()-start})
        print('COMPLETE gate',flush=True)


def finish():
    protocol=verify()
    load_data()  # Rechecks all raw hashes and original source freeze before release.
    assert set(read_json(OUT/'fold_replay.json'))=={'F1','F3'}
    inputs=read_json(OUT/'ranking_inputs.json')
    for name,digest in inputs['files'].items():
        assert sha256(OUT/name)==digest
    paths={}
    for name in [*protocol['final_trees'],'gate']:
        path=OUT/'components'/name
        record=read_json(path/'record.json')
        assert record['status']=='complete' and record['protocol_sha256']==sha256(OUT/'protocol.json')
        assert record['model_sha256']==sha256(path/'model.cbm')
        paths[name]=path/'model.cbm'
    x=pd.read_parquet(OUT/'ranking_base.parquet')
    sx=pd.read_parquet(OUT/'ranking_schedule.parquet')
    meta=pd.read_parquet(OUT/'ranking_meta.parquet')
    assert np.array_equal(x.index,meta[ID]) and np.array_equal(sx.index,x.index)
    means=read_json(OUT/'rome_means.json')
    result=predict(x,sx,meta,paths,means)
    result.to_parquet(OUT/'ranking_predictions.parquet',index=False)
    # Repeat all ranking predictions from serialized components, also in different batches.
    parts=[]
    for start in range(0,len(x),50000):
        pos=slice(start,start+50000)
        parts.append(predict(x.iloc[pos],sx.iloc[pos],meta.iloc[pos].reset_index(drop=True),paths,means))
    repeated=pd.concat(parts,ignore_index=True)
    assert np.array_equal(result[ID],repeated[ID])
    assert np.array_equal(result.route,repeated.route)
    assert np.array_equal(result.prediction_sec,repeated.prediction_sec)
    file=OUT/NAME
    if file.exists():
        raise ValueError('Submission file already exists; no overwrite')
    receipt=build_submission(RAW/'submitting.parquet',result,file)
    assert receipt['rows']==344841
    validate_submission(file,RAW/'submitting.parquet')
    receipt.update(pipeline_verified=True,protocol_sha256=sha256(OUT/'protocol.json'),
        components={k:sha256(v) for k,v in paths.items()},
        exact_batch_replay=True,routes={str(k):int(v) for k,v in result.route.value_counts().items()},
        ranking_predictions_sha256=sha256(OUT/'ranking_predictions.parquet'),
        model_prediction_quantiles={str(q):float(v) for q,v in result.prediction_sec.quantile([0,.01,.5,.95,.99,1]).items()},
        local_validation_rmse_sec=protocol['local_seasonal_rmse_sec'],
        external_score=None,prepared_utc=utc_now())
    write_json(OUT/'submission_ready.json',receipt)
    print('SUBMISSION READY',receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','train','finish'])
    p.add_argument('--group',choices=['gpu','direct','fallback','rome','gate'])
    args=p.parse_args()
    if args.action=='prepare': setup()
    elif args.action=='train': train(args.group)
    else: finish()
