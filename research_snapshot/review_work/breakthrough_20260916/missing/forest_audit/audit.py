"""Independent immutable-forest replay, sensitivity and tune complementarity."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import sys
import importlib.util
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
import run_missing_models as missing
import run_id_context as context
from taxiout.artifacts import read_json,write_json,sha256,object_hash,utc_now
from taxiout.schema import ID,TARGET,MOVEMENT
from taxiout.metrics import season_score


def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


forest=import_path('review_forest',ROOT/'review_work/breakthrough_20260916/models/missing_forest/adapter.py')
prioraudit=import_path('review_prior',ROOT/'review_work/breakthrough_20260916/models/verification_id_template/audit.py')
OUT=ROOT/'private_runs/breakthrough_20260916/missing/forest_audit'
FORESTS=ROOT/'private_runs/breakthrough_20260916/models/missing_forest'
TEMPLATES=ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models'


def checked(path):
    manifest=read_json(path/'manifest.json')
    if manifest['status']!='complete':
        raise ValueError(f'Incomplete artifact {path}')
    for name,digest in manifest['outputs'].items():
        assert sha256(path/name)==digest,(path,name)
    return manifest


def check_sources(record):
    mapping={'prior_common.py':ROOT/'review_work/campaign_20260916/common.py'}
    for name,digest in record['source_hashes'].items():
        path=mapping.get(name,ROOT/name if name.startswith('review_work') else ROOT/'review_work/breakthrough_20260916/models'/name)
        assert sha256(path)==digest,(name,'sourcechanged')


def replay(path,manifest,x,meta,idx):
    unavailable=~np.isfinite(meta.proxy_sec.to_numpy(float))
    rows={stage:positions[unavailable[positions]] for stage,positions in idx.items()}
    evidence={}
    for stage,filename,query in [('fit','fit_model.joblib','tune'),('refit','model.joblib','score')]:
        assert manifest['fit_ids'][stage]==dict(n=len(rows[stage]),hash=object_hash(meta.iloc[rows[stage]][ID].tolist()))
        saved=joblib.load(path/filename)
        saved['estimator'].n_jobs=1
        features=x.loc[meta.iloc[rows[query]][ID]]
        prediction=forest.predict(saved,features)
        output=pd.read_parquet(path/('tune_predictions.parquet' if stage=='fit' else 'candidate.parquet')).set_index(ID)
        expected=output.loc[features.index].prediction_sec.to_numpy(float)
        delta=float(np.max(np.abs(prediction-expected)))
        assert delta<=1e-8
        if stage=='fit':
            choices=manifest['fit']['tune_candidates']
            best=min(choices,key=lambda item:(item['tune_mse_sec2'],[1,5,20].index(item['min_samples_leaf'])))
            assert saved['steps']==best['min_samples_leaf']==manifest['fit']['steps']
        else:
            assert saved['steps']==manifest['fit']['steps']==manifest['refit']['steps']
        assert saved['estimator'].criterion=='squared_error' and len(saved['estimator'].estimators_)==300
        expected_prior=missing.HistoricalTemplate().fit(forest.input_frame(x.loc[meta.iloc[rows[stage]][ID]]),
            meta.iloc[rows[stage]][TARGET].to_numpy(float),meta.iloc[rows[stage]][MOVEMENT])
        assert saved['prior'].history_n==expected_prior.history_n==len(rows[stage])
        assert saved['prior'].last_fit==expected_prior.last_fit
        assert saved['prior'].last_fit<pd.to_datetime(meta.iloc[rows[query]][MOVEMENT],utc=True).min()
        assert saved['prior'].global_mean==expected_prior.global_mean
        for (keys,table),(ekeys,etable) in zip(saved['prior'].tables,expected_prior.tables):
            assert keys==ekeys
            pd.testing.assert_frame_equal(table,etable)
        evidence[stage]=dict(rows=len(rows[stage]),query_rows=len(rows[query]),reload_max_abs_delta=delta,
            selected_leaf=saved['steps'],prior_tables_exact=True,last_prior_time=str(saved['prior'].last_fit),
            first_query_time=str(pd.to_datetime(meta.iloc[rows[query]][MOVEMENT],utc=True).min()),
            estimator_class=type(saved['estimator']).__name__)
        del saved
        gc.collect()
    return evidence


def tune_evidence(template,other,labels,days):
    y=np.asarray(labels,float)
    t=np.asarray(template,float)
    r=np.asarray(other,float)
    te=t-y
    re=r-y
    difference=r-t
    denominator=float(difference@difference)
    optimal=float(np.clip(-(te@difference)/denominator,0,1)) if denominator else .5
    half=.5*(t+r)
    keep=np.ones(len(y),bool)
    tail=y>=86400
    largest=np.argsort(te**2)[-2:]
    no2=keep.copy()
    no2[largest]=False
    return dict(n=len(y),residual_correlation=float(np.corrcoef(te,re)[0,1]),
        template=prioraudit.metric(y,t),forest=prioraudit.metric(y,r),fixed_half=prioraudit.metric(y,half),
        tune_optimal_forest_weight=optimal,tune_optimal=prioraudit.metric(y,t+optimal*difference),
        fixed_half_vs_template=prioraudit.comparison(y,t,half,days),
        forest_vs_template=prioraudit.comparison(y,t,r,days),
        remove_dayplus={name:prioraudit.metric(y,p,~tail) for name,p in [('template',t),('forest',r),('half',half)]},
        remove_template_top2={name:prioraudit.metric(y,p,no2) for name,p in [('template',t),('forest',r),('half',half)]},
        warning='Alltuneheads alreadyusedtune foriterations/leaves; tuneoptimalweight isdiagnostic optimistic, not adopted. No scoreensemble fitted.')


def main():
    if (OUT/'audit.json').exists():
        raise ValueError('Completedforest auditretained')
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',dict(created_utc=utc_now(),source_sha256=sha256(__file__),
        scope='Read-onlysavedforest audit, alloriginalscorelabels andprotectedroutes, source/data/foldpurges/replay/priors/sensitivity.',
        sensitivity='Dayandtoprowremovals arediagnostic only; no scorederivedpredictionrules or newmodel.',
        tune='Compare originalmissingtune expert residuals andfixed50/50 blend. Analytictuneoptimum isdiagnostic; no fittedensemble persisted.',
        sources='RandomForest complete; ExtraTrees included onlyif bothF1/F3 arecomplete atstart.',cpu_threads=1))
    x,meta=missing.load_data()
    peer=pd.read_parquet(context.CACHE/'features.parquet').set_index(ID)
    audit=read_json(context.CACHE/'audit.json')
    assert sha256(context.CACHE/'features.parquet')==audit['feature_sha256']
    np.testing.assert_array_equal(peer.index,meta[ID])
    times=meta.set_index(ID).loc[x.index,MOVEMENT]
    extension=context.id_context_features(x,peer.loc[x.index],times)
    fx=pd.concat([x,extension],axis=1)
    fx[forest.TIME_COLUMN]=times
    families=['randomforest']
    extra=[FORESTS/f'extratrees_missing_template_idcontext_{fold}_s20260916'/'manifest.json' for fold in ['F1','F3']]
    if all(path.exists() and read_json(path)['status']=='complete' for path in extra):
        families.append('extratrees')
    allfolds={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        reference,reference_record=common.reference(fold)
        np.testing.assert_array_equal(reference[ID],meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(reference[TARGET],meta.iloc[idx['score']][TARGET])
        assert object_hash(split)==object_hash(reference_record['split'])
        tpath=TEMPLATES/f'historical_template_{fold}_s20260916'
        tm=checked(tpath)
        assert object_hash(tm['split'])==object_hash(split)
        template=pd.read_parquet(tpath/'candidate.parquet')
        templateblend=pd.read_parquet(tpath/'blend25.parquet')
        guardpath=ROOT/'private_runs/breakthrough_20260916/models/support_guard'/fold
        gm=checked(guardpath)
        guard=pd.read_parquet(guardpath/'fit_min_floor.parquet')
        floor=float(meta.iloc[idx['fit']][TARGET].min())
        assert gm['fit_min_target_sec']==floor
        missing_score=~np.isfinite(meta.iloc[idx['score']].proxy_sec.to_numpy(float))
        y=reference[TARGET].to_numpy(float)
        day=reference.day.to_numpy()
        preds={'reference':reference.prediction_sec.to_numpy(float),'guarded_reference':guard.prediction_sec.to_numpy(float),
            'template_candidate':template.prediction_sec.to_numpy(float),'template_blend25':templateblend.prediction_sec.to_numpy(float)}
        preds['template_guardedblend']=preds['guarded_reference'].copy()
        preds['template_guardedblend'][missing_score]+=.25*(preds['template_candidate'][missing_score]-preds['guarded_reference'][missing_score])
        evidence={}
        tune={}
        mt=~np.isfinite(meta.iloc[idx['tune']].proxy_sec.to_numpy(float))
        tunerows=idx['tune'][mt]
        template_tune=pd.read_parquet(tpath/'tune_predictions.parquet')
        np.testing.assert_array_equal(template_tune[ID],meta.iloc[tunerows][ID])
        for family in families:
            path=FORESTS/f'{family}_missing_template_idcontext_{fold}_s20260916'
            manifest=checked(path)
            check_sources(manifest)
            protocol=FORESTS/f'protocol_{family}_s20260916.json'
            assert manifest['protocol_sha256']==sha256(protocol)
            assert object_hash(manifest['split'])==object_hash(split)
            for stage,positions in idx.items():
                selected=positions[~np.isfinite(meta.iloc[positions].proxy_sec.to_numpy(float))]
                assert manifest['fit_ids'][stage]==dict(n=len(selected),hash=object_hash(meta.iloc[selected][ID].tolist()))
            for variant in ['candidate','blend25']:
                frame=pd.read_parquet(path/f'{variant}.parquet')
                np.testing.assert_array_equal(frame[ID],reference[ID])
                np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
                pred=frame.prediction_sec.to_numpy(float)
                np.testing.assert_array_equal(pred[~missing_score],preds['reference'][~missing_score])
                preds[family+'_'+variant]=pred
                assert abs(prioraudit.metric(y,pred)['rmse_sec']-manifest['reports'][variant]['metrics']['overall']['rmse_sec'])<1e-10
            np.testing.assert_allclose(preds[family+'_blend25'],preds['reference']+.25*(preds[family+'_candidate']-preds['reference']),rtol=0,atol=1e-9)
            guarded=preds['guarded_reference'].copy()
            guarded[missing_score]+=.25*(preds[family+'_candidate'][missing_score]-guarded[missing_score])
            preds[family+'_guardedblend']=guarded
            evidence[family]=dict(manifest_sha256=sha256(path/'manifest.json'),all_artifact_hashes_valid=True,
                source_hashes_valid=True,original_purged_row_ids_exact=True,replay=replay(path,manifest,fx,meta,idx))
            ft=pd.read_parquet(path/'tune_predictions.parquet')
            np.testing.assert_array_equal(ft[ID],template_tune[ID])
            tune[family]=tune_evidence(template_tune.prediction_sec,ft.prediction_sec,meta.iloc[tunerows][TARGET],
                pd.to_datetime(meta.iloc[tunerows][MOVEMENT],utc=True).dt.strftime('%Y-%m-%d').to_numpy())
        tail=missing_score&(y>=86400)
        taildays=np.isin(day,np.unique(day[tail]))
        reference_top2=np.argsort((preds['reference']-y)**2)[-2:]
        gain=(preds['reference']-y)**2-(preds['randomforest_blend25']-y)**2
        gain_top2=np.argsort(gain)[-2:]
        scenarios={'all':np.ones(len(y),bool),'remove_missing_dayplus':~tail,'remove_missing_dayplus_days':~taildays,
            'remove_reference_top2':~np.isin(np.arange(len(y)),reference_top2),'remove_rf_gain_top2':~np.isin(np.arange(len(y)),gain_top2),
            'remove_support_guard_rows':preds['reference']>=floor}
        metrics={scenario:{name:prioraudit.metric(y,pred,keep) for name,pred in preds.items()} for scenario,keep in scenarios.items()}
        pairs=[('reference','randomforest_blend25'),('template_blend25','randomforest_blend25'),
            ('guarded_reference','randomforest_guardedblend'),('template_guardedblend','randomforest_guardedblend')]
        if 'extratrees' in families:
            pairs.extend([('reference','extratrees_blend25'),('randomforest_blend25','extratrees_blend25')])
        rows=reference[[ID,TARGET,'day','ADEP_mvt']].copy()
        for name,pred in preds.items():
            rows[name]=pred
        rows['missing_source']=missing_score
        rows['rf_sse_gain_vs_reference']=gain
        rows.to_parquet(OUT/f'{fold}_comparison.parquet',index=False)
        rows.iloc[np.unique(np.r_[reference_top2,gain_top2,np.flatnonzero(tail)])].to_parquet(OUT/f'{fold}_sensitivity_rows.parquet',index=False)
        allfolds[fold]=dict(score_rows=len(y),missing_score_rows=int(missing_score.sum()),dayplus_rows=int(tail.sum()),
            fit_support_floor=floor,evidence=evidence,tune=tune,metrics=metrics,
            pairs={b+'_vs_'+a:prioraudit.comparison(y,preds[a],preds[b],day) for a,b in pairs},
            reference_top2_ids=reference.iloc[reference_top2][ID].tolist(),rf_gain_top2_ids=reference.iloc[gain_top2][ID].tolist(),
            top2_sse_gain=float(gain[gain_top2].sum()),total_sse_gain=float(gain.sum()),
            score_labels_unchanged=True,all_finite_routes_exact=True)
        print('FOREST_AUDIT',fold,{name:values['rmse_sec'] for name,values in metrics['all'].items()},flush=True)
    seasonal={scenario:{name:season_score(allfolds['F1']['metrics'][scenario][name],allfolds['F3']['metrics'][scenario][name])
        for name in allfolds['F1']['metrics'][scenario]} for scenario in allfolds['F1']['metrics']}
    write_json(OUT/'audit.json',dict(created_utc=utc_now(),source_sha256=sha256(__file__),families=families,
        folds=allfolds,seasonal=seasonal,no_new_predictions_or_rules=True,
        caveat='Removaldiagnostics retainoriginalartifacts; foldsdevelopment-exposed; tuneweightoptimum diagnosticnotadopted.'))
    print('FOREST_SEASONAL',seasonal,flush=True)


if __name__=='__main__':
    main()
