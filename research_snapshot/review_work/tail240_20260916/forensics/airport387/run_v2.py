"""All-airport full387 residual bank, frozen tune-only comparison."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import lightgbm as lgb
import argparse
import gc
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/state/risk'))
import run_risk as risk
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/models'))
import schema_discovery
common, ID, TARGET = risk.common, risk.ID, risk.TARGET
OUT = ROOT/'private_runs/tail240_20260916/forensics/airport387/tune_v2'
NM_PROTOCOL = ROOT/'private_runs/tail240_20260916/models/nm_clock_peers_tune_v2/protocol.json'
BASE = ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
NEURAL = {f:ROOT/'private_runs/tail240_20260916/state/neural_context'/v/f for f,v in [('F1','v1'),('F3','v3')]}
CONTROL = {f:ROOT/'private_runs/tail240_20260916/models'/v/f/'control387'
    for f,v in [('F1','following_groups_tune_v1'),('F3','following_groups_tune_v2')]}
ORIGINAL_SOURCES = risk.feature_sources
DISCOVERY = []


def guard(_env=None):
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert memory.rss < 8*1024**3 and peak < 8*1024**3, (memory.rss, peak)
    assert psutil.virtual_memory().available >= 8*1024**3, 'Host reserve below8GiB'
    return peak


def sources(columns):
    result, record = schema_discovery.cached_discovery(ORIGINAL_SOURCES, columns)
    DISCOVERY.append(record)
    return result


def declare():
    nm = common.read_json(NM_PROTOCOL)
    columns = nm['columns'][:387]
    controls = {f:common.read_json(risk.union_folder(f)/'manifest.json') for f in ('F1','F3')}
    assert len(columns) == len(set(columns)) == 387
    assert all(m['feature_columns']==columns and m['fit']['params']==nm['params'] for m in controls.values())
    assert all(common.sha256(risk.union_folder(f)/'manifest.json')==nm['controls'][f] for f in controls)
    assert all(common.sha256(NEURAL[f]/'manifest.json')==nm['neural_controls'][f] for f in controls)
    record = dict(source_sha256=common.sha256(__file__),
        sources={str(Path(p).relative_to(ROOT)):common.sha256(p) for p in [risk.__file__,schema_discovery.__file__]},
        columns=columns, params=nm['params'], original_controls=nm['controls'], neural_controls=nm['neural_controls'],
        fresh_controls={f:common.sha256(p/'manifest.json') for f,p in CONTROL.items()},
        baseline_protocol_sha256=common.sha256(NM_PROTOCOL), global9_preparation_sha256=common.sha256(BASE/'preparation.json'),
        primary=nm['primary'], baseline_definition='Identical formula and artifacts as nm_clock_peers_tune_v2: oldglobal9+wPLE*(savedPLE387-oldPLE225). Primary evaluated on original ordinary finite tune rows.',
        bank='All airport categories present in original finite fit, sorted deterministically; no selected airports. Same387 and globalfit-only vocabulary. RawY-P residual, no clipping/sampling. Same leaf63 params/cap2500/150patience/seed2CPU perairport.',
        fallback='Zero-fit queryairport uses verified global387 prediction. Zero-tune fitairport uses global387 selected round count withoutlocalearlystop. Unknown airport normalized only for null to __MISSING__. No selectedroute exclusion.',
        gate='Bothmonths positive matchedallfinite,matchedordinary andprimaryordinaryfixed25, everyone-dayremovalpositive; ranking-season-weighted primaryordinaryfixed25 gain>=2seconds. No alternatepromotionroute. F1 only initially authorized; F3 centrally scheduled.',
        resources='2CPU;currentandOShistoricalpeak<8GiB;hostreserve8GiB;startup>=20GiB pluscentralaggregateallocation. Load387float32matrix onceperfold, oneairportcopy/model atatime.',
        evidence='Prior115CatBoostairportbank worse thanmatchedglobal115 inbothhistoricalscoremonths. Global387 canalreadyconditiononairport; thistestsallocationonly.',
        scope='Originalfullfinitefit/tune andflightIDpurges. No score/refit/rankingprediction. Repeatedlyexposedtune andglobal9weightstune-fit: adaptivedevelopment.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record
    else:common.write_json(path,record)
    return record


def airport_values(frame):
    return frame['ADEP_mvt'].astype('string').fillna('__MISSING__').astype(str).to_numpy()


def partitions(fit_airports, tune_airports):
    fit_airports, tune_airports = np.asarray(fit_airports), np.asarray(tune_airports)
    return [(a,np.flatnonzero(fit_airports==a),np.flatnonzero(tune_airports==a)) for a in sorted(set(fit_airports))]


def train_bank(matrix, nfit, target_fit, target_tune, proxy_tune, fit_ids, tune_ids,
               fit_airports, tune_airports, fallback_prediction, columns, vocab, params, global_steps, dest):
    prediction = np.asarray(fallback_prediction,float).copy()
    assigned = np.zeros(len(tune_ids),bool)
    records = {}
    for number,(airport,fi,ti) in enumerate(partitions(fit_airports,tune_airports)):
        guard()
        options = dict(params)
        if not len(ti):options['n_estimators']=global_steps
        xf = np.asarray(matrix[fi],dtype=np.float32,order='C')
        xt = np.asarray(matrix[nfit+ti],dtype=np.float32,order='C')
        model = lgb.LGBMRegressor(**options)
        kwargs = dict(categorical_feature=[columns.index(c) for c in vocab],feature_name=columns,
            callbacks=[guard,lgb.log_evaluation(500)])
        if len(ti):
            kwargs.update(eval_X=xt,eval_y=target_tune[ti],eval_metric='rmse')
            kwargs['callbacks'].append(lgb.early_stopping(150,verbose=False))
        print('AIRPORT_START',airport,'fit',len(fi),'tune',len(ti),flush=True)
        model.fit(xf,target_fit[fi],**kwargs)
        steps=int(model.best_iteration_ or model.n_estimators_)
        filename=f'airport_{number:02d}.txt'
        model.booster_.save_model(str(dest/filename),num_iteration=steps)
        delta=0.
        if len(ti):
            values=model.predict(xt,num_iteration=steps)+proxy_tune[ti]
            native=lgb.Booster(model_file=str(dest/filename))
            replay=native.predict(xt,num_threads=2)+proxy_tune[ti]
            np.testing.assert_array_equal(values,replay)
            assert not assigned[ti].any() and np.isfinite(values).all()
            prediction[ti]=values
            assigned[ti]=True
            del native,values,replay
        records[airport]=dict(fit_n=len(fi),tune_n=len(ti),fit_id_hash=common.object_hash(np.asarray(fit_ids)[fi].tolist()),
            tune_id_hash=common.object_hash(np.asarray(tune_ids)[ti].tolist()),steps=steps,
            fixed_global_rounds_no_tune=not bool(len(ti)),params=options,model_file=filename,
            model_sha256=common.sha256(dest/filename),native_replay_max_abs_delta=delta,peak_bytes=guard())
        common.write_json(dest/'airport_progress.json',records)
        print('AIRPORT_DONE',airport,steps,'peak',records[airport]['peak_bytes'],flush=True)
        del xf,xt,model
        gc.collect()
        guard()
    expected=np.isin(tune_airports,np.unique(fit_airports))
    np.testing.assert_array_equal(assigned,expected)
    np.testing.assert_array_equal(prediction[~assigned],np.asarray(fallback_prediction)[~assigned])
    assert np.isfinite(prediction).all()
    return prediction,records,~assigned


def compare(y,prediction,reference,dates):
    gain=(y-reference)**2-(y-prediction)**2
    days,codes=np.unique(dates,return_inverse=True)
    counts=np.bincount(codes)
    deletion=(gain.sum()-np.bincount(codes,weights=gain))/(len(y)-counts)
    rmse=float(np.sqrt(np.mean((y-prediction)**2)))
    base=float(np.sqrt(np.mean((y-reference)**2)))
    return dict(n=len(y),rmse=rmse,reference_rmse=base,gain=base-rmse,mse_gain=float(gain.mean()),
        all_day_removals_improve=bool((deletion>0).all()),day_removal_min_gain=float(deletion.min()),days=len(days))


def baseline(fold,frame,protocol):
    assert common.sha256(BASE/'preparation.json')==protocol['global9_preparation_sha256']
    prep=common.read_json(BASE/'preparation.json')['folds'][fold]
    path=BASE/f'{fold}_aligned_tune.parquet'
    assert common.sha256(path)==prep['aligned_tune_sha256']
    weights=common.read_json(BASE/f'{fold}_weights.json')
    assert weights==prep['weights']
    aligned=pd.read_parquet(path)
    ordinary=frame.proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID],frame.loc[ordinary,ID])
    np.testing.assert_array_equal(aligned[TARGET],frame.loc[ordinary,TARGET])
    marker=common.read_json(NEURAL[fold]/'manifest.json')
    assert common.sha256(NEURAL[fold]/'manifest.json')==protocol['neural_controls'][fold]
    neural_path=NEURAL[fold]/'tune_predictions.parquet'
    assert common.sha256(neural_path)==marker['outputs'][neural_path.name]
    neural=pd.read_parquet(neural_path).set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET],aligned[TARGET])
    old=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    weight=weights['global'][weights['experts'].index('tabm_ple8')]
    values=old+weight*(neural.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    return ordinary,values,dict(aligned_sha256=common.sha256(path),neural_sha256=common.sha256(neural_path),
        ple_weight=weight,weights=weights)


def run(fold):
    protocol=declare()
    assert psutil.virtual_memory().available>=20*1024**3
    guard()
    dest=OUT/fold
    assert not dest.exists()
    dest.mkdir()
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    frames={s:meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy() for s,p in idx.items() if s in ('fit','tune')}
    del meta
    cohort={s:dict(n=len(f),hash=common.object_hash(f[ID].tolist())) for s,f in frames.items()}
    marker=common.read_json(CONTROL[fold]/'manifest.json')
    assert common.sha256(CONTROL[fold]/'manifest.json')==protocol['fresh_controls'][fold]
    assert marker['status']=='complete' and marker['params']==protocol['params'] and marker['ids']==cohort
    assert common.object_hash(marker['split'])==common.object_hash(split)
    assert marker['original_control_max_abs_delta']<=1e-7
    for filename,h in marker['outputs'].items():assert common.sha256(CONTROL[fold]/filename)==h
    control=pd.read_parquet(CONTROL[fold]/'tune.parquet')
    np.testing.assert_array_equal(control[ID],frames['tune'][ID])
    old=common.read_json(risk.union_folder(fold)/'manifest.json')
    assert common.sha256(risk.union_folder(fold)/'manifest.json')==protocol['original_controls'][fold]
    old_path=risk.union_folder(fold)/'tune_predictions.parquet'
    assert common.sha256(old_path)==old['outputs'][old_path.name]
    original=pd.read_parquet(old_path)
    np.testing.assert_array_equal(control[ID],original[ID])
    np.testing.assert_allclose(control.prediction_sec,original.prediction_sec,rtol=0,atol=1e-7)
    encoder=common.read_json(CONTROL[fold]/'encoder.json')
    assert encoder['columns']==protocol['columns']
    ids=pd.Index(pd.concat([frames[s][ID] for s in ('fit','tune')],ignore_index=True))
    risk.feature_sources,risk.guard=sources,guard
    matrix,vocab,receipts=risk.load_matrix(ids,len(frames['fit']),protocol['columns'],dest)
    assert vocab==encoder['vocab'] and receipts==marker['feature_receipts']
    common.write_json(dest/'discovery.json',DISCOVERY)
    common.write_json(dest/'encoder.json',dict(columns=protocol['columns'],vocab=vocab))
    targets={s:f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s,f in frames.items()}
    prediction,bank,fallback=train_bank(matrix,len(frames['fit']),targets['fit'],targets['tune'],frames['tune'].proxy_sec.to_numpy(float),
        frames['fit'][ID],frames['tune'][ID],airport_values(frames['fit']),airport_values(frames['tune']),
        control.prediction_sec.to_numpy(float),protocol['columns'],vocab,protocol['params'],marker['steps'],dest)
    pd.DataFrame({ID:frames['tune'][ID], 'prediction_sec':prediction,'global_fallback':fallback}).to_parquet(dest/'tune.parquet',index=False)
    ordinary,current,base_receipt=baseline(fold,frames['tune'],protocol)
    pd.DataFrame({ID:frames['tune'].loc[ordinary,ID],'baseline_sec':current,
        'prediction_sec':.75*current+.25*prediction[ordinary]}).to_parquet(dest/'ordinary_fixed25.parquet',index=False)
    y=frames['tune'][TARGET].to_numpy(float)
    days=frames['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    results=dict(matched_all_finite=compare(y,prediction,control.prediction_sec.to_numpy(),days),
        matched_ordinary=compare(y[ordinary],prediction[ordinary],control.prediction_sec.to_numpy()[ordinary],days[ordinary]),
        ordinary_fixed25=compare(y[ordinary],.75*current+.25*prediction[ordinary],current,days[ordinary]))
    del matrix
    gc.collect()
    (dest/'matrix.float32').unlink()
    record=dict(status='complete',fold=fold,source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),
        split=split,ids=cohort,params=protocol['params'],feature_columns=protocol['columns'],feature_receipts=receipts,
        bank=bank,global_fallback_rows=int(fallback.sum()),baseline_receipt=base_receipt,results=results,
        fold_positive_day_gate=all(v['gain']>0 and v['all_day_removals_improve'] for v in results.values()),
        seasonal_materiality_pending=True,no_score_prediction=True,peak_bytes=guard(),native_replay_max_abs_delta=0.,
        outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file()})
    common.write_json(dest/'manifest.json',record)
    print('RESULT',fold,results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    if args.declare_only:declare()
    else:
        assert args.fold
        run(args.fold)


