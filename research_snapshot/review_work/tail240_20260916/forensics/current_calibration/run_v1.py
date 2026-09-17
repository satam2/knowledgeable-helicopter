"""One current-nine additive calibration on purged early/late original tune."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import lightgbm as lgb
import argparse
import ast
import gc
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/state/risk'))
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/models'))
sys.path.insert(0, str(ROOT/'review_work/breakthrough_20260916/models/context_gate'))
import run_risk as risk
import schema_discovery
import gate

common, ID, TARGET = risk.common, risk.ID, risk.TARGET
TIME = common.MOVEMENT
FID = 'FLIGHT_ID_mvt'
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/forensics/current_calibration/v1')
ENSEMBLE = ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
NEURAL = {f:ROOT/'private_runs/tail240_20260916/state/neural_context'/version/f
          for f,version in [('F1','v1'),('F3','v3')]}
META = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
OLD_SOURCE = ROOT/'review_work/breakthrough_20260916/models/context_gate/residual_calibration/run.py'
EXPERTS = ['tabm_source','tabm_combined','lgb63','lgb63_sequence','tabm_ple8',
           'lgb63_sequence8','lgb63_union','catboost_combined','xgb_combined']
PARAMS = dict(n_estimators=200, max_depth=4, num_leaves=16, min_child_samples=500,
              learning_rate=.03, reg_lambda=20., objective='regression', random_state=20260916,
              n_jobs=2, verbosity=-1, subsample=1., colsample_bytree=1., deterministic=True,
              force_col_wise=True)
ORIGINAL_SOURCES = risk.feature_sources
CAP = 4*1024**3
WEIGHTS = {'F1':192122/344841, 'F3':152719/344841}


def guard(_env=None):
    info = psutil.Process().memory_info()
    peak = getattr(info,'peak_wset',info.rss)
    assert max(info.rss,peak) < CAP, ('4GiB calibration ceiling',info)
    assert psutil.virtual_memory().available >= 8*1024**3, 'Host reserve below8GiB'
    return int(peak)


def inherited_parameters():
    tree = ast.parse(OLD_SOURCE.read_text())
    assignment = next(node for node in tree.body if isinstance(node,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in node.targets))
    assert isinstance(assignment.value,ast.Call) and assignment.value.func.id=='dict'
    return {item.arg:ast.literal_eval(item.value) for item in assignment.value.keywords}


def partitions(frame):
    assert len(frame) and frame[ID].is_unique
    order = frame.sort_values([TIME,ID],kind='stable')
    cutoff = order.iloc[int(.6*len(order))][TIME]
    early = frame[TIME].lt(cutoff).to_numpy()
    late = ~early
    later_flights = frame.loc[late,FID].dropna().unique()
    purged = early & frame[FID].notna().to_numpy() & frame[FID].isin(later_flights).to_numpy()
    fit = early & ~purged
    assert fit.any() and late.any()
    return fit, early, late, purged, cutoff


def restore_numeric(values, filled):
    output = np.asarray(values,dtype=np.float32).copy()
    output[~np.isfinite(output)] = np.nan
    if filled:
        output[output == -999999.] = np.nan
    return output


def contrasts(predictions):
    p = np.asarray(predictions,dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] == 9 and np.isfinite(p).all()
    return (p-p.mean(axis=1,keepdims=True)).astype(np.float32)


def declaration():
    assert PARAMS == inherited_parameters()
    prep = common.read_json(ENSEMBLE/'preparation.json')
    columns = common.read_json(risk.union_folder('F1')/'manifest.json')['feature_columns']
    assert len(columns)==387 and len(set(columns))==387
    folds = {}
    for fold in ('F1','F3'):
        neural = common.read_json(NEURAL[fold]/'manifest.json')
        assert neural['status']=='complete' and neural['feature_columns']==columns
        weights = common.read_json(ENSEMBLE/f'{fold}_weights.json')
        assert weights==prep['folds'][fold]['weights'] and weights['experts']==EXPERTS
        folds[fold] = dict(aligned_tune_sha256=prep['folds'][fold]['aligned_tune_sha256'],
            weights_sha256=common.sha256(ENSEMBLE/f'{fold}_weights.json'),
            neural_manifest_sha256=common.sha256(NEURAL[fold]/'manifest.json'),
            neural_tune_sha256=neural['outputs']['tune_predictions.parquet'],
            union_manifest_sha256=common.sha256(risk.union_folder(fold)/'manifest.json'),
            split=neural['split'])
    audit = common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    return dict(source_sha256=common.sha256(__file__), tests_sha256=common.sha256(Path(__file__).with_name('test_v1.py')),
        dependencies={str(Path(p).relative_to(ROOT)):common.sha256(p) for p in
            [risk.__file__,common.__file__,schema_discovery.__file__,gate.__file__,OLD_SOURCE]},
        preparation_sha256=common.sha256(ENSEMBLE/'preparation.json'), meta_sha256=audit['artifacts'][META.name],
        columns=columns, contrast_columns=[f'current_expert_centered_{name}' for name in EXPERTS],
        experts=EXPERTS, params=PARAMS, folds=folds, seasonal_weights=WEIGHTS,
        cohort='All original ordinary tune rows only. Timestamp boundary at floor(.6*N) aftertime/IDsort. Purge early rows sharing nonnull flightID with any late row; late unchanged.',
        predictors='Exact387 supplied-batch features plus9 p_j-rowmean expert contrasts. Only original PLE225 expert replacedbyPLE387 FIT-model TUNE predictions.',
        preprocessing='Fit-only categorical vocab on purgedearly; numericNaNs preserved; -999999toNaN onlysources declaredfillTrue. Rawbaseclock sentinelvalue retained; no scaling/imputation.',
        target='RawY minus freshconstant simplex fitted on exact samepurgedearly rows.200fixedtrees additive correction, no evalset/earlystop/grid/blend/clipping.',
        primary='Late rawMSE vs sameearlyonlyconstant. Bothmonths positive,everydayremoval positive,and>=2second weightedlateordinaryRMSE gain required. No complete-score claim.',
        diagnostic='Frozenfulltune currentweights usingPLE387 substitution; weights already exposed to late labels, notprimarycontrol.',
        resources='2CPUthreads,currentandOShistoricalpeak<4GiB,start12GiB,reserve8GiB;sampledloading/everyroundguard,notOShardlimit.',
        exposure='Expert tune predictions excludedfromgradientfit but fulltunelabelsselectedupstreamstopping. Earlylate isdevelopment-exposed,notfreshOOF. No scorelabels/predictions/refit/ranking.',
        later='No advancement/refit/score automaticallyauthorized. Modelonlyordinary; all missing/nonordinary conceptually unchanged.')


def declare():
    protocol = declaration()
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==protocol,'Frozen protocol changed'
    else:
        common.write_json(path,protocol)
    return protocol


def preflight(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 12*1024**3
    dest = OUT/fold
    dest.mkdir(exist_ok=False)
    binding = protocol['folds'][fold]
    assert common.sha256(META)==protocol['meta_sha256']
    path = ENSEMBLE/f'{fold}_aligned_tune.parquet'
    assert common.sha256(path)==binding['aligned_tune_sha256']
    frame = pd.read_parquet(path)
    assert frame[ID].is_unique and len(frame)==common.read_json(ENSEMBLE/f'{fold}_weights.json')['tune_rows']
    start,end = binding['split']['spec']['tune']
    meta = pd.read_parquet(META,columns=[ID,FID,TIME,'ADEP_mvt','proxy_sec'],
        filters=[(TIME,'>=',pd.Timestamp(start,tz='UTC')),(TIME,'<',pd.Timestamp(end,tz='UTC'))]).set_index(ID)
    expected = meta.index[np.isfinite(meta.proxy_sec)&meta.proxy_sec.between(0,7200)]
    np.testing.assert_array_equal(frame[ID],expected)
    source_meta = meta.loc[frame[ID]]
    np.testing.assert_array_equal(pd.to_datetime(frame[TIME],utc=True),source_meta[TIME])
    np.testing.assert_array_equal(frame.proxy_sec,source_meta.proxy_sec)
    frame[FID] = source_meta[FID].to_numpy()
    np.testing.assert_array_equal(frame.ADEP_mvt.astype(str),source_meta.ADEP_mvt.astype(str))
    neural_path = NEURAL[fold]/'tune_predictions.parquet'
    assert common.sha256(neural_path)==binding['neural_tune_sha256']
    neural = pd.read_parquet(neural_path).set_index(ID)
    assert neural.index.is_unique
    np.testing.assert_array_equal(neural.index,meta.index[np.isfinite(meta.proxy_sec)])
    neural = neural.loc[frame[ID]]
    np.testing.assert_array_equal(neural[TARGET],frame[TARGET])
    old_predictions = frame[EXPERTS].to_numpy(float)
    weights = np.asarray(common.read_json(ENSEMBLE/f'{fold}_weights.json')['global'])
    old_baseline = old_predictions@weights
    frame['tabm_ple8'] = neural.prediction_sec.to_numpy()
    current_predictions = frame[EXPERTS].to_numpy(float)
    current = current_predictions@weights
    expected_current = old_baseline+weights[EXPERTS.index('tabm_ple8')]*(
        neural.prediction_sec.to_numpy()-old_predictions[:,EXPERTS.index('tabm_ple8')])
    np.testing.assert_allclose(current,expected_current,rtol=0,atol=1e-8)
    frame['frozen_current_sec'] = current
    fit,early,late,purged,cutoff = partitions(frame)
    for name,values in [('fit',fit),('early',early),('late',late),('purged',purged)]:
        frame[name]=values
    assert np.isfinite(frame[[TARGET,*EXPERTS,'frozen_current_sec']].to_numpy(float)).all()
    source_tuples,discovery = schema_discovery.cached_discovery(ORIGINAL_SOURCES,protocol['columns'])
    receipts = [dict(path=str(path),sha256=digest,fill=fill,columns=names) for path,digest,fill,names in source_tuples]
    frame.to_parquet(dest/'inputs.parquet',index=False)
    common.write_json(dest/'preflight.json',dict(status='passed',source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),inputs_sha256=common.sha256(dest/'inputs.parquet'),
        rows=len(frame),early_rows=int(early.sum()),fit_rows=int(fit.sum()),late_rows=int(late.sum()),
        purged_early_rows=int(purged.sum()),cutoff=str(cutoff),
        fit_ids_hash=common.object_hash(frame.loc[fit,ID].tolist()),
        late_ids_hash=common.object_hash(frame.loc[late,ID].tolist()),
        original_ids_hash=common.object_hash(frame[ID].tolist()),
        raw_labels_hash=common.object_hash(frame[TARGET].tolist()),
        source_receipts=receipts,discovery=discovery,peak_bytes=guard(),model_fit=False,
        no_score_label_or_prediction_read=True,expert_predictions='Saved fit-model originaltune;PLE387replacementonly'))
    print('PREFLIGHT',fold,'rows',len(frame),'fit',fit.sum(),'late',late.sum(),'purged',purged.sum(),'peak',guard(),flush=True)


def metrics(y,prediction):
    error = np.asarray(prediction,float)-np.asarray(y,float)
    return dict(rows=len(error),rmse=float(np.sqrt(np.mean(error**2))),mse=float(np.mean(error**2)),
        mae=float(np.mean(np.abs(error))),bias=float(np.mean(error)),p95=float(np.quantile(np.abs(error),.95)))


def comparison(y,prediction,baseline,dates):
    y,prediction,baseline = [np.asarray(v,float) for v in (y,prediction,baseline)]
    a,b = (y-baseline)**2,(y-prediction)**2
    codes,days = pd.factorize(pd.DatetimeIndex(pd.to_datetime(dates,utc=True)).floor('D'),sort=True)
    deletion = []
    for i,day in enumerate(days):
        keep = codes!=i
        deletion.append(dict(day=str(day),gain=float(np.sqrt(a[keep].mean())-np.sqrt(b[keep].mean()))))
    top = np.argsort(a-b,kind='stable')[::-1]
    influence = {}
    for n in (1,2,5,10):
        keep = np.ones(len(y),bool)
        keep[top[:n]]=False
        influence[str(n)] = float(np.sqrt(a[keep].mean())-np.sqrt(b[keep].mean()))
    boot = np.random.default_rng(20260916).multinomial(len(days),np.full(len(days),1/len(days)),size=300)
    counts = np.bincount(codes,minlength=len(days))
    ag = np.bincount(codes,weights=a,minlength=len(days))
    bg = np.bincount(codes,weights=b,minlength=len(days))
    gains = np.sqrt((boot@ag)/(boot@counts))-np.sqrt((boot@bg)/(boot@counts))
    return dict(candidate=metrics(y,prediction),baseline=metrics(y,baseline),
        gain=float(np.sqrt(a.mean())-np.sqrt(b.mean())),day_removal=deletion,
        every_day_positive=all(item['gain']>0 for item in deletion),
        remove_largest_gain_rows=influence,gain_ci95=np.quantile(gains,[.025,.975]).tolist(),
        bootstrap='300fixedseed dayclusters; trainedmodels fixed; no adaptive-selection correction')


def fit(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 12*1024**3
    dest = OUT/fold
    preflight_record = common.read_json(dest/'preflight.json')
    assert preflight_record['status']=='passed' and preflight_record['protocol_sha256']==common.sha256(OUT/'protocol.json')
    assert common.sha256(dest/'inputs.parquet')==preflight_record['inputs_sha256']
    assert not (dest/'model.txt').exists() and not (dest/'manifest.json').exists()
    started = time.monotonic()
    frame = pd.read_parquet(dest/'inputs.parquet')
    fit_mask = frame.fit.to_numpy(bool)
    late = frame.late.to_numpy(bool)
    order = np.r_[np.flatnonzero(fit_mask),np.flatnonzero(~fit_mask)]
    inverse = np.argsort(order)
    nfit = int(fit_mask.sum())
    sources,discovery = schema_discovery.cached_discovery(ORIGINAL_SOURCES,protocol['columns'])
    assert [dict(path=str(p),sha256=h,fill=fill,columns=c) for p,h,fill,c in sources]==preflight_record['source_receipts']
    risk.feature_sources = lambda columns:sources if columns==protocol['columns'] else (_ for _ in ()).throw(ValueError('Schema changed'))
    risk.guard=guard
    matrix,vocab,receipts = risk.load_matrix(pd.Index(frame.iloc[order][ID]),nfit,protocol['columns'],dest)
    fill_by_name = {}
    for _,_,filled,names in sources:
        for name in names:
            if name in fill_by_name:
                assert fill_by_name[name]==filled
            fill_by_name[name]=filled
    for j,name in enumerate(protocol['columns']):
        if name not in vocab:
            matrix[:,j]=restore_numeric(matrix[:,j],fill_by_name[name])
    p = frame[EXPERTS].to_numpy(float)
    y = frame[TARGET].to_numpy(float)
    with threadpool_limits(2):
        weights = gate.constant_weights(y[fit_mask],p[fit_mask])
    baseline = p@weights
    x = np.empty((len(frame),396),dtype=np.float32,order='F')
    x[:,:387]=matrix
    x[:,387:]=contrasts(p)[order]
    del matrix
    gc.collect()
    guard()
    column_names=protocol['columns']+protocol['contrast_columns']
    common.write_json(dest/'encoder.json',dict(columns=column_names,vocab=vocab,
        sentinel_to_nan={name:filled for name,filled in fill_by_name.items() if name not in vocab},
        fit_ids_hash=preflight_record['fit_ids_hash']))
    common.write_json(dest/'weights.json',dict(experts=EXPERTS,weights=weights.tolist(),
        objective='PurgedearlyrawSSE simplex,nointercept',fit_rows=nfit,fit_ids_hash=preflight_record['fit_ids_hash']))
    target=(y-baseline)[order]
    model=lgb.LGBMRegressor(**PARAMS)
    train_started=time.monotonic()
    model.fit(x[:nfit],target[:nfit],feature_name=column_names,
        categorical_feature=[column_names.index(name) for name in vocab],callbacks=[guard,lgb.log_evaluation(50)])
    fit_seconds=time.monotonic()-train_started
    assert model.n_estimators_==200
    prediction=baseline+model.predict(x,num_iteration=200,num_threads=2)[inverse]
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest/'model.txt'),num_iteration=200)
    native=lgb.Booster(model_file=str(dest/'model.txt'))
    replay=baseline+native.predict(x,num_iteration=200,num_threads=2)[inverse]
    np.testing.assert_array_equal(prediction,replay)
    output=frame[[ID,FID,TIME,TARGET,'fit','early','late','purged','frozen_current_sec']].copy()
    output['early_constant_sec']=baseline
    output['prediction_sec']=prediction
    output['correction_sec']=prediction-baseline
    output['squared_error']=(prediction-y)**2
    output.to_parquet(dest/'predictions.parquet',index=False)
    result=dict(primary=comparison(y[late],prediction[late],baseline[late],frame.loc[late,TIME]),
        frozen_current_diagnostic=comparison(y[late],prediction[late],frame.loc[late,'frozen_current_sec'],frame.loc[late,TIME]),
        early_training=dict(constant=metrics(y[fit_mask],baseline[fit_mask]),candidate=metrics(y[fit_mask],prediction[fit_mask])))
    common.write_json(dest/'metrics.json',result)
    assert declaration()==protocol
    common.write_json(dest/'manifest.json',dict(status='complete',source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),preflight_sha256=common.sha256(dest/'preflight.json'),
        feature_receipts=receipts,discovery=discovery,feature_columns=column_names,
        fit_rows=nfit,late_rows=int(late.sum()),purged_rows=preflight_record['purged_early_rows'],
        params=model.get_params(),steps=model.n_estimators_,native_replay_max_abs_delta=0.,
        metrics=result,fit_seconds=fit_seconds,total_seconds=time.monotonic()-started,peak_bytes=guard(),
        outputs={name:common.sha256(dest/name) for name in ['inputs.parquet','encoder.json','weights.json','model.txt','predictions.parquet','metrics.json']}))
    print('CALIBRATION',fold,result,'peak',guard(),flush=True)


def assess():
    protocol=declare()
    summaries={fold:common.read_json(OUT/fold/'manifest.json') for fold in ('F1','F3')}
    assert all(r['status']=='complete' and r['protocol_sha256']==common.sha256(OUT/'protocol.json') for r in summaries.values())
    old=np.sqrt(sum(WEIGHTS[f]*r['metrics']['primary']['baseline']['mse'] for f,r in summaries.items()))
    new=np.sqrt(sum(WEIGHTS[f]*r['metrics']['primary']['candidate']['mse'] for f,r in summaries.items()))
    passed=old-new>=2 and all(r['metrics']['primary']['gain']>0 and r['metrics']['primary']['every_day_positive'] for r in summaries.values())
    record=dict(status='complete',primary_gate_passed=bool(passed),late_ordinary_seasonal_constant=float(old),
        late_ordinary_seasonal_candidate=float(new),gain=float(old-new),seasonal_weights=WEIGHTS,
        folds={f:dict(primary=r['metrics']['primary'],manifest_sha256=common.sha256(OUT/f/'manifest.json')) for f,r in summaries.items()},
        limitation='Only lateordinarytune; NOT complete-score impact. Exposedupstreamstopping; noautomaticadvancement,refit,ranking orscoreaccess.')
    assert not (OUT/'assessment.json').exists()
    common.write_json(OUT/'assessment.json',record)
    print('ASSESSMENT',record,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['declare','preflight','fit','assess'])
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    if args.stage=='declare':
        declare()
        print('DECLARED',guard(),flush=True)
    elif args.stage=='assess':
        assess()
    else:
        assert args.fold
        (preflight if args.stage=='preflight' else fit)(args.fold)
