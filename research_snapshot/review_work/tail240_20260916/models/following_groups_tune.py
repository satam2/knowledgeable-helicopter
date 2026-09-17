"""Matched full-cohort ordinary expert with same-runway/stand following clocks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm as lgb
from pathlib import Path
import sys
import argparse
import gc
import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/risk'))
import run_risk as risk
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/source_distinctions'))
import following_groups as addition
import normalized_catboost_tune as metric
common,ID,TARGET=risk.common,risk.ID,risk.TARGET
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1')
ARMS=['control387','following415']
ORIGINAL_SOURCES=risk.feature_sources


def sources(desired):
    result=ORIGINAL_SOURCES(desired)
    chosen=[c for c in desired if c in addition.COLUMNS]
    if chosen:
        marker=common.read_json(addition.OUT/'manifest.json')
        assert marker['status']=='complete' and marker['source_sha256']==common.sha256(addition.__file__)
        assert common.sha256(addition.OUT/'features.parquet')==marker['feature_sha256']
        result.append((addition.OUT/'features.parquet',marker['feature_sha256'],True,chosen))
    return result


def guard():
    rss=psutil.Process().memory_info().rss
    assert rss<10*1024**3, f'Main CPU10GiB sampled RSS exceeded: {rss}'
    assert psutil.virtual_memory().available>=8*1024**3
    return rss


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    control={f:common.read_json(risk.union_folder(f)/'manifest.json') for f in ('F1','F3')}
    assert control['F1']['fit']['params']==control['F3']['fit']['params']
    record=dict(source_sha256=common.sha256(__file__),arms=ARMS,params=control['F1']['fit']['params'],
        full_columns=control['F1']['feature_columns'],extra_columns=addition.COLUMNS,
        sources={str(Path(p).relative_to(ROOT)):common.sha256(p) for p in [risk.__file__,addition.__file__,metric.__file__]},
        cache_manifest_sha256=common.sha256(addition.OUT/'manifest.json'),
        control_manifests={f:common.sha256(risk.union_folder(f)/'manifest.json') for f in control},
        model='Same frozen leaf63 LGBM2500cap/.03/min50/L2=10, 150patience, seed20260916,2CPU. RawY-minusfiniteNM target, no clipping/sampling.',
        availability='Inherited retrospective387 plus declaredfollowing28 observation-onlycache. Alloriginalfinitefit/tune with flightIDpurges. No score/refit.',
        encoder='Risk audited streaming original387 feature sources with fit-onlynativecategorycodes; cache-specific missing sentinel behavior preserved.',
        gate='Fresh387control first must match original tune model predictions within1e-7; otherwise stopdiagnose before candidate. Candidate mustbeatcontrol andfixed25complementglobal9 ordinary tune inbothmonths, everyUTCdayremoval positive; materiality >2seconds blendgain preferred.',
        resources='10GiB sampledprocessRSS,8GiBhostreserve,2CPU; centrallyserialized with20GiBneural full-load phase.',
        caution='Global9 weights were learned on same tune month; comparison is adaptivedevelopment, not freshvalidation.')
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


def run(fold):
    protocol=declare()
    assert psutil.virtual_memory().available>=18*1024**3
    risk.feature_sources=sources
    risk.guard=guard
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path)==audit['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    selected={s:p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())] for s,p in idx.items() if s in ('fit','tune')}
    frames={s:meta.iloc[p].copy() for s,p in selected.items()}
    ids=pd.Index(pd.concat([f[ID] for f in frames.values()],ignore_index=True))
    nfit=len(frames['fit'])
    old_folder=risk.union_folder(fold)
    old_manifest=common.read_json(old_folder/'manifest.json')
    assert common.sha256(old_folder/'tune_predictions.parquet')==old_manifest['outputs']['tune_predictions.parquet']
    old=pd.read_parquet(old_folder/'tune_predictions.parquet')
    np.testing.assert_array_equal(old[ID],frames['tune'][ID])
    target={s:f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s,f in frames.items()}
    dates=frames['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    y=frames['tune'][TARGET].to_numpy(float)
    predictions={}
    summary={}
    for arm in ARMS:
        dest=OUT/fold/arm
        dest.mkdir(parents=True,exist_ok=False)
        columns=protocol['full_columns']+(addition.COLUMNS if arm=='following415' else [])
        matrix,vocab,receipts=risk.load_matrix(ids,nfit,columns,dest)
        cats=[columns.index(c) for c in vocab]
        model=lgb.LGBMRegressor(**protocol['params'])
        model.fit(matrix[:nfit],target['fit'],categorical_feature=cats,feature_name=columns,
            eval_X=matrix[nfit:],eval_y=target['tune'],eval_metric='rmse',
            callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(250)])
        steps=int(model.best_iteration_ or model.n_estimators_)
        pred=model.predict(matrix[nfit:],num_iteration=steps)+frames['tune'].proxy_sec.to_numpy(float)
        assert np.isfinite(pred).all()
        model.booster_.save_model(str(dest/'model.txt'),num_iteration=steps)
        replay=lgb.Booster(model_file=str(dest/'model.txt')).predict(matrix[nfit:])+frames['tune'].proxy_sec.to_numpy(float)
        np.testing.assert_array_equal(pred,replay)
        pd.DataFrame({ID:frames['tune'][ID],'prediction_sec':pred}).to_parquet(dest/'tune.parquet',index=False)
        common.write_json(dest/'encoder.json',dict(columns=columns,vocab=vocab))
        record=dict(status='complete',fold=fold,arm=arm,steps=steps,params=protocol['params'],split=split,
            ids={s:dict(n=len(f),hash=common.object_hash(f[ID].tolist())) for s,f in frames.items()},
            source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),
            feature_receipts=receipts,rmse=float(np.sqrt(np.mean((pred-y)**2))),native_replay_max_abs_delta=0.,
            original_control_max_abs_delta=float(np.max(np.abs(pred-old.prediction_sec.to_numpy()))) if arm=='control387' else None,
            outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.name in ['model.txt','tune.parquet','encoder.json']})
        common.write_json(dest/'manifest.json',record)
        predictions[arm]=pred
        summary[arm]=record['rmse']
        del model,matrix
        gc.collect()
        (dest/'matrix.float32').unlink()
        print('RESULT',fold,arm,record['rmse'],'controlparity',record['original_control_max_abs_delta'],flush=True)
        if arm=='control387':
            assert record['original_control_max_abs_delta']<1e-7, 'Fresh streamed control differs; preserve and diagnose before candidate'
    summary['matched']=metric.comparison(y,predictions['following415'],predictions['control387'],dates)
    common.write_json(OUT/fold/'summary.json',dict(status='complete',results=summary,no_score_prediction=True))
    print('SUMMARY',fold,summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    if args.declare_only:
        declare()
    else:
        assert args.fold
        run(args.fold)
