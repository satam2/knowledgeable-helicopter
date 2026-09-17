"""Independent finite-source classifier replay and diagnostic metric oracle."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm as lgb
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
import validate_candidate as validation

ROOT=validation.ROOT
ID,TARGET=validation.ID,validation.TARGET
BASE=ROOT/'private_runs/tail240_20260916/state/risk/v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/risk_postfit_v1'
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def close(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    parser.add_argument('--arm',choices=['own','full387'],required=True)
    args=parser.parse_args()
    folder=BASE/args.fold/args.arm
    record=read(folder/'manifest.json')
    protocol=read(BASE/'protocol.json')
    assert record['status']=='complete' and record['protocol_sha256']==sha(BASE/'protocol.json')
    assert record['source_sha256']==protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/state/risk/run_risk.py')
    for name,digest in record['outputs'].items():
        assert sha(folder/name)==digest
    binding=read(validation.BINDING)['folds'][args.fold]
    assert oh(record['split'])==binding['split_hash']
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(path)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt',TARGET,'proxy_sec'])
    idx,_,_=validation.common.fold_data(meta,args.fold,full=True)
    stages={stage:meta.iloc[idx[stage]].loc[lambda frame:np.isfinite(frame.proxy_sec)].copy() for stage in ('fit','tune')}
    for stage,frame in stages.items():
        assert {'n':len(frame),'hash':oh(frame[ID].tolist())}==binding['cohorts']['finite_nm'][stage]
        assert record[stage+'_ids_hash']==oh(frame[ID].tolist())
        assert record[stage+'_label_hash']==oh(frame[TARGET].tolist())
    fit,tune=stages['fit'],stages['tune']
    yfit=np.abs(fit[TARGET].to_numpy()-fit.proxy_sec.to_numpy())>1800
    assert int(yfit.sum())==record['fit_positive_count']
    assert yfit.mean()==record['fit_prevalence']
    table=pd.read_parquet(folder/'tune_probabilities.parquet')
    for column in [ID,'ADEP_mvt','MVT_TIME_UTC_mvt',TARGET,'proxy_sec']:
        np.testing.assert_array_equal(table[column],tune[column])
    ytune=(np.abs(tune[TARGET].to_numpy()-tune.proxy_sec.to_numpy())>1800).astype('int8')
    np.testing.assert_array_equal(table.large_gap,ytune)
    enc=read(folder/'encoder.json')
    columns=enc['columns']
    assert columns==record['feature_columns']==protocol['full_columns' if args.arm=='full387' else 'own_columns']
    ids=pd.Index(tune[ID])
    matrix=np.full((len(ids),len(columns)),np.nan,dtype='float32',order='F')
    counts=np.zeros(len(columns),dtype=int)
    for source in record['feature_receipts']:
        path=Path(source['path'])
        assert sha(path)==source['sha256']
        names=source['columns']
        seen=np.zeros(len(ids),dtype=bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas()
            rows=ids.get_indexer(frame[ID])
            keep=rows>=0
            rows=rows[keep]
            assert not seen[rows].any() and len(np.unique(rows))==len(rows)
            seen[rows]=True
            frame=frame.loc[keep]
            for name in names:
                if name in enc['vocab']:
                    mapping={value:i+2 for i,value in enumerate(enc['vocab'][name])}
                    values=frame[name].astype('string').map(mapping).fillna(1).where(frame[name].notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(frame[name]).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    base_features='screening_230/data/interim/features' in path.as_posix()
                    if not base_features and 'sequence_flatten' not in str(path):
                        values[np.isnan(values)]=-999999
                matrix[rows,columns.index(name)]=values
        for name in names:
            counts[columns.index(name)]+=int(seen.sum())
        validation.guard()
    assert np.all(counts==len(ids))
    model=lgb.Booster(model_file=str(folder/'model.txt'))
    probabilities=model.predict(matrix,num_threads=2)
    np.testing.assert_array_equal(probabilities,table.probability)
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    union=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{args.fold}_s20260916'
    assert sha(union/'manifest.json')==record['union_manifest_sha256']
    assert sha(union/'tune_predictions.parquet')==record['union_tune_sha256']
    previous=pd.read_parquet(union/'tune_predictions.parquet').set_index(ID)
    np.testing.assert_array_equal(previous.loc[ids,'prediction_sec'],table.union_fit_prediction_sec)
    sse=(tune[TARGET].to_numpy()-table.union_fit_prediction_sec.to_numpy())**2
    reports=read(folder/'metrics.json')
    checks={}
    for name,mask in [('ordinary_proxy',tune.proxy_sec.between(0,7200).to_numpy()),('all_finite',np.ones(len(tune),bool))]:
        expected=reports[name]
        y,p,e=ytune[mask],probabilities[mask],sse[mask]
        prevalence=yfit.mean()
        assert expected['rows']==len(y) and expected['positives']==int(y.sum())
        values={'prevalence':y.mean(),'average_probability':p.mean(),'roc_auc':roc_auc_score(y,p),
            'pr_auc_average_precision':average_precision_score(y,p),'brier':np.mean((y-p)**2),
            'logloss':log_loss(y,p,labels=[0,1]),'baseline_brier':np.mean((y-prevalence)**2),
            'baseline_logloss':log_loss(y,np.full(len(y),prevalence),labels=[0,1]),'union_sse':e.sum()}
        for key,value in values.items():
            close(value,expected[key])
        order=np.argsort(-p,kind='stable')
        for fraction in (.01,.05,.1):
            selected=order[:max(1,int(np.ceil(len(y)*fraction))) ]
            top=expected['top_risk'][str(fraction)]
            assert len(selected)==top['rows']
            for key,value in {'threshold':p[selected].min(),'precision':y[selected].mean(),'positive_capture':y[selected].sum()/y.sum(),'union_sse_capture':e[selected].sum()/e.sum()}.items():
                close(value,top[key])
        binids=np.minimum(np.searchsorted(protocol['calibration_bins'],p,side='right')-1,len(protocol['calibration_bins'])-2)
        for i,b in enumerate(expected['calibration']):
            members=binids==i
            assert b['rows']==int(members.sum())
            if members.any():
                close(p[members].mean(),b['mean_probability'])
                close(y[members].mean(),b['observed_rate'])
        checks[name]={'rows':len(y),'positives':int(y.sum()),'pr_auc':float(values['pr_auc_average_precision']),
            'roc_auc':float(values['roc_auc']),'brier':float(values['brier']),'baseline_brier':float(values['baseline_brier']),
            'top1pct_positive_capture':expected['top_risk']['0.01']['positive_capture'],'top1pct_union_sse_capture':expected['top_risk']['0.01']['union_sse_capture']}
    dest=OUT/args.fold/args.arm
    dest.mkdir(parents=True,exist_ok=False)
    receipt={'status':'passed','source_sha256':sha(__file__),'producer_manifest_sha256':sha(folder/'manifest.json'),
        'fold':args.fold,'arm':args.arm,'native_full_tune_probability_replay_max_abs_delta':0.,'checks':checks,
        'peak_rss_bytes':validation.guard(),'scope':'Original complete finite fit/tune IDs/rawlabels independently verified, features rebuilt from bound raw caches with saved fitencoder; all probability replay and point metrics/calibration/rank captures exact. Daybootstrap source reviewed but not independently recomputed. Classification diagnostic only; no taxi prediction or score evaluation.'}
    write(dest/'receipt.json',receipt)
    print(args.fold,args.arm,checks,flush=True)


if __name__=='__main__':
    main()
