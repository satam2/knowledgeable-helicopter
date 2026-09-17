"""Direct cache-to-CatBoost encoding and independent saved native CPU replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import lightgbm
import argparse
import gc
from pathlib import Path
import sys
import joblib
import catboost
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT,read,sha,write,object_hash,guard
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import encoders
import validate_candidate as v
from verify_linear_finite import check_result

BASE=ROOT/'private_runs/tail240_20260916/models/catboost_union387_v2'
ID,TIME,TARGET='MVT_ID_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt'


def main(fold):
    folder=BASE/fold;marker=read(folder/'manifest.json');protocol=read(BASE/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==sha(BASE/'protocol.json')
    for relative,expected in protocol['source_hashes'].items():assert sha(ROOT/relative)==expected
    for name,expected in marker['outputs'].items():assert sha(folder/name)==expected
    binding=protocol['controls'][fold]
    model=joblib.load(folder/'fit_model.joblib');encoder=model['encoder'];columns=encoder.columns
    assert columns==marker['columns']==protocol['columns'] and len(columns)==387
    if fold=='F1':
        prior=ROOT/'private_runs/tail240_20260916/validation/catboost387_inputs_F1_v1'
        receipt=read(prior/'receipt.json');assert receipt['status']=='passed'
        assert receipt['encoder387_sha256']==sha(prior/'encoder387.joblib')
        verified=joblib.load(prior/'encoder387.joblib')
        for name in ('columns','numeric','categories'):assert getattr(verified,name)==getattr(encoder,name)
    evidence=read(folder/'fit_evidence.json');assert evidence['params']==protocol['parameters']
    assert model['steps']==evidence['steps']==model['estimator'].tree_count_
    del model
    gc.collect()
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(metadata)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    assert not any(marker['split']['purged_related_departures'].values())
    pieces=[]
    for batch in pq.ParquetFile(metadata).iter_batches(batch_size=8192,columns=[ID,TIME,TARGET,'proxy_sec'],use_threads=False):
        data=batch.to_pandas();start,stop=[pd.Timestamp(value,tz='UTC') for value in marker['split']['spec']['refit']]
        pieces.append(data.loc[data[TIME].ge(start)&data[TIME].lt(stop)&np.isfinite(data.proxy_sec)])
    meta=pd.concat(pieces,ignore_index=True);boundary=pd.Timestamp(marker['split']['spec']['tune'][0],tz='UTC')
    fit=meta.loc[meta[TIME].lt(boundary)];tune=meta.loc[meta[TIME].ge(boundary)].reset_index(drop=True)
    assert {stage:dict(n=len(part),hash=object_hash(part[ID].tolist())) for stage,part in [('fit',fit),('tune',tune)]}==marker['ids']=={stage:binding['fit_ids'][stage] for stage in ('fit','tune')}
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True));nfit=len(fit)
    fit_label_hash=object_hash(fit[TARGET].tolist());tune_label_hash=object_hash(tune[TARGET].tolist())
    if fold=='F1':
        assert fit_label_hash==receipt['fit_raw_label_hash'] and tune_label_hash==receipt['tune_raw_label_hash']
    del meta,pieces,fit
    gc.collect()
    matrix=np.empty((len(tune),len(columns)),dtype='float32',order='F')
    vocab={name:set() for name in encoder.categories};base_categories=set(columns[:11]);totals={name:0 for name in columns}
    for source in marker['source_receipts']:
        path=Path(source['path']);assert sha(path)==source['sha256'];names=source['columns'];seen=np.zeros(len(ids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            data=batch.to_pandas();positions=ids.get_indexer(data[ID]);keep=positions>=0;positions=positions[keep];data=data.loc[keep]
            assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
            selected=positions>=nfit
            for name in names:
                values=data[name]
                if name in vocab:
                    raw=values.astype('string')
                    if name not in base_categories:raw=raw.fillna('MISSING')
                    vocab[name].update(raw.iloc[np.flatnonzero(positions<nfit)].dropna().tolist())
                    values=raw.map(encoder.categories[name]).fillna(1).where(raw.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan)
                    values[(values==-999999.)|~np.isfinite(values)]=np.nan
                matrix[positions[selected]-nfit,columns.index(name)]=values[selected]
                totals[name]+=len(positions)
            assert guard()<3*1024**3
        assert int(seen.sum())==source['rows']
    assert all(value==len(ids) for value in totals.values())
    for name,values in vocab.items():assert {value:i+2 for i,value in enumerate(sorted(values))}==encoder.categories[name]
    data={name:(pd.Categorical(matrix[:,j].astype('int32'),categories=range(len(encoder.categories[name])+2)) if name in vocab else matrix[:,j]) for j,name in enumerate(columns)}
    encoded=pd.DataFrame(data,copy=False)[columns]
    native=catboost.CatBoostRegressor();native.load_model(str(folder/'model.cbm'))
    assert native.feature_names_==columns and native.tree_count_==evidence['steps']
    prediction=native.predict(encoded,task_type='CPU',thread_count=1)+tune.proxy_sec.to_numpy(float)
    stored=pd.read_parquet(folder/'tune_predictions.parquet')
    for name in (ID,TIME,TARGET,'proxy_sec'):np.testing.assert_array_equal(stored[name],tune[name])
    np.testing.assert_array_equal(prediction,stored.prediction_sec)
    del native
    oldroot=ROOT/f'private_runs/breakthrough_20260916/combined_catboost/catboost_combined_aobt_allfinite_{fold}_s20260916'
    assert sha(oldroot/'manifest.json')==binding['manifest_sha256']
    for name in ('fit_model.joblib','tune_predictions.parquet'):assert sha(oldroot/name)==binding['outputs'][name]
    oldmodel=joblib.load(oldroot/'fit_model.joblib')
    assert oldmodel['encoder'].numeric==[name for name in columns[:225] if name not in vocab]
    for name,mapping in oldmodel['encoder'].categories.items():assert mapping==encoder.categories[name]
    reference=oldmodel['estimator'].predict(encoded[columns[:225]],task_type='CPU',thread_count=1)+tune.proxy_sec.to_numpy(float)
    old=pd.read_parquet(oldroot/'tune_predictions.parquet').set_index(ID).loc[tune[ID],'prediction_sec'].to_numpy(float)
    np.testing.assert_array_equal(reference,old)
    del oldmodel,encoded,matrix,data
    gc.collect();assert guard()<3*1024**3
    ensemble=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    assert sha(ensemble/f'{fold}_weights.json')==binding['weights_sha256']
    assert sha(ensemble/f'{fold}_aligned_tune.parquet')==binding['aligned_sha256']
    weights=read(ensemble/f'{fold}_weights.json');aligned=pd.read_parquet(ensemble/f'{fold}_aligned_tune.parquet')
    ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID],tune.loc[ordinary,ID]);np.testing.assert_array_equal(aligned[TARGET],tune.loc[ordinary,TARGET])
    np.testing.assert_array_equal(aligned.catboost_combined,old[ordinary])
    neural=ROOT/'private_runs/tail240_20260916/state/neural_context'/('v1' if fold=='F1' else 'v3')/fold
    assert sha(neural/'manifest.json')==binding['neural_manifest_sha256']
    assert sha(neural/'tune_predictions.parquet')==read(neural/'manifest.json')['outputs']['tune_predictions.parquet']
    ple=pd.read_parquet(neural/'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(ple[TARGET],aligned[TARGET])
    current=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    current+=weights['global'][weights['experts'].index('tabm_ple8')]*(ple.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    coefficient=weights['global'][weights['experts'].index('catboost_combined')]
    candidate=current+coefficient*(prediction[ordinary]-old[ordinary])
    y=tune[TARGET].to_numpy(float);days=tune[TIME].dt.floor('D').to_numpy();metrics=read(folder/'metrics.json');assert metrics==marker['metrics']
    reports=dict(matched=check_result(y,prediction,old,days,metrics['matched_all_finite']),
        primary=check_result(y[ordinary],candidate,current,days[ordinary],metrics['primary_current387']))
    out=ROOT/f'private_runs/tail240_20260916/validation/catboost387_result_{fold}_v2';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',source_sha256=sha(Path(__file__)),producer_manifest_sha256=sha(folder/'manifest.json'),
        protocol_sha256=sha(BASE/'protocol.json'),all387_tune_inputs_and15fit_vocabularies_rebuilt=True,
        candidate_native_cpu_max_abs_delta=0.,old225_native_cpu_max_abs_delta=0.,steps=evidence['steps'],
        fit_rows=nfit,tune_rows=len(tune),fit_label_hash=fit_label_hash,tune_label_hash=tune_label_hash,
        metrics=reports,coefficient=coefficient,peak_bytes=guard(),no_GPU_or_fitting=True,
        limitation='PreviouslyverifiedF1fitinputpreflight reusedbyexactencoderparity;newtunesources/vocab/rawlabels/nativeCPU/metricsindependent. Exposedtunegate,notcomplete-scoreclaim.')
    v.write_json(out/'receipt.json',result);print('PASSED',fold,metrics,'peak',result['peak_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    with threadpool_limits(1):main(args.fold)
