"""Native neural replay plus independent full-fit columnwise statistics/bins."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import torch
import lightgbm
import argparse
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT=v.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/neural_context'))
import run as producer
common=producer.common
ID,TARGET=producer.ID,producer.TARGET


def guard():
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    assert peak<8*1024**3,peak
    assert psutil.virtual_memory().available>=8*1024**3
    return int(peak)



def bounded_load(ids,nfit,desired,folder):
    sources=producer.risk.feature_sources(desired)
    print('SOURCE_HASH_MEMORY',guard(),flush=True)
    matrix=np.memmap(folder/'matrix.float32',mode='w+',dtype='float32',shape=(len(ids),len(desired)),order='F')
    vocab_sets={};receipts=[];counts=np.zeros(len(desired),dtype=np.int64)
    for path,expected,fill,names in sources:
        parquet=pq.ParquetFile(path)
        selected=None
        for name in names:
            dtype=parquet.schema_arrow.field(name).type
            if pa.types.is_dictionary(dtype):dtype=dtype.value_type
            categorical=not(pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype))
            if categorical:
                known=vocab_sets.setdefault(name,set())
                for batch in parquet.iter_batches(batch_size=8192,columns=[ID,name],use_threads=False):
                    frame=batch.to_pandas();positions=ids.get_indexer(frame[ID])
                    mask=(positions>=0)&(positions<nfit)
                    known.update(frame.loc[mask,name].dropna().astype(str).tolist())
            else:
                seen=np.zeros(len(ids),bool);column_index=desired.index(name)
                for batch in parquet.iter_batches(batch_size=8192,columns=[ID,name],use_threads=False):
                    frame=batch.to_pandas();positions=ids.get_indexer(frame[ID]);keep=positions>=0;positions=positions[keep]
                    assert len(np.unique(positions))==len(positions) and not seen[positions].any()
                    seen[positions]=True
                    values=pd.to_numeric(frame.loc[keep,name]).to_numpy(dtype=np.float32,na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if fill:values[np.isnan(values)]=-999999.
                    matrix[positions,column_index]=values
                if selected is None:selected=int(seen.sum())
                else:assert selected==int(seen.sum())
                counts[column_index]+=int(seen.sum())
        receipts.append(dict(path=str(path),sha256=expected,rows=selected,columns=names))
        print('NUMERIC_SOURCE',path.parent.name,len(names),guard(),flush=True)
    vocab={name:sorted(values) for name,values in vocab_sets.items()}
    for source_index,(path,expected,fill,names) in enumerate(sources):
        parquet=pq.ParquetFile(path)
        for name in [name for name in names if name in vocab]:
            known={value:i+2 for i,value in enumerate(vocab[name])};seen=np.zeros(len(ids),bool)
            for batch in parquet.iter_batches(batch_size=8192,columns=[ID,name],use_threads=False):
                frame=batch.to_pandas();positions=ids.get_indexer(frame[ID]);keep=positions>=0;positions=positions[keep]
                assert len(np.unique(positions))==len(positions) and not seen[positions].any()
                seen[positions]=True;values=frame.loc[keep,name]
                matrix[positions,desired.index(name)]=values.astype('string').map(known).fillna(1).where(values.notna(),0).to_numpy(np.float32)
            counts[desired.index(name)]+=int(seen.sum())
            assert int(seen.sum())==receipts[source_index]['rows']
        guard()
    assert np.all(counts==len(ids))
    matrix.flush()
    return matrix,vocab,receipts


def main(fold,version):
    out=ROOT/f'private_runs/tail240_20260916/validation/neural_context_{fold}_{version}_verifier_v3'
    out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/f'private_runs/tail240_20260916/state/neural_context/{version}/{fold}'
    manifest=v.read_json(folder/'manifest.json')
    protocol=v.read_json(folder.parent/'protocol.json')
    assert manifest['status']=='complete' and manifest['source_sha256']==v.sha256(producer.__file__)
    assert manifest['protocol_sha256']==v.sha256(folder.parent/'protocol.json')
    for filename,digest in manifest['outputs'].items():
        assert v.sha256(folder/filename)==digest
    print('PREFLIGHT_MEMORY',guard(),flush=True)
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    indices,split,_=common.fold_data(meta,fold,full=True)
    assert v.object_hash(split)==v.object_hash(manifest['split'])
    parts={stage:meta.iloc[indices[stage][np.isfinite(meta.iloc[indices[stage]].proxy_sec)]].copy() for stage in ['fit','tune']}
    fit,tune=parts['fit'],parts['tune']
    for stage,frame in parts.items():
        assert len(frame)==manifest[f'full_finite_{stage}_rows']
        assert v.object_hash(frame[ID].tolist())==manifest[f'{stage}_ids_hash']
        assert v.object_hash(frame[TARGET].tolist())==manifest[f'{stage}_label_hash']
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True))
    del meta,indices,parts
    gc.collect()
    print('COHORT_MEMORY',guard(),flush=True)
    desired=manifest['feature_columns']
    assert desired==protocol['feature_columns'] and len(desired)==387
    producer.risk.guard=guard
    matrix,vocab,receipts=bounded_load(ids,len(fit),desired,out)
    assert receipts==manifest['feature_receipts']
    model=joblib.load(folder/'fit_model.joblib')
    encoder=model['encoder']
    assert encoder.columns==desired
    residual=(fit[TARGET]-fit.proxy_sec).to_numpy(float)
    assert model['y_mean']==float(residual.mean())
    assert model['y_scale']==max(float(residual.std()),1.)
    bins=[]; embedded=[]
    for index,name in enumerate(encoder.numeric):
        values=np.asarray(matrix[:len(fit),desired.index(name)]).copy()
        values[(values==-999999)|~np.isfinite(values)]=np.nan
        median=np.float32(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.float32(0)
        filled=np.where(np.isfinite(values),values,median)
        mean=np.float32(filled.mean(dtype=np.float64))
        scale=filled.std(dtype=np.float64)
        scale=np.float32(scale if scale>1e-6 else 1.)
        assert median==encoder.medians[index],name
        assert mean==encoder.means[index],name
        assert scale==encoder.scales[index],name
        numbers=(filled-mean)/scale
        if np.any(numbers!=numbers[0]):
            embedded.append(index)
            bins.extend(producer.adapter.rtdl_num_embeddings.compute_bins(torch.from_numpy(numbers[:,None]),n_bins=48))
        if index%50==0:
            print('FIT_COLUMN_VERIFIED',index,name,guard(),flush=True)
    for name,mapping in encoder.categories.items():
        column=producer.decode_frame(matrix[:len(fit),[desired.index(name)]],{name:vocab[name]},[name])[name].astype('string')
        expected={value:i+2 for i,value in enumerate(sorted(column.dropna().unique()))}
        assert expected==mapping,name
    expected_passthrough=[i for i in range(len(encoder.numeric)*2) if i not in set(embedded)]
    np.testing.assert_array_equal(model['estimator'].embedded.numpy(),embedded)
    np.testing.assert_array_equal(model['estimator'].passthrough.numpy(),expected_passthrough)
    rebuilt=producer.adapter.rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
    left=rebuilt.impl.state_dict();right=model['estimator'].numeric_embeddings.impl.state_dict()
    assert list(left)==list(right)
    for key in left:
        assert torch.equal(left[key],right[key]),key
    del rebuilt,values,filled,numbers,column,residual,bins
    gc.collect();guard()
    xt=producer.decode_frame(matrix[len(fit):],vocab,desired)
    saved=pd.read_parquet(folder/'tune_predictions.parquet')
    np.testing.assert_array_equal(saved[ID],tune[ID])
    np.testing.assert_array_equal(saved[TARGET],tune[TARGET])
    np.testing.assert_array_equal(saved.proxy_sec,tune.proxy_sec)
    pred=producer.tabm_gpu.predict(model,xt)+tune.proxy_sec.to_numpy(float)
    candidate_delta=float(np.max(np.abs(pred-saved.prediction_sec.to_numpy())))
    assert candidate_delta<=1e-6
    control_folder=producer.control_folder(fold)
    control_manifest=v.read_json(control_folder/'manifest.json')
    for name in ['fit_model.joblib','tune_predictions.parquet']:
        assert v.sha256(control_folder/name)==control_manifest['outputs'][name]
    control_model=joblib.load(control_folder/'fit_model.joblib')
    control=producer.tabm_gpu.predict(control_model,xt[desired[:225]])+tune.proxy_sec.to_numpy(float)
    old=pd.read_parquet(control_folder/'tune_predictions.parquet').set_index(ID).loc[tune[ID],'prediction_sec'].to_numpy()
    control_delta=float(np.max(np.abs(control-old)))
    assert control_delta<=1e-6
    np.testing.assert_array_equal(old,saved.control225_prediction_sec)
    torch.cuda.empty_cache()
    print('GPU_REPLAY_COMPLETE',fold,candidate_delta,control_delta,flush=True)
    y=tune[TARGET].to_numpy(float);dates=tune[common.MOVEMENT].dt.floor('D').to_numpy()
    ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    comparisons={'all_finite':(y,pred,control,dates),'ordinary_proxy':(y[ordinary],pred[ordinary],control[ordinary],dates[ordinary])}
    ensemble=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    preparation=v.read_json(ensemble/'preparation.json')['folds'][fold]
    aligned_path=ensemble/(fold+'_aligned_tune.parquet')
    assert v.sha256(aligned_path)==preparation['aligned_tune_sha256']
    weights=v.read_json(ensemble/(fold+'_weights.json'))
    assert weights==preparation['weights']
    aligned=pd.read_parquet(aligned_path).set_index(ID)
    assert set(aligned.index)==set(tune.loc[ordinary,ID])
    own=saved.set_index(ID).loc[aligned.index]
    np.testing.assert_array_equal(own[TARGET],aligned[TARGET])
    baseline=aligned[weights['experts']].to_numpy()@np.array(weights['global'])
    replacement=baseline+weights['global'][weights['experts'].index('tabm_ple8')]*(own.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    blend=.25*own.prediction_sec.to_numpy()+.75*baseline
    ensemble_dates=aligned[common.MOVEMENT].dt.floor('D').to_numpy()
    comparisons['global9_replacement']=(aligned[TARGET].to_numpy(),replacement,baseline,ensemble_dates)
    comparisons['global9_blend25']=(aligned[TARGET].to_numpy(),blend,baseline,ensemble_dates)
    recorded=v.read_json(folder/'metrics.json')
    reports={}
    for name,(target,candidate,base,days) in comparisons.items():
        actual=v.metrics(target,candidate); reference=v.metrics(target,base)
        assert abs(actual['rmse']-recorded[name]['candidate_rmse'])<1e-10
        assert abs(reference['rmse']-recorded[name]['control_rmse'])<1e-10
        reports[name]={'candidate':actual,'control':reference,'paired':v.paired(target,base,candidate,days),
            'gain':reference['rmse']-actual['rmse']}
    evidence=v.read_json(folder/'fit_evidence.json')
    best=min(evidence['history'],key=lambda row:row['tune_mse_sec2'])
    assert model['steps']==best['epoch']==manifest['selected_epochs']
    assert abs(best['tune_mse_sec2']-reports['all_finite']['candidate']['mse'])<1e-7
    v.write_json(out/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),
        'producer_manifest_sha256':v.sha256(folder/'manifest.json'),'fold':fold,'fit_rows':len(fit),'tune_rows':len(tune),
        'candidate_native_max_abs_delta_sec':candidate_delta,'control_native_max_abs_delta_sec':control_delta,
        'fit_only_numeric_stats_exact':len(encoder.numeric),'fit_only_category_mappings_exact':len(encoder.categories),
        'fit_only_bin_impl_state_exact':True,'embedded_numeric_count':len(embedded),'selected_epochs':model['steps'],
        'metrics':reports,'peak_rss_bytes':guard(),'global9_caveat':'Frozen ensemble weights were fitted on same tune labels; this remains diagnostic.'})
    print('COMPLETE',fold,{k:r['gain'] for k,r in reports.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3']);parser.add_argument('--version',default='v1')
    args=parser.parse_args();pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(2)
    with threadpool_limits(2):main(args.fold,args.version)
