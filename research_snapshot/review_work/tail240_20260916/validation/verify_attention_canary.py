"""CPU-only provenance and fullfit preprocessing check of the GPU canary."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import torch
import lightgbm
import gc
import sys
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT=v.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/neural_attention'))
import run_attention as subject
OUT=ROOT/'private_runs/tail240_20260916/validation/neural_attention_canary_F1'
BASE=subject.OUT.parent/'canary_v1/F1'


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    r=v.read_json(BASE/'receipt.json')
    assert r['status']=='passed' and r['optimizer_updates']==0 and not r['full_training_started']
    assert r['gpu_peak_bytes']<8*1024**3 and r['zero_head_static_exact'] and r['finite_gradients']
    assert r['protocol_sha256']==v.sha256(BASE.parent/'protocol.json')
    assert r['producer_protocol_sha256']==v.sha256(subject.OUT/'protocol.json')
    protocol=v.read_json(subject.OUT/'protocol.json')
    for path,digest in protocol['source_hashes'].items():assert v.sha256(ROOT/path)==digest
    assert r['preparation_sha256']==v.sha256(BASE/'fit_preparation.joblib')
    assert r['control_receipt_sha256']==v.sha256(BASE/'control_replay.json')
    control=v.read_json(BASE/'control_replay.json')
    assert control['max_abs_delta_sec']==0 and control['before_candidate_fit']
    payload=joblib.load(BASE/'fit_preparation.joblib');encoder=payload['encoder']
    binding=protocol['controls']['F1']
    oldpath=subject.CONTROLS['F1']/'fit_model.joblib'
    assert v.sha256(oldpath)==binding['fit_model_sha256']==control['model_sha256']
    old=joblib.load(oldpath)
    for name in ['numeric','columns','categories']:assert getattr(encoder,name)==getattr(old['encoder'],name)
    for name in ['medians','means','scales']:np.testing.assert_array_equal(getattr(encoder,name),getattr(old['encoder'],name))
    np.testing.assert_array_equal(payload['embedded'],old['estimator'].embedded)
    np.testing.assert_array_equal(payload['passthrough'],old['estimator'].passthrough)
    rebuilt=subject.adapter.ple.rtdl_num_embeddings.PiecewiseLinearEmbeddings(payload['bins'],d_embedding=16,activation=False,version='B')
    left,right=rebuilt.impl.state_dict(),old['estimator'].numeric_embeddings.impl.state_dict()
    assert set(left)==set(right)
    for key in left:assert torch.equal(left[key],right[key]),key
    del old,rebuilt,left,right;gc.collect();guard()
    assert v.object_hash([b.tolist() for b in payload['bins']])==r['bin_hash']
    assert v.object_hash(dict(means=encoder.means.tolist(),scales=encoder.scales.tolist(),medians=encoder.medians.tolist(),categories=encoder.categories,numeric=encoder.numeric))==r['encoder_hash']
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(metadata)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata,columns=[v.ID,'FLIGHT_ID_mvt',v.common.MOVEMENT,'proxy_sec']);meta[v.TARGET]=np.nan
    idx,_,_=v.common.fold_data(meta,'F1',full=True)
    rows={s:meta.iloc[idx[s]].loc[lambda x:np.isfinite(x.proxy_sec)] for s in ['fit','tune']}
    for stage,frame in rows.items():
        assert len(frame)==binding[stage+'_rows'] and v.object_hash(frame[v.ID].tolist())==binding[stage+'_ids_hash']
    nfit,ntotal=len(rows['fit']),sum(len(f) for f in rows.values())
    assert nfit==r['fit_rows'] and r['inference_rows']==8192
    assert v.object_hash(rows['fit'][v.ID].tolist())==r['fit_ids_hash']
    assert v.object_hash(rows['fit'][v.ID].iloc[:8192].tolist())==r['canary_ids_hash']
    columns=protocol['feature_columns'];assert columns==r['columns']==encoder.columns
    path=BASE/'matrix.float32';assert path.stat().st_size==ntotal*len(columns)*4
    required=subject.adapter.OWN_CLOCKS+[p+f for p in subject.adapter.PREFIXES for f in subject.adapter.FIELDS]
    counts=np.zeros(24,np.int64);sums=np.zeros(24);squares=np.zeros(24)
    def read_block(handle,names,start,count):
        a=np.empty((count,len(names)),np.float64)
        for j,name in enumerate(names):
            handle.seek((columns.index(name)*ntotal+start)*4)
            a[:,j]=np.fromfile(handle,dtype=np.float32,count=count)
        a[(a==-999999)|~np.isfinite(a)]=np.nan
        return a
    with path.open('rb') as handle:
        for start in range(0,nfit,65536):
            n=min(65536,nfit-start)
            allraw=read_block(handle,required,start,n)
            own=allraw[:,:5]
            for token in range(8):
                raw=allraw[:,5+token*14:5+(token+1)*14]
                present=raw[:,13]==0
                duration=own-raw[:,:5]
                stamp=duration-raw[:,8:9]
                valid=present[:,None]&np.isfinite(own)&np.isfinite(raw[:,:5])&np.isfinite(raw[:,8:9])
                if token>=4:valid[:]=False
                duration=np.where(valid,duration,np.nan);stamp=np.where(valid,stamp,np.nan)
                values=np.concatenate([raw,duration,stamp],axis=1)
                exists=np.isfinite(values)&present[:,None]
                values=np.where(exists,values,0.)
                counts+=exists.sum(axis=0);sums+=values.sum(axis=0);squares+=(values**2).sum(axis=0)
            del allraw,own,raw,values,duration,stamp,valid,exists
            guard()
    means=sums/np.maximum(counts,1);variance=np.maximum(squares/np.maximum(counts,1)-means**2,0.)
    scales=np.where(variance>1e-12,np.sqrt(variance),1.)
    stats=payload['stats'];np.testing.assert_array_equal(counts,stats['count'])
    np.testing.assert_array_equal(means.astype('float32'),stats['means'])
    np.testing.assert_array_equal(scales.astype('float32'),stats['scales'])
    assert v.object_hash(stats)==r['stats_hash']
    result=dict(status='passed',source_sha256=v.sha256(__file__),producer_receipt_sha256=v.sha256(BASE/'receipt.json'),all_source_and_preparation_hashes_verified=True,encoder_exact_against_previously_independently_verified_PLE387=True,bins_exact_against_verified_PLE387=True,fullfit_shared_token_statistics_independently_exact=True,fit_rows=nfit,tune_rows=len(rows['tune']),native_control_reported_delta=control['max_abs_delta_sec'],gpu_canary_reported_peak=r['gpu_peak_bytes'],gpu_canary_rerun=False,no_model_fit=True,peak_bytes=guard(),limitation='CPU preflight reuses independently verified PLE387 encoder/bin geometry and reconstructs token stats from retained canary matrix; does not rerun GPUcanary or reload every raw source into a new matrix. Full candidate native verification follows completed training.')
    v.write_json(OUT/'receipt.json',result)
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):main()
