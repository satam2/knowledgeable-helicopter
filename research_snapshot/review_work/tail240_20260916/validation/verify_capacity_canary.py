"""CPU-only PLE32 canary provenance and verified full-fit preprocessing parity."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import torch
import lightgbm
import gc
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT=v.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/neural_capacity'))
import run_capacity as subject
BASE=subject.OUT/'canary/F1'
OUT=ROOT/'private_runs/tail240_20260916/validation/neural_capacity_canary_F1'


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,m.peak_wset)
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    r=v.read_json(BASE/'receipt.json');protocol=v.read_json(subject.OUT/'protocol.json')
    assert r['status']=='passed' and r['optimizer_updates']==0 and not r['full_training_started']
    assert r['members']==32 and r['inference_rows']==8192 and r['backward_rows']==4096
    assert r['gpu_peak_allocated_bytes']<12*1024**3 and r['finite_outputs'] and r['finite_gradients']
    assert r['resources']['peak_wset_bytes']<12*1024**3 and r['resources']['available_bytes']>=8*1024**3
    assert r['protocol_sha256']==v.sha256(subject.OUT/'protocol.json')
    for path,digest in protocol['source_hashes'].items():assert v.sha256(ROOT/path)==digest
    assert r['preparation_sha256']==v.sha256(BASE/'preparation.joblib')
    assert r['control_receipt_sha256']==v.sha256(BASE/'control_replay.json')
    control=v.read_json(BASE/'control_replay.json')
    assert control['max_abs_delta_sec']==0 and control['rows']==180640
    preparation=joblib.load(BASE/'preparation.joblib');encoder=preparation['encoder']
    binding=protocol['controls']['F1'];oldroot=subject.CONTROLS['F1']
    prior=v.read_json(ROOT/'private_runs/tail240_20260916/validation/neural_context_F1_v1_verifier_v4/receipt.json')
    assert prior['status']=='passed' and prior['producer_manifest_sha256']==binding['manifest_sha256']
    assert prior['fit_only_numeric_stats_exact']==372 and prior['fit_only_category_mappings_exact']==15 and prior['fit_only_bin_impl_state_exact']
    for name,key in [('manifest.json','manifest_sha256'),('fit_model.joblib','model_sha256'),('tune_predictions.parquet','predictions_sha256')]:assert v.sha256(oldroot/name)==binding[key]
    assert control['model_sha256']==binding['model_sha256']
    old=joblib.load(oldroot/'fit_model.joblib')
    for name in ['numeric','columns','categories']:assert getattr(encoder,name)==getattr(old['encoder'],name)
    for name in ['medians','means','scales']:np.testing.assert_array_equal(getattr(encoder,name),getattr(old['encoder'],name))
    np.testing.assert_array_equal(preparation['embedded'],old['estimator'].embedded)
    np.testing.assert_array_equal(preparation['passthrough'],old['estimator'].passthrough)
    rebuilt=subject.adapter.ple.rtdl_num_embeddings.PiecewiseLinearEmbeddings(preparation['bins'],d_embedding=16,activation=False,version='B')
    left,right=rebuilt.impl.state_dict(),old['estimator'].numeric_embeddings.impl.state_dict()
    assert set(left)==set(right)
    for key in left:assert torch.equal(left[key],right[key]),key
    assert v.object_hash([b.tolist() for b in preparation['bins']])==r['bin_hash']
    network=subject.adapter.network(preparation)
    assert sum(p.numel() for p in network.parameters())==r['parameter_count']
    del old,network,rebuilt,left,right;gc.collect();guard()
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[v.ID,'FLIGHT_ID_mvt',v.common.MOVEMENT,v.TARGET,'proxy_sec'])
    idx,split,_=v.common.fold_data(meta,'F1',full=True)
    assert v.object_hash(split)==v.object_hash(r['split'])
    parts={s:meta.iloc[idx[s]].loc[lambda f:np.isfinite(f.proxy_sec)].copy() for s in ['fit','tune']}
    for stage,frame in parts.items():
        assert len(frame)==binding[stage+'_rows']==r[stage+'_rows']
        assert v.object_hash(frame[v.ID].tolist())==binding[stage+'_ids_hash']
        assert v.object_hash(frame[v.TARGET].tolist())==binding[stage+'_label_hash']
    assert r['fit_label_hash']==binding['fit_label_hash'] and r['fit_ids_hash']==binding['fit_ids_hash']
    fitids=pd.Index(parts['fit'][v.ID]);nfit=len(fitids);ntotal=sum(len(f) for f in parts.values())
    assert v.object_hash(fitids[:8192].tolist())==r['canary_ids_hash']
    oldmarker=v.read_json(oldroot/'manifest.json')
    assert r['feature_receipts']==oldmarker['feature_receipts']
    columns=protocol['feature_columns'];assert columns==encoder.columns and len(columns)==387
    del meta,idx,parts;gc.collect();guard()
    vocab={name:set() for name in encoder.categories}
    for receipt in r['feature_receipts']:
        assert v.sha256(receipt['path'])==receipt['sha256']
        cats=[name for name in receipt['columns'] if name in vocab]
        if not cats:continue
        for batch in pq.ParquetFile(receipt['path']).iter_batches(batch_size=8192,columns=[v.ID,*cats],use_threads=False):
            frame=batch.to_pandas();keep=fitids.get_indexer(frame[v.ID])>=0
            for name in cats:vocab[name].update(frame.loc[keep,name].dropna().astype(str).tolist())
        guard()
    vocab={name:sorted(values) for name,values in vocab.items()}
    path=BASE/'matrix.float32';assert path.stat().st_size==ntotal*len(columns)*4
    block=np.empty((8192,len(columns)),np.float32)
    with path.open('rb') as handle:
        for j in range(len(columns)):
            handle.seek(j*ntotal*4);block[:,j]=np.fromfile(handle,dtype=np.float32,count=8192)
    frame=subject.context.decode_frame(block,vocab,columns)
    values=subject.adapter.frozen.tensors(encoder,frame)
    assert v.object_hash([t.tolist() for t in values])==r['input_hash']
    result=dict(status='passed',source_sha256=v.sha256(__file__),producer_receipt_sha256=v.sha256(BASE/'receipt.json'),protocol_sha256=r['protocol_sha256'],all_source_cohort_label_and_preparation_hashes_verified=True,encoder_exact_against_independently_verified_PLE387=True,bins_exact_against_verified_PLE387=True,canary_input_hash_reconstructed=True,fit_rows=nfit,tune_rows=ntotal-nfit,numeric_stats=372,category_mappings=15,native_control_reported_delta=control['max_abs_delta_sec'],gpu_canary_reported_peak=r['gpu_peak_allocated_bytes'],gpu_canary_rerun=False,no_model_fit=True,peak_bytes=guard(),limitation='CPU preflight uses exact parity to previously independently reconstructed PLE387 encoder/bin state, matches all raw source bindings and reconstructs canary inputs from retained matrix using rebuilt fit vocabularies. No repeat GPUcanary or full native replay; full candidate audit follows training.')
    v.write_json(OUT/'receipt.json',result)
    print('VERIFIED',result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(1):main()
