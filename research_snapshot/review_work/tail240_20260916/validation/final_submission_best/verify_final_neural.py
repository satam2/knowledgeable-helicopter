"""Separated CPU preprocessing and GPU native replay of frozen final neural fits."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import torch
import lightgbm
from preflight import ROOT, OUT, ID, TARGET, read, sha
from rebuild_final_frame import reconstruct,guard
import argparse
import gc
import json
import joblib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import tabm_gpu
import tabm_ple_gpu
import common


def folder_for(name):
    release=read(ROOT/'private_runs/tail240_20260916/final_submission_v3_v2/protocol.json')
    return Path(release['component_directories']['ple']) if name=='ple387' else Path(release['component_directories']['ordinary'])/'tabm225'


def inputs(folder,name):
    marker=read(folder/'manifest.json')
    protocol_path=folder.parent/'protocol.json' if name=='ple387' else folder/'protocol.json'
    protocol=read(protocol_path)
    assert marker['status']=='complete' and marker['protocol_sha256']==sha(protocol_path)
    for filename,digest in marker['outputs'].items():assert sha(folder/filename)==digest
    for relative,digest in protocol['sources'].items():assert sha(ROOT/relative)==digest
    source=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(source)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][source.name]
    train=pd.read_parquet(source,columns=[ID,TARGET,'proxy_sec'])
    train=train.loc[np.isfinite(train.proxy_sec)].copy()
    rankpath=ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    assert sha(rankpath)==read(rankpath.parent/'ranking_inputs.json')['files'][rankpath.name]
    ranking=pd.read_parquet(rankpath,columns=[ID,'proxy_sec']);ranking=ranking.loc[np.isfinite(ranking.proxy_sec)].copy()
    assert len(train)==2062577 and len(ranking)==339551
    return marker,protocol,train,ranking


def cpu(name):
    dest=OUT/f'{name}_cpu';dest.mkdir(exist_ok=False)
    folder=folder_for(name);marker,protocol,train,ranking=inputs(folder,name)
    columns=protocol['columns'];nfit=len(train);total=nfit+len(ranking)
    frame,vocab,sources=reconstruct(columns,train[ID],ranking[ID])
    model=joblib.load(folder/'model.joblib');encoder=model['encoder']
    assert encoder.columns==columns and encoder.neural
    for column,known in vocab.items():assert encoder.categories[column]=={value:i+2 for i,value in enumerate(known)}
    assert encoder.numeric==[column for column in columns if column not in vocab]
    ids=pd.Index(pd.concat([train[ID],ranking[ID]],ignore_index=True));assert ids.is_unique
    matrix_path=folder/'matrix.float32'
    coverage={column:np.zeros((total+7)//8,dtype=np.uint8) for column in encoder.numeric}
    # Remap each source separately so resident matrix pages remain bounded.
    for source in sources:
        names=[column for column in source['columns'] if column in encoder.numeric]
        if not names:continue
        matrix=np.memmap(matrix_path,dtype='float32',mode='r',shape=(total,len(columns)),order='F')
        for batch in pq.ParquetFile(source['path']).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            part=batch.to_pandas();positions=ids.get_indexer(part[ID]);keep=positions>=0;positions=positions[keep]
            assert len(np.unique(positions))==len(positions)
            bytes_index=positions//8
            bits=(1 << (positions%8)).astype(np.uint8)
            for column in names:
                assert not (coverage[column][bytes_index]&bits).any()
                np.bitwise_or.at(coverage[column],bytes_index,bits)
                values=pd.to_numeric(part.loc[keep,column]).to_numpy(dtype='float32',na_value=np.nan)
                values[~np.isfinite(values)]=np.nan
                if source['fill']:values[np.isnan(values)]=-999999
                np.testing.assert_array_equal(matrix[positions,columns.index(column)],values)
        del matrix,part
        gc.collect();guard()
    assert all(np.unpackbits(value,bitorder='little')[:total].all() for value in coverage.values())
    del coverage
    embedded=[];bins=[]
    for j,column in enumerate(encoder.numeric):
        matrix=np.memmap(matrix_path,dtype='float32',mode='r',shape=(total,len(columns)),order='F')
        values=np.array(matrix[:nfit,columns.index(column)],copy=True);del matrix
        values[(values==-999999)|~np.isfinite(values)]=np.nan
        median=np.float32(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.float32(0)
        filled=np.where(np.isfinite(values),values,median)
        mean=np.float32(filled.mean(dtype=np.float64));scale=filled.std(dtype=np.float64);scale=np.float32(scale if scale>1e-6 else 1.)
        assert median==encoder.medians[j] and mean==encoder.means[j] and scale==encoder.scales[j],column
        if name=='ple387':
            normalized=(filled-mean)/scale
            if np.any(normalized!=normalized[0]):
                embedded.append(j)
                bins.extend(tabm_ple_gpu.rtdl_num_embeddings.compute_bins(torch.from_numpy(normalized[:,None]),n_bins=48))
        if j%50==0:print('STATS',name,j,guard(),flush=True)
    if name=='ple387':
        np.testing.assert_array_equal(model['estimator'].embedded.numpy(),embedded)
        np.testing.assert_array_equal(model['estimator'].passthrough.numpy(),[j for j in range(2*len(encoder.numeric)) if j not in set(embedded)])
        rebuilt=tabm_ple_gpu.rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
        actual=model['estimator'].numeric_embeddings.impl.state_dict()
        for key,value in rebuilt.impl.state_dict().items():assert torch.equal(value,actual[key]),key
    residual=(train[TARGET]-train.proxy_sec).to_numpy(float)
    assert model['y_mean']==float(residual.mean()) and model['y_scale']==max(float(residual.std()),1.)
    evidence=read(folder/'fit_evidence.json')
    steps=protocol['final_epochs'] if name=='ple387' else protocol['final_steps']
    assert model['steps']==steps==len(evidence['history']) and evidence['rows']==nfit
    assert all('tune_mse_sec2' not in row for row in evidence['history'])
    frame.to_parquet(dest/'ranking_frame.parquet')
    receipt=dict(status='passed',name=name,manifest_sha256=sha(folder/'manifest.json'),protocol_sha256=marker['protocol_sha256'],
        verifier_sha256=sha(__file__),training_rows=nfit,ranking_rows=len(ranking),steps=steps,
        numeric_stats_exact=len(encoder.numeric),all_training_ranking_numeric_source_cells_exact=True,
        fit_categories_exact=len(vocab),official_bins_exact=name=='ple387',target_scale_exact=True,
        prediction_sha256=sha(folder/'ranking_predictions.parquet'),ranking_frame_sha256=sha(dest/'ranking_frame.parquet'),peak_bytes=guard(),gpu_used=False)
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


def gpu(name):
    dest=OUT/f'{name}_cpu';receipt=read(dest/'receipt.json');assert receipt['status']=='passed'
    folder=folder_for(name);marker,protocol,train,ranking=inputs(folder,name)
    assert sha(folder/'manifest.json')==receipt['manifest_sha256']
    assert sha(dest/'ranking_frame.parquet')==receipt['ranking_frame_sha256']
    frame=pd.read_parquet(dest/'ranking_frame.parquet');np.testing.assert_array_equal(frame.index,ranking[ID])
    model=joblib.load(folder/'model.joblib')
    prediction=tabm_gpu.predict(model,frame)+ranking.proxy_sec.to_numpy(float)
    saved=pd.read_parquet(folder/'ranking_predictions.parquet');np.testing.assert_array_equal(saved[ID],ranking[ID])
    np.testing.assert_array_equal(prediction,saved.prediction_sec)
    final=dict(receipt,cpu_receipt_sha256=sha(dest/'receipt.json'),gpu_used=True,native_max_abs_delta_sec=0.,
               cpu_threads=2,native_batch_size=8192,gpu_peak_bytes=torch.cuda.max_memory_allocated(),peak_bytes=guard())
    target=OUT/f'{name}_native_receipt.json';assert not target.exists()
    target.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['ple387','tabm225'],required=True);parser.add_argument('--stage',choices=['cpu','gpu'],required=True)
    args=parser.parse_args();torch.set_num_threads(1 if args.stage=='cpu' else 2);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(1 if args.stage=='cpu' else 2):(cpu if args.stage=='cpu' else gpu)(args.name)
