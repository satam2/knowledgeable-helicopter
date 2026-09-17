"""Independently rebuild sources/vocabulary and replay a completed final tree."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import lightgbm
from preflight import ROOT, OUT, ID, TARGET, read, sha
from rebuild_final_frame import reconstruct,guard,BASE_CATS
import argparse
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from threadpoolctl import threadpool_limits


def verify_matrix(folder,columns,sources,fit_ids,rank_ids,categories):
    ids=pd.Index(pd.concat([fit_ids,rank_ids],ignore_index=True));total=len(ids)
    numeric=[name for name in columns if name not in categories]
    coverage={name:np.zeros((total+7)//8,dtype=np.uint8) for name in numeric}
    for source in sources:
        names=[name for name in source['columns'] if name in numeric]
        parquet=pq.ParquetFile(source['path'])
        for offset in range(0,len(names),32):
            chunk=names[offset:offset+32]
            matrix=np.memmap(folder/'matrix.float32',dtype='float32',mode='r',shape=(total,len(columns)),order='F')
            for batch in parquet.iter_batches(batch_size=8192,columns=[ID,*chunk],use_threads=False):
                frame=batch.to_pandas();positions=ids.get_indexer(frame[ID]);keep=positions>=0;positions=positions[keep]
                assert len(np.unique(positions))==len(positions)
                locations=positions//8;bits=(1<<(positions%8)).astype(np.uint8)
                for name in chunk:
                    assert not (coverage[name][locations]&bits).any();np.bitwise_or.at(coverage[name],locations,bits)
                    values=pd.to_numeric(frame.loc[keep,name]).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if source['fill']:values[np.isnan(values)]=-999999
                    np.testing.assert_array_equal(values,matrix[positions,columns.index(name)])
            del matrix
            gc.collect();guard()
    assert all(np.unpackbits(value,bitorder='little')[:total].all() for value in coverage.values())
    return total*len(numeric)


def main(name):
    output=OUT/f'{name}_native_receipt.json'
    assert not output.exists()
    code=ROOT/'review_work/tail240_20260916/final_ordinary/run.py'
    spec=importlib.util.spec_from_file_location('ordinary_validation',code)
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    folder=subject.OUT/name
    marker,protocol=read(folder/'manifest.json'),read(folder/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==sha(folder/'protocol.json')
    for relative,digest in protocol['sources'].items():assert sha(ROOT/relative)==digest
    for filename,digest in marker['outputs'].items():assert sha(folder/filename)==digest
    train,ranking=subject.data()
    assert len(train)==2062577 and len(ranking)==339551
    assert marker['training_rows']==len(train) and marker['ranking_rows']==len(ranking)
    assert marker['train_id_hash']==subject.common.object_hash(train[ID].tolist())
    assert marker['train_label_hash']==subject.common.object_hash(train[TARGET].tolist())
    assert marker['ranking_id_hash']==subject.common.object_hash(ranking[ID].tolist())
    assert protocol['final_steps']==int(np.median(protocol['fold_steps']))==marker['steps']
    frame,vocab,receipts=reconstruct(protocol['columns'],train[ID],ranking[ID])
    checked_cells=verify_matrix(folder,protocol['columns'],receipts,train[ID],ranking[ID],vocab)
    model=joblib.load(folder/'model.joblib')
    assert model['steps']==protocol['final_steps']
    mapping=model['encoder'].maps if name.startswith('lgb') else model['encoder'].categories
    assert set(mapping)==set(vocab)
    for column,known in vocab.items():
        # Non-base missing inputs are the frozen explicit string MISSING.
        desired={value:i+2 for i,value in enumerate(known)}
        assert mapping[column]==desired,(column,'fit vocabulary changed')
    expected=pd.read_parquet(folder/'ranking_predictions.parquet')
    np.testing.assert_array_equal(expected[ID],ranking[ID])
    predicted=np.empty(len(ranking))
    for start in range(0,len(ranking),8192):
        stop=min(start+8192,len(ranking))
        encoded=model['encoder'].transform(frame.iloc[start:stop])
        if name.startswith('lgb'):
            raw=model['estimator'].predict(encoded,num_iteration=model['steps'],num_threads=1)
        else:
            raw=model['estimator'].predict(encoded,ntree_end=model['steps'],thread_count=1)
        predicted[start:stop]=raw+ranking.proxy_sec.iloc[start:stop].to_numpy(float)
        guard()
    np.testing.assert_array_equal(predicted,expected.prediction_sec)
    receipt=dict(status='passed',name=name,manifest_sha256=sha(folder/'manifest.json'),protocol_sha256=sha(folder/'protocol.json'),
        source_sha256=sha(__file__),reconstruction_source_sha256=sha(ROOT/'review_work/tail240_20260916/validation/final_submission_best/rebuild_final_frame.py'),
        training_rows=len(train),ranking_rows=len(ranking),features=len(frame.columns),fit_category_vocabularies=len(vocab),
        fit_vocabularies_exact=True,ranking_sources_independently_rebuilt=True,native_max_abs_delta_sec=0.,
        all_training_ranking_numeric_source_cells_exact=checked_cells,
        steps=model['steps'],prediction_sha256=sha(folder/'ranking_predictions.parquet'),
        peak_bytes=guard(),gpu_used=False,fitting_used=False,source_bindings=receipts)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_bindings'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['lgb225','lgb449','lgb387','cat225'],required=True)
    args=parser.parse_args();pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(1):main(args.name)
