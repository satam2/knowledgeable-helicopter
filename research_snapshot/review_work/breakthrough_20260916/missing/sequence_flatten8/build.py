"""Streaming last8-per-phase extension; original last4 producer remains frozen."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
import importlib.util
from pathlib import Path
import gc
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[4]
ORIGINAL=Path(__file__).resolve().parent.parent/'sequence_flatten/build.py'
spec=importlib.util.spec_from_file_location('frozen_flatten4',ORIGINAL)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
cache=old.cache
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten8'
COLUMNS=[f'flat_{phase}{rank}_{name}' for phase in ['dep','arr'] for rank in range(1,9) for name in old.FIELDS]
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


class Flatten8(old.Flatten):
    def indices(self,neighbors):
        valid=neighbors>=0
        safe=np.maximum(neighbors,0)
        selected=[]
        for phase in [True,False]:
            positions=np.where(valid&(self.dep[safe]==phase),np.arange(neighbors.shape[1]),-1)
            recent=np.sort(positions,axis=1)[:,-8:][:,::-1]
            found=np.take_along_axis(neighbors,np.maximum(recent,0),axis=1)
            selected.append(np.where(recent>=0,found,-1))
        return np.concatenate(selected,axis=1)

    def batch(self,neighbors,start):
        selected=self.indices(neighbors)
        mask=selected>=0
        safe=np.maximum(selected,0)
        age=(self.query_times[start:start+len(neighbors),None]-self.times[safe])/1e9
        landing=(self.query_times[start:start+len(neighbors),None]-self.movement[safe])/1e9
        landing[self.dep[safe]]=np.nan
        numeric=np.concatenate([self.numeric[safe],age[:,:,None],landing[:,:,None]],axis=2).astype('float32')
        numeric[~mask]=np.nan
        equality=np.stack([(self.event_eq[c][safe]==self.query_eq[c][start:start+len(neighbors),None])
            &(self.event_eq[c][safe]>=0)&(self.query_eq[c][start:start+len(neighbors),None]>=0)&mask
            for c in ['runway','stand','operator']],axis=2).astype('float32')
        return np.concatenate([numeric,equality,(~mask).astype('float32')[:,:,None]],axis=2).reshape(len(neighbors),len(COLUMNS))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    original=cache.json.loads((old.OUT/'manifest.json').read_text())
    assert original['status']=='complete' and original['source_sha256']==cache.sha(ORIGINAL)
    manifest=cache.json.loads((cache.OUT/'manifest.json').read_text())
    verified=json.loads((old.VERIFY/'verification.json').read_text())
    bounds=json.loads((old.VERIFY/'encoding_bounds.json').read_text())
    assert manifest['status']=='complete' and verified['status']==bounds['status']=='passed'
    digest=cache.sha(cache.OUT/'manifest.json')
    assert digest==verified['manifest_sha256']==bounds['cache_manifest_sha256']==original['source_cache_manifest_sha256']
    assert cache.sha(cache.OUT/'neighbors.npy')==manifest['neighbors_sha256']
    assert cache.sha(old.SOURCE/'cache.py')==manifest['source_sha256']
    protocol=dict(source_sha256=cache.sha(__file__),frozen_last4_source_sha256=cache.sha(ORIGINAL),
        original_last4_manifest_sha256=cache.sha(old.OUT/'manifest.json'),source_cache_manifest_sha256=digest,
        availability_receipt_sha256=cache.sha(old.VERIFY/'verification.json'),bounds_receipt_sha256=cache.sha(old.VERIFY/'encoding_bounds.json'),
        features=COLUMNS,rows=verified['all_query_rows'],selection='Fixedlast8DEP+last8completedARRfromverified32eventcache;nearestfirst;directextensionoflast4.',
        encoding='Samefrozenlast4rawnumerics/age/landingage/known-onlyequalities/padding;no learnedcategories.',
        availability=manifest['policy'],resources='2CPU,4096rowsperbatch,oneeventpartition,<2GiBpeakRSS',
        motivation='Adaptive single history-depthfollow-up afterrobustF1last4gain; nohypergrid,noadditionaltrees,no labelreads.')
    cache.write_json(OUT/'protocol.json',protocol)
    neighbors=np.load(cache.OUT/'neighbors.npy',mmap_mode='r')
    writer=None
    records=[]
    output=OUT/'training_features.parquet'
    try:
        for record in manifest['records']:
            old.guard()
            for key in ['events','queries']:
                assert cache.sha(cache.OUT/record[key+'_file'])==record[key+'_sha256']
            events=pd.read_parquet(cache.OUT/record['events_file'],dtype_backend='pyarrow')
            queries=pd.read_parquet(cache.OUT/record['queries_file'],dtype_backend='pyarrow')
            engine=Flatten8(events,queries)
            for start in range(0,len(queries),4096):
                end=min(start+4096,len(queries))
                idx=np.asarray(neighbors[record['query_offset']+start:record['query_offset']+end])
                idx=np.where(idx>=0,idx-record['event_offset'],-1)
                assert np.all((idx==-1)|((idx>=0)&(idx<len(events))))
                frame=pd.DataFrame(engine.batch(idx,start),columns=COLUMNS)
                frame.insert(0,cache.ID,queries[cache.ID].iloc[start:end].to_numpy())
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None:
                    writer=pq.ParquetWriter(output,table.schema,compression='zstd')
                writer.write_table(table)
                old.guard()
            records.append(dict(file=record['file'],rows=len(queries),peak_rss_bytes=old.guard()))
            print('FLATTEN8',record['file'],len(queries),'peak',old.guard(),flush=True)
            del events,queries,engine,frame,table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    result=dict(status='complete',source_sha256=cache.sha(__file__),frozen_last4_source_sha256=cache.sha(ORIGINAL),
        protocol_sha256=cache.sha(OUT/'protocol.json'),source_cache_manifest_sha256=digest,
        original_last4_manifest_sha256=protocol['original_last4_manifest_sha256'],
        availability_receipt_sha256=protocol['availability_receipt_sha256'],features=COLUMNS,rows=sum(r['rows'] for r in records),
        outputs={output.name:cache.sha(output)},runtime_sec=time.monotonic()-began,peak_rss_bytes=old.guard(),records=records)
    cache.write_json(OUT/'manifest.json',result)
    print('COMPLETE_FLATTEN8',result['rows'],len(COLUMNS),result['runtime_sec'],result['peak_rss_bytes'],flush=True)


if __name__=='__main__':
    main()
