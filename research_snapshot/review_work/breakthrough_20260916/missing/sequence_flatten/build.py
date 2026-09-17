"""Streaming fixed event-token flattening from the independently verified cache."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='2'
from pathlib import Path
import gc
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import sys
ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'review_work/breakthrough_20260916/sequence_context'
sys.path.insert(0,str(SOURCE))
import cache
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten'
VERIFY=ROOT/'private_runs/breakthrough_20260916/missing/sequence_independent_audit'
NUMERIC=[*cache.NUMS,'age_sec','landing_age_sec']
FIELDS=[*NUMERIC,'same_runway','same_stand','same_operator','padding_missing']
COLUMNS=[f'flat_{phase}{rank}_{name}' for phase in ['dep','arr'] for rank in range(1,5) for name in FIELDS]
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def normalized(values):
    return values.astype('string').replace({'<missing>':pd.NA,'MISSING':pd.NA,'m:':pd.NA})


class Flatten:
    def __init__(self,events,queries):
        self.numeric=events[cache.NUMS].to_numpy('float32')
        self.times=events.time_ns.to_numpy('int64')
        self.movement=events.movement_ns.to_numpy('int64')
        self.dep=events.phase.eq('DEP').to_numpy()
        self.query_times=queries.time_ns.to_numpy('int64')
        self.event_eq={}
        self.query_eq={}
        for c in ['runway','stand','operator']:
            codes,vocab=pd.factorize(normalized(events[c]),sort=False)
            self.event_eq[c]=codes.astype('int32')
            self.query_eq[c]=pd.Index(vocab).get_indexer(normalized(queries[c])).astype('int32')

    def indices(self,neighbors):
        safe=np.maximum(neighbors,0)
        valid=neighbors>=0
        selected=[]
        for phase in [True,False]:
            positions=np.where(valid&(self.dep[safe]==phase),np.arange(neighbors.shape[1]),-1)
            latest=np.sort(positions,axis=1)[:,-4:][:,::-1]
            found=np.take_along_axis(neighbors,np.maximum(latest,0),axis=1)
            selected.append(np.where(latest>=0,found,-1))
        return np.concatenate(selected,axis=1)

    def batch(self,neighbors,start):
        chosen=self.indices(neighbors)
        mask=chosen>=0
        safe=np.maximum(chosen,0)
        base=self.numeric[safe]
        age=(self.query_times[start:start+len(neighbors),None]-self.times[safe])/1e9
        landing=(self.query_times[start:start+len(neighbors),None]-self.movement[safe])/1e9
        landing[self.dep[safe]]=np.nan
        numeric=np.concatenate([base,age[:,:,None],landing[:,:,None]],axis=2).astype('float32')
        numeric[~mask]=np.nan
        equality=np.stack([(self.event_eq[c][safe]==self.query_eq[c][start:start+len(neighbors),None])
            &(self.event_eq[c][safe]>=0)&(self.query_eq[c][start:start+len(neighbors),None]>=0)&mask
            for c in ['runway','stand','operator']],axis=2).astype('float32')
        output=np.concatenate([numeric,equality,(~mask).astype('float32')[:,:,None]],axis=2)
        return output.reshape(len(neighbors),len(COLUMNS))


def guard():
    process=psutil.Process()
    peak=getattr(process.memory_info(),'peak_wset',process.memory_info().rss)
    if peak>=2*1024**3:
        raise MemoryError('Flatten builder exceeded2GiB RSS cap')
    return peak


def main():
    if OUT.exists():
        raise ValueError('Preserve completed or failed flatten attempt')
    OUT.mkdir(parents=True)
    began=time.monotonic()
    manifest=cache.json.loads((cache.OUT/'manifest.json').read_text())
    verification=json.loads((VERIFY/'verification.json').read_text())
    bounds=json.loads((VERIFY/'encoding_bounds.json').read_text())
    assert manifest['status']=='complete' and verification['status']=='passed' and bounds['status']=='passed'
    digest=cache.sha(cache.OUT/'manifest.json')
    assert verification['manifest_sha256']==bounds['cache_manifest_sha256']==digest
    assert cache.sha(SOURCE/'cache.py')==manifest['source_sha256']
    assert cache.sha(cache.OUT/'neighbors.npy')==manifest['neighbors_sha256']
    protocol=dict(source_sha256=cache.sha(__file__),source_cache_manifest_sha256=digest,
        availability_receipt_sha256=cache.sha(VERIFY/'verification.json'),bounds_receipt_sha256=cache.sha(VERIFY/'encoding_bounds.json'),
        source_policy=manifest['policy'],features=COLUMNS,shape=[verification['all_query_rows'],len(COLUMNS)],
        extraction='Last4DEP andlast4completedARR fromverified32neighbors, nearestfirst withinphase; nootherselection.',
        encoding='Rawfloat32; numericmissingNaN; paddingflag1 onlynoevent; samecategory1 onlybothknownandexact; nolearnedfrequency/vocabulary.',
        leakage='No raw/label reads; retains originalretrospective finalNMpublication-time limitation.',
        resource='4096queryblocks,oneeventpartition,2CPUthreads,2GiBRSS cap')
    cache.write_json(OUT/'protocol.json',protocol)
    output=OUT/'training_features.parquet'
    neighbors=np.load(cache.OUT/'neighbors.npy',mmap_mode='r')
    writer=None
    checks=[]
    try:
        for record in manifest['records']:
            guard()
            for field in ['events','queries']:
                assert cache.sha(cache.OUT/record[field+'_file'])==record[field+'_sha256']
            events=pd.read_parquet(cache.OUT/record['events_file'],dtype_backend='pyarrow')
            queries=pd.read_parquet(cache.OUT/record['queries_file'],dtype_backend='pyarrow')
            engine=Flatten(events,queries)
            qoffset=record['query_offset']
            for start in range(0,len(queries),4096):
                stop=min(start+4096,len(queries))
                global_indices=np.asarray(neighbors[qoffset+start:qoffset+stop])
                indices=np.where(global_indices>=0,global_indices-record['event_offset'],-1)
                assert np.all((indices==-1)|((indices>=0)&(indices<len(events))))
                flat=engine.batch(indices,start)
                frame=pd.DataFrame(flat,columns=COLUMNS)
                frame.insert(0,cache.ID,queries[cache.ID].iloc[start:stop].to_numpy())
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None:
                    writer=pq.ParquetWriter(output,table.schema,compression='zstd')
                writer.write_table(table)
                guard()
            checks.append(dict(file=record['file'],rows=len(queries),peak_rss_bytes=guard()))
            print('FLATTEN',record['file'],len(queries),'peak',guard(),flush=True)
            del engine,events,queries,flat,frame,table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    result=dict(status='complete',protocol_sha256=cache.sha(OUT/'protocol.json'),source_sha256=cache.sha(__file__),
        source_cache_manifest_sha256=digest,availability_receipt_sha256=protocol['availability_receipt_sha256'],
        outputs={output.name:cache.sha(output)},features=COLUMNS,rows=sum(c['rows'] for c in checks),
        runtime_sec=time.monotonic()-began,peak_rss_bytes=guard(),partitions=checks)
    cache.write_json(OUT/'manifest.json',result)
    print('COMPLETE_FLATTEN',result['rows'],len(COLUMNS),result['peak_rss_bytes'],flush=True)


if __name__=='__main__':
    main()
