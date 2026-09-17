"""Independent flatten oracle; reuse upstream strict availability proof."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten'
CACHE=ROOT/'private_runs/breakthrough_20260916/sequence_context'
ID=common.ID
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def same(a,b):
    bad={'<missing>','MISSING','m:'}
    return float(pd.notna(a) and pd.notna(b) and str(a) not in bad and str(b) not in bad and a==b)


def main():
    assert not (OUT/'verification.json').exists()
    manifest=common.read_json(OUT/'manifest.json')
    source=common.read_json(CACHE/'manifest.json')
    assert manifest['status']=='complete'
    assert common.sha256(CACHE/'manifest.json')==manifest['source_cache_manifest_sha256']
    assert common.sha256(OUT/'training_features.parquet')==manifest['outputs']['training_features.parquet']
    assert common.sha256(Path(__file__).with_name('build.py'))==manifest['source_sha256']
    upstream=ROOT/'private_runs/breakthrough_20260916/missing/sequence_independent_audit/verification.json'
    assert common.sha256(upstream)==manifest['availability_receipt_sha256']
    nums=source['event_numeric']
    fields=nums+['age_sec','landing_age_sec','same_runway','same_stand','same_operator','padding_missing']
    expected_columns=[f'flat_{phase}{rank}_{field}' for phase in ['dep','arr'] for rank in range(1,5) for field in fields]
    assert expected_columns==manifest['features']
    parquet=pq.ParquetFile(OUT/'training_features.parquet')
    assert parquet.schema_arrow.names==[ID,*expected_columns]
    ids=pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[ID],use_threads=False).column(0).to_numpy()
    neighbors=np.load(CACHE/'neighbors.npy',mmap_mode='r')
    rowgroup=0
    checks=[]
    for i,part in enumerate(source['records']):
        events=pd.read_parquet(CACHE/part['events_file'],dtype_backend='pyarrow')
        queries=pd.read_parquet(CACHE/part['queries_file'],dtype_backend='pyarrow')
        phase=events.phase.to_numpy()
        rng=np.random.default_rng(20260916+i)
        chosen=set(np.r_[np.arange(8),np.arange(len(queries)-8,len(queries)),rng.choice(len(queries),24,replace=False)].tolist())
        offset=part['query_offset']
        checked=0
        for start in range(0,len(queries),4096):
            stop=min(start+4096,len(queries))
            flat=parquet.read_row_group(rowgroup).to_pandas()
            rowgroup+=1
            assert len(flat)==stop-start
            np.testing.assert_array_equal(flat[ID],ids[offset+start:offset+stop])
            values=flat[expected_columns].to_numpy().reshape(len(flat),8,14)
            index=np.asarray(neighbors[offset+start:offset+stop])
            index=np.where(index>=0,index-part['event_offset'],-1)
            safe=np.maximum(index,0)
            for p,section in [('DEP',slice(0,4)),('ARR',slice(4,8))]:
                count=np.minimum(((index>=0)&(phase[safe]==p)).sum(axis=1),4)
                missing=np.arange(4)[None,:]>=count[:,None]
                np.testing.assert_array_equal(values[:,section,13],missing.astype('float32'))
                assert np.isnan(values[:,section,:10][missing]).all()
                assert np.all(values[:,section,10:13][missing]==0)
            for local in sorted(chosen.intersection(range(start,stop))):
                selected=index[local-start]
                selected=selected[selected>=0].tolist()
                query=queries.iloc[local]
                expected=[]
                for p in ['DEP','ARR']:
                    eligible=[j for j in reversed(selected) if events.iloc[j].phase==p][:4]
                    for k in range(4):
                        if k>=len(eligible):
                            expected.append([np.nan]*10+[0.,0.,0.,1.])
                        else:
                            event=events.iloc[eligible[k]]
                            numeric=[float(event[c]) if pd.notna(event[c]) else np.nan for c in nums]
                            age=(query.time_ns-event.time_ns)/1e9
                            landing=(query.time_ns-event.movement_ns)/1e9 if p=='ARR' else np.nan
                            equality=[same(event[c],query[c]) for c in ['runway','stand','operator']]
                            expected.append(numeric+[age,landing,*equality,0.])
                np.testing.assert_allclose(values[local-start],np.asarray(expected,dtype='float32'),rtol=0,atol=0,equal_nan=True)
                checked+=1
        checks.append(dict(file=part['file'],all_id_padding_rows=len(queries),oracle_queries=checked))
        print('FLATTEN_VERIFIED',part['file'],checked,flush=True)
    assert rowgroup==parquet.num_row_groups
    result=dict(status='passed',verifier_sha256=common.sha256(__file__),manifest_sha256=common.sha256(OUT/'manifest.json'),
        all_rows=sum(p['all_id_padding_rows'] for p in checks),oracle_queries=sum(p['oracle_queries'] for p in checks),
        features=len(expected_columns),partitions=checks,upstream_availability_reused=True,raw_data_read=False,
        exact_stored_values_and_known_only_equalities=True,all_padding_flags_values_verified=True)
    common.write_json(OUT/'verification.json',result)


if __name__=='__main__':
    main()
