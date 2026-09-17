"""All-row last4 identity/padding plus independent raw-field token reconstruction."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import gc
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TIME=common.ID,common.MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten8'
OLD=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten'
CACHE=ROOT/'private_runs/breakthrough_20260916/sequence_context'
RAW=ROOT/'data/09-15-2026-18-55-03_files_list'
FIELDS=[ID,'FLIGHT_ID_mvt','PHASE_mvt',TIME,'ADEP_mvt','ADES_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt']
CLOCKS=['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def eq(a,b):
    missing={'<missing>','MISSING','m:'}
    return float(pd.notna(a) and pd.notna(b) and str(a) not in missing and str(b) not in missing and a==b)


def peak():
    info=psutil.Process().memory_info()
    value=getattr(info,'peak_wset',info.rss)
    if value>=2*1024**3:
        raise MemoryError('Verifier exceeded2GiBRSS')
    return value


def main():
    assert not (OUT/'verification.json').exists()
    manifest=common.read_json(OUT/'manifest.json')
    prior=common.read_json(OLD/'manifest.json')
    source=common.read_json(CACHE/'manifest.json')
    assert manifest['status']==prior['status']==source['status']=='complete'
    assert common.sha256(OUT/'training_features.parquet')==manifest['outputs']['training_features.parquet']
    assert common.sha256(OLD/'training_features.parquet')==prior['outputs']['training_features.parquet']
    assert common.sha256(CACHE/'manifest.json')==manifest['source_cache_manifest_sha256']==prior['source_cache_manifest_sha256']
    assert common.sha256(OLD/'manifest.json')==manifest['original_last4_manifest_sha256']
    assert common.sha256(Path(__file__).with_name('build.py'))==manifest['source_sha256']
    p8=pq.ParquetFile(OUT/'training_features.parquet')
    p4=pq.ParquetFile(OLD/'training_features.parquet')
    assert p8.num_row_groups==p4.num_row_groups
    ids=pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[ID],use_threads=False).column(0).to_numpy()
    neighbors=np.load(CACHE/'neighbors.npy',mmap_mode='r')
    assert common.sha256(CACHE/'neighbors.npy')==source['neighbors_sha256']
    rowgroup=0
    rawfiles=sorted(RAW.glob('training_*.parquet'))
    checks=[]
    for partindex,part in enumerate(source['records']):
        events=pd.read_parquet(CACHE/part['events_file'],columns=[ID,'phase'])
        queries=pd.read_parquet(CACHE/part['queries_file'],columns=[ID])
        rng=np.random.default_rng(20260916+partindex)
        sample=np.unique(np.r_[np.arange(8),np.arange(len(queries)-8,len(queries)),rng.choice(len(queries),24,replace=False)])
        offset=part['query_offset']
        sample_indices=np.asarray(neighbors[offset+sample])
        sample_indices=np.where(sample_indices>=0,sample_indices-part['event_offset'],-1)
        selected_ids=set(queries.iloc[sample][ID].tolist())
        selected_ids.update(events.iloc[np.unique(sample_indices[sample_indices>=0])][ID].tolist())
        public=[]
        arrivals=[]
        for path in rawfiles[max(0,partindex-1):min(len(rawfiles),partindex+2)]:
            assert common.sha256(path)==source['raw_hashes'][path.name]
            public.append(pq.read_table(path,columns=FIELDS+CLOCKS,filters=[(ID,'in',list(selected_ids))],use_threads=False).to_pandas())
            arrivals.append(pq.read_table(path,columns=[ID,'PHASE_mvt','BLOCK_TIME_UTC_mvt'],filters=[('PHASE_mvt','=','ARR'),(ID,'in',list(selected_ids))],use_threads=False).to_pandas())
        raw=pd.concat(public).drop_duplicates(ID).set_index(ID)
        completion=pd.concat(arrivals).drop_duplicates(ID).set_index(ID)
        assert completion.PHASE_mvt.eq('ARR').all()
        checked=0
        phase=events.phase.to_numpy()
        for start in range(0,len(queries),4096):
            stop=min(start+4096,len(queries))
            new=p8.read_row_group(rowgroup).to_pandas()
            old=p4.read_row_group(rowgroup).to_pandas()
            rowgroup+=1
            np.testing.assert_array_equal(new[ID],ids[offset+start:offset+stop])
            pd.testing.assert_frame_equal(new[[ID,*prior['features']]],old,check_exact=True)
            values=new[manifest['features']].to_numpy().reshape(len(new),16,14)
            index=np.asarray(neighbors[offset+start:offset+stop])
            index=np.where(index>=0,index-part['event_offset'],-1)
            safe=np.maximum(index,0)
            for ph,section in [('DEP',slice(0,8)),('ARR',slice(8,16))]:
                count=np.minimum(((index>=0)&(phase[safe]==ph)).sum(axis=1),8)
                padding=np.arange(8)[None,:]>=count[:,None]
                np.testing.assert_array_equal(values[:,section,13],padding)
                assert np.isnan(values[:,section,:10][padding]).all()
                assert np.all(values[:,section,10:13][padding]==0)
            for pos in sample[(sample>=start)&(sample<stop)]:
                query=raw.loc[queries.iloc[pos][ID]]
                qt=pd.Timestamp(query[TIME])
                candidates=index[pos-start]
                candidates=candidates[candidates>=0]
                expect=[]
                for ph in ['DEP','ARR']:
                    # Reverse the already-proven chronological source list; reconstruct values only from raw fields.
                    chosen=[j for j in candidates[::-1] if phase[j]==ph][:8]
                    for rank in range(8):
                        if rank>=len(chosen):
                            expect.append([np.nan]*10+[0.,0.,0.,1.])
                            continue
                        event_id=events.iloc[chosen[rank]][ID]
                        event=raw.loc[event_id]
                        movement=pd.Timestamp(event[TIME])
                        eventtime=movement if ph=='DEP' else pd.Timestamp(completion.loc[event_id,'BLOCK_TIME_UTC_mvt'])
                        assert eventtime<qt and eventtime>=qt-pd.Timedelta(hours=1)
                        assert event_id!=queries.iloc[pos][ID]
                        if pd.notna(query.FLIGHT_ID_mvt):
                            assert event.FLIGHT_ID_mvt!=query.FLIGHT_ID_mvt
                        assert (event.ADEP_mvt if ph=='DEP' else event.ADES_mvt)==query.ADEP_mvt
                        assert movement.strftime('%Y-%m')==qt.strftime('%Y-%m')
                        if ph=='DEP':
                            numeric=[(movement-pd.Timestamp(event[c])).total_seconds() if pd.notna(event[c]) else np.nan for c in CLOCKS]
                            numeric+=[pd.Timestamp(event.AOBT_3_flt).second if pd.notna(event.AOBT_3_flt) else np.nan,movement.second,np.nan]
                        else:
                            numeric=[np.nan]*6+[movement.second,(eventtime-movement).total_seconds()]
                        numeric.extend([(qt-eventtime).total_seconds(),(qt-movement).total_seconds() if ph=='ARR' else np.nan])
                        numeric.extend([eq(event[c],query[c]) for c in ['RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt']])
                        numeric.append(0.)
                        expect.append(numeric)
                np.testing.assert_allclose(values[pos-start],np.asarray(expect,dtype='float32'),rtol=0,atol=0,equal_nan=True)
                checked+=1
            peak()
        checks.append(dict(file=part['file'],rows=len(queries),raw_oracle_queries=checked))
        print('VERIFIED8',part['file'],checked,'peak',peak(),flush=True)
        del events,queries,raw,completion,public,arrivals
        gc.collect()
    assert rowgroup==p8.num_row_groups
    common.write_json(OUT/'verification.json',dict(status='passed',verifier_sha256=common.sha256(__file__),manifest_sha256=common.sha256(OUT/'manifest.json'),
        all_rows=sum(c['rows'] for c in checks),features=len(manifest['features']),all_last4_subsets_exact=True,
        all_IDs_and_padding_values_exact=True,raw_oracle_queries=sum(c['raw_oracle_queries'] for c in checks),
        departure_hidden_columns_read=[],arrival_block_filter=['PHASE_mvt','=','ARR'],raw_public_columns=FIELDS+CLOCKS,
        upstream_selection_proof_reused=True,peak_rss_bytes=peak(),partitions=checks))


if __name__=='__main__':
    main()
