"""Independent sequence raw oracle and exhaustive cached availability predicates."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'review_work/breakthrough_20260916/sequence_context'
CACHE=ROOT/'private_runs/breakthrough_20260916/sequence_context'
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_independent_audit'
ID='MVT_ID_mvt'
FLIGHT='FLIGHT_ID_mvt'
PHASE='PHASE_mvt'
TIME='MVT_TIME_UTC_mvt'
BLOCK='BLOCK_TIME_UTC_mvt'
CLOCKS=['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
FIELDS=[ID,FLIGHT,PHASE,TIME,'ADEP_mvt','ADES_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','WK_TBL_CAT_flt','AIRCRAFT_OPERATOR_flt',*CLOCKS]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):
            h.update(data)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def raw_events(files):
    pieces=[]
    arrivals=[]
    for path in files:
        pieces.append(pq.read_table(path,columns=FIELDS,use_threads=False).to_pandas())
        arrival=pq.read_table(path,columns=[ID,PHASE,BLOCK],filters=[(PHASE,'=','ARR')],use_threads=False).to_pandas()
        assert arrival[PHASE].eq('ARR').all()
        arrivals.append(arrival[[ID,BLOCK]])
    raw=pd.concat(pieces,ignore_index=True).drop_duplicates(ID).sort_values(ID).reset_index(drop=True)
    completion=pd.concat(arrivals,ignore_index=True).drop_duplicates(ID).set_index(ID)[BLOCK]
    raw['arrival_completion']=pd.to_datetime(raw[ID].map(completion),utc=True)
    movement=pd.to_datetime(raw[TIME],utc=True)
    dep=raw[PHASE].eq('DEP')
    raw['event_time']=movement.where(dep,raw.arrival_completion)
    raw['airport']=raw.ADEP_mvt.where(dep,raw.ADES_mvt).astype('string').fillna('<missing>')
    raw['month']=movement.dt.strftime('%Y-%m')
    raw['arr_duration']=(raw.arrival_completion-movement).dt.total_seconds()
    valid=dep|((~dep)&raw.arrival_completion.notna()&raw.arr_duration.ge(0))
    events=raw.loc[valid].copy()
    known=events[FLIGHT].notna()
    duplicate=events.loc[known].duplicated([FLIGHT,'airport',PHASE,TIME])
    events=events.drop(index=duplicate.index[duplicate]).reset_index(drop=True)
    return events


def exhaustive(events,queries,local):
    count=0
    times=events.time_ns.to_numpy(np.int64)
    ids=events[ID].to_numpy(float)
    flight=events.flight.to_numpy(float)
    airports=events.airport.to_numpy()
    months=events.month.to_numpy()
    phase=events.phase.to_numpy()
    for start in range(0,len(queries),4096):
        stop=min(start+4096,len(queries))
        selected=local[start:stop]
        mask=selected>=0
        safe=np.maximum(selected,0)
        query=queries.iloc[start:stop]
        qt=query.time_ns.to_numpy(np.int64)[:,None]
        assert np.all(~mask|((times[safe]<qt)&(times[safe]>=qt-3600*10**9)))
        assert np.all(~mask|(ids[safe]!=query[ID].to_numpy(float)[:,None]))
        qf=query.flight.to_numpy(float)[:,None]
        assert np.all(~mask|~(np.isfinite(qf)&(flight[safe]==qf)))
        assert np.all(~mask|(airports[safe]==query.airport.to_numpy()[:,None]))
        assert np.all(~mask|(months[safe]==query.month.to_numpy()[:,None]))
        assert np.all(np.sum(mask&(phase[safe]=='DEP'),axis=1)<=16)
        assert np.all(np.sum(mask&(phase[safe]=='ARR'),axis=1)<=16)
        assert not np.any((~mask[:,:-1])&mask[:,1:])
        ordered=np.where(mask,times[safe],np.iinfo(np.int64).max)
        assert np.all(ordered[:,1:]>=ordered[:,:-1])
        for row in selected:
            real=row[row>=0]
            assert len(real)==len(np.unique(ids[real]))
        count+=int(mask.sum())
    return count


def oracle(events,queries,stored,local,sample):
    checked=0
    for position in sample:
        query=queries.iloc[position]
        qt=pd.Timestamp(query.time_ns,tz='UTC')
        eligible=events.airport.eq(query.airport)&events.month.eq(query.month)&events.event_time.lt(qt)&events.event_time.ge(qt-pd.Timedelta(hours=1))&events[ID].ne(query[ID])
        if np.isfinite(query.flight):
            eligible &= events[FLIGHT].ne(query.flight)
        pieces=[]
        for phase in ['DEP','ARR']:
            selected=events.loc[eligible&events[PHASE].eq(phase)].sort_values(['event_time',ID],ascending=[False,False],kind='stable').head(16)
            pieces.append(selected)
        expected=pd.concat(pieces).sort_values('event_time',kind='stable')
        index=local[position]
        actual=stored.iloc[index[index>=0]]
        np.testing.assert_array_equal(actual[ID],expected[ID])
        for (_,got),(_,raw) in zip(actual.iterrows(),expected.iterrows()):
            assert got.time_ns==raw.event_time.value
            assert got.movement_ns==pd.Timestamp(raw[TIME]).value
            if raw[PHASE]=='ARR':
                assert got.arrival_duration==np.float32(raw.arr_duration)
                assert all(np.isnan(got['offset_'+clock]) for clock in CLOCKS)
                assert np.isnan(got.aobt_second)
            else:
                assert np.isnan(got.arrival_duration)
                for clock in CLOCKS:
                    value=(pd.Timestamp(raw[TIME])-pd.Timestamp(raw[clock])).total_seconds() if pd.notna(raw[clock]) else np.nan
                    np.testing.assert_allclose(got['offset_'+clock],np.float32(value),rtol=0,atol=0,equal_nan=True)
        checked+=1
    return checked


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'verification.json'
    if output.exists():
        raise ValueError('Completed independentaudit retained')
    marker=json.loads((CACHE/'manifest.json').read_text())
    assert marker['status']=='complete'
    assert sha(SOURCE/'cache.py')==marker['source_sha256']
    assert sha(CACHE/'neighbors.npy')==marker['neighbors_sha256']
    audit=json.loads((ROOT/'private_runs/screening_230/reports/data_audit.json').read_text())
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(metadata)==audit['artifacts'][metadata.name]
    ids=pq.read_table(metadata,columns=[ID],use_threads=False).column(0).to_numpy()
    neighbors=np.load(CACHE/'neighbors.npy',mmap_mode='r')
    files=sorted((ROOT/'data/09-15-2026-18-55-03_files_list').glob('training*.parquet'))
    checks=[]
    prior_event_ids=np.array([],float)
    for i,record in enumerate(marker['records']):
        for name in ['events','queries']:
            assert sha(CACHE/record[name+'_file'])==record[name+'_sha256']
        events=pd.read_parquet(CACHE/record['events_file'])
        queries=pd.read_parquet(CACHE/record['queries_file'])
        offset=record['query_offset']
        np.testing.assert_array_equal(queries[ID],ids[offset:offset+len(queries)])
        global_index=np.asarray(neighbors[offset:offset+len(queries)])
        local=np.where(global_index>=0,global_index-record['event_offset'],-1)
        assert np.all((local<0)|(local<len(events)))
        assert events[ID].is_unique
        available=exhaustive(events,queries,local)
        duplicate_cross=int(len(np.intersect1d(prior_event_ids,events[ID].to_numpy())))
        prior_event_ids=events[ID].to_numpy()
        window=files[max(0,i-1):min(len(files),i+2)]
        for path in window:
            assert sha(path)==marker['raw_hashes'][path.name]
        raw=raw_events(window)
        rng=np.random.default_rng(20260916+i)
        unusual=np.flatnonzero(queries.month.to_numpy()!=record['file'][9:16])
        first=np.argsort(queries.time_ns.to_numpy())[:8]
        last=np.argsort(queries.time_ns.to_numpy())[-8:]
        random=rng.choice(len(queries),24,replace=False)
        sample=np.unique(np.r_[first,last,random,unusual])
        n=oracle(raw,queries,events,local,sample)
        arrivals=events.phase.eq('ARR')
        crosscompletion=int((np.floor_divide(events.loc[arrivals,'time_ns'].to_numpy(np.int64),86400*10**9)!=np.floor_divide(events.loc[arrivals,'movement_ns'].to_numpy(np.int64),86400*10**9)).sum())
        checks.append(dict(file=record['file'],query_rows=len(queries),selected_tokens_checked=available,raw_oracle_queries=n,
            unusual_query_month_rows=len(unusual),event_ids_repeated_from_previous_segment=duplicate_cross,
            arrival_completion_next_calendar_day_rows=crosscompletion,duplicate_events_within_query=0))
        print('INDEPENDENT_SEQUENCE',record['file'],checks[-1],flush=True)
        del events,queries,raw,local
        gc.collect()
    result=dict(status='passed',manifest_sha256=sha(CACHE/'manifest.json'),source_sha256=sha(__file__),
        all_query_rows=len(ids),all_selected_tokens=sum(c['selected_tokens_checked'] for c in checks),
        raw_oracle_queries=sum(c['raw_oracle_queries'] for c in checks),checks=checks,
        hidden_departure_columns_read=[],arrival_predicate=[PHASE,'=','ARR'],
        all_actual_event_times_strict_before_query=True,sameflight_and_self_excluded=True,
        exact_airport_and_movement_month=True,padded_mask_contiguous=True,context_monotone_event_times=True,
        policy_limit='ARRmovementmonth retained evenif completioncrossesmonth; finalNMmetadata publicationtime unknown. No globalcachede-dup neededperquery; samephysicalevents can appearinseparatequerypack segments.')
    write(output,result)
    print('PASSED_SEQUENCE',result['all_query_rows'],result['raw_oracle_queries'],flush=True)


if __name__=='__main__':
    main()
