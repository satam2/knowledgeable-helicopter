"""Independent full ARR-only calendar audit with raw anomaly witnesses."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil
import validate_candidate as v

BASE=v.ROOT/'private_runs/tail240_20260916/state/arrival_calendar/v1'
OUT=v.ROOT/'private_runs/tail240_20260916/validation/arrival_calendar_v1'
ID,TIME=v.ID,v.common.MOVEMENT
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');r=v.read_json(BASE/'receipt.json')
    assert r['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert protocol['source_sha256']==v.sha256(v.ROOT/'review_work/tail240_20260916/state/arrival_calendar/audit.py')
    for name,digest in r['artifacts'].items():assert v.sha256(BASE/name)==digest
    expected_counts=pd.read_csv(BASE/'file_counts.csv')
    names=list(expected_counts.columns[1:])
    groups=[];witnesses=[];records=[]
    for filename,digest in protocol['raw_hashes'].items():
        path=v.common.RAW/filename
        assert v.sha256(path)==digest
        scanner=ds.dataset(path,format='parquet').scanner(columns=protocol['columns'],filter=ds.field('PHASE_mvt')=='ARR',batch_size=8192,batch_readahead=1,fragment_readahead=1,use_threads=False)
        totals=np.zeros(len(names),np.int64)
        for batch in scanner.to_batches():
            frame=batch.to_pandas()
            assert frame.PHASE_mvt.eq('ARR').all()
            t=pd.to_datetime(frame[TIME],utc=True);b=pd.to_datetime(frame.BLOCK_TIME_UTC_mvt,utc=True);s=pd.to_datetime(frame.SCHED_TIME_UTC_mvt,utc=True)
            seconds=(b-t).dt.total_seconds()
            valid=t.notna()&b.notna()&s.notna()
            utc=valid&b.dt.date.eq(s.dt.date)&b.dt.date.ne(t.dt.date)
            td,bd,sd=[a.dt.tz_convert('Europe/Rome').dt.date for a in [t,b,s]]
            rome=valid&frame.ADES_mvt.isin(['LIRF','LIRA'])&bd.eq(sd)&bd.ne(td)
            neg=seconds.lt(0);extreme=seconds.abs().gt(43200)
            data=dict(rows=np.ones(len(frame),int),finite_duration=np.isfinite(seconds),known_three_clocks=valid,negative=neg,abs_gt_12h=extreme,negative_near_day=seconds.ge(-86400)&seconds.le(-79200),positive_near_day=seconds.ge(86400)&seconds.le(93600),utc_block_schedule_date_carry=utc,utc_carry_negative=utc&neg,utc_carry_extreme=utc&extreme,rome_block_schedule_date_carry=rome,rome_carry_negative=rome&neg,rome_carry_extreme=rome&extreme)
            for name,value in data.items():frame[name]=np.asarray(value,np.int64)
            frame['source']=filename
            frame['month']=t.dt.strftime('%Y-%m').fillna('MISSING')
            frame['airport']=frame.ADES_mvt.fillna('MISSING')
            frame['prefix']=frame.FLIGHT_mvt.astype('string').str.extract(r'^([A-Za-z]+)',expand=False).str.upper().fillna('MISSING')
            frame['duration_sec']=seconds
            frame['block_minus_movement_days_utc']=(b.dt.floor('D')-t.dt.floor('D')).dt.total_seconds()/86400
            totals+=frame[names].sum().to_numpy(np.int64)
            groups.append(frame.groupby(['source','month','airport','prefix'],dropna=False)[names].sum())
            unusual=neg|extreme|utc|rome
            if unusual.any():witnesses.append(frame.loc[unusual].copy())
            guard()
        records.append(dict(source=filename,**dict(zip(names,totals.tolist()))))
    actual=pd.DataFrame(records).sort_values('source').reset_index(drop=True)
    expected=expected_counts.sort_values('source').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns],expected)
    agg=pd.concat(groups).groupby(level=[0,1,2,3]).sum().reset_index()
    expected=pd.read_parquet(BASE/'airport_prefix_month.parquet')
    keys=['source','month','airport','prefix']
    pd.testing.assert_frame_equal(agg.sort_values(keys).reset_index(drop=True)[expected.columns],expected.sort_values(keys).reset_index(drop=True),check_dtype=False)
    witnesses=pd.concat(witnesses,ignore_index=True)
    expected=pd.read_parquet(BASE/'anomalous_arrivals.parquet')
    keys=['source',ID]
    pd.testing.assert_frame_equal(witnesses.sort_values(keys).reset_index(drop=True)[expected.columns],expected.sort_values(keys).reset_index(drop=True),check_dtype=False)
    totals=actual[names].sum().to_dict()
    assert totals==r['totals']
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),producer_receipt_sha256=v.sha256(BASE/'receipt.json'),all_13_files_replayed=True,all_source_month_airport_prefix_counts_exact=True,all_anomaly_witness_fields_exact=True,totals=totals,anomaly_rows=len(witnesses),private_departure_labels_or_block_read=False,phase_predicate_before_projection='ARR',peak_bytes=guard(),limitation='Calendar associations and anomalous ARR durations are descriptive; no automatic DEP label correction or source-date repair justified.')
    v.write_json(OUT/'receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':main()
