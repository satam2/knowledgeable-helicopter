"""Observed departure-clock summaries for strictly following stand/runway peers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from taxiout.schema import ID,FLIGHT_ID,PHASE,MOVEMENT,CLOCKS
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/source_distinctions/following_groups_v1')
RAW=ROOT/'data/09-15-2026-18-55-03_files_list'
INPUTS=[ID,FLIGHT_ID,PHASE,MOVEMENT,'ADEP_mvt','RUNWAY_mvt','STAND_mvt',*CLOCKS]
NAMES=['nm','est','init','last','sched']
STATS=[*[f'window_{c}_mean_sec' for c in NAMES],'window_count','window_nm_missing_share',
       *[f'nearest_{c}_mean_sec' for c in NAMES],'nearest_count','nearest_gap_sec']
COLUMNS=[f'following_{scope}_{s}' for scope in ('runway','stand') for s in STATS]
HORIZON=3600*10**9


def prepare(raw):
    x=raw.loc[raw[PHASE].eq('DEP'),INPUTS].copy()
    stamp=pd.to_datetime(x[MOVEMENT],utc=True)
    assert stamp.notna().all() and x[ID].is_unique
    x['time']=stamp.dt.as_unit('ns').astype('int64').to_numpy()
    x['month']=stamp.dt.strftime('%Y-%m').to_numpy()
    for dest,src in [('airport','ADEP_mvt'),('runway','RUNWAY_mvt'),('stand','STAND_mvt')]:
        x[dest]=x[src].astype('string').fillna('').str.strip().str.upper()
    keys=[('f',float(f)) if pd.notna(f) else ('m',float(i)) for f,i in zip(x[FLIGHT_ID],x[ID])]
    x['flightcode']=pd.factorize(keys,sort=False)[0]
    for name,column in zip(NAMES,CLOCKS):
        x[name]=(stamp-pd.to_datetime(x[column],utc=True)).dt.total_seconds().to_numpy()
    return x[[ID,'time','month','airport','runway','stand','flightcode',*NAMES]].reset_index(drop=True)


def prefix(values):
    return np.vstack([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0,dtype=float)])


def keys(flight,time):
    result=np.empty(len(time),dtype=[('flight','i8'),('time','i8')])
    result['flight']=flight
    result['time']=time
    return result


def summarize(query,events):
    n=len(query)
    result=np.full((n,len(STATS)),np.nan)
    result[:,5]=result[:,12]=0.
    if not len(events):
        return result
    events=events.sort_values('time',kind='stable')
    times=events.time.to_numpy(np.int64)
    codes=events.flightcode.to_numpy(np.int64)
    values=events[NAMES].to_numpy(float)
    finite=np.isfinite(values)
    sufficient=np.column_stack([np.ones(len(events)),finite.astype(float),np.where(finite,values,0.)])
    sums=prefix(sufficient)
    qt=query.time.to_numpy(np.int64)
    qf=query.flightcode.to_numpy(np.int64)
    lower=np.searchsorted(times,qt,side='right')
    upper=np.searchsorted(times,qt+HORIZON,side='right')
    total=sums[upper]-sums[lower]
    order=np.lexsort((times,codes))
    byflight=keys(codes[order],times[order])
    flight_sums=prefix(sufficient[order])
    left=np.searchsorted(byflight,keys(qf,qt),side='right')
    right=np.searchsorted(byflight,keys(qf,qt+HORIZON),side='right')
    total-=flight_sums[right]-flight_sums[left]
    assert np.all(total[:,:6]>=0)
    result[:,:5]=np.divide(total[:,6:],total[:,1:6],out=np.full((n,5),np.nan),where=total[:,1:6]>0)
    result[:,5]=total[:,0]
    result[:,6]=np.divide(total[:,0]-total[:,1],total[:,0],out=np.full(n,np.nan),where=total[:,0]>0)
    unique,starts,counts=np.unique(times,return_index=True,return_counts=True)
    bin_sums=sums[starts+counts]-sums[starts]
    chosen=np.searchsorted(unique,qt,side='right')
    pending=np.flatnonzero(total[:,0]>0)
    # Skip a tied event group only when every event belongs to the query flight.
    while len(pending):
        assert np.all(chosen[pending]<len(unique))
        selected_time=unique[chosen[pending]]
        assert np.all(selected_time<=qt[pending]+HORIZON)
        lo=np.searchsorted(byflight,keys(qf[pending],selected_time),side='left')
        hi=np.searchsorted(byflight,keys(qf[pending],selected_time),side='right')
        nearest=bin_sums[chosen[pending]]-(flight_sums[hi]-flight_sums[lo])
        valid=nearest[:,0]>0
        take=pending[valid]
        result[take,7:12]=np.divide(nearest[valid,6:],nearest[valid,1:6],out=np.full((len(take),5),np.nan),where=nearest[valid,1:6]>0)
        result[take,12]=nearest[valid,0]
        result[take,13]=(selected_time[valid]-qt[take])/1e9
        pending=pending[~valid]
        chosen[pending]+=1
    return result


def build_frame(x):
    events=x.sort_values(ID,kind='stable').drop_duplicates(['airport','month','flightcode','time'])
    result=np.full((len(x),len(COLUMNS)),np.nan,dtype=np.float32)
    for k,scope in enumerate(('runway','stand')):
        result[:,k*14+5]=result[:,k*14+12]=0.
        valid=x.airport.ne('') & x[scope].ne('')
        valid_events=events.airport.ne('') & events[scope].ne('')
        groups=events.loc[valid_events].groupby(['airport','month',scope],observed=True).indices
        pool=events.loc[valid_events]
        for group,positions in x.loc[valid].groupby(['airport','month',scope],observed=True).groups.items():
            query=x.loc[positions]
            peer=pool.iloc[groups.get(group,np.array([],dtype=int))]
            result[np.asarray(positions),k*14:(k+1)*14]=summarize(query,peer).astype(np.float32)
    return pd.DataFrame(result,index=x[ID].to_numpy(),columns=COLUMNS).rename_axis(ID).reset_index()


def declaration():
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(source_sha256=common.sha256(__file__),features=COLUMNS,input_columns=INPUTS,
        policy='Retrospective supplied final DEP records. LogicalUTCmovementmonth andairport isolated; sameknownstand/runway; unknown group no peers. Strict(T,T+3600s] horizon; nearest tiedtimestamp means withinhorizon. Allsameflight excluded, missingflightIDs distinctbyMVTID. Identicalairport/month/flight/time dedup smallestMVTID beforegrouping. No departureblock/target columns loaded.',
        values='Unclipped finite takeoff-minus-fiveobservedclocks. Count0, unavailablemeans/shares/gap NaN. Allrawrows retained asqueries. No label-derived choices.',
        resources='2CPU, <6GiB RSS and >=8GiB available guards; all2025DEP training queries only, no ranking build inthispilot.',
        raw_hashes=common.read_json(ROOT/'private_runs/submission_v2/protocol.json')['raw_hashes'])
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


def guard():
    assert psutil.Process().memory_info().rss<6*1024**3
    assert psutil.virtual_memory().available>=8*1024**3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    record=declaration()
    if args.declare_only:
        return
    assert not (OUT/'features.parquet').exists()
    frames=[]
    for path in sorted(RAW.glob('training_*.parquet')):
        assert common.sha256(path)==record['raw_hashes'][path.name]
        raw=pq.read_table(path,columns=INPUTS,filters=[(PHASE,'=','DEP')],use_threads=False).to_pandas()
        frames.append(raw)
        guard()
    raw=pd.concat(frames,ignore_index=True)
    del frames
    x=prepare(raw)
    del raw
    guard()
    assert len(x)==2085047
    chunks=[]
    for month,part in x.groupby('month',sort=True):
        part=part.reset_index(drop=True)
        chunks.append(build_frame(part))
        print('BUILT',month,len(part),flush=True)
        guard()
    result=pd.concat(chunks,ignore_index=True).set_index(ID).loc[x[ID]].reset_index()
    assert result[ID].is_unique and len(result)==len(x)
    result.to_parquet(OUT/'features.parquet',index=False)
    common.write_json(OUT/'manifest.json',dict(status='complete',source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),feature_sha256=common.sha256(OUT/'features.parquet'),
        rows=len(result),features=COLUMNS,ids_hash=common.object_hash(result[ID].tolist())))
    print('COMPLETE following grouped clocks',len(result),flush=True)


if __name__=='__main__':
    main()
