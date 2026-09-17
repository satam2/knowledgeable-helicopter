"""Independent brute-force oracle for following stand/runway clock aggregates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import validate_candidate as validation

ROOT=validation.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/source_distinctions'))
import following_groups as producer
ID,FID,TIME,PHASE=producer.ID,producer.FLIGHT_ID,producer.MOVEMENT,producer.PHASE
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def independent(raw):
    data=raw.loc[raw[PHASE].eq('DEP')].copy()
    data['timestamp']=pd.to_datetime(data[TIME],utc=True)
    data['month']=data.timestamp.dt.strftime('%Y-%m')
    for dest,src in [('airport','ADEP_mvt'),('stand','STAND_mvt'),('runway','RUNWAY_mvt')]:
        data[dest]=data[src].astype('string').fillna('').str.strip().str.upper()
    data['flightkey']=[('flight',float(fid)) if pd.notna(fid) else ('movement',float(mid)) for fid,mid in zip(data[FID],data[ID])]
    for name,clock in zip(producer.NAMES,producer.CLOCKS):
        data[name]=(data.timestamp-pd.to_datetime(data[clock],utc=True)).dt.total_seconds()
    return data.set_index(ID,drop=False)


def brute(data,ids):
    events=data.sort_index(kind='stable').drop_duplicates(['airport','month','flightkey','timestamp'])
    output=[]
    for movement_id in ids:
        query=data.loc[movement_id]
        row=[]
        for scope in ('runway','stand'):
            if query.airport=='' or query[scope]=='':
                row.extend([np.nan]*5+[0,np.nan]+[np.nan]*5+[0,np.nan])
                continue
            peers=events.loc[events.airport.eq(query.airport)&events.month.eq(query.month)&events[scope].eq(query[scope])]
            peers=peers.loc[(peers.timestamp>query.timestamp)&(peers.timestamp<=query.timestamp+pd.Timedelta(hours=1))]
            peers=peers.loc[peers.flightkey.map(lambda key:key!=query.flightkey)]
            window=peers[producer.NAMES].replace([np.inf,-np.inf],np.nan)
            means=window.mean().to_list()
            share=float(window.nm.isna().mean()) if len(window) else np.nan
            nearest=peers.loc[peers.timestamp.eq(peers.timestamp.min())]
            nearestvalues=nearest[producer.NAMES].replace([np.inf,-np.inf],np.nan)
            gap=(nearest.timestamp.iloc[0]-query.timestamp).total_seconds() if len(nearest) else np.nan
            row.extend(means+[len(peers),share]+nearestvalues.mean().to_list()+[len(nearest),gap])
        output.append(row)
    return pd.DataFrame(output,index=pd.Index(ids,name=ID),columns=producer.COLUMNS,dtype='float32')


def synthetic():
    rng=np.random.default_rng(20260916)
    n=480
    seconds=rng.choice([0,1,30,60,120,1800,3599,3600,3601,7200],n)
    months=rng.choice([0,31],n)
    times=pd.Timestamp('2025-01-01',tz='UTC')+pd.to_timedelta(months,unit='D')+pd.to_timedelta(seconds,unit='s')
    raw=pd.DataFrame({ID:np.arange(n,dtype=float)+1,FID:rng.choice([np.nan,1.,2.,3.,4.,5.,6.,7.],n),PHASE:'DEP',TIME:times,
        'ADEP_mvt':rng.choice(['LIRF','EGLL',''],n),'STAND_mvt':rng.choice(['A','B',' a ',None],n),
        'RUNWAY_mvt':rng.choice(['25','36',' 25 ',None],n)})
    for clock in producer.CLOCKS:
        raw[clock]=times-pd.to_timedelta(rng.integers(-10000,100000,n),unit='s')
        raw.loc[rng.random(n)<.2,clock]=pd.NaT
    expected=brute(independent(raw),raw[ID].tolist())
    actual=producer.build_frame(producer.prepare(raw)).set_index(ID)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-4,equal_nan=True)
    return {'queries':n,'cells':n*28,'max_abs_difference':float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy())))}


def cache():
    manifest=read(producer.OUT/'manifest.json')
    protocol=read(producer.OUT/'protocol.json')
    assert manifest['status']=='complete' and manifest['source_sha256']==sha(producer.__file__)
    assert manifest['protocol_sha256']==sha(producer.OUT/'protocol.json')
    assert sha(producer.OUT/'features.parquet')==manifest['feature_sha256']
    features=pd.read_parquet(producer.OUT/'features.parquet').set_index(ID)
    assert features.index.is_unique and len(features)==2085047 and list(features)==producer.COLUMNS
    assert oh(features.index.tolist())==manifest['ids_hash']
    raw=[]
    for path in sorted(producer.RAW.glob('training_*.parquet')):
        assert sha(path)==protocol['raw_hashes'][path.name]
        raw.append(pq.read_table(path,columns=producer.INPUTS,filters=[(PHASE,'=','DEP')],use_threads=False).to_pandas())
        validation.guard()
    data=independent(pd.concat(raw,ignore_index=True))
    del raw
    np.testing.assert_array_equal(data.index,features.index)
    comparisons=[]
    count=0
    for month,part in data.groupby('month'):
        sample=[]
        for _,airport in part.groupby('airport'):
            positions=np.unique(np.linspace(0,len(airport)-1,min(8,len(airport)),dtype=int))
            sample.extend(airport.iloc[positions].index.tolist())
        sample=list(dict.fromkeys(sample))
        expected=brute(part,sample)
        actual=features.loc[sample]
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-3,equal_nan=True)
        delta=float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy())))
        comparisons.append({'month':month,'queries':len(sample),'max_abs_difference':delta})
        count+=len(sample)
        print('RAW_ORACLE',month,len(sample),delta,flush=True)
        validation.guard()
    return {'query_count':count,'cell_count':count*28,'months':comparisons,'producer_manifest_sha256':sha(producer.OUT/'manifest.json'),
        'full_query_raw_ID_order_verified':True,'all_original_queries':len(data)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',action='store_true')
    args=parser.parse_args()
    out=ROOT/f'private_runs/tail240_20260916/validation/following_groups_{"cache" if args.cache else "synthetic"}_v2'
    assert not out.exists()
    result=cache() if args.cache else synthetic()
    out.mkdir(parents=True)
    write(out/'receipt.json',{'status':'passed','source_sha256':sha(__file__),'producer_source_sha256':sha(producer.__file__),
        'result':result,'peak_rss_bytes':validation.guard(),'scope':'No target/privateblock fields loaded; directfilteroracle excludesall sameflight, strictfuture/horizoninclusive, logicalUTCmonth/group, smallestIDduplicates, tiednearestmean. Cacheverification samples deterministic airport/month rows; source reviewed separately.'})
    print(result,flush=True)


if __name__=='__main__':
    main()
