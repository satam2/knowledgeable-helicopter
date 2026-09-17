"""Bounded all-ID and stratified raw-row oracle for NM-clock peer cache."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import validate_candidate as v

ROOT,ID,TIME=v.ROOT,v.ID,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/forensics/nm_clock_peers/cache_v2'
OUT=ROOT/'private_runs/tail240_20260916/validation/nm_peer_cache_v2'
CLOCKS=['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
FIELDS=['nm','est','init','last','sched']
INPUTS=[ID,'FLIGHT_ID_mvt',TIME,'PHASE_mvt','ADEP_mvt','STAND_mvt',*CLOCKS]
STATS=['count',*[n+'_mean_sec' for n in FIELDS],'nm_std_sec','exact_nm_tie_fraction',*[n+'_finite_count' for n in FIELDS[1:]]]
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(source_sha256=v.sha256(__file__),producer_protocol_sha256=v.sha256(BASE/'protocol.json'),private_columns=INPUTS,
        sample='Each logical2025UTCmonth:5randomqueries per airport xNMavailability using seed20260916, allrowsifgroup<5; originalrawqueryorder then sort selectedpositions. Fullrawmonth all12packs filteredDEP beforeprojection.',
        oracle='Direct inclusive +/-15/60minute rawNMtimestamp membership; sameairport/month/knownstand; dedup smallestmovementID perairport/nonnullflight/movementtime, missingflightmovement-specific; excludequery andevery nonnullsameflight. Rawmeans/populationstd/ties/finitecounts. No targets/DEPblocks.',
        tolerance='Float32 counts/flags/finitecounts exact;meansrtol2e-6atol.002seconds;std rtol2e-5atol.02seconds;tie rtol2e-6atol1e-7;NaNpatternexact. Frozen before sampledoutputs.',
        full_checks='Complete originaldepartureIDorder/rowcount/hash, no duplicateIDs, all49declaredcolumns,source/rawhashes andcompletedmanifest, v1/v2 featureParquetsha256 parity.',resources='1CPU/2GiBpeak/8GiBhostreserve/start10GiB; noGPU/model/labels.')
    path=OUT/'protocol.json'
    if path.exists():assert v.read_json(path)==record
    else:v.write_json(path,record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    protocol=declare()
    if args.declare_only:print('DECLARED',v.sha256(OUT/'protocol.json'),flush=True);return
    assert psutil.virtual_memory().available>=10*1024**3
    assert not (OUT/'receipt.json').exists()
    marker=v.read_json(BASE/'manifest.json');producer=v.read_json(BASE/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==protocol['producer_protocol_sha256']
    assert marker['source_sha256']==producer['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/nm_clock_peers/build_v2.py')
    assert producer['input_columns']==INPUTS
    featurepath=BASE/'features.parquet'
    assert v.sha256(featurepath)==marker['outputs']['features.parquet']
    previous=BASE.parent/'cache_v1/features.parquet'
    assert v.sha256(previous)==v.sha256(featurepath)
    ids=pd.read_parquet(featurepath,columns=[ID])[ID]
    original=pd.read_parquet(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[ID])[ID]
    np.testing.assert_array_equal(ids,original)
    assert len(ids)==marker['rows']==2085047 and ids.is_unique
    assert v.object_hash(ids.tolist())==marker['id_hash']
    assert pq.read_schema(featurepath).names==[ID,*producer['feature_columns']]
    rawfiles=sorted(v.common.RAW.glob('training_*.parquet'))
    for path in rawfiles:assert v.sha256(path)==marker['raw_hashes'][path.name]
    rng=np.random.default_rng(20260916);records=[];witnesses=[]
    for month in pd.date_range('2025-01-01','2025-12-01',freq='MS',tz='UTC'):
        end=month+pd.offsets.MonthBegin(1)
        chunks=[pd.read_parquet(p,columns=INPUTS,filters=[('PHASE_mvt','=','DEP'),(TIME,'>=',month),(TIME,'<',end)]) for p in rawfiles]
        raw=pd.concat(chunks,ignore_index=True);del chunks
        assert raw.PHASE_mvt.eq('DEP').all() and raw[ID].is_unique
        for name in ['ADEP_mvt','STAND_mvt']:raw[name]=raw[name].astype('string').fillna('').str.strip().str.upper()
        raw['_missing']=raw[CLOCKS[0]].isna()
        positions=[]
        for _,part in raw.groupby(['ADEP_mvt','_missing'],observed=True):positions.extend(rng.choice(part.index,size=min(5,len(part)),replace=False).tolist())
        query=raw.loc[sorted(positions)].copy()
        cached=pd.read_parquet(featurepath,filters=[(ID,'in',query[ID].tolist())]).set_index(ID)
        events=raw.sort_values(ID,kind='stable').copy()
        events['_flight_key']=[('f',f) if pd.notna(f) else ('m',ident) for f,ident in zip(events.FLIGHT_ID_mvt,events[ID])]
        events=events.drop_duplicates(['ADEP_mvt','_flight_key',TIME])
        events=events.loc[events[CLOCKS[0]].notna()]
        groups={airport:part for airport,part in events.groupby('ADEP_mvt',observed=True)}
        checked=0;maxmean=0.;maxstd=0.
        for _,q in query.iterrows():
            actual=cached.loc[q[ID]]
            assert actual.nmpeer_anchor_missing==float(pd.isna(q[CLOCKS[0]]))
            group=groups.get(q.ADEP_mvt,events.iloc[:0])
            for scope in ['airport','stand']:
                for width in [15,60]:
                    selected=group
                    if pd.isna(q[CLOCKS[0]]) or q.ADEP_mvt=='' or (scope=='stand' and q.STAND_mvt==''):
                        selected=selected.iloc[:0]
                    else:
                        selected=selected.loc[(selected[CLOCKS[0]]-q[CLOCKS[0]]).abs().le(pd.Timedelta(minutes=width))]
                    if scope=='stand':selected=selected.loc[selected.STAND_mvt.eq(q.STAND_mvt)]
                    selected=selected.loc[selected[ID].ne(q[ID])]
                    if pd.notna(q.FLIGHT_ID_mvt):selected=selected.loc[selected.FLIGHT_ID_mvt.ne(q.FLIGHT_ID_mvt)]
                    values=np.column_stack([(selected[TIME]-selected[c]).dt.total_seconds().to_numpy() for c in CLOCKS])
                    means=np.array([a[np.isfinite(a)].mean() if np.isfinite(a).any() else np.nan for a in values.T])
                    std=float(values[:,0].std()) if len(selected) else np.nan
                    tie=float(selected[CLOCKS[0]].eq(q[CLOCKS[0]]).mean()) if len(selected) else np.nan
                    expected=np.array([len(selected),*means,std,tie,*np.isfinite(values[:,1:]).sum(axis=0)],float)
                    names=[f'nmpeer_{scope}_{width}m_{s}' for s in STATS]
                    observed=actual[names].to_numpy(float)
                    np.testing.assert_array_equal(np.isnan(expected),np.isnan(observed))
                    np.testing.assert_array_equal(observed[[0,8,9,10,11]],expected[[0,8,9,10,11]])
                    np.testing.assert_allclose(observed[1:6],expected[1:6],rtol=2e-6,atol=.002,equal_nan=True)
                    np.testing.assert_allclose(observed[6],std,rtol=2e-5,atol=.02,equal_nan=True)
                    np.testing.assert_allclose(observed[7],tie,rtol=2e-6,atol=1e-7,equal_nan=True)
                    if len(selected):maxmean=max(maxmean,float(np.nanmax(abs(observed[1:6]-means))));maxstd=max(maxstd,abs(observed[6]-std))
                    checked+=len(names)
            witnesses.append({ID:q[ID],'month':month.strftime('%Y-%m'),'airport':q.ADEP_mvt,'missing_anchor':bool(q['_missing'])})
        records.append(dict(month=month.strftime('%Y-%m'),raw_dep_rows=len(raw),sample_queries=len(query),statistic_cells=checked,maximum_mean_error=maxmean,maximum_std_error=maxstd,peak_bytes=guard()))
        print('MONTH_ORACLE',records[-1],flush=True)
        del raw,query,cached,events,groups
        gc.collect()
    pd.DataFrame(witnesses).to_parquet(OUT/'query_ids.parquet',index=False)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),protocol_sha256=v.sha256(OUT/'protocol.json'),producer_manifest_sha256=v.sha256(BASE/'manifest.json'),all_original_ids_exact=True,v1_v2_parquet_byte_identical=True,records=records,sample_queries=sum(r['sample_queries'] for r in records),statistic_cells=sum(r['statistic_cells'] for r in records),query_ids_sha256=v.sha256(OUT/'query_ids.parquet'),private_targets_or_dep_block_read=False,peak_bytes=guard(),limitation='AllIDs and source/hash integrity checked; values independently checked on fixed stratified sample plus synthetic boundaries, not every103millioncachevalue.')
    v.write_json(OUT/'receipt.json',receipt)
    print('NM_CACHE_VERIFIED',receipt['sample_queries'],receipt['statistic_cells'],receipt['peak_bytes'],flush=True)


if __name__=='__main__':main()
