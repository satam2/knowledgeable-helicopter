"""Independent full-cache ATFM formula, metadata-join and missingness oracle."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import gc
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import validate_candidate as v

ROOT=v.ROOT
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
BASE=ROOT/'private_runs/tail240_20260916/forensics/atfm'
CACHE=BASE/'features_v1'
TABLES=BASE/'inspection_v1/tables'
OUT=ROOT/'private_runs/tail240_20260916/validation/atfm_cache_v1'
CAUSES=['A','C','D','E','G','I','M','N','O','P','R','S','T','V','W','NA']
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def divide(a,b):
    return a/b if pd.notna(a) and pd.notna(b) and b>0 else np.nan


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=v.read_json(CACHE/'manifest.json')
    assert manifest['status']=='complete'
    assert manifest['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/atfm/build_features.py')
    assert manifest['protocol_sha256']==v.sha256(CACHE/'protocol.json')
    for name,digest in manifest['outputs'].items():
        assert v.sha256(CACHE/name)==digest,name
    for name,digest in manifest['source_table_hashes'].items():
        assert v.sha256(TABLES/name)==digest,name
    slot=pd.read_parquet(TABLES/'ATFM_Slot_Adherence_airports.parquet')
    arrival=pd.read_parquet(TABLES/'Airport_Arrival_ATFM_Delay_airports.parquet')
    keys=['APT_ICAO','date_utc']
    s={tuple(row[k] for k in keys):row for row in slot.to_dict('records')}
    a={tuple(row[k] for k in keys):row for row in arrival.to_dict('records')}
    assert len(s)==len(slot) and len(a)==len(arrival)
    allkeys=sorted(set(s)|set(a))
    rows=[]
    for key in allkeys:
        sr=s.get(key,{}); ar=a.get(key,{})
        dep=sr.get('FLT_DEP_1',np.nan); reg=sr.get('FLT_DEP_REG_1',np.nan)
        arr=ar.get('FLT_ARR_1',np.nan); delay=ar.get('DLY_APT_ARR_1',np.nan)
        rows.append([float(key in s),float(key in a),dep,divide(reg,dep),
            *[divide(sr.get(c,np.nan),reg) for c in ['FLT_DEP_OUT_EARLY_1','FLT_DEP_IN_1','FLT_DEP_OUT_LATE_1']],
            arr,delay,divide(delay,arr),float(pd.notna(delay)) if key in a else np.nan,
            divide(ar.get('FLT_ARR_1_DLY',np.nan),arr),divide(ar.get('FLT_ARR_1_DLY_15',np.nan),arr),
            *[divide(ar.get('DLY_APT_ARR_'+c+'_1',np.nan),arr) for c in CAUSES]])
    features=['atfm_slot_row_present','atfm_arrival_row_present','atfm_departures','atfm_regulated_fraction',
        'atfm_early_fraction','atfm_within_fraction','atfm_late_fraction','atfm_arrivals',
        'atfm_arrival_delay_minutes','atfm_arrival_delay_minutes_per_arrival','atfm_arrival_delay_populated',
        'atfm_arrival_delayed_fraction','atfm_arrival_delayed_over15_fraction']+[
        'atfm_arrival_cause_'+c+'_minutes_per_arrival' for c in CAUSES]
    assert features==manifest['features']
    oracle=pd.DataFrame(rows,index=pd.MultiIndex.from_tuples(allkeys,names=keys),columns=features,dtype='float32')
    cached_days=pd.read_parquet(CACHE/'airport_days.parquet').set_index(keys).sort_index()
    pd.testing.assert_frame_equal(cached_days,oracle,check_exact=True)
    reports={}
    for split in ['training','ranking']:
        path=(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet') if split=='training' else common.RAW/'ranking.parquet'
        expected_hash=manifest['metadata_sha256' if split=='training' else 'ranking_sha256']
        assert v.sha256(path)==expected_hash
        query=pq.read_table(path,columns=[common.ID,'ADEP_mvt',common.MOVEMENT],
            filters=None if split=='training' else [('PHASE_mvt','=','DEP')],use_threads=False).to_pandas()
        cache=pd.read_parquet(CACHE/(split+'_features.parquet'))
        assert len(query)==manifest['splits'][split]['rows'] and len(cache)==len(query)
        assert query[common.ID].is_unique and query[common.ID].notna().all()
        np.testing.assert_array_equal(cache[common.ID],query[common.ID])
        assert v.object_hash(cache[common.ID].tolist())==manifest['splits'][split]['id_hash']
        timestamps=pd.to_datetime(query[common.MOVEMENT],utc=True)
        dates=timestamps.dt.strftime('%Y-%m-%d')
        querykeys=pd.MultiIndex.from_arrays([query.ADEP_mvt.astype('string'),dates],names=keys)
        indices=oracle.index.get_indexer(querykeys)
        assert (indices>=0).all()
        missing={}
        for feature in features:
            expected=oracle[feature].to_numpy()[indices]
            actual=cache[feature].to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=0,atol=0,equal_nan=True,err_msg=split+':'+feature)
            missing[feature]=int(np.isnan(actual).sum())
            assert missing[feature]==manifest['splits'][split]['missing_cells'][feature]
        coverage=[]
        groups=query.assign(month=timestamps.dt.strftime('%Y-%m')).groupby(['ADEP_mvt','month'],observed=True).indices
        for (airport,month),positions in groups.items():
            coverage.append({'airport':str(airport),'month':month,'rows':len(positions),
                'slot_matched':int(cache.atfm_slot_row_present.iloc[positions].sum()),
                'arrival_matched':int(cache.atfm_arrival_row_present.iloc[positions].sum()),
                'arrival_delay_populated':int(cache.atfm_arrival_delay_populated.iloc[positions].sum())})
        assert sorted(coverage,key=lambda r:(r['airport'],r['month']))==sorted(manifest['splits'][split]['coverage'],key=lambda r:(r['airport'],r['month']))
        reports[split]={'rows':len(query),'cells_verified':len(query)*len(features),'exact_ids_order':True,
            'all_source_rows_present':bool(cache.atfm_slot_row_present.eq(1).all() and cache.atfm_arrival_row_present.eq(1).all()),
            'missing_cells':missing,'coverage_groups_verified':len(coverage),'max_abs_difference':0}
        print('ATFM_CACHE_VERIFIED',split,len(query),len(query)*len(features),flush=True)
        del query,cache,timestamps,dates,querykeys,indices,groups
        gc.collect();v.guard()
    v.write_json(OUT/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),
        'cache_manifest_sha256':v.sha256(CACHE/'manifest.json'),'workbook_oracle_sha256':v.sha256(ROOT/'private_runs/tail240_20260916/validation/atfm_workbooks_v2/receipt.json'),
        'airport_days':len(oracle),'airport_day_cells_verified':len(oracle)*len(features),'splits':reports,
        'private_input_columns':[common.ID,'ADEP_mvt',common.MOVEMENT], 'private_targets_read':False,
        'time_semantics':'Exact UTC movement date join verified; source Date of flight timezone is unspecified. Whole-day published-in-arrears context is retrospective, not causal.',
        'blank_semantics':'Blank delay values retain NaN; finite source zeros retain zero; total-delay-populated distinguishes these cases.',
        'peak_rss_bytes':v.guard()})
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
