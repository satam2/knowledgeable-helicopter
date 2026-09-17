"""Independent all-year raw clock reconstruction with integer timestamp arithmetic."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_union387_sources import ROOT,read,sha,write,object_hash,guard

BASE=ROOT/'private_runs/tail240_20260916/forensics/clock_year_audit/v1'
ID,TIME='MVT_ID_mvt','MVT_TIME_UTC_mvt'
CLOCKS=['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
PAIRS=[('takeoff_minus_'+name,TIME,name) for name in CLOCKS]+[
    ('nm_actual_minus_estimated','AOBT_3_flt','EOBT_1_flt'),('last_minus_initial','LOBT_flt','IOBT_flt')]


def main():
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    protocol,receipt=read(BASE/'protocol.json'),read(BASE/'receipt.json')
    assert receipt['status']=='passed' and receipt['protocol_sha256']==sha(BASE/'protocol.json')
    assert protocol['input_columns']==[ID,'PHASE_mvt',TIME,*CLOCKS]
    assert protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/forensics/clock_year_audit/run_v1.py')
    assert protocol['base_pipeline_sha256']==sha(ROOT/'knowledgeable-helicopter-screening/src/taxiout/features/pipeline.py')
    previous=ROOT/'private_runs/tail240_20260916/source_distinctions/clock_preprocessing_audit_v2'
    assert sha(previous/'protocol.json')==protocol['preserved_original_protocol_sha256']
    assert sha(previous/'receipt.json')==protocol['preserved_original_receipt_sha256']
    rawhash=read(ROOT/'private_runs/submission_v2/protocol.json')['raw_hashes']
    result=[]
    for recorded in receipt['records']:
        filename=recorded['file'];rawpath=ROOT/'data/09-15-2026-18-55-03_files_list'/filename
        cachepath=ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'/filename
        assert sha(rawpath)==rawhash[filename]==protocol['raw_files'][filename]
        assert sha(cachepath)==recorded['cache_sha256']==read(cachepath.with_suffix('.json'))['sha256']
        table=pq.read_table(rawpath,columns=protocol['input_columns'],filters=[('PHASE_mvt','=','DEP')],use_threads=False)
        ids=table[ID].to_pandas();assert ids.is_unique
        cached=pd.read_parquet(cachepath,columns=[ID,*[name for name,_,_ in PAIRS]]).set_index(ID)
        assert cached.index.is_unique and set(cached.index)==set(ids)
        cached=cached.loc[ids]
        assert len(ids)==recorded['rows'] and object_hash(ids.tolist())==recorded['raw_ids_hash']==recorded['cache_ids_after_alignment_hash']
        ticks={};valid={}
        for name in [TIME,*CLOCKS]:
            column=table[name]
            assert pa.types.is_timestamp(column.type) and column.type.tz=='UTC'
            multiplier={'s':1000000000,'ms':1000000,'us':1000,'ns':1}[column.type.unit]
            values=pc.fill_null(pc.cast(column,pa.int64()),0).to_numpy().astype(np.int64)
            valid[name]=column.is_valid().to_numpy()
            ticks[name]=values*multiplier
            stats=recorded['clocks'][name]
            assert int(valid[name].sum())==stats['original_nonnull'] and stats['coerced_nonnull']==0
            assert str(column.to_pandas().dtype)==stats['source_dtype']
            assert str(pd.Timestamp(pc.min(column).as_py()))==stats['minimum']
            assert str(pd.Timestamp(pc.max(column).as_py()))==stats['maximum']
        comparisons={}
        for name,end,start in PAIRS:
            keep=valid[end]&valid[start]
            seconds=(ticks[end]-ticks[start]).astype(np.float64)/1e9;seconds[~keep]=np.nan
            rounded=seconds.astype(np.float32).astype(float)
            expected=np.where(np.isfinite(rounded),rounded,-999999.)
            np.testing.assert_array_equal(cached[name].to_numpy(float),expected)
            values=dict(rows=len(ids),finite=int(keep.sum()),mismatches=0,
                finite_to_nonfinite=int((keep&~np.isfinite(rounded)).sum()),encoded_missing=int((~keep).sum()),
                rounded_to_sentinel=int((keep&(seconds!=-999999.)&(rounded==-999999.)).sum()),
                sentinel_collision=int((seconds==-999999.).sum()),
                within_one_float32_ulp_of_sentinel=int((keep&(seconds!=-999999.)&(np.abs(seconds+999999.)<=np.spacing(np.float32(999999.)))).sum()),
                negative_values=int((seconds<0).sum()),absolute_over_day=int((np.abs(seconds)>86400).sum()),
                max_float32_rounding_sec=float(np.abs(rounded[keep]-seconds[keep]).max()) if keep.any() else 0.)
            assert values==recorded['comparisons'][name],(filename,name)
            comparisons[name]=values
        assert guard()<1024**3
        result.append(dict(file=filename,rows=len(ids),comparisons=comparisons))
        print('RAW_INTEGER_CLOCKS_EXACT',filename,len(ids),flush=True)
    assert sum(r['rows'] for r in result)==receipt['departure_rows']==2085047
    out=ROOT/'private_runs/tail240_20260916/validation/clock_year_audit_v1';out.mkdir(parents=True,exist_ok=False)
    final=dict(status='passed',source_sha256=sha(Path(__file__)),producer_receipt_sha256=sha(BASE/'receipt.json'),
        protocol_sha256=sha(BASE/'protocol.json'),projection=protocol['input_columns'],rows=2085047,clock_values=2085047*7,
        all12raw_and_basecache_hashes_exact=True,independent_int64_timestamp_differences_exact=True,
        all_parser_nulls_minmax_negative_long_sentinel_and_cast_statistics_exact=True,peak_bytes=guard(),
        no_target_block_ranking_or_model_access=True,no_GPU=True,
        limitation='Allraw-to-base7clock arithmetic only. Does not prove operational timestamp truth or every derivedcontextfeature.')
    write(out/'receipt.json',final);print(final,flush=True)


if __name__=='__main__':main()
