"""Independent raw-whitelist enumeration of every frozen exact-arrival link."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TIME=common.ID,common.MOVEMENT
FLIGHT='FLIGHT_ID_mvt'
ORIGIN,DEST='ADEP_mvt','ADES_mvt'
BASE=ROOT/'private_runs/tail240_20260916/state/exact_arrival/v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/exact_arrival_joins_v1'
ARR=[ID,FLIGHT,TIME,'PHASE_mvt',ORIGIN,DEST]
DEP=ARR+['ARVT_3_flt','AIRCRAFT_OPERATOR_flt']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    info=psutil.Process().memory_info()
    peak=max(info.rss,getattr(info,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def raw_month(path,phase,start,end):
    data=ds.dataset(path,format='parquet')
    dtype=data.schema.field(TIME).type
    predicate=(ds.field('PHASE_mvt')==phase)&(ds.field(TIME)>=pa.scalar(pd.Timestamp(start,tz='UTC').to_pydatetime(),type=dtype))&(ds.field(TIME)<pa.scalar(pd.Timestamp(end,tz='UTC').to_pydatetime(),type=dtype))
    return data.to_table(columns=ARR if phase=='ARR' else DEP,filter=predicate,use_threads=False).to_pandas()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=common.read_json(BASE/'protocol.json')
    manifest=common.read_json(BASE/'join_manifest.json')
    assert manifest['status']=='complete' and manifest['labels_loaded'] is False
    assert manifest['protocol_sha256']==common.sha256(BASE/'protocol.json')
    for name,digest in protocol['source_hashes'].items():assert common.sha256(ROOT/name)==digest
    for name,digest in manifest['outputs'].items():assert common.sha256(BASE/name)==digest
    raw=sorted(common.RAW.glob('training_*.parquet'))
    assert set(p.name for p in raw)==set(protocol['raw_hashes'])
    for path in raw:assert common.sha256(path)==protocol['raw_hashes'][path.name]
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==protocol['metadata_sha256']
    meta=pd.read_parquet(meta_path,columns=[ID,TIME,'proxy_sec']).set_index(ID)
    checks={}
    for fold,(start,end) in protocol['folds'].items():
        saved=pd.read_parquet(BASE/f'{fold}_join.parquet')
        control=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        assert common.sha256(control/'tune_predictions.parquet')==protocol['controls'][fold]['predictions_sha256']
        ids=pd.read_parquet(control/'tune_predictions.parquet',columns=[ID])[ID]
        original=meta.loc[ids]
        expected=original.loc[original.proxy_sec.between(0,7200)].index
        np.testing.assert_array_equal(saved[ID],expected)
        dep=pd.concat([raw_month(p,'DEP',start,end) for p in raw],ignore_index=True).drop_duplicates()
        arr=pd.concat([raw_month(p,'ARR',start,end) for p in raw],ignore_index=True).drop_duplicates()
        assert dep[ID].is_unique and arr[ID].is_unique
        dep=dep.set_index(ID).loc[expected].reset_index()
        pd.testing.assert_frame_equal(saved[DEP[:3]+[ORIGIN,DEST,'AIRCRAFT_OPERATOR_flt','ARVT_3_flt']].reset_index(drop=True),dep[DEP[:3]+[ORIGIN,DEST,'AIRCRAFT_OPERATOR_flt','ARVT_3_flt']].reset_index(drop=True),check_dtype=False)
        # Independent tuple dictionary: physical duplicates share one key; distinct keys cause ambiguity.
        groups={}
        for identity,fid,landing,phase,origin,dest in arr.itertuples(index=False,name=None):
            if pd.isna(fid):continue
            key=(origin if pd.notna(origin) else None,dest if pd.notna(dest) else None,landing)
            physical=groups.setdefault(int(fid),{})
            physical[key]=min(identity,physical.get(key,identity))
        statuses=[];deltas=[];counts=[]
        for row in dep.itertuples(index=False,name=None):
            identity,fid,when,phase,origin,dest,arvt,operator=row
            matches=groups.get(int(fid),{}) if pd.notna(fid) else {}
            counts.append(len(matches));delta=np.nan
            if pd.isna(fid):status='missing_flight'
            elif not matches:status='no_arrival_in_month'
            elif len(matches)>1:status='ambiguous_arrival'
            else:
                (aorigin,adest,landing),aid=next(iter(matches.items()))
                if any(pd.isna(x) or not str(x).strip() for x in [origin,dest,aorigin,adest]) or origin!=aorigin or dest!=adest:status='route_mismatch'
                elif not landing>when:status='not_strictly_later'
                elif aid==identity:status='same_movement'
                elif pd.isna(arvt):status='missing_own_arvt'
                else:status='linked';delta=(arvt-landing).total_seconds()
            statuses.append(status);deltas.append(delta)
        np.testing.assert_array_equal(saved.status,statuses)
        np.testing.assert_array_equal(saved.arrival_count,counts)
        np.testing.assert_array_equal(saved.linked,np.asarray(statuses)=='linked')
        np.testing.assert_allclose(saved.delta_sec,deltas,rtol=0,atol=0,equal_nan=True)
        assert saved.status.value_counts().to_dict()==manifest['folds'][fold]['status_counts']
        checks[fold]=dict(rows=len(saved),linked=int(saved.linked.sum()),all_statuses_counts_deltas_exact=True,original_ordinary_cohort_exact=True)
        print('VERIFIED_JOINS',fold,checks[fold],flush=True)
        guard()
    common.write_json(OUT/'receipt.json',dict(status='passed',source_sha256=common.sha256(__file__),join_manifest_sha256=common.sha256(BASE/'join_manifest.json'),protocol_sha256=common.sha256(BASE/'protocol.json'),folds=checks,labels_loaded=False,forbidden_columns_projected=False,peak_bytes=guard()))


if __name__=='__main__':main()
