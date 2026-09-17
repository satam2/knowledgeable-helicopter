"""All-departure airport-only features and label-independent support checks."""
import lightgbm
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
import run_missing_models as original
from common import ID,TARGET,MOVEMENT,read_json,write_json,sha256,object_hash,utc_now
from taxiout.schema import PHASE
from taxiout.paths import external_path
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT=external_path(ROOT/'private_runs/breakthrough_20260916/missing/domain_adaptation')
RAW_COLUMNS=[ID,PHASE,MOVEMENT,'SCHED_TIME_UTC_mvt','ADEP_mvt','ADES_mvt',
    'STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt','FLIGHT_RULE_mvt']
TRAFFIC=['dep_prior_15m','dep_prior_60m','arr_prior_15m','arr_prior_60m',
    'observed_window_15m_sec','observed_window_60m_sec','month_edge']


def check_features(x):
    forbidden=[c for c in x if c in [TARGET,'BLOCK_TIME_UTC_mvt','proxy_missing','FLIGHT_ID_mvt']
        or c.endswith('_flt') or c.startswith(('nmid_','batch_','conv_','takeoff_minus_'))]
    if forbidden:
        raise ValueError(f'Forbidden propensity/regression predictors: {forbidden}')


def overlap(x,missing,idx):
    fit=x.iloc[idx['fit']]
    known=fit.loc[~missing[idx['fit']]]
    report={}
    for stage in ['fit','tune']:
        query=x.iloc[idx[stage][missing[idx[stage]]]]
        summary={}
        for keys in [['airport'],['airport','stand'],['airport','flight'],['airport','schedule_bucket']]:
            counts=known.groupby(keys,observed=True,dropna=False).size().rename('known_n').reset_index()
            merged=query[keys].merge(counts,on=keys,how='left',validate='many_to_one')
            support=merged.known_n.fillna(0).to_numpy()
            summary['+'.join(keys)]=dict(rows=len(query),zero_support_n=int((support==0).sum()),
                support_under20_n=int((support<20).sum()),median_known_support=float(np.median(support)))
        report[stage]=dict(missing_rows=len(query),known_fit_rows=len(known),support=summary,
            missing_by_airport=query.airport.value_counts().to_dict())
    return report


def load():
    marker=read_json(OUT/'features_manifest.json')
    assert sha256(OUT/'features.parquet')==marker['feature_sha256']
    for path,digest in marker['source_hashes'].items():
        assert sha256(ROOT/path)==digest
    audit=read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(metadata)==audit['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata)
    x=pd.read_parquet(OUT/'features.parquet').set_index(ID)
    np.testing.assert_array_equal(x.index,meta[ID])
    check_features(x)
    return x,meta,marker


def main():
    if (OUT/'features_manifest.json').exists():
        raise ValueError('Completed airport-only featurecache retained')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=read_json(ROOT/'private_runs/submission_v2/protocol.json')
    base,meta=common.load_data()
    base=base[TRAFFIC]
    parts=[]
    for path in sorted(common.RAW.glob('training*.parquet')):
        assert sha256(path)==frozen['raw_hashes'][path.name]
        raw=pq.read_table(path,columns=RAW_COLUMNS,use_threads=False).to_pandas()
        frame=original.airport_features(raw.loc[raw[PHASE].eq('DEP')])
        parts.append(frame)
    x=pd.concat(parts).loc[meta[ID]]
    np.testing.assert_array_equal(x.index,base.index)
    x=pd.concat([x,base],axis=1)
    for column in x.select_dtypes(['object','string']):
        x[column]=x[column].astype('category')
    check_features(x)
    x.reset_index().to_parquet(OUT/'features.parquet',index=False)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    feasibility={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        feasibility[fold]=dict(split=split,overlap=overlap(x,missing,idx))
    marker=dict(created_utc=utc_now(),rows=len(x),features=list(x),raw_columns=RAW_COLUMNS,
        feature_sha256=sha256(OUT/'features.parquet'),original_id_order_verified=True,
        source_hashes={str(path.relative_to(ROOT)):sha256(path) for path in [Path(__file__),Path(original.__file__),Path(common.__file__)]},
        availability='Ownairport record schedule/calendar/categories plus existing strictprior15/60min movementcounts. NoNM clocks, ID, availabilityflag, target or BLOCK inputs.',
        label_use='No training model launched; missingavailability only for support diagnostics; noY calculations.',
        feasibility=feasibility)
    write_json(OUT/'features_manifest.json',marker)
    print('PREPARED',x.shape,{f:v['overlap'] for f,v in feasibility.items()},flush=True)


if __name__=='__main__':
    main()
