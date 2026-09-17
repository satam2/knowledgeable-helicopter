"""Observed flight identity and schedule context for finite-source experts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'campaign_20260916'))
import common
from taxiout.schema import MOVEMENT,PHASE
from taxiout.features.pipeline import token

OUT=common.external_path(common.WORKSPACE/'private_runs/breakthrough_20260916/flight_identity')
COLS=[common.ID,PHASE,MOVEMENT,'FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','SCHED_TIME_UTC_mvt']


def transform(raw):
    dep=raw.loc[raw[PHASE].eq('DEP')].reset_index(drop=True)
    out=dep[[common.ID]].copy()
    flight=dep.FLIGHT_mvt.astype('string')
    callsign=dep.CALLSIGN_flt.astype('string')
    airport=token(dep.ADEP_mvt)
    for key,values in [('flight',flight),('callsign',callsign)]:
        prefix=values.str.extract(r'^([A-Za-z]{2,3})(?=[0-9])',expand=False)
        out['identity_'+key]=token(values).astype('category')
        out['identity_airport_'+key]=(airport+token(values)).astype('category')
        out['identity_'+key+'_prefix']=token(prefix).astype('category')
        out['identity_airport_'+key+'_prefix']=(airport+token(prefix)).astype('category')
        out['identity_'+key+'_length']=values.str.len().astype('float32')
    out['identity_exact_callsign_flight_match']=(flight.eq(callsign)&flight.notna()&callsign.notna()).fillna(False).astype('float32')
    schedule=pd.to_datetime(dep.SCHED_TIME_UTC_mvt,utc=True)
    movement=pd.to_datetime(dep[MOVEMENT],utc=True)
    for name in ['hour','minute','second','dayofweek']:
        out['identity_schedule_'+name]=getattr(schedule.dt,name).astype('float32')
    out['identity_schedule_day_offset']=(movement.dt.normalize()-schedule.dt.normalize()).dt.total_seconds().div(86400).astype('float32')
    return out


if __name__=='__main__':
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    frozen=common.read_json(common.WORKSPACE/'private_runs/submission_v2/protocol.json')
    frames=[]
    for path in sorted(common.RAW.glob('training_*.parquet')):
        assert common.sha256(path)==frozen['raw_hashes'][path.name]
        frames.append(transform(pd.read_parquet(path,columns=COLS)))
    training=pd.concat(frames,ignore_index=True)
    for column in training.select_dtypes('object'):
        training[column]=training[column].astype('category')
    ids=pd.read_parquet(common.WORKSPACE/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[common.ID])
    training=training.set_index(common.ID).loc[ids[common.ID]].reset_index()
    assert np.array_equal(training[common.ID],ids[common.ID])
    ranking_path=common.RAW/'ranking.parquet'
    assert common.sha256(ranking_path)==frozen['raw_hashes'][ranking_path.name]
    ranking=transform(pd.read_parquet(ranking_path,columns=COLS))
    training.to_parquet(OUT/'training_features.parquet',index=False)
    ranking.to_parquet(OUT/'ranking_features.parquet',index=False)
    common.write_json(OUT/'manifest.json',{'status':'complete','script_sha256':common.sha256(__file__),
        'created_utc':common.utc_now(),'rows':len(training),'ranking_rows':len(ranking),'columns':list(training),
        'hidden_columns_loaded':[],'raw_hashes':frozen['raw_hashes'],
        'categorical_values':'Observed identity strings only; no frequency/label/category mapping fitted on ranking or score.',
        'novelty':'Flight/schedule fields previously used in Rome missing branch; this tests all finite-NM source residuals and adds unused callsign identity.',
        'outputs':{p.name:common.sha256(p) for p in OUT.glob('*.parquet')}})
    print('IDENTITY_CACHE',training.shape,ranking.shape,flush=True)
