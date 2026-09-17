"""Independent physical q10-reference plus mean delay on non-clock features."""
import numpy as np
import pandas as pd
from common import ID, TARGET, MOVEMENT
from taxiout.schema import PHASE, FLIGHT_ID
from aviation.arrival_features import StandRunwayReference, chronological_reference
import lgbm_adapter


def observations(x):
    frame=x[['ADEP_mvt','RUNWAY_mvt','STAND_mvt']].copy()
    frame[ID]=x.index.to_numpy()
    frame[FLIGHT_ID]=np.nan
    frame[PHASE]='DEP'
    frame[MOVEMENT]=pd.to_datetime(x['__movement_ns'].to_numpy(dtype=np.int64),utc=True)
    return frame.reset_index(drop=True)


def raw_features(x):
    return x.drop(columns='__movement_ns')


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    obs=observations(x)
    labels=pd.DataFrame({ID:x.index,TARGET:y})
    cross=chronological_reference(obs,labels)
    prior=StandRunwayReference().fit(obs,labels)
    features=pd.concat([raw_features(x),cross],axis=1)
    offset=cross.physical_reference_q10_sec.to_numpy(float)
    tune=None
    if tuning is not None:
        tx,ty=tuning
        tprior=prior.transform(observations(tx))
        tune=(pd.concat([raw_features(tx),tprior],axis=1),np.asarray(ty)-tprior.physical_reference_q10_sec.to_numpy(float))
    estimator,evidence=lgbm_adapter.fit(features,np.asarray(y)-offset,tune,steps=steps,seed=seed,threads=threads)
    evidence.update(target='raw taxi time minus chronological q10 reference',
        prior='q10 earlier-month labels; Jan no-history fixed900s; min_support100 shrinkage50',
        clocks='excluded all NM/schedule predictors',prior_fit_rows=len(obs))
    return {'estimator':estimator,'prior':prior},evidence


def predict(model,x):
    prior=model['prior'].transform(observations(x))
    return lgbm_adapter.predict(model['estimator'],pd.concat([raw_features(x),prior],axis=1))+prior.physical_reference_q10_sec.to_numpy(float)
