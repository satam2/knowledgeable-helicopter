"""Higher-capacity follow-up for the verified combined-information branch."""
import time
import numpy as np
import lightgbm as lgb
from lgbm_adapter import NativeFrameEncoder

PRESET='leaf63'
PARAMS={
    'leaf63':dict(n_estimators=2500,num_leaves=63,max_depth=-1,learning_rate=.03,min_child_samples=50,reg_lambda=10.),
    'leaf127':dict(n_estimators=3000,num_leaves=127,max_depth=-1,learning_rate=.025,min_child_samples=100,reg_lambda=20.)}


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=2):
    y=np.asarray(y,float)
    assert len(x)==len(y) and np.isfinite(y).all()
    assert steps is None or (steps>=1 and tuning is None)
    started=time.monotonic()
    encoder=NativeFrameEncoder().fit(x)
    params=dict(PARAMS[PRESET])
    if steps is not None:
        params['n_estimators']=int(steps)
    estimator=lgb.LGBMRegressor(**params,objective='regression',n_jobs=threads,random_state=seed,
        verbosity=-1,deterministic=True,force_col_wise=True,colsample_bytree=1.,subsample=1.)
    options={}
    if tuning is not None:
        options={'eval_X':encoder.transform(tuning[0]),'eval_y':np.asarray(tuning[1],float),'eval_metric':'rmse',
                 'callbacks':[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(250)]}
    estimator.fit(encoder.transform(x),y,categorical_feature=list(encoder.maps),**options)
    selected=int(estimator.best_iteration_ or estimator.n_estimators_)
    model={'encoder':encoder,'estimator':estimator,'steps':selected}
    evidence={'steps':selected,'preset':PRESET,'rows':len(x),'features':len(x.columns),
              'runtime_sec':time.monotonic()-started,'params':estimator.get_params(),
              'objective':'Original raw squared loss; no label truncation, weighting, exclusion or transform'}
    return model,evidence


def predict(model,x):
    return np.asarray(model['estimator'].predict(model['encoder'].transform(x),num_iteration=model['steps']),float)
