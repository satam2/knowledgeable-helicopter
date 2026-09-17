"""Chronological availability-propensity weights and raw-label mean regression."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from lgbm_adapter import NativeFrameEncoder

RATIOS=[0.,.25,1.,4.]
CAP=20.
COLD_PRIOR=.01


def propensity_params(seed=20260916,threads=2):
    return dict(n_estimators=150,learning_rate=.05,num_leaves=15,max_depth=5,
        min_child_samples=100,reg_lambda=10.,objective='binary',n_jobs=threads,
        random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)


def regression_params(seed=20260916,threads=2,steps=600):
    return dict(n_estimators=steps,learning_rate=.05,num_leaves=31,max_depth=6,
        min_child_samples=30,reg_lambda=30.,objective='regression',n_jobs=threads,
        random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)


def fit_propensity(x,missing,seed=20260916,threads=2,params=None):
    missing=np.asarray(missing,bool)
    if len(missing)!=len(x) or not len(x):
        raise ValueError('Availability labels must match nonempty fit rows')
    encoder=NativeFrameEncoder().fit(x)
    if len(np.unique(missing))==1:
        return dict(encoder=encoder,constant=float(missing.mean()),rows=len(x))
    estimator=lgb.LGBMClassifier(**(params or propensity_params(seed,threads)))
    estimator.fit(encoder.transform(x),missing.astype(int),categorical_feature=list(encoder.maps))
    return dict(encoder=encoder,estimator=estimator,rows=len(x))


def predict_propensity(model,x):
    if 'constant' in model:
        return np.full(len(x),model['constant'])
    return np.asarray(model['estimator'].predict_proba(model['encoder'].transform(x))[:,1],float)


def crossfit(x,missing,timestamps,seed=20260916,threads=2,fit_function=None):
    times=pd.to_datetime(timestamps,utc=True)
    months=times.year*100+times.month if isinstance(times,pd.DatetimeIndex) else times.dt.year*100+times.dt.month
    months=np.asarray(months)
    missing=np.asarray(missing,bool)
    probabilities=np.empty(len(x),float)
    records=[]
    fitting=fit_function or (lambda fx,fm:fit_propensity(fx,fm,seed,threads))
    for month in np.unique(months):
        before=months<month
        query=months==month
        if not before.any():
            probabilities[query]=COLD_PRIOR
            records.append(dict(month=int(month),history_rows=0,query_rows=int(query.sum()),cold_prior=COLD_PRIOR))
        else:
            model=fitting(x.loc[before],missing[before])
            probabilities[query]=predict_propensity(model,x.loc[query])
            records.append(dict(month=int(month),history_rows=int(before.sum()),query_rows=int(query.sum()),
                history_missing=int(missing[before].sum()),max_history_month=int(months[before].max())))
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite crossfit propensity')
    return probabilities,records


def weights(probabilities,missing,ratio):
    probabilities=np.asarray(probabilities,float)
    missing=np.asarray(missing,bool)
    if len(probabilities)!=len(missing) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Finite probabilities in[0,1] required')
    if ratio not in RATIOS or not missing.any():
        raise ValueError('Supportedratio and nonempty missingcohort required')
    result=missing.astype(float)
    if ratio==0:
        return result,dict(ratio=0.,missing_total=float(missing.sum()),known_total=0.,known_ess=0.,known_capped=0)
    p=np.clip(probabilities[~missing],1e-6,1-1e-6)
    odds=np.minimum(CAP,p/(1-p))
    target=ratio*missing.sum()
    if target>CAP*len(odds):
        raise ValueError('Requestedmass exceeds cappedknown support')
    lower,upper=0.,max(1.,target/odds.sum())
    while np.minimum(CAP,upper*odds).sum()<target:
        upper*=2
    for _ in range(70):
        middle=(lower+upper)/2
        if np.minimum(CAP,middle*odds).sum()<target:
            lower=middle
        else:
            upper=middle
    known=np.minimum(CAP,upper*odds)
    result[~missing]=known
    if not np.allclose(known.sum(),target,rtol=1e-10,atol=1e-8):
        raise ValueError('Weightmass normalization failed')
    return result,dict(ratio=ratio,missing_total=float(missing.sum()),known_total=float(known.sum()),
        known_ess=float(known.sum()**2/(known@known)),known_capped=int((known>=CAP-1e-9).sum()),
        known_max=float(known.max()),known_median=float(np.median(known)),odds_scale=float(upper),
        odds_cap=CAP,final_weight_cap=CAP)


def overlap(probabilities,missing):
    p=np.asarray(probabilities,float)
    missing=np.asarray(missing,bool)
    report={name:dict(n=int(mask.sum()),quantiles=np.quantile(p[mask],[0,.01,.1,.5,.9,.99,1]).tolist())
        for name,mask in [('missing',missing),('known',~missing)] if mask.any()}
    report['brier']=float(brier_score_loss(missing,p))
    if len(np.unique(missing))==2:
        report['auc']=float(roc_auc_score(missing,p))
        q=np.quantile(p[~missing],[.01,.99])
        report['missing_outside_known_central98_pct']=float(np.mean((p[missing]<q[0])|(p[missing]>q[1]))*100)
    return report


def fit_regression(x,y,sample_weight,tuning=None,steps=None,seed=20260916,threads=2):
    y=np.asarray(y,float)
    weight=np.asarray(sample_weight,float)
    if len(y)!=len(x) or len(weight)!=len(x) or not np.isfinite(y).all() or not np.isfinite(weight).all() or np.any(weight<0):
        raise ValueError('Alignedfinite unalteredlabels andnonnegativeweights required')
    keep=weight>0
    encoder=NativeFrameEncoder().fit(x.loc[keep])
    model=lgb.LGBMRegressor(**regression_params(seed,threads,steps or 600))
    options={}
    if tuning is not None:
        if steps is not None:
            raise ValueError('Refit must not tune')
        tx,ty=tuning
        options=dict(eval_X=encoder.transform(tx),eval_y=np.asarray(ty,float),eval_metric='rmse',
            callbacks=[lgb.early_stopping(60,verbose=False)])
    model.fit(encoder.transform(x.loc[keep]),y[keep],sample_weight=weight[keep],
        categorical_feature=list(encoder.maps),**options)
    selected=int(model.best_iteration_ or model.n_estimators_)
    return dict(estimator=model,encoder=encoder,steps=selected),dict(steps=selected,rows=int(keep.sum()),
        original_rows=len(x),weight_sum=float(weight.sum()),params=model.get_params())


def predict_regression(model,x):
    return np.asarray(model['estimator'].predict(model['encoder'].transform(x),num_iteration=model['steps']),float)
