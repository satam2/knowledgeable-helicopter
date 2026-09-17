"""Full raw-label physical reference-plus-conditional-mean model."""
import lightgbm as lgb
import time
import numpy as np
import pandas as pd

from domain_adapter import observations
from aviation.arrival_features import StandRunwayReference, chronological_reference
from lgbm_adapter import NativeFrameEncoder
from common import ID, TARGET


def parameters(seed=20260916, threads=2, steps=600):
    return dict(n_estimators=steps, learning_rate=.05, num_leaves=31, max_depth=8,
        objective='regression', n_jobs=threads, random_state=seed, verbosity=-1,
        deterministic=True, force_col_wise=True, reg_lambda=5., min_child_samples=30,
        subsample=1., colsample_bytree=1.)


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
    started = time.monotonic()
    y = np.asarray(y, dtype=float)
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError('Complete finite raw labels required')
    if steps is not None and (steps < 1 or tuning is not None):
        raise ValueError('Refit requires positive steps and no tuning')
    obs = observations(x)
    labels = pd.DataFrame({ID: x.index, TARGET: y})
    history = chronological_reference(obs, labels)
    prior = StandRunwayReference().fit(obs, labels)
    features = pd.concat([x.drop(columns='__movement_ns'), history], axis=1)
    offset = history.physical_reference_q10_sec.to_numpy(float)
    encoder = NativeFrameEncoder().fit(features)
    estimator = lgb.LGBMRegressor(**parameters(seed, threads, steps or 600))
    options = {}
    if tuning is not None:
        tx, ty = tuning
        reference = prior.transform(observations(tx))
        tf = pd.concat([tx.drop(columns='__movement_ns'), reference], axis=1)
        options = dict(eval_X=encoder.transform(tf), eval_y=np.asarray(ty, float) - reference.physical_reference_q10_sec.to_numpy(float),
            eval_metric='rmse', callbacks=[lgb.early_stopping(60, verbose=False)])
    estimator.fit(encoder.transform(features), y - offset,
        categorical_feature=list(encoder.maps), **options)
    selected = int(estimator.best_iteration_ or estimator.n_estimators_)
    model = dict(prior=prior, estimator=estimator, encoder=encoder, steps=selected)
    return model, dict(steps=selected, rows=len(x), features=len(features.columns),
        seconds=time.monotonic()-started, params=estimator.get_params(), lightgbm_version=lgb.__version__,
        target='Raw Y minus earlier-month stand/runway q10; no clipping or exclusions')


def predict(model, x):
    reference = model['prior'].transform(observations(x))
    features = pd.concat([x.drop(columns='__movement_ns'), reference], axis=1)
    residual = model['estimator'].predict(model['encoder'].transform(features), num_iteration=model['steps'])
    return np.asarray(residual, float) + reference.physical_reference_q10_sec.to_numpy(float)
