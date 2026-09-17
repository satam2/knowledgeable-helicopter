"""Direct raw-mixture squared-error boosting without divided pseudo-targets."""
import lightgbm as lgb
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
from lgbm_adapter import NativeFrameEncoder

PARAMS = {'learning_rate':.05,'num_leaves':4,'max_depth':2,'min_data_in_leaf':500,
          'lambda_l2':1.,'verbosity':-1,'num_threads':2,'deterministic':True,'force_col_wise':True,
          'seed':20260916,'feature_pre_filter':False}
TREES = 100


def derivatives(alpha, difference, residual, normalization):
    difference=np.asarray(difference,dtype=np.float64)
    residual=np.asarray(residual,dtype=np.float64)
    alpha=np.asarray(alpha,dtype=np.float64)
    gradient=2*difference*(difference*alpha-residual)/normalization
    hessian=2*difference*difference/normalization
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise FloatingPointError('Nonfinite direct mixture derivatives')
    return gradient,hessian


def fit(frame, labels, first, second, initial_alpha):
    difference=np.asarray(second,dtype=np.float64)-np.asarray(first,dtype=np.float64)
    residual=np.asarray(labels,dtype=np.float64)-np.asarray(first,dtype=np.float64)
    if not np.isfinite(difference).all() or not np.isfinite(residual).all():
        raise ValueError('Nonfinite original mixture inputs')
    nonzero=difference!=0
    if not nonzero.any():
        return {'estimator':None,'encoder':None,'initial_alpha':float(initial_alpha)}, {'constant_only':True,'zero_disagreement_rows':len(frame)}
    normal=float(np.mean(difference[nonzero]**2))
    if not np.isfinite(normal) or normal<=0:
        raise FloatingPointError('Invalid common Hessian normalization; no epsilon clipping permitted')
    encoder=NativeFrameEncoder().fit(frame.iloc[np.flatnonzero(nonzero)])
    d,r=difference[nonzero],residual[nonzero]
    observed=[]

    def objective(alpha,dataset):
        if not observed:
            observed.append(float(np.max(np.abs(alpha-initial_alpha))))
        return derivatives(alpha,d,r,normal)

    dataset=lgb.Dataset(encoder.transform(frame.iloc[np.flatnonzero(nonzero)]),label=r,
                        init_score=np.full(len(r),initial_alpha),categorical_feature=list(encoder.maps),free_raw_data=False)
    estimator=lgb.train({**PARAMS,'objective':objective},dataset,num_boost_round=TREES)
    if observed[0]>1e-12:
        raise AssertionError('LightGBM custom objective did not receive declared initial global alpha')
    return {'estimator':estimator,'encoder':encoder,'initial_alpha':float(initial_alpha)}, {
        'constant_only':False,'zero_disagreement_rows':int((~nonzero).sum()),'fit_rows':len(r),'normalization_mean_d2':normal,
        'first_objective_init_max_delta':observed[0],'trees':estimator.current_iteration(),'declared_trees':TREES,
        'parameters':PARAMS,'raw_labels_unmodified':True}


def predict(model, frame, first, second):
    correction=np.zeros(len(frame)) if model['estimator'] is None else model['estimator'].predict(model['encoder'].transform(frame),num_threads=2)
    unbounded=model['initial_alpha']+correction
    alpha=np.clip(unbounded,0.,1.)
    prediction=np.asarray(first,dtype=np.float64)+alpha*(np.asarray(second,dtype=np.float64)-np.asarray(first,dtype=np.float64))
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite clipped mixture prediction')
    return prediction,alpha,unbounded
