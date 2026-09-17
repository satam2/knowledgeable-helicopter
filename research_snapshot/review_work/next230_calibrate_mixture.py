"""Tune-only RMSE calibration of a soft schedule reliability probability."""

import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import least_squares

from next230_common import OUT, load_data, load_reference, load_extension, fit, start_run, finish_run, same_split
from next230_schedule_mixture import CONFIG, predict_mixture
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import ID, TARGET


def calibrate_probability(p, params):
    return np.clip(params['scale'] * np.asarray(p) + params['shift'], 0, 1)


def fit_probability_calibration(p, offset, good_mean, bad, y):
    arrays = [np.asarray(a, float) for a in [p, offset, bad, y]]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite calibration inputs')
    p, offset, bad, y = arrays
    if len(y) < 30:
        return {'scale': 1., 'shift': 0., 'insufficient_support': True, 'n': len(y)}
    # Fixed ridge keeps weakly identified scale/shift near the identity map.
    def residual(theta):
        probability = np.clip(theta[0]*p+theta[1], 0, 1)
        error = predict_mixture(offset, probability, good_mean, bad)-y
        return np.r_[error/3000, np.sqrt(10)*(theta[0]-1), np.sqrt(10)*theta[1]]
    solution = least_squares(residual, [1., 0.], bounds=([0., -.5], [2., .5]), max_nfev=1000)
    if not solution.success:
        raise ValueError('Probability calibration did not converge')
    return {'scale': float(solution.x[0]), 'shift': float(solution.x[1]), 'n': len(y),
            'insufficient_support': False, 'ridge_strength': 10, 'residual_unit_sec': 3000}


def execute(fold, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    mixture_path = OUT/'models'/f'rome_schedule_mixture_{fold}_s20260910'
    mixture = read_json(mixture_path/'manifest.json')
    if mixture['status'] != 'complete' or not same_split(mixture['split'], split):
        raise ValueError('Uncalibrated mixture must be complete')
    for file, digest in mixture['outputs'].items():
        if sha256(mixture_path/file) != digest:
            raise ValueError('Mixture source corrupt')
    name = 'rome_mixture_calibrated'
    path, record, reused = start_run(name, fold, 20260910, CONFIG, reference, split)
    if reused:
        return
    x = load_extension(x, 'schedule')
    y = labels[TARGET].to_numpy(float)
    offset = meta.schedule_sec.to_numpy(float)
    target = y-offset
    status = proxy_status(meta.proxy_sec, CONFIG)
    eligible = (status == 'missing') & np.isfinite(offset)
    sf = idx['fit'][eligible[idx['fit']]]
    st = idx['tune'][eligible[idx['tune']] & meta.iloc[idx['tune']].ADEP_mvt.eq('LIRF').to_numpy()]
    consistent = np.abs(target) <= CONFIG['schedule_consistent_threshold_sec']
    classifier = CatBoostClassifier(iterations=mixture['classifier_trees'], depth=CONFIG['classifier_depth'],
        learning_rate=.05, l2_leaf_reg=20, loss_function='Logloss', random_seed=20260910,
        thread_count=4, allow_writing_files=False, verbose=200)
    classifier.fit(x.iloc[sf], consistent[sf].astype(int), cat_features=list(x.select_dtypes('category').columns))
    regressor, training = fit(x.iloc[sf[~consistent[sf]]], target[sf[~consistent[sf]]], CONFIG, trees=mixture['trees'])
    good_mean = float(target[sf][consistent[sf]].mean())
    p = classifier.predict_proba(x.iloc[st], thread_count=4)[:,1]
    bad = regressor.predict(x.iloc[st], thread_count=4)
    params = fit_probability_calibration(p, offset[st], good_mean, bad, y[st])
    classifier.save_model(str(path/'fit_only_reliability.cbm'))
    regressor.save_model(str(path/'fit_only_inconsistent.cbm'))
    pd.DataFrame({ID:x.index[st], TARGET:y[st], 'probability':p, 'offset':offset[st],
                  'bad_correction':bad}).to_parquet(path/'calibration_predictions.parquet',index=False)
    write_json(path/'calibration.json',params)
    classifier.load_model(str(mixture_path/'reliability.cbm'))
    regressor.load_model(str(mixture_path/'inconsistent.cbm'))
    schema = read_json(mixture_path/'mixture.json')
    sx=x.iloc[idx['score']]
    calibrated = calibrate_probability(classifier.predict_proba(sx,thread_count=4)[:,1],params)
    predictions=baseline.copy()
    changed=baseline.route.eq('rome_schedule_residual').to_numpy()
    values=predict_mixture(offset[idx['score']],calibrated,schema['consistent_mean'],regressor.predict(sx,thread_count=4))
    predictions.loc[changed,'prediction_sec']=values[changed]
    predictions['schedule_consistency_probability']=calibrated
    finish_run(path,record,baseline,predictions,labels.iloc[idx['score']],changed,
        {'calibrated_mixture_path':str(mixture_path),'mixture_manifest_sha256':sha256(mixture_path/'manifest.json'),
         'calibration':params,'fit_consistent_mean_sec':good_mean,'runner_sha256':sha256(__file__),
         'supplemental_protocol_sha256':sha256(OUT/'mixture_calibration_protocol.json'),
         'calibration_id_hash':object_hash(x.index[st].tolist()),'training':training,'reload_max_abs_delta':0.})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',default=['F1','F3'])
    args=parser.parse_args()
    file=OUT/'mixture_calibration_protocol.json'
    protocol={'runner_sha256':sha256(__file__),'bounds':{'scale':[0,2],'shift':[-.5,.5]},
              'ridge_strength':10,'residual_unit_sec':3000,'minimum_tune_rome_rows':30,
              'selection':'Declared after uncalibrated mixture F1/F3; optimize tuning Rome RMSE only; no score labels used.'}
    if file.exists() and read_json(file)!=protocol:
        raise ValueError('Calibration protocol changed')
    write_json(file,protocol)
    data=load_data()
    for fold in args.folds:
        execute(fold,data)
