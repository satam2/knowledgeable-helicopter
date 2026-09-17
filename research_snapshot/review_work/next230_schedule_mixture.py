"""Learn schedule reliability on fit labels, then route by observed probabilities."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from next230_common import OUT, config_for, load_data, load_reference, load_extension, fit, start_run, finish_run
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import TARGET

CONFIG = {**config_for('baseline'), 'task_type': 'CPU', 'iterations': 1200, 'depth': 6,
          'learning_rate': .05, 'l2_leaf_reg': 20, 'classifier_depth': 4,
          'classifier_iterations': 800, 'schedule_consistent_threshold_sec': 60,
          'seed': 20260910, 'train_sample': None}


def predict_mixture(offset, probability, consistent_mean, inconsistent_correction):
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid schedule reliability probability')
    return np.asarray(offset) + probability * consistent_mean + (1-probability) * inconsistent_correction


def train(fold, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    path, record, reused = start_run('rome_schedule_mixture', fold, CONFIG['seed'], CONFIG, reference, split)
    if reused:
        return
    x = load_extension(x, 'schedule')
    target = labels[TARGET].to_numpy(float) - meta.schedule_sec.to_numpy(float)
    eligible = (proxy_status(meta.proxy_sec, CONFIG) == 'missing') & np.isfinite(meta.schedule_sec)
    sf, st, sr = [p[eligible[p]] for p in [idx['fit'], idx['tune'], idx['refit']]]
    consistent = np.abs(target) <= CONFIG['schedule_consistent_threshold_sec']
    cat = list(x.select_dtypes('category').columns)
    params = dict(iterations=CONFIG['classifier_iterations'], depth=CONFIG['classifier_depth'],
                  learning_rate=.05, l2_leaf_reg=20, loss_function='Logloss', random_seed=CONFIG['seed'],
                  thread_count=4, allow_writing_files=False, verbose=200)
    classifier = CatBoostClassifier(**params)
    classifier.fit(x.iloc[sf], consistent[sf].astype(int), cat_features=cat,
                   eval_set=(x.iloc[st], consistent[st].astype(int)), early_stopping_rounds=80, use_best_model=True)
    class_trees = classifier.tree_count_
    bad_fit, bad_tune, bad_refit = [p[~consistent[p]] for p in [sf, st, sr]]
    regressor, tune_evidence = fit(x.iloc[bad_fit], target[bad_fit], CONFIG,
                                 tuning=(x.iloc[bad_tune], target[bad_tune]))
    reg_trees = regressor.tree_count_
    del classifier, regressor
    gc.collect()
    params['iterations'] = class_trees
    classifier = CatBoostClassifier(**params)
    classifier.fit(x.iloc[sr], consistent[sr].astype(int), cat_features=cat)
    regressor, refit_evidence = fit(x.iloc[bad_refit], target[bad_refit], CONFIG, trees=reg_trees)
    good_mean = float(target[sr][consistent[sr]].mean())
    sx = x.iloc[idx['score']]
    probability = classifier.predict_proba(sx, thread_count=4)[:, 1]
    correction = regressor.predict(sx, thread_count=4)
    offset = meta.iloc[idx['score']].schedule_sec.to_numpy()
    values = predict_mixture(offset, probability, good_mean, correction)
    classifier.save_model(str(path / 'reliability.cbm'))
    regressor.save_model(str(path / 'inconsistent.cbm'))
    saved_c, saved_r = CatBoostClassifier(), CatBoostRegressor()
    saved_c.load_model(str(path / 'reliability.cbm'))
    saved_r.load_model(str(path / 'inconsistent.cbm'))
    repeated = predict_mixture(offset, saved_c.predict_proba(sx, thread_count=4)[:, 1], good_mean,
                               saved_r.predict(sx, thread_count=4))
    delta = float(np.max(np.abs(repeated-values)))
    if delta > 1e-9:
        raise ValueError('Mixture serialization failure')
    changed = baseline.route.eq('rome_schedule_residual').to_numpy()
    predictions = baseline.copy()
    predictions.loc[changed, 'prediction_sec'] = values[changed]
    predictions['schedule_consistency_probability'] = probability
    write_json(path / 'mixture.json', {'consistent_mean': good_mean, 'columns': list(x),
        'dtypes': {c: str(x[c].dtype) for c in x}, 'feature_block': 'schedule',
        'threshold_training_only': CONFIG['schedule_consistent_threshold_sec']})
    finish_run(path, record, baseline, predictions, labels.iloc[idx['score']], changed,
        {'mixture': True, 'runner_sha256': sha256(__file__),
         'supplemental_protocol_sha256': sha256(OUT/'schedule_mixture_protocol.json'),
         'trees': reg_trees, 'classifier_trees': class_trees, 'consistent_mean_sec': good_mean,
         'training': {'tune': tune_evidence, 'refit': refit_evidence},
         'fit_id_hash': object_hash(x.index[sf].tolist()), 'tune_id_hash': object_hash(x.index[st].tolist()),
         'refit_id_hash': object_hash(x.index[sr].tolist()),
         'fit_consistent_rows': int(consistent[sf].sum()), 'fit_inconsistent_rows': len(bad_fit),
         'reload_max_abs_delta': delta})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    args = p.parse_args()
    protocol = {'config': CONFIG, 'runner_sha256': sha256(__file__),
                'rationale': 'Fit-period Rome missing residuals cluster near zero and hours apart; learn a soft observable reliability mixture.',
                'selection': 'Adaptive follow-up after Rome context failed; no score labels used to train, calibrate or route.'}
    file = OUT / 'schedule_mixture_protocol.json'
    if file.exists() and read_json(file) != protocol:
        raise ValueError('Mixture protocol changed')
    write_json(file, protocol)
    data = load_data()
    for fold in args.folds:
        train(fold, data)
