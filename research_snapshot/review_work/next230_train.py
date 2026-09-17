"""Candidate component training with untouched scoring folds and protected routes."""

import argparse
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from next230_common import (OUT, OLD, RECIPES, config_for, load_data, load_reference,
    load_extension, fit, start_run, finish_run, reload_parity, same_split)
from next230_features import designator_support, known_designator, fit_blend, blend_weights, apply_blend
from taxiout.artifacts import object_hash, read_json, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import ID, TARGET


def recipe(name, seed):
    return {**config_for('baseline'), 'task_type': 'GPU', 'learning_rate': .05,
            'depth': 8, 'iterations': 2500, 'seed': seed, 'feature_block': RECIPES[name].get('features', 'base'),
            **{k: v for k, v in RECIPES[name].items() if k != 'features'}}


def train_component(name, fold, seed, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    config = recipe(name, seed)
    path, record, reused = start_run(name, fold, seed, config, reference, split)
    if reused:
        return
    if config['feature_block'] != 'base':
        x = load_extension(x, config['feature_block'])
    status = proxy_status(meta.proxy_sec, config)
    rome = name.startswith('rome_')
    mask = ((status == 'missing') & np.isfinite(meta.schedule_sec)) if rome else status == 'present'
    sf, st, sr = [p[mask[p]] for p in [idx['fit'], idx['tune'], idx['refit']]]
    offset = meta.schedule_sec.to_numpy() if rome else meta.proxy_sec.to_numpy()
    target = labels[TARGET].to_numpy(float) - offset
    print(f'TRAIN {path.name}: fit={len(sf)}, tune={len(st)}, refit={len(sr)}', flush=True)
    tuned, te = fit(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
    trees = tuned.tree_count_
    tune_predictions = pd.DataFrame({ID: x.index[st], 'target': target[st], 'prediction': tuned.predict(x.iloc[st], thread_count=4)})
    tune_predictions.to_parquet(path / 'tune_predictions.parquet', index=False)
    del tuned
    gc.collect()
    model, re = fit(x.iloc[sr], target[sr], config, trees=trees)
    score_x = x.iloc[idx['score']]
    changed = baseline.route.eq('rome_schedule_residual').to_numpy() if rome else baseline.route.isin(['residual', 'residual_long_proxy']).to_numpy()
    values, delta = reload_parity(model, path / 'component.cbm', score_x)
    predicted = baseline.copy()
    predicted.loc[changed, 'prediction_sec'] = offset[idx['score']][changed] + values[changed]
    support = {c: set(x.iloc[sr][c].astype(str)) for c in x.select_dtypes('category')}
    unknown = np.zeros(len(score_x), dtype=bool)
    for col, known in support.items():
        unknown |= ~score_x[col].astype(str).isin(known).to_numpy()
    predicted['component_unseen_category'] = unknown
    schema = {'columns': list(x), 'dtypes': {c: str(x[c].dtype) for c in x},
              'vocabulary': {k: sorted(v) for k, v in support.items()},
              'base_bundle': str(OLD / 'models' / reference['run_id']), 'component_route': 'rome' if rome else 'residual',
              'feature_block': config['feature_block']}
    if config['feature_block'] == 'schedule':
        fallback_path = OUT / 'models' / f'rome_base_1200_{fold}_s{seed}'
        fallback_record = read_json(fallback_path / 'manifest.json')
        if fallback_record['status'] != 'complete' or not same_split(fallback_record['split'], split):
            raise ValueError('Context fallback not complete or incompatible')
        fallback = CatBoostRegressor()
        fallback.load_model(str(fallback_path / 'component.cbm'))
        known = known_designator(score_x, designator_support(x.iloc[sr]))
        fallback_values = fallback.predict(data[0].iloc[idx['score']], thread_count=4)
        use_fallback = changed & ~known
        predicted.loc[use_fallback, 'prediction_sec'] = offset[idx['score']][use_fallback] + fallback_values[use_fallback]
        predicted['component_known_flight'] = known
        schema.update(designator_support=designator_support(x.iloc[sr]), fallback=str(fallback_path),
                      unseen_fallback_rows=int(use_fallback.sum()))
    write_json(path / 'component_schema.json', schema)
    write_json(path / 'feature_importance.json', dict(zip(x.columns, model.feature_importances_.tolist())))
    finish_run(path, record, baseline, predicted, labels.iloc[idx['score']], changed,
        {'training': {'tune': te, 'refit': re}, 'trees': trees, 'reload_max_abs_delta': delta,
         'fit_id_hash': object_hash(x.index[sf].tolist()), 'tune_id_hash': object_hash(x.index[st].tolist()),
         'refit_id_hash': object_hash(x.index[sr].tolist())})


def train_clocks(fold, seed, data):
    x, meta, labels = data
    reference, baseline, idx, split = load_reference(fold, meta)
    config = recipe('clock_experts', seed)
    path, record, reused = start_run('clock_experts', fold, seed, config, reference, split)
    if reused:
        return
    status = proxy_status(meta.proxy_sec, config)
    sf, st, sr = [p[status[p] == 'present'] for p in [idx['fit'], idx['tune'], idx['refit']]]
    y, offset = labels[TARGET].to_numpy(float), meta.proxy_sec.to_numpy(float)
    # The existing saved residual has seen tune labels; reconstruct its fit-period expert.
    residual, residual_evidence = fit(x.iloc[sf], y[sf] - offset[sf], config, trees=reference['iterations']['residual'])
    residual.save_model(str(path / 'fit_only_residual.cbm'))
    residual_tune = offset[st] + residual.predict(x.iloc[st], thread_count=4)
    del residual
    gc.collect()
    direct, direct_tune_evidence = fit(x.iloc[sf], y[sf], config, tuning=(x.iloc[st], y[st]))
    trees = direct.tree_count_
    direct.save_model(str(path / 'fit_only_direct.cbm'))
    direct_tune = direct.predict(x.iloc[st], thread_count=4)
    gate = fit_blend(y[st], residual_tune, direct_tune, meta.iloc[st].ADEP_mvt.to_numpy(), shrinkage=1000)
    write_json(path / 'gate.json', gate)
    pd.DataFrame({ID: x.index[st], TARGET: y[st], 'residual': residual_tune, 'direct': direct_tune,
                  'ADEP_mvt': meta.iloc[st].ADEP_mvt.to_numpy()}).to_parquet(path / 'tune_experts.parquet', index=False)
    del direct
    gc.collect()
    direct, direct_refit_evidence = fit(x.iloc[sr], y[sr], config, trees=trees)
    values, delta = reload_parity(direct, path / 'component.cbm', x.iloc[idx['score']])
    pd.DataFrame({ID: x.index[idx['score']], 'direct': values}).to_parquet(path / 'score_expert.parquet', index=False)
    changed = baseline.route.eq('residual').to_numpy()
    for conditioned in [False, True]:
        child_name = 'clock_airport' if conditioned else 'clock_global'
        child_path, child_record, child_reused = start_run(child_name, fold, seed, config, reference, split)
        if child_reused:
            continue
        predicted = baseline.copy()
        predicted['prediction_sec'] = apply_blend(baseline.prediction_sec, values, baseline.route,
            blend_weights(gate, baseline.ADEP_mvt, conditioned))
        finish_run(child_path, child_record, baseline, predicted, labels.iloc[idx['score']], changed,
                   {'expert_path': str(path), 'gate': gate, 'conditioned': conditioned, 'reload_max_abs_delta': delta})
    # Expert container uses the airport result as its pipeline prediction receipt.
    airport_result = pd.read_parquet(OUT / 'models' / f'clock_airport_{fold}_s{seed}' / 'score_predictions.parquet')
    finish_run(path, record, baseline, airport_result, labels.iloc[idx['score']], changed,
        {'training': {'residual_fit_only': residual_evidence, 'direct_tune': direct_tune_evidence,
                      'direct_refit': direct_refit_evidence}, 'trees': trees, 'gate': gate,
         'reload_max_abs_delta': delta, 'fit_id_hash': object_hash(x.index[sf].tolist()),
         'tune_id_hash': object_hash(x.index[st].tolist()), 'refit_id_hash': object_hash(x.index[sr].tolist())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+', choices=list(RECIPES), required=True)
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F2', 'F3', 'G1'], default=['F1', 'F3'])
    parser.add_argument('--seed', type=int, default=20260910)
    args = parser.parse_args()
    data = load_data()
    for candidate in args.candidates:
        for fold in args.folds:
            if candidate == 'clock_experts':
                train_clocks(fold, args.seed, data)
            else:
                train_component(candidate, fold, args.seed, data)
