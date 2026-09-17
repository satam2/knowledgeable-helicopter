"""All-airport missing-NM soft schedule regimes; CPU only, no submission path."""

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[1]
sys.path.insert(0, str(CAMPAIGN))

import common
import lgbm_adapter
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET

OUT = external_path(ROOT / 'private_runs/campaign_20260916/missing_mixture')
THRESHOLD = 60.


def regime_labels(y, schedule):
    return (np.abs(np.asarray(y, float) - np.asarray(schedule, float)) <= THRESHOLD).astype(np.int32)


def combine(schedule, probability, good_correction, bad_direct):
    schedule, probability, good_correction, bad_direct = [np.asarray(v, float) for v in [schedule, probability, good_correction, bad_direct]]
    if not all(np.isfinite(v).all() for v in [schedule, probability, good_correction, bad_direct]):
        raise ValueError('Mixture inputs must be finite')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Mixture probability must lie in [0,1]')
    return probability * (schedule + good_correction) + (1 - probability) * bad_direct


def replace_eligible(reference, missing_nm, schedule, eligible_prediction):
    eligible = np.asarray(missing_nm, bool) & np.isfinite(np.asarray(schedule, float))
    result = np.asarray(reference, float).copy()
    if len(eligible_prediction) != int(eligible.sum()):
        raise ValueError('Mixture predictions do not match eligible cohort')
    result[eligible] = eligible_prediction
    return result


def _fit_head(x, y, tuning, steps, seed, threads, fallback):
    if len(y) < 30 or (len(y) and np.std(y) == 0):
        mean = float(np.mean(y)) if len(y) else float(fallback)
        return {'constant': mean}, {'steps': 1, 'rows': len(y), 'fallback_mean': mean}
    model, evidence = lgbm_adapter.fit(x, y, tuning, steps=steps, seed=seed, threads=threads)
    return {'model': model}, evidence


def _predict_head(head, x):
    return np.full(len(x), head['constant']) if 'constant' in head else lgbm_adapter.predict(head['model'], x)


def fit_mixture(x, y, schedule, tuning=None, *, steps=None, seed=20260916, threads=2):
    y, schedule = np.asarray(y, float), np.asarray(schedule, float)
    if not len(x) or len(x) != len(y) or len(x) != len(schedule) or not np.isfinite(y).all() or not np.isfinite(schedule).all():
        raise ValueError('Mixture fit needs nonempty aligned finite labels and schedules')
    if steps is not None and tuning is not None:
        raise ValueError('Fixed refit steps cannot use tuning labels')
    consistent = regime_labels(y, schedule)
    evidence = {'fit_rows': len(x), 'consistent_n': int(consistent.sum()), 'inconsistent_n': int((1 - consistent).sum())}
    if np.unique(consistent).size == 1:
        classifier = {'constant': float(consistent[0])}
        classifier_steps = 1
    else:
        encoder = lgbm_adapter.NativeFrameEncoder().fit(x)
        classifier_model = lgb.LGBMClassifier(n_estimators=400 if steps is None else steps['classifier'],
            learning_rate=.05, num_leaves=15, max_depth=4, objective='binary', n_jobs=threads,
            random_state=seed, verbosity=-1, deterministic=True, force_col_wise=True,
            reg_lambda=10., min_child_samples=30)
        options = {}
        if tuning is not None and len(tuning[0]):
            tx, ty, tschedule = tuning
            options.update(eval_X=encoder.transform(tx), eval_y=regime_labels(ty, tschedule),
                           eval_metric='binary_logloss', callbacks=[lgb.early_stopping(60, verbose=False)])
        classifier_model.fit(encoder.transform(x), consistent, categorical_feature=list(encoder.maps), **options)
        classifier_steps = int(classifier_model.best_iteration_ or classifier_model.n_estimators_)
        classifier = {'model': classifier_model, 'encoder': encoder, 'steps': classifier_steps}
    heads = {}
    selected_steps = {'classifier': classifier_steps}
    for name, group in [('good', consistent == 1), ('bad', consistent == 0)]:
        target = y - schedule if name == 'good' else y
        tune = None
        if tuning is not None:
            tx, ty, tschedule = tuning
            ty, tschedule = np.asarray(ty, float), np.asarray(tschedule, float)
            tune_consistent = regime_labels(ty, tschedule)
            use = tune_consistent == (1 if name == 'good' else 0)
            if use.any():
                tune = (tx.loc[use], (ty - tschedule if name == 'good' else ty)[use])
        head, head_evidence = _fit_head(x.loc[group], target[group], tune,
            None if steps is None else steps[name], seed, threads,
            fallback=0. if name == 'good' else np.mean(y))
        heads[name] = head
        evidence[name] = head_evidence
        selected_steps[name] = int(head_evidence['steps'])
    evidence['steps'] = selected_steps
    evidence['classifier_objective'] = 'binary logloss; no post-hoc score calibration'
    return {'classifier': classifier, **heads}, evidence


def predict_mixture(model, x, schedule):
    classifier = model['classifier']
    if 'constant' in classifier:
        probability = np.full(len(x), classifier['constant'])
    else:
        probability = classifier['model'].predict_proba(classifier['encoder'].transform(x), num_iteration=classifier['steps'])[:, 1]
    return combine(schedule, probability, _predict_head(model['good'], x), _predict_head(model['bad'], x))


def declare(seed, threads):
    sources = [Path(__file__), CAMPAIGN / 'common.py', CAMPAIGN / 'lgbm_adapter.py', ROOT / 'review_work/next230_features.py']
    proposal = {'seed': seed, 'threads': threads, 'gpu': False,
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'rationale': 'All-airport missing-NM application plus LightGBM reliability and two learned conditional-mean heads.',
        'prior_difference': 'Prior CatBoost trained all-airport missing schedules but applied Rome only; good correction was a constant and bad head predicted schedule residual.',
        'training': 'All permitted fit/refit missing-NM rows with finite schedule; original flight-ID purges; no 200k subsample.',
        'features': 'Existing verified base plus unchanged schedule extension; no new aviation block in this formulation experiment.',
        'label_rule_training_only': 'abs(taxi_target - schedule_proxy) <= 60 seconds',
        'inference': 'Probability-weighted conditional means; no hard target-derived routing or target clipping.',
        'fallback': 'Reference prediction unchanged for observed NM or missing/nonfinite schedule, including all such score rows.',
        'classifier': {'iterations': 400, 'leaves': 15, 'depth': 4, 'learning_rate': .05, 'l2': 10, 'patience': 60},
        'regressors': 'Shared LightGBM adapter fixed configuration, tune RMSE within each training-defined regime.',
        'variants': ['candidate', 'blend25'], 'blend_weight': .25,
        'folds': ['F1', 'F3'], 'selection': 'Adaptive development test; previous folds already exposed; no tuning or calibration on score labels.'}
    path = OUT / 'protocol.json'
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read_json(path)
        if old['declaration'] != proposal:
            raise ValueError('Declared mixture sources/configuration changed')
        return old
    result = {'created_utc': utc_now(), 'declaration': proposal}
    write_json(path, result)
    return result


def load_schedule_extension(x):
    folder = ROOT / 'private_runs/next_230/features'
    marker = read_json(folder / 'manifest.json')
    if marker['feature_source_sha256'] != sha256(ROOT / 'review_work/next230_features.py'):
        raise ValueError('Frozen schedule extension source changed')
    path = folder / 'extensions.parquet'
    if sha256(path) != marker['sha256']:
        raise ValueError('Frozen schedule extension cache changed')
    extension = pd.read_parquet(path).set_index(ID)
    if not np.array_equal(extension.index, x.index):
        raise ValueError('Schedule extension ID order differs')
    columns = [c for c in extension if not c.startswith('rw_')]
    return pd.concat([x, extension[columns]], axis=1), sha256(folder / 'manifest.json')


def run_fold(fold, x, meta, seed, threads, extension_hash):
    name = f'lightgbm_soft_missing_{fold}_s{seed}'
    dest = external_path(OUT / 'models' / name)
    if dest.exists():
        old = read_json(dest / 'manifest.json')
        if old['status'] != 'complete':
            raise ValueError('Prior incomplete mixture run retained')
        for filename, digest in old['outputs'].items():
            assert sha256(dest / filename) == digest
        print('REUSED', name, flush=True)
        return
    dest.mkdir(parents=True)
    idx, split, _ = common.fold_data(meta, fold, full=True)
    reference, reference_record = common.reference(fold)
    assert object_hash(split) == object_hash(reference_record['split'])
    assert np.array_equal(reference[ID], meta.iloc[idx['score']][ID])
    assert np.array_equal(reference[TARGET], meta.iloc[idx['score']][TARGET])
    y = meta[TARGET].to_numpy(float)
    schedule = meta.schedule_sec.to_numpy(float)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    eligible = missing & np.isfinite(schedule)
    selected = {stage: rows[eligible[rows]] for stage, rows in idx.items()}
    features = {stage: x.loc[meta.iloc[rows][ID]].copy() for stage, rows in selected.items()}
    record = {'status': 'running', 'name': name, 'family': 'lightgbm_soft_missing', 'target': 'conditional_mean_mixture',
        'features': 'base_plus_frozen_schedule', 'full': True, 'seed': seed, 'fold': fold, 'threads': threads,
        'created_utc': utc_now(), 'split': split, 'protocol_sha256': sha256(OUT / 'protocol.json'),
        'schedule_extension_manifest_sha256': extension_hash,
        'fit_ids': {stage: {'n': len(rows), 'hash': object_hash(meta.iloc[rows][ID].tolist())} for stage, rows in selected.items()},
        'reference_manifest_sha256': object_hash(reference_record)}
    write_json(dest / 'manifest.json', record)
    started = time.monotonic()
    try:
        fit_rows, tune_rows, refit_rows, score_rows = [selected[key] for key in ['fit', 'tune', 'refit', 'score']]
        fitted, tune_info = fit_mixture(features['fit'], y[fit_rows], schedule[fit_rows],
            (features['tune'], y[tune_rows], schedule[tune_rows]), seed=seed, threads=threads)
        tune_pred = predict_mixture(fitted, features['tune'], schedule[tune_rows])
        pd.DataFrame({ID: meta.iloc[tune_rows][ID].to_numpy(), 'prediction_sec': tune_pred}).to_parquet(dest / 'tune_predictions.parquet', index=False)
        joblib.dump(fitted, dest / 'fit_model.joblib')
        del fitted
        gc.collect()
        model, refit_info = fit_mixture(features['refit'], y[refit_rows], schedule[refit_rows], steps=tune_info['steps'], seed=seed, threads=threads)
        joblib.dump(model, dest / 'model.joblib')
        infer_start = time.monotonic()
        prediction = predict_mixture(model, features['score'], schedule[score_rows])
        inference_sec = time.monotonic() - infer_start
        replay = predict_mixture(joblib.load(dest / 'model.joblib'), features['score'], schedule[score_rows])
        replay_delta = float(np.max(np.abs(prediction - replay)))
        assert replay_delta <= 1e-9
        score = idx['score']
        candidate = replace_eligible(reference.prediction_sec, missing[score], schedule[score], prediction)
        changed = eligible[score]
        assert np.array_equal(candidate[~changed], reference.prediction_sec.to_numpy()[~changed])
        reports = {}
        for variant, values in {'candidate': candidate, 'blend25': .75 * reference.prediction_sec.to_numpy() + .25 * candidate}.items():
            pred = reference.drop(columns=[TARGET, 'error_sec', 'squared_error', 'label_bin', 'month', 'day'], errors='ignore').copy()
            pred['prediction_sec'] = values
            metrics, errors = evaluate(pred, meta.iloc[score][[ID, TARGET]])
            errors.to_parquet(dest / f'{variant}.parquet', index=False)
            reports[variant] = {'metrics': metrics, 'stability': paired_stability(reference, errors, repetitions=500)}
        record.update(status='complete', completed_utc=utc_now(), tune=tune_info, refit=refit_info,
            reports=reports, changed_rows=int(changed.sum()), protected_routes_equal=True,
            unchanged_missing_schedule_rows=int((missing[score] & ~np.isfinite(schedule[score])).sum()),
            complete_score_rows=len(score), runtime_sec=time.monotonic() - started,
            inference_runtime_sec=inference_sec, reload_max_abs_delta=replay_delta,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            features_used=list(x), libraries={'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
        record['outputs'] = {p.name: sha256(p) for p in dest.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(dest / 'manifest.json', record)
        print('RESULT', name, {key: round(value['metrics']['overall']['rmse_sec'], 6) for key, value in reports.items()}, flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), runtime_sec=time.monotonic() - started)
        write_json(dest / 'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare(args.seed, args.threads)
    if args.declare_only:
        print('Mixture experiment declared; no training launched', flush=True)
        return
    x, meta = common.load_data()
    x, extension_hash = load_schedule_extension(x)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    x = x.loc[meta.loc[missing, ID]].copy()
    gc.collect()
    for fold in args.folds:
        run_fold(fold, x, meta, args.seed, args.threads, extension_hash)


if __name__ == '__main__':
    main()
