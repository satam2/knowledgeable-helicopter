"""Three declared airport-record-only models for the fully missing NM cohort."""

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import joblib
from catboost import CatBoostClassifier, CatBoostRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.availability import assert_observations
from taxiout.features.pipeline import token
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, MOVEMENT, PHASE, utc

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
DAY = 86400.
ARMS = ['direct_airport', 'day_decomposition', 'historical_template']
PARAMS = {'iterations': 600, 'depth': 5, 'learning_rate': .05, 'l2_leaf_reg': 30,
          'loss_function': 'RMSE', 'task_type': 'CPU', 'allow_writing_files': False,
          'verbose': False, 'bootstrap_type': 'No'}


def airport_features(obs):
    assert_observations(obs)
    if not obs[PHASE].eq('DEP').all():
        raise ValueError('Missing-source feature queries must be departures')
    x = pd.DataFrame(index=pd.Index(obs[ID].to_numpy(), name=ID))
    columns = {'airport': 'ADEP_mvt', 'destination': 'ADES_mvt', 'stand': 'STAND_mvt',
        'runway': 'RUNWAY_mvt', 'aircraft': 'AIRCRAFT_TYPE_mvt', 'flight': 'FLIGHT_mvt',
        'flight_rule': 'FLIGHT_RULE_mvt'}
    for name, column in columns.items():
        x[name] = token(obs[column]).to_numpy()
        x[name + '_missing'] = obs[column].isna().to_numpy(dtype=np.float32)
    prefix = obs.FLIGHT_mvt.astype('string').str.extract(r'^([A-Za-z]{2,3})(?=[0-9])', expand=False)
    x['flight_prefix'] = token(prefix).to_numpy()
    movement = utc(obs[MOVEMENT], required=True)
    schedule = utc(obs.SCHED_TIME_UTC_mvt)
    seconds = (movement - schedule).dt.total_seconds().to_numpy()
    x['schedule_proxy_sec'] = seconds
    x['schedule_signed_log'] = np.sign(seconds) * np.log1p(np.abs(seconds))
    x['schedule_absolute_days'] = np.floor(seconds / DAY)
    x['schedule_mod_day_sec'] = np.mod(seconds, DAY)
    x['schedule_calendar_day_delta'] = (movement.dt.normalize() - schedule.dt.normalize()).dt.total_seconds().to_numpy() / DAY
    x['schedule_bucket'] = pd.cut(seconds, [-np.inf, 0, 1800, 7200, 43200, np.inf],
        labels=['negative', '0_30m', '30m_2h', '2h_12h', 'over12h']).astype('string').fillna('missing').to_numpy()
    x['schedule_hhmm'] = (schedule.dt.hour * 60 + schedule.dt.minute).fillna(-1).to_numpy()
    x['schedule_missing'] = schedule.isna().to_numpy(dtype=np.float32)
    for prefix, timestamp in [('movement', movement), ('schedule', schedule)]:
        for attr in ['hour', 'minute', 'second', 'dayofweek', 'month']:
            x[prefix + '_' + attr] = getattr(timestamp.dt, attr).to_numpy()
    for name, left, right in [('airport_flight', 'airport', 'flight'), ('airport_stand', 'airport', 'stand'),
                              ('airport_runway', 'airport', 'runway'), ('airport_schedule_regime', 'airport', 'schedule_bucket')]:
        x[name] = x[left].astype(str) + '|' + x[right].astype(str)
    x['source_record_regime'] = x.airport_schedule_regime + '|stand_missing=' + x.stand_missing.astype(str)
    for column in x.select_dtypes('number'):
        x[column] = x[column].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
    return x


def day_parts(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).all():
        raise ValueError('Original targets must be finite')
    day = np.maximum(0, np.floor(y / DAY)).astype(np.int32)
    return day, y - DAY * day


def expected_days(probability, classes):
    probability = np.asarray(probability, float)
    classes = np.asarray(classes, float)
    if probability.ndim != 2 or probability.shape[1] != len(classes) or not np.isfinite(probability).all():
        raise ValueError('Malformed day probability matrix')
    if np.any(probability < 0) or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError('Day probabilities must be normalized')
    return probability @ classes


def calibrate_days(expected, remainder_prediction, target):
    delta = DAY * np.asarray(expected, float)
    residual = np.asarray(target, float) - np.asarray(remainder_prediction, float)
    if not all(np.isfinite(array).all() for array in [delta, residual]):
        raise ValueError('Nonfinite calibration data')
    # Equivalent to 10 observations with a one-day prediction effect shrinking
    # the scale toward one. Only tune-period out-of-sample predictions enter.
    ridge = 10 * DAY ** 2
    scale = float(max(0., (delta @ residual + ridge) / (delta @ delta + ridge)))
    return {'scale': scale, 'ridge': ridge, 'rows': len(delta), 'ridge_day_equivalent_rows': 10}


class HistoricalTemplate:
    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def fit(self, x, y, timestamps):
        if len(x) != len(y) or not np.isfinite(y).all():
            raise ValueError('Template training labels are not aligned and finite')
        data = x.copy()
        data['_target'] = np.asarray(y, float)
        data['_time'] = pd.to_datetime(timestamps.to_numpy(), utc=True)
        self.history_n = len(data)
        self.last_fit = data._time.max() if len(data) else pd.Timestamp('1900-01-01', tz='UTC')
        self.global_mean = float(data._target.mean()) if len(data) else 900.
        self.tables = []
        for keys in [['airport', 'schedule_bucket'], ['airport', 'flight'],
                     ['airport', 'stand', 'schedule_bucket'], ['airport', 'flight', 'schedule_hhmm']]:
            grouped = data.sort_values('_time').groupby(keys, observed=True, dropna=False)
            table = grouped.agg(mean=('_target', 'mean'), n=('_target', 'size'), last=('_target', 'last'), time=('_time', 'max')).reset_index()
            self.tables.append((keys, table))
        return self

    def transform(self, x, timestamps):
        timestamps = pd.to_datetime(timestamps.to_numpy(), utc=True)
        if len(x) and (timestamps <= self.last_fit).any():
            raise ValueError('Historical template queries must follow every fit record')
        mean = np.full(len(x), self.global_mean)
        support = np.zeros(len(x))
        last = mean.copy()
        age = np.full(len(x), -1.)
        for keys, table in self.tables:
            matched = x.reset_index().merge(table, on=keys, how='left', validate='many_to_one')
            n = matched.n.fillna(0).to_numpy(float)
            weight = n / (n + self.shrinkage)
            mean = weight * matched['mean'].fillna(self.global_mean).to_numpy() + (1 - weight) * mean
            observed = n > 0
            support = np.where(observed, n, support)
            last = np.where(observed, matched['last'].fillna(self.global_mean), last)
            difference = (timestamps - pd.DatetimeIndex(pd.to_datetime(matched.time, utc=True))).total_seconds() / DAY
            age = np.where(observed, difference, age)
        return pd.DataFrame({'template_mean_sec': mean.astype('float32'), 'template_group_n': support.astype('float32'),
            'template_last_sec': last.astype('float32'), 'template_age_days': age.astype('float32'),
            'template_history_n': np.full(len(x), self.history_n, dtype='float32')}, index=x.index)


def crossfit_templates(x, y, timestamps):
    months = pd.to_datetime(timestamps, utc=True).dt.year * 100 + pd.to_datetime(timestamps, utc=True).dt.month
    result = []
    for month in sorted(months.unique()):
        before, current = (months < month).to_numpy(), (months == month).to_numpy()
        model = HistoricalTemplate().fit(x.loc[before], np.asarray(y)[before], timestamps.loc[before])
        result.append(model.transform(x.loc[current], timestamps.loc[current]))
    return pd.concat(result).loc[x.index]


def cats(x):
    return list(x.select_dtypes(['object', 'string', 'category']).columns)


def fit_regressor(x, y, tune=None, steps=None, seed=20260916, threads=2):
    options = dict(PARAMS, iterations=steps or PARAMS['iterations'], random_seed=seed, thread_count=threads)
    model = CatBoostRegressor(**options)
    kwargs = {'cat_features': cats(x)}
    if tune is not None:
        kwargs.update(eval_set=tune, early_stopping_rounds=80, use_best_model=True)
    model.fit(x, y, **kwargs)
    return model, {'steps': int(model.tree_count_), 'rows': len(x)}


def fit_arm(arm, x, y, timestamps, tune=None, selected=None, seed=20260916, threads=2):
    if arm == 'direct_airport':
        tuned = None if tune is None else (tune[0], tune[1])
        model, evidence = fit_regressor(x, y, tuned, None if selected is None else selected['steps'], seed, threads)
        return {'kind': arm, 'model': model}, evidence
    if arm == 'historical_template':
        cross = crossfit_templates(x, y, timestamps)
        prior = HistoricalTemplate().fit(x, y, timestamps)
        offset = cross.template_mean_sec.to_numpy(float)
        features = pd.concat([x, cross], axis=1)
        tuned = None
        if tune is not None:
            tx, ty, tt = tune
            tp = prior.transform(tx, tt)
            tuned = (pd.concat([tx, tp], axis=1), np.asarray(ty) - tp.template_mean_sec.to_numpy(float))
        model, evidence = fit_regressor(features, np.asarray(y) - offset, tuned, None if selected is None else selected['steps'], seed, threads)
        return {'kind': arm, 'model': model, 'prior': prior}, evidence
    if arm != 'day_decomposition':
        raise ValueError(arm)
    day, remainder = day_parts(y)
    classes = np.unique(day)
    if len(classes) < 2:
        classifier = {'constant': float(classes[0])}
        class_steps = 1
    else:
        classifier = CatBoostClassifier(iterations=400 if selected is None else selected['classifier_steps'], depth=4,
            learning_rate=.05, l2_leaf_reg=30, loss_function='MultiClass', task_type='CPU',
            random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False, bootstrap_type='No')
        kwargs = {'cat_features': cats(x)}
        if tune is not None:
            td, _ = day_parts(tune[1])
            if np.isin(td, classes).all():
                kwargs.update(eval_set=(tune[0], td), early_stopping_rounds=60, use_best_model=True)
        classifier.fit(x, day, **kwargs)
        class_steps = int(classifier.tree_count_)
    within_tune = None
    if tune is not None:
        _, within = day_parts(tune[1])
        within_tune = (tune[0], within)
    model, evidence = fit_regressor(x, remainder, within_tune, None if selected is None else selected['steps'], seed, threads)
    result = {'kind': arm, 'model': model, 'classifier': classifier, 'calibration': {'scale': 1.}}
    if tune is not None:
        expected = predict_days(result, tune[0])
        within = model.predict(tune[0], thread_count=threads)
        result['calibration'] = calibrate_days(expected, within, tune[1])
    elif selected is not None:
        result['calibration'] = selected['calibration']
    evidence.update(classifier_steps=class_steps, calibration=result['calibration'], day_counts={str(c): int((day == c).sum()) for c in classes})
    return result, evidence


def predict_days(model, x):
    classifier = model['classifier']
    if isinstance(classifier, dict):
        return np.full(len(x), classifier['constant'])
    return expected_days(classifier.predict_proba(x, thread_count=2), classifier.classes_)


def predict_arm(model, x, timestamps):
    if model['kind'] == 'historical_template':
        prior = model['prior'].transform(x, timestamps)
        return prior.template_mean_sec.to_numpy(float) + model['model'].predict(pd.concat([x, prior], axis=1), thread_count=2)
    within = np.asarray(model['model'].predict(x, thread_count=2), float)
    if model['kind'] == 'day_decomposition':
        return within + DAY * model['calibration']['scale'] * predict_days(model, x)
    return within


def load_data():
    frozen = read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    audit = read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(path) == audit['artifacts'][path.name]
    meta = pd.read_parquet(path)
    parts = []
    for path in sorted(RAW.glob('training_*.parquet')):
        assert sha256(path) == frozen['raw_hashes'][path.name]
        fields = [ID, PHASE, MOVEMENT, 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt', 'ADEP_mvt', 'ADES_mvt',
                  'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt']
        raw = pq.read_table(path, columns=fields, use_threads=False).to_pandas(strings_to_categorical=True)
        missing = raw.loc[raw[PHASE].eq('DEP') & raw.AOBT_3_flt.isna()]
        parts.append(airport_features(missing))
    x = pd.concat(parts)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    assert np.array_equal(x.index, meta.loc[missing, ID]) and len(x) == 22470
    return x, meta


def declaration(seed, threads):
    sources = [Path(__file__), ROOT / 'review_work/campaign_20260916/common.py']
    config = {'arms': ARMS, 'seed': seed, 'threads': threads, 'params': PARAMS,
        'classifier': '400 depth4 MultiClass logloss; exact day expectation; 60-round stopping',
        'day_definition': 'D=max(0,floor(Y/86400)); R=Y-86400D; negatives unchanged in R; predict E[R]+86400E[D]',
        'day_calibration': 'Tune-only scalar >=0 ridge-shrunk toward1 with10 one-day-equivalent observations; optimize raw-second squared error',
        'template': 'Missing-record history only; earlier UTC month crossfit; airport/regime, airport/flight, airport/stand/regime, airport/flight/schedule_hhmm; shrinkage10; Jan900fallback',
        'training': 'Full permitted missing-NM fit and refit cohorts; original flight-ID purges; no sampling, clipping or ranking outcomes',
        'features': 'Airport record fields only: flight text/stand/runway/aircraft/rules, UTC calendar, schedule date+signedlog+modday+regime and categoric interactions',
        'prior_difference': 'Old Rome1200 schedule residual only on Rome. A direct Y across all missing; B exact day decomposition+mean calibration; C chronology-safe template plus residual.',
        'variants': ['candidate', 'blend25'], 'blend_weight': .25, 'folds': ['F1', 'F3'],
        'availability': 'Per-record observed fields and earlier training labels only. No retrospective batch context or exact aircraft rotation inference.',
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    path = OUT / 'protocol.json'
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_json(path)
        if existing['declaration'] != config:
            raise ValueError('Frozen missing-model declaration changed')
        return existing
    record = {'created_utc': utc_now(), 'declaration': config}
    write_json(path, record)
    return record


def run_arm(arm, fold, x, meta, seed, threads):
    name = f'{arm}_{fold}_s{seed}'
    dest = external_path(OUT / 'models' / name)
    if dest.exists():
        record = read_json(dest / 'manifest.json')
        assert record['status'] == 'complete'
        for filename, digest in record['outputs'].items():
            assert sha256(dest / filename) == digest
        print('REUSED', name, flush=True)
        return
    dest.mkdir(parents=True)
    idx, split, _ = common.fold_data(meta, fold, full=True)
    reference, reference_record = common.reference(fold)
    assert object_hash(split) == object_hash(reference_record['split'])
    assert np.array_equal(reference[ID], meta.iloc[idx['score']][ID])
    assert np.array_equal(reference[TARGET], meta.iloc[idx['score']][TARGET])
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    selected = {stage: rows[missing[rows]] for stage, rows in idx.items()}
    frames = {stage: x.loc[meta.iloc[rows][ID]].copy() for stage, rows in selected.items()}
    times = {stage: pd.Series(pd.to_datetime(meta.iloc[rows][MOVEMENT], utc=True).to_numpy(), index=frames[stage].index) for stage, rows in selected.items()}
    y = meta[TARGET].to_numpy(float)
    record = {'status': 'running', 'name': name, 'arm': arm, 'family': 'catboost_missing', 'fold': fold,
        'target': arm, 'features': 'airport_record_only', 'full': True, 'seed': seed, 'threads': threads,
        'created_utc': utc_now(), 'split': split, 'protocol_sha256': sha256(OUT / 'protocol.json'),
        'source_sha256': sha256(__file__), 'fit_ids': {stage: {'n': len(rows), 'hash': object_hash(meta.iloc[rows][ID].tolist())} for stage, rows in selected.items()}}
    write_json(dest / 'manifest.json', record)
    began = time.monotonic()
    try:
        model, tuning = fit_arm(arm, frames['fit'], y[selected['fit']], times['fit'],
            tune=(frames['tune'], y[selected['tune']], times['tune']), seed=seed, threads=threads)
        joblib.dump(model, dest / 'fit_model.joblib')
        tune_prediction = predict_arm(model, frames['tune'], times['tune'])
        pd.DataFrame({ID: frames['tune'].index, 'prediction_sec': tune_prediction}).to_parquet(dest / 'tune_predictions.parquet', index=False)
        del model
        gc.collect()
        model, refit = fit_arm(arm, frames['refit'], y[selected['refit']], times['refit'], selected=tuning, seed=seed, threads=threads)
        joblib.dump(model, dest / 'model.joblib')
        inference_start = time.monotonic()
        prediction = predict_arm(model, frames['score'], times['score'])
        inference_seconds = time.monotonic() - inference_start
        repeated = predict_arm(joblib.load(dest / 'model.joblib'), frames['score'], times['score'])
        delta = float(np.max(np.abs(prediction - repeated)))
        assert delta <= 1e-9 and np.isfinite(prediction).all()
        changed = missing[idx['score']]
        reference_values = reference.prediction_sec.to_numpy()
        values = reference_values.copy()
        values[changed] = prediction
        reports = {}
        for variant, output in {'candidate': values, 'blend25': reference_values + .25 * (values - reference_values)}.items():
            assert np.array_equal(output[~changed], reference_values[~changed])
            pred = reference.drop(columns=[TARGET, 'error_sec', 'squared_error', 'month', 'day', 'label_bin'], errors='ignore').copy()
            pred['prediction_sec'] = output
            metrics, errors = evaluate(pred, meta.iloc[idx['score']][[ID, TARGET]])
            errors.to_parquet(dest / f'{variant}.parquet', index=False)
            reports[variant] = {'metrics': metrics, 'stability': paired_stability(reference, errors, repetitions=500)}
        record.update(status='complete', completed_utc=utc_now(), tune=tuning, refit=refit, reports=reports,
            changed_rows=int(changed.sum()), complete_score_rows=len(reference), protected_routes_equal=True,
            reload_max_abs_delta=delta, runtime_sec=time.monotonic() - began, inference_runtime_sec=inference_seconds,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            features_used=list(x), seed_scope='one complete fit+tune+refit pipeline seed')
        record['outputs'] = {p.name: sha256(p) for p in dest.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(dest / 'manifest.json', record)
        print('RESULT', name, {k: round(v['metrics']['overall']['rmse_sec'], 6) for k, v in reports.items()}, flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), runtime_sec=time.monotonic() - began)
        write_json(dest / 'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declaration(args.seed, args.threads)
    if args.declare_only:
        print('Declared missing models; no private training launched', flush=True)
        return
    x, meta = load_data()
    print('Loaded', len(x), 'missing-source rows and', len(x.columns), 'airport features', flush=True)
    for arm in args.arms:
        for fold in args.folds:
            run_arm(arm, fold, x, meta, args.seed, args.threads)


if __name__ == '__main__':
    main()
