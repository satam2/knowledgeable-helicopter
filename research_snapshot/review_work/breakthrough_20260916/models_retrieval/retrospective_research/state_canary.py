"""Bounded observation-only GaussianHMM diagnostic with temporal calibration."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.special
import hmmlearn
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import persistence_audit as common
from taxiout.schema import ID, FLIGHT_ID, PHASE, MOVEMENT, TARGET, CLOCKS
from taxiout.splits import make_fold

ROOT, BASE, RAW = common.ROOT, common.BASE, common.RAW
OUT = common.external_path(BASE / 'retrospective_research/state_canary')
OBS = [ID, FLIGHT_ID, PHASE, MOVEMENT, 'ADEP_mvt', *CLOCKS]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def extract(raw):
    x = raw[OBS].copy()
    time = pd.to_datetime(x[MOVEMENT], utc=True)
    x['airport'] = x.ADEP_mvt.astype('string').fillna('<missing>')
    x['day'] = time.dt.strftime('%Y-%m-%d')
    x['time'] = time.dt.as_unit('ns').astype('int64') / 1e9
    proxies = [(time - pd.to_datetime(x[c], utc=True)).dt.total_seconds().to_numpy() for c in CLOCKS]
    features = [proxies[0], proxies[1] - proxies[0], proxies[3] - proxies[0], proxies[3] - proxies[2], proxies[4] - proxies[0]]
    for j, values in enumerate(features):
        x[f'emission{j}'] = np.arcsinh(values / 300)
        x[f'missing{j}'] = ~np.isfinite(values)
    return x


def ordered(frame):
    data = frame.sort_values(['day', 'time', ID], kind='stable').reset_index(drop=True)
    return data, data.groupby('day', sort=False).size().to_numpy()


def transform(frame, parameters):
    a = frame[parameters['columns']].to_numpy(float)
    a = np.where(np.isfinite(a), a, parameters['median'])
    return (a - parameters['mean']) / parameters['scale']


def posteriors(model, frame, parameters):
    data, lengths = ordered(frame)
    x = transform(data, parameters)
    smooth = model.predict_proba(x, lengths)
    independent = deepcopy(model)
    stationary = model.get_stationary_distribution()
    independent.startprob_ = stationary
    independent.transmat_ = np.tile(stationary, (model.n_components, 1))
    emission = independent.predict_proba(x, lengths)
    assert np.isfinite(smooth).all() and np.isfinite(emission).all()
    np.testing.assert_allclose(smooth.sum(axis=1), 1, atol=1e-10)
    return data, smooth, emission


def tests():
    model = GaussianHMM(2, covariance_type='diag', init_params='', params='')
    model.startprob_ = np.array([.5, .5])
    model.transmat_ = np.array([[.98, .02], [.02, .98]])
    model.means_ = np.array([[-1.], [1.]])
    model.covars_ = np.array([[1.], [1.]])
    a = np.array([[-1.], [-.1], [.1], [1.]])
    b = np.array([[5.], [5.]])
    isolated = model.predict_proba(a, [len(a)])
    together = model.predict_proba(np.vstack([a, b]), [len(a), len(b)])
    np.testing.assert_array_equal(isolated, together[:len(a)])
    perm = np.array([2, 0, 3, 1])
    independent = deepcopy(model)
    independent.transmat_[:] = .5
    np.testing.assert_allclose(independent.predict_proba(a), independent.predict_proba(a[perm])[np.argsort(perm)], atol=1e-10)
    assert not np.allclose(isolated, model.predict_proba(a[perm])[np.argsort(perm)])
    print('TESTS_PASS HMMdayreset; independentposteriororderinvariance; sequenceordersensitivity', flush=True)


def metric(error):
    return {'n': len(error), 'rmse_sec': float(np.sqrt(np.mean(error**2))), 'bias_sec': float(np.mean(error)), 'mae_sec': float(np.abs(error).mean())}


def main():
    tests()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise ValueError('Preserve earlier state canary')
    protocol = {'created_utc': common.datetime.now(common.timezone.utc).isoformat(), 'source_sha256': common.sha(__file__),
        'hypothesis': 'Observable clock-pattern persistence can improve residual correction over identical nonsequential emission posterior',
        'fit': 'F1 originalflightpurged Jan-May observations only, full days sampled to max20000perairport,3states20EMiterationsdiagGaussian',
        'calibration': 'June1-15 originaltune residual of independentlyfittedcombinedLGB600; Ridgealpha100, airportintercept andstateposterior',
        'evaluation': 'June16-30 andOctober originalF3tunerows, no retuning; inheritedbasemodelchangesbetweenF1/F3 disclosed',
        'comparator': 'Identical learnedGaussianemissions+stationaryprior, transitionrows allstationary so stateposterior ignoresorder; separate identicallyconfiguredRidge',
        'observations': 'Five signedasinh clockdifferences plusmissing masks, fitonlymedian/mean/scale; no clipping',
        'availability': 'Retrospective suppliedDEPclock sequence; futureobservedclocks permitted; dayreset; ownemissions intentionallyincluded',
        'limits': 'Diagnostic only, not originalF1/F3fullscore model. Latentstates notknownprovenance; no ARRmeasurements or hiddenDEPpeerlabels ininference. No scorelabels read.',
        'library': {'hmmlearn': hmmlearn.__version__, 'repository': 'https://github.com/hmmlearn/hmmlearn', 'license': 'BSD3Clause; installedLICENSEretainedmetadata', 'maintenance': 'AuthorREADME sayslimitedmaintenance; no active-maintenance claim'},
        'resource': 'oneCPUthread,2GiBRSSceiling,8GiBhostreserve, noGPU/network in privatecanary'}
    common.write(OUT / 'protocol.json', protocol)
    frozen = common.read(ROOT / 'private_runs/submission_v2/protocol.json')
    own_marker = common.read(BASE / 'missing/provenance_audit/matched_models/control/lightgbm_ordinary_source_residual_F1_s20260916/manifest.json')
    meta = pq.read_table(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID, FLIGHT_ID, MOVEMENT], use_threads=False).to_pandas()
    indices, split = make_fold(meta, own_marker['split']['spec'])
    assert split == own_marker['split']
    permitted = set(meta.iloc[indices['fit']][ID])
    frames = []
    for path in sorted(RAW.glob('training_2025-0[1-5]-01_*.parquet')):
        assert common.sha(path) == frozen['raw_hashes'][path.name]
        raw = pq.read_table(path, columns=OBS, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
        frames.append(extract(raw.loc[raw[ID].isin(permitted)]))
    fit = pd.concat(frames, ignore_index=True)
    del frames, raw, meta, permitted
    gc.collect()
    query = {}
    receipts = {}
    for fold, month in [('F1', '06'), ('F3', '10')]:
        path = next(RAW.glob(f'training_2025-{month}-01_*.parquet'))
        assert common.sha(path) == frozen['raw_hashes'][path.name]
        raw = pq.read_table(path, columns=[*OBS, TARGET], filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
        ordinary, _ = common.load_predictions(f'missing/provenance_audit/matched_models/control/lightgbm_ordinary_source_residual_{fold}_s20260916')
        prediction, receipts[fold] = common.load_predictions(f'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T/lightgbm_aobt_allfinite_{fold}_s20260916')
        selected = raw.set_index(ID).loc[ordinary[ID]].reset_index()
        query[fold] = extract(selected)
        query[fold]['error'] = selected[TARGET].to_numpy(float) - prediction.set_index(ID).loc[selected[ID], 'prediction_sec'].to_numpy(float)
    common.guard()
    results, models = {}, {}
    started = time.monotonic()
    for airport, available in fit.groupby('airport', observed=True):
        common.guard()
        rng = np.random.default_rng(20260916)
        days = rng.permutation(sorted(available.day.unique()))
        sizes = available.groupby('day', observed=True).size()
        chosen, total = [], 0
        for day in days:
            if total and total + sizes[day] > 20000:
                continue
            chosen.append(day)
            total += sizes[day]
        sample, lengths = ordered(available.loc[available.day.isin(chosen)])
        columns = [c for c in sample if c.startswith(('emission', 'missing'))]
        numeric = sample[columns].to_numpy(float)
        median = np.nanmedian(np.where(np.isfinite(numeric), numeric, np.nan), axis=0)
        median = np.where(np.isfinite(median), median, 0)
        filled = np.where(np.isfinite(numeric), numeric, median)
        scale = np.std(filled, axis=0)
        parameters = {'columns': columns, 'median': median, 'mean': filled.mean(axis=0), 'scale': np.where(scale > .01, scale, 1)}
        model = GaussianHMM(3, covariance_type='diag', n_iter=20, tol=.1, min_covar=.05,
            covars_prior=.1, random_state=20260916, implementation='log', verbose=False)
        model.fit(transform(sample, parameters), lengths)
        predictions = {}
        for fold, frame in query.items():
            data, smooth, emission = posteriors(model, frame.loc[frame.airport.eq(airport)], parameters)
            predictions[fold] = (data, smooth, emission)
        june, sm, em = predictions['F1']
        calibration = june.day.lt('2025-06-16').to_numpy()
        model_record = {'training_rows': len(sample), 'training_days': len(lengths), 'iterations': model.monitor_.iter,
            'converged': model.monitor_.converged, 'transition': model.transmat_.tolist(), 'calibration_rows': int(calibration.sum()), 'results': {}}
        fitters = {'airport_bias': float(june.error.to_numpy()[calibration].mean())}
        for label, posterior in [('smooth', sm), ('emission', em)]:
            fitters[label] = Ridge(alpha=100).fit(posterior[calibration], june.error.to_numpy()[calibration])
        for fold, (data, smooth, emission) in predictions.items():
            eligible = data.day.ge('2025-06-16').to_numpy() if fold == 'F1' else np.ones(len(data), bool)
            target = data.error.to_numpy()[eligible]
            row = {'base': metric(target), 'airport_bias': metric(target - fitters['airport_bias'])}
            for label, posterior in [('smooth', smooth), ('emission', emission)]:
                adjustment = fitters[label].predict(posterior[eligible])
                row[label] = metric(target - adjustment)
                saved = data.loc[eligible, [ID, 'day']].copy()
                saved['baseline_error'] = target
                saved['adjustment'] = adjustment
                saved.to_parquet(OUT / f'{airport}_{fold}_{label}.parquet', index=False)
            model_record['results'][fold] = row
        results[str(airport)] = model_record
        models[str(airport)] = {'hmm': model, 'parameters': parameters, 'calibrators': fitters}
        common.write(OUT / 'progress.json', results)
        print('STATE', airport, len(sample), {f: {k: round(v['rmse_sec'], 3) for k, v in r.items()} for f, r in model_record['results'].items()}, flush=True)
    joblib.dump(models, OUT / 'state_models.joblib')
    replay = joblib.load(OUT / 'state_models.joblib')
    airport = next(iter(models))
    frame = query['F1'].loc[query['F1'].airport.eq(airport)]
    a = posteriors(models[airport]['hmm'], frame, models[airport]['parameters'])[1]
    b = posteriors(replay[airport]['hmm'], frame, replay[airport]['parameters'])[1]
    np.testing.assert_array_equal(a, b)
    aggregate = {}
    for fold in ['F1', 'F3']:
        aggregate[fold] = {}
        for label in ['base', 'airport_bias', 'smooth', 'emission']:
            items = [r['results'][fold][label] for r in results.values()]
            n = sum(v['n'] for v in items)
            aggregate[fold][label] = {'n': n, 'rmse_sec': float(np.sqrt(sum(v['n'] * v['rmse_sec']**2 for v in items) / n))}
    common.write(OUT / 'analysis.json', {'status': 'complete', 'source_sha256': common.sha(__file__), 'protocol_sha256': common.sha(OUT / 'protocol.json'),
        'receipts': receipts, 'records': results, 'aggregate': aggregate, 'runtime_sec': time.monotonic() - started,
        'saved_replay_exact': True, 'future_observations_only': True, 'score_labels_used': False, 'outputs': {p.name: common.sha(p) for p in OUT.glob('*.parquet')}})
    print('COMPLETE', aggregate, flush=True)


if __name__ == '__main__':
    main()
