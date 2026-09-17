"""Fixed-leaf ExtraTrees output-scaling interaction on complete missing cohorts."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import gc
import importlib.util
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import normalized_missing_tune as normalized
common = normalized.common
ID, TARGET, TIME = normalized.ID, normalized.TARGET, normalized.TIME
path = ROOT / 'review_work/breakthrough_20260916/models/missing_forest/adapter.py'
spec = importlib.util.spec_from_file_location('original_forest', path)
forest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forest)
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/risk'))
import run_risk as risk
from taxiout.metrics import scores, season_score
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/normalized_forest/v1')
BASE = ROOT / 'private_runs/breakthrough_20260916'
ENSEMBLE = BASE / 'models/context_gate/final_simplex9_v1'


def original_folder(fold):
    return BASE / 'models/missing_forest' / f'extratrees_missing_template_idcontext_{fold}_s20260916'


def norm_folder(fold):
    return normalized.OUT / fold / 'observed_schedule_scale'


def guard():
    info = psutil.Process().memory_info()
    assert max(info.rss, info.peak_wset) < 3 * 1024**3, 'Normalized forest 3GiB budget exceeded'
    assert psutil.virtual_memory().available >= 8 * 1024**3


def declare():
    records = {}
    for fold in ['F1', 'F3']:
        old = common.read_json(original_folder(fold) / 'manifest.json')
        assert old['family'] == 'extratrees' and old['fit']['params']['n_estimators'] == 300
        assert old['fit']['params']['max_features'] == .7
        for source in [Path(forest.__file__), Path(forest.missing.__file__), Path(normalized.shared.identity.__file__)]:
            matches = [digest for name, digest in old['source_hashes'].items()
                       if name.replace('\\', '/') == str(source.relative_to(ROOT)).replace('\\', '/')]
            assert matches == [common.sha256(source)], source
        records[fold] = dict(min_samples_leaf=old['fit']['steps'],
            original_manifest_sha256=common.sha256(original_folder(fold) / 'manifest.json'),
            normalized_lgb_manifest_sha256=common.sha256(norm_folder(fold) / 'manifest.json'),
            union_manifest_sha256=common.sha256(risk.union_folder(fold) / 'manifest.json'),
            global9_weights_sha256=common.sha256(ENSEMBLE / f'{fold}_weights.json'),
            global9_aligned_tune_sha256=common.sha256(ENSEMBLE / f'{fold}_aligned_tune.parquet'))
    record = dict(source_sha256=common.sha256(__file__),
        dependency_hashes={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in
            [forest.__file__, forest.missing.__file__, normalized.__file__, normalized.shared.__file__, normalized.shared.identity.__file__, risk.__file__]},
        folds=records, stages=['fit', 'tune'],
        models='ExtraTrees300,maxfeatures.7,bootstrapFalse,squared_error,seed20260916,1CPU. Per-fold original tune-selected minleaf fixed; no new grid.',
        inputs='Original76airport+ID fields plus5 chronological rawY templates; exact originalonehot/median/missingindicator encoder. TIME column only forstrictlyearlier templates. Both arms share rawY priors andfit-onlyencoder.',
        control='Saved rawET full tune native replay, then freshrawET selectedleaf refit on originalfitcohort; max absolute tune delta<=1e-7 before scaledarm.',
        scale='Exact frozen normalized_missing_tune s=sqrt(3600^2+(schedule_proxy-900)^2), missing/sentinel =>3600; Z=(rawY-900)/s, weights=s^2/meanfit(s^2); rawprediction900+sZ. Chronologicaltemplates ALWAYSrawY.',
        objective='Weighted Z squarederror equals rawY squarederror divided bypositivefitnormalizer; all original missingfit/tune labels retained. No newinformation claim: nonboosting variance/outputparameterization interaction.',
        reference='Explicit FULLTUNE diagnostic composition: ordinary finiteNM originalglobal9 fixedweights; finite nonordinary originalunion387 FIT-model tune predictions; missing frozen normalizedLGB FIT-model tune predictions. This is NOT bound272 or269 scorecomposition.',
        candidate='FULLTUNE reference with missing rows only replaced by .75normalizedLGB+.25scaledET. Fixed25, no blendgrid. All finite predictions identical.',
        gate='Both months scaledET missingRMSE beats freshrawET AND normalizedLGB, both fulltune blend gains positive, every fulltune day removal positive, >=2seconds seasonal FULLTUNE RMSE gain (192122summer/152719winter MSE weights). Failure => no score refit; passing does not automatically authorize scoring.',
        evaluation='Report raw/scaled ET and normalizedLGB missing comparisons; fixed25 fulltune composition, bootstrap/dayremoval/topbeneficialrow sensitivity. All used months exposeddevelopment.',
        resources='1CPU,<3GiBhistoricalpeak checkpoints,8GiBhostreserve,start11GiB; noGPU. Parent allocates; neuralF3 priority.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def paired(y, candidate, control, dates):
    a, b = (y-candidate)**2, (y-control)**2
    codes, days = pd.factorize(dates, sort=True)
    weights = np.random.default_rng(20260916).multinomial(len(days), np.full(len(days), 1/len(days)), size=300)
    removals = []
    for i, day in enumerate(days):
        keep = codes != i
        removals.append(dict(day=str(day), gain=float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))))
    influence = {}
    ranks = np.argsort(-(b-a), kind='stable')
    for count in [1,5,10]:
        keep = np.ones(len(y), bool)
        keep[ranks[:count]] = False
        influence[str(count)] = float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))
    return dict(candidate=scores(y, candidate), control=scores(y, control),
        gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())), day_removals=removals,
        all_day_removals_improve=all(x['gain'] > 0 for x in removals),
        mse_gain_ci95=risk.bootstrap_interval(b-a, np.ones(len(y)), codes, weights),
        remove_top_beneficial_rows_gain=influence)


def run(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 11 * 1024**3
    folder = OUT / fold
    folder.mkdir(exist_ok=False)
    normalized.shared.state.guard = guard
    x, meta = normalized.shared.load_missing()
    guard()
    idx, split, _ = common.fold_data(meta, fold, full=True)
    is_missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    selected = {s: idx[s][is_missing[idx[s]]] for s in ['fit','tune']}
    frames = {s: x.loc[meta.iloc[positions][ID]].copy() for s, positions in selected.items()}
    for stage in frames:
        frames[stage][forest.TIME_COLUMN] = meta.set_index(ID).loc[frames[stage].index, TIME]
    y = {s: meta.iloc[p][TARGET].to_numpy(float) for s,p in selected.items()}
    old_path = original_folder(fold)
    old = common.read_json(old_path / 'manifest.json')
    assert common.sha256(old_path / 'manifest.json') == protocol['folds'][fold]['original_manifest_sha256']
    assert common.object_hash(old['split']) == common.object_hash(split)
    assert list(frames['fit']) == old['feature_columns'] and len(frames['fit'].columns) == 77
    for stage, positions in selected.items():
        assert common.object_hash(meta.iloc[positions][ID].tolist()) == old['fit_ids'][stage]['hash']
    for name in ['fit_model.joblib', 'tune_predictions.parquet']:
        assert common.sha256(old_path / name) == old['outputs'][name]
    old_model = joblib.load(old_path / 'fit_model.joblib')
    old_model['estimator'].set_params(n_jobs=1)
    saved = pd.read_parquet(old_path / 'tune_predictions.parquet')
    assert np.array_equal(saved[ID], frames['tune'].index)
    old_pred = forest.predict(old_model, frames['tune'])
    assert np.max(np.abs(old_pred-saved.prediction_sec.to_numpy())) <= 1e-7
    del old_model
    gc.collect()
    control, evidence = forest.fit(frames['fit'], y['fit'], steps=protocol['folds'][fold]['min_samples_leaf'], seed=20260916, threads=1)
    raw_pred = forest.predict(control, frames['tune'])
    delta = float(np.max(np.abs(raw_pred-saved.prediction_sec.to_numpy())))
    assert delta <= 1e-7, f'Fresh raw ExtraTrees control mismatch: {delta}'
    joblib.dump(control, folder / 'raw_control.joblib')
    common.write_json(folder / 'control_replay.json', dict(max_abs_delta=delta, native_original_max_abs_delta=float(np.max(np.abs(old_pred-saved.prediction_sec.to_numpy()))), rows=len(raw_pred), evidence=evidence))
    common.write_json(folder / 'control_pass_before_scaled_fit.json', dict(passed=True, created_utc=common.utc_now()))
    times = pd.Series(pd.to_datetime(frames['fit'][forest.TIME_COLUMN], utc=True).to_numpy(), index=frames['fit'].index)
    raw = forest.input_frame(frames['fit'])
    cross = forest.missing.crossfit_templates(raw, y['fit'], times)
    training_frame = pd.concat([raw, cross], axis=1)
    assert list(training_frame) == control['feature_columns'] and len(training_frame.columns) == 81
    values = control['encoder'].transform(training_frame)
    shared_bundle = {k:v for k,v in control.items() if k != 'estimator'}
    del control
    gc.collect()
    guard()
    scales = {s: normalized.scale_of(frames[s]) for s in frames}
    normalizer = float(np.mean(scales['fit']**2))
    weights = scales['fit']**2 / normalizer
    target = normalized.transformed(y['fit'], scales['fit'])
    estimator = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=protocol['folds'][fold]['min_samples_leaf'],
        max_features=.7, criterion='squared_error', n_jobs=1, random_state=20260916, bootstrap=False)
    estimator.fit(values, target, sample_weight=weights)
    model = dict(shared_bundle, estimator=estimator, normalization='observed_schedule_scale', fit_scale_squared_normalizer=normalizer)
    z = forest.predict(model, frames['tune'])
    pred = normalized.reconstruct(z, scales['tune'])
    np.testing.assert_allclose(scales['tune']**2/normalizer*(z-normalized.transformed(y['tune'], scales['tune']))**2,
        (pred-y['tune'])**2/normalizer, rtol=1e-8, atol=1e-6)
    joblib.dump(model, folder / 'scaled_model.joblib')
    del model, estimator
    gc.collect()
    reload = joblib.load(folder / 'scaled_model.joblib')
    replay = normalized.reconstruct(forest.predict(reload, frames['tune']), scales['tune'])
    np.testing.assert_array_equal(pred, replay)
    guard()
    norm_path = norm_folder(fold)
    norm_manifest = common.read_json(norm_path / 'manifest.json')
    assert common.sha256(norm_path / 'manifest.json') == protocol['folds'][fold]['normalized_lgb_manifest_sha256']
    assert common.sha256(norm_path / 'tune.parquet') == norm_manifest['outputs']['tune.parquet']
    norm_pred = pd.read_parquet(norm_path / 'tune.parquet')
    assert np.array_equal(norm_pred[ID], frames['tune'].index)
    assert np.array_equal(norm_pred.raw_target_sec, y['tune'])
    dates = meta.iloc[selected['tune']][TIME].dt.floor('D').to_numpy()
    reports = dict(vs_raw=paired(y['tune'],pred,raw_pred,dates), vs_normalized_lgb=paired(y['tune'],pred,norm_pred.prediction_sec.to_numpy(),dates))
    full = meta.iloc[idx['tune']].copy()
    finite = np.isfinite(full.proxy_sec).to_numpy()
    ordinary = full.proxy_sec.between(0,7200).to_numpy()
    baseline = np.full(len(full), np.nan)
    union_path = risk.union_folder(fold)
    union_marker = common.read_json(union_path / 'manifest.json')
    assert common.sha256(union_path / 'manifest.json') == protocol['folds'][fold]['union_manifest_sha256']
    assert common.sha256(union_path / 'tune_predictions.parquet') == union_marker['outputs']['tune_predictions.parquet']
    union = pd.read_parquet(union_path / 'tune_predictions.parquet')
    assert np.array_equal(union[ID], full.loc[finite, ID])
    baseline[finite] = union.prediction_sec
    assert common.sha256(ENSEMBLE / f'{fold}_weights.json') == protocol['folds'][fold]['global9_weights_sha256']
    assert common.sha256(ENSEMBLE / f'{fold}_aligned_tune.parquet') == protocol['folds'][fold]['global9_aligned_tune_sha256']
    w = common.read_json(ENSEMBLE / f'{fold}_weights.json')
    aligned = pd.read_parquet(ENSEMBLE / f'{fold}_aligned_tune.parquet')
    assert np.array_equal(aligned[ID], full.loc[ordinary,ID])
    assert np.array_equal(full.loc[~finite,ID], frames['tune'].index)
    baseline[ordinary] = aligned[w['experts']].to_numpy() @ np.asarray(w['global'])
    baseline[~finite] = norm_pred.prediction_sec
    candidate = baseline.copy()
    candidate[~finite] = .75*baseline[~finite] + .25*pred
    assert np.array_equal(candidate[finite], baseline[finite])
    reports['full_tune_fixed25'] = paired(full[TARGET].to_numpy(float),candidate,baseline,full[TIME].dt.floor('D').to_numpy())
    full.assign(reference_prediction_sec=baseline,prediction_sec=candidate).to_parquet(folder / 'full_tune.parquet', index=False)
    pd.DataFrame({ID:frames['tune'].index, 'prediction_sec':pred, 'raw_control_prediction_sec':raw_pred, 'normalized_lgb_prediction_sec':norm_pred.prediction_sec.to_numpy(), 'raw_target_sec':y['tune']}).to_parquet(folder / 'missing_tune.parquet', index=False)
    common.write_json(folder / 'metrics.json', reports)
    common.write_json(folder / 'manifest.json', dict(status='complete', fold=fold, protocol_sha256=common.sha256(OUT / 'protocol.json'),
        split=split, fixed_leaf=protocol['folds'][fold]['min_samples_leaf'], template_targets='rawY unchanged',
        fit_ids={s:dict(n=len(p),hash=common.object_hash(meta.iloc[p][ID].tolist())) for s,p in selected.items()},
        full_tune_id_hash=common.object_hash(full[ID].tolist()), full_tune_rows=len(full),
        fit_weight_ess=float(weights.sum()**2/np.dot(weights,weights)), native_scaled_reload_delta=0.,
        peak_wset_bytes=psutil.Process().memory_info().peak_wset,
        outputs={p.name:common.sha256(p) for p in folder.iterdir() if p.is_file()}))
    print('COMPLETE',fold,{k:v['gain'] for k,v in reports.items()},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT/'protocol.json'),flush=True)
    else:
        assert args.fold
        with threadpool_limits(1):
            run(args.fold)
