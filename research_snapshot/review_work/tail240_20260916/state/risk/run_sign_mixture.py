"""Three supervised residual regimes with an observable probability-weighted mean."""
import lightgbm as lgb
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
import run_risk as risk

common, ROOT, ID, TARGET = risk.common, risk.ROOT, risk.ID, risk.TARGET
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/risk/sign_mixture_v1')
NAMES = ['negative', 'ordinary', 'positive']
GATE_PARAMS = dict(risk.PARAMS, objective='multiclass', num_class=3, metric='multi_logloss')
REG_PARAMS = dict(risk.PARAMS, objective='regression', metric='l2', lambda_l2=30.)


def classes(residual):
    values = np.asarray(residual)
    return np.where(values < -1800, 0, np.where(values > 1800, 2, 1)).astype(np.int32)


def mixture_mean(probabilities, means):
    p, m = np.asarray(probabilities), np.asarray(means)
    assert p.shape == m.shape and p.shape[1] == 3
    assert np.isfinite(p).all() and np.isfinite(m).all()
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.)
    return np.sum(p*m, axis=1)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    prior = risk.freeze()
    protocol = dict(source_sha256=common.sha256(source),
        frozen_loader_sha256=common.sha256(risk.SOURCE),
        binary_risk_protocol_sha256=common.sha256(risk.OUT/'protocol.json'),
        feature_columns=prior['own_columns'], folds=['F1', 'F3'], stages=['fit', 'tune'],
        formula='r=Y-P; c=negative if r<-1800, positive if r>1800, ordinary otherwise. Prediction=P+sum_c p(c|X)*mean(r|c,X).',
        inference='Only X and finite proxy P; no observed class or target at prediction time. No clipping or hard routing.',
        cohort='Original purged full finite-NM fit/tune cohorts. Every raw row/label retained; class partition only supervises fit heads.',
        gate_params=GATE_PARAMS, gate_rounds=150,
        regressor_params=REG_PARAMS, maximum_rounds=600, early_stopping_rounds=50,
        min_leaf={'matched_single':100, 'ordinary':100, 'negative':30, 'positive':30},
        head_selection='Raw squared loss on corresponding class tune subset; absent tune class fixed600fallback. Matchedsingle rawMSE on allfinite tune.',
        calibration='No posthoc probability calibration, weighting, resampling, hyperparameter grid, or oracle inference.',
        comparisons=['mixture versus matched single regressor', 'fixed25%mixture+75%union FIT-model tune prediction versus union',
                     'fixed25%matchedsingle+75%union FIT-model tune prediction versus union'],
        reporting='Allfinite primary and ordinaryproxy0..7200 separate; paired day bootstrap300 of MSE differences; all outcomes retained.',
        followup='Tune diagnostics only. Any score/refit requires separate declaration. A possible advance requires matched gain bothmonths or fixed25union gain>2seconds bothmonths, plus independent audit.',
        limitation='June/October already exposed development; raw conditional-head labels may select steps but cannot select inference class. Union reference is not global9.',
        resources='2CPU sequential, sampledRSS6GiB, hostreserve8GiB.')
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path) == protocol
    else:
        common.write_json(path, protocol)
    return protocol


def regression_fit(xfit, residual, xtune, tune_residual, columns, categories, min_leaf):
    params = dict(REG_PARAMS, min_data_in_leaf=min_leaf)
    dataset = lgb.Dataset(xfit, label=residual, feature_name=columns,
                          categorical_feature=categories, free_raw_data=True)
    kwargs = {}
    if len(tune_residual):
        valid = lgb.Dataset(xtune, label=tune_residual, reference=dataset, free_raw_data=True)
        kwargs = dict(valid_sets=[valid], callbacks=[lgb.early_stopping(50, verbose=False)])
    model = lgb.train(params, dataset, num_boost_round=600, **kwargs)
    return model, params


def evaluate(y, predictions, proxy, dates):
    reports = {}
    for name, mask in [('all_finite', np.ones(len(y), bool)), ('ordinary_proxy', (proxy >= 0)&(proxy <= 7200))]:
        codes, days = pd.factorize(dates[mask], sort=True)
        weights = np.random.default_rng(20260916).multinomial(len(days), np.full(len(days), 1/len(days)), size=300)
        sse = {key:(y[mask]-values[mask])**2 for key, values in predictions.items()}
        pairs = [('mixture', 'matched_single'), ('blend25_mixture_union', 'union'), ('blend25_single_union', 'union')]
        reports[name] = dict(rows=int(mask.sum()), rmse={key:float(np.sqrt(values.mean())) for key, values in sse.items()}, paired={})
        for candidate, reference in pairs:
            gain = sse[reference]-sse[candidate]
            reports[name]['paired'][candidate+'_vs_'+reference] = dict(
                mse_improvement=float(gain.mean()),
                mse_improvement_ci95=risk.bootstrap_interval(gain, np.ones(len(gain)), codes, weights),
                rmse_improvement=reports[name]['rmse'][reference]-reports[name]['rmse'][candidate])
    return reports


def run(fold):
    protocol = freeze()
    folder = common.external_path(OUT/fold)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta_path = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    expected = common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    assert common.sha256(meta_path) == expected
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    indices, split, _ = common.fold_data(meta, fold, full=True)
    parts = {}
    for stage in ['fit', 'tune']:
        positions = indices[stage]
        parts[stage] = meta.iloc[positions[np.isfinite(meta.iloc[positions].proxy_sec)]].copy()
    fit, tune = parts['fit'], parts['tune']
    rf = (fit[TARGET]-fit.proxy_sec).to_numpy(float)
    rt = (tune[TARGET]-tune.proxy_sec).to_numpy(float)
    assert np.isfinite(rf).all() and np.isfinite(rt).all()
    cf, ct = classes(rf), classes(rt)
    ids = pd.Index(pd.concat([fit[ID], tune[ID]], ignore_index=True))
    columns = protocol['feature_columns']
    del meta, parts
    gc.collect()
    x, vocab, receipts = risk.load_matrix(ids, len(fit), columns, folder)
    cats = [columns.index(name) for name in vocab]
    common.write_json(folder/'encoder.json', dict(columns=columns, vocab=vocab))
    xf, xt = x[:len(fit)], x[len(fit):]
    dataset = lgb.Dataset(xf, label=cf, feature_name=columns, categorical_feature=cats, free_raw_data=True)
    gate = lgb.train(GATE_PARAMS, dataset, num_boost_round=150)
    p = gate.predict(xt, num_threads=2)
    gate.save_model(str(folder/'gate.txt'))
    assert np.array_equal(p, lgb.Booster(model_file=str(folder/'gate.txt')).predict(xt, num_threads=2))
    del gate, dataset
    gc.collect()
    heads, training = [], {}
    for k, name in enumerate(NAMES):
        fmask, tmask = cf == k, ct == k
        assert fmask.sum() >= 60, 'Declared tails require at least60fit rows'
        model, params = regression_fit(xf[fmask], rf[fmask], xt[tmask], rt[tmask], columns, cats, protocol['min_leaf'][name])
        values = model.predict(xt, num_threads=2)
        path = folder/(name+'.txt')
        model.save_model(str(path))
        assert np.array_equal(values, lgb.Booster(model_file=str(path)).predict(xt, num_threads=2))
        heads.append(values)
        training[name] = dict(fit_rows=int(fmask.sum()), tune_rows=int(tmask.sum()),
            raw_residual_mean=float(rf[fmask].mean()), raw_residual_min=float(rf[fmask].min()),
            raw_residual_max=float(rf[fmask].max()), selected_rounds=model.best_iteration or model.current_iteration(), params=params)
        print('HEAD', fold, name, training[name], 'RSS', risk.guard(), flush=True)
        del model
        gc.collect()
    means = np.column_stack(heads)
    pred_mix = tune.proxy_sec.to_numpy()+mixture_mean(p, means)
    model, params = regression_fit(xf, rf, xt, rt, columns, cats, 100)
    pred_single = tune.proxy_sec.to_numpy()+model.predict(xt, num_threads=2)
    model.save_model(str(folder/'matched_single.txt'))
    assert np.array_equal(pred_single, tune.proxy_sec.to_numpy()+lgb.Booster(model_file=str(folder/'matched_single.txt')).predict(xt, num_threads=2))
    training['matched_single'] = dict(fit_rows=len(fit), tune_rows=len(tune),
        selected_rounds=model.best_iteration or model.current_iteration(), params=params)
    union_folder = risk.union_folder(fold)
    union_manifest = common.read_json(union_folder/'manifest.json')
    union_path = union_folder/'tune_predictions.parquet'
    assert common.sha256(union_path) == union_manifest['outputs'][union_path.name]
    ref = pd.read_parquet(union_path).set_index(ID)
    assert ref.index.is_unique and set(ref.index) == set(tune[ID])
    union = ref.loc[tune[ID], 'prediction_sec'].to_numpy(float)
    preds = dict(mixture=pred_mix, matched_single=pred_single, union=union,
                 blend25_mixture_union=.25*pred_mix+.75*union,
                 blend25_single_union=.25*pred_single+.75*union)
    results = evaluate(tune[TARGET].to_numpy(), preds, tune.proxy_sec.to_numpy(), tune[common.MOVEMENT].dt.floor('D').to_numpy())
    save = tune[[ID, common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec']].copy()
    for k, name in enumerate(NAMES):
        save['probability_'+name] = p[:, k]
        save['residual_mean_'+name] = means[:, k]
    for name, values in preds.items():
        save[name] = values
    save.to_parquet(folder/'tune_predictions.parquet', index=False)
    common.write_json(folder/'metrics.json', results)
    files = ['gate.txt', 'negative.txt', 'ordinary.txt', 'positive.txt', 'matched_single.txt',
             'encoder.json', 'tune_predictions.parquet', 'metrics.json']
    common.write_json(folder/'manifest.json', dict(status='complete', fold=fold,
        source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT/'protocol.json'),
        split=split, feature_receipts=receipts, training=training,
        fit_ids_hash=common.object_hash(fit[ID].tolist()), tune_ids_hash=common.object_hash(tune[ID].tolist()),
        fit_label_hash=common.object_hash(fit[TARGET].tolist()), tune_label_hash=common.object_hash(tune[TARGET].tolist()),
        union_manifest_sha256=common.sha256(union_folder/'manifest.json'), union_tune_sha256=common.sha256(union_path),
        native_reload_exact=True, sampled_rss_bytes=risk.guard(), runtime_sec=time.monotonic()-started,
        outputs={name:common.sha256(folder/name) for name in files}))
    del model, xf, xt, x
    gc.collect()
    (folder/'matrix.float32').unlink()
    print('COMPLETE', fold, results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare:
            print(freeze())
        else:
            assert args.fold
            run(args.fold)
