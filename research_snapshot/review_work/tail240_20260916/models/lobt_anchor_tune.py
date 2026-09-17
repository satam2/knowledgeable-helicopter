"""Same union387 model with fixed last-known-clock residual origin."""
import following_groups_tune as shared
import schema_discovery
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

ROOT, common, risk, ID, TARGET = shared.ROOT, shared.common, shared.risk, shared.ID, shared.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/lobt_anchor_tune_v1')
ORIGINAL = risk.feature_sources
COLUMN = 'takeoff_minus_LOBT_flt'
FOLD = None
NEURAL = {f: ROOT / 'private_runs/tail240_20260916/state/neural_context' / v / f
          for f, v in [('F1', 'v1'), ('F3', 'v3')]}


def anchor_of(observed_lobt, original_proxy):
    observed = np.asarray(observed_lobt, dtype=float)
    proxy = np.asarray(original_proxy, dtype=float)
    assert observed.shape == proxy.shape and np.isfinite(proxy).all()
    return np.where(np.isfinite(observed), observed, proxy)


def declaration():
    controls = {f: common.read_json(risk.union_folder(f) / 'manifest.json') for f in ('F1', 'F3')}
    assert controls['F1']['feature_columns'] == controls['F3']['feature_columns']
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    assert len(controls['F1']['feature_columns']) == 387 and COLUMN in controls['F1']['feature_columns']
    record = dict(source_sha256=common.sha256(__file__),
        sources={str(Path(p).relative_to(ROOT)): common.sha256(p)
                 for p in [shared.__file__, risk.__file__, schema_discovery.__file__]},
        columns=controls['F1']['feature_columns'], params=controls['F1']['fit']['params'],
        controls={f: common.sha256(risk.union_folder(f) / 'manifest.json') for f in controls},
        neural_controls={f: common.sha256(p / 'manifest.json') for f, p in NEURAL.items()},
        anchor='Existing raw base-feature T-LOBT float32 value promoted tofloat64; if nonfinite use original finiteNM proxy. Preserve negative/long values, including legitimate -999999; no cap or inferredclock.',
        target='Raw Y minus this observed anchor, then add sameanchor at inference; raw squared-error objective unchanged.',
        rows='Every original finiteNM fit/tune after originalflightpurges; cohort/routing defined by originalNM, neveralternateanchor or trueY.',
        difference='Multi-anchor previously choosesNM whenpresent; earlier115clockfusion/MDN hadlearnedLOBTcomponent, notfixedLOBT387. Same observations/modelparameters; tests finitecapacity/extrapolation of targetorigin.',
        training='Same union387 fit-only categories, leaf63/2500cap/150patience/seed/2CPU; no resampling/clipping/labelweights.',
        primary='Ordinary originalNM[0,7200] fixed25candidate blend vs current387neuralglobal9 = oldglobal9+wPLE*(savedPLE387-oldPLE225).',
        gate='Both months matchedallfinite andordinaryfixed25 improve, all dayremovalspositive; >=2seasonalordinaryfixed25seconds. No alternatepromotionpath.',
        resources='2CPU, sampled10GiBprocessRSS,8GiBhostreserve,18GiBstartup; centralallocation.',
        limitation='Exposed adaptive tune; no score/refit/ranking stage. Last-known clock isnotguaranteed actualoffblock or availablelive.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def sources(columns):
    result, receipt = schema_discovery.cached_discovery(ORIGINAL, columns)
    lobt_sources = [(p, fill) for p, _, fill, names in result if COLUMN in names]
    assert len(lobt_sources) == 12 and all(not fill for _, fill in lobt_sources)
    common.write_json(OUT / FOLD / 'discovery.json', receipt)
    return result


def compare(y, prediction, reference, dates):
    record = shared.metric.comparison(y, prediction, reference, dates)
    record.update(rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                  reference_rmse=float(np.sqrt(np.mean((reference-y)**2))))
    record['gain'] = record['reference_rmse'] - record['rmse']
    return record


def run(fold):
    global FOLD
    FOLD = fold
    protocol = declaration()
    assert psutil.virtual_memory().available >= 18 * 1024**3
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    risk.feature_sources, risk.guard = sources, shared.guard
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(path) == audit['artifacts'][path.name]
    meta = pd.read_parquet(path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    parts = {s: meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy()
             for s, p in idx.items() if s in ('fit', 'tune')}
    ids = pd.Index(pd.concat([parts['fit'][ID], parts['tune'][ID]], ignore_index=True))
    nfit = len(parts['fit'])
    matrix, vocab, receipts = risk.load_matrix(ids, nfit, protocol['columns'], dest)
    observed = np.asarray(matrix[:, protocol['columns'].index(COLUMN)], dtype=float)
    proxy = np.concatenate([parts['fit'].proxy_sec.to_numpy(float), parts['tune'].proxy_sec.to_numpy(float)])
    anchor = anchor_of(observed, proxy)
    anchors = {'fit': anchor[:nfit], 'tune': anchor[nfit:]}
    target = {s: f[TARGET].to_numpy(float)-anchors[s] for s, f in parts.items()}
    diagnostics = dict(lobt_finite_fit=int(np.isfinite(observed[:nfit]).sum()),
        lobt_finite_tune=int(np.isfinite(observed[nfit:]).sum()),
        fallback_fit=int((~np.isfinite(observed[:nfit])).sum()), fallback_tune=int((~np.isfinite(observed[nfit:])).sum()),
        anchor_min=float(anchor.min()), anchor_max=float(anchor.max()),
        fit_anchor_hash=common.object_hash(anchors['fit'].tolist()), tune_anchor_hash=common.object_hash(anchors['tune'].tolist()))
    common.write_json(dest / 'anchor_diagnostics.json', diagnostics)
    model = lgb.LGBMRegressor(**protocol['params'])
    model.fit(matrix[:nfit], target['fit'], categorical_feature=[protocol['columns'].index(c) for c in vocab],
        feature_name=protocol['columns'], eval_X=matrix[nfit:], eval_y=target['tune'], eval_metric='rmse',
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
    steps = int(model.best_iteration_ or model.n_estimators_)
    residual = model.predict(matrix[nfit:], num_iteration=steps)
    prediction = residual + anchors['tune']
    y = parts['tune'][TARGET].to_numpy(float)
    np.testing.assert_allclose(prediction-y, residual-target['tune'], rtol=1e-10, atol=1e-8)
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest / 'model.txt'), num_iteration=steps)
    replay = lgb.Booster(model_file=str(dest / 'model.txt')).predict(matrix[nfit:]) + anchors['tune']
    np.testing.assert_array_equal(prediction, replay)
    pd.DataFrame({ID: parts['tune'][ID], 'prediction_sec': prediction, 'anchor_sec': anchors['tune']}).to_parquet(dest / 'tune.parquet', index=False)
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab))
    oldfolder = risk.union_folder(fold)
    oldmanifest = common.read_json(oldfolder / 'manifest.json')
    assert common.sha256(oldfolder / 'tune_predictions.parquet') == oldmanifest['outputs']['tune_predictions.parquet']
    control = pd.read_parquet(oldfolder / 'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID], parts['tune'][ID])
    dates = parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    matched = compare(y, prediction, control.prediction_sec.to_numpy(), dates)
    base = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep = common.read_json(base / 'preparation.json')['folds'][fold]
    assert common.sha256(base / f'{fold}_aligned_tune.parquet') == prep['aligned_tune_sha256']
    weights = common.read_json(base / f'{fold}_weights.json')
    assert weights == prep['weights']
    aligned = pd.read_parquet(base / f'{fold}_aligned_tune.parquet')
    ordinary = parts['tune'].proxy_sec.between(0, 7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID], parts['tune'].loc[ordinary, ID])
    np.testing.assert_array_equal(aligned[TARGET], y[ordinary])
    neuralmanifest = common.read_json(NEURAL[fold] / 'manifest.json')
    assert common.sha256(NEURAL[fold] / 'tune_predictions.parquet') == neuralmanifest['outputs']['tune_predictions.parquet']
    neural = pd.read_parquet(NEURAL[fold] / 'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET], aligned[TARGET])
    oldbase = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    pleweight = weights['global'][weights['experts'].index('tabm_ple8')]
    baseline = oldbase + pleweight * (neural.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    primary = compare(y[ordinary], .75*baseline+.25*prediction[ordinary], baseline, dates[ordinary])
    common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold, steps=steps, split=split,
        source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
        feature_receipts=receipts, params=protocol['params'], anchor_diagnostics=diagnostics,
        ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s, f in parts.items()},
        matched=matched, ordinary_fixed25=primary, native_replay_max_abs_delta=0., no_score_prediction=True,
        outputs={p.name: common.sha256(p) for p in dest.iterdir() if p.name in
                 ['model.txt', 'encoder.json', 'tune.parquet', 'discovery.json', 'anchor_diagnostics.json']}))
    del matrix, model
    gc.collect()
    (dest / 'matrix.float32').unlink()
    print('COMPLETE', fold, matched, primary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        assert args.fold
        run(args.fold)
