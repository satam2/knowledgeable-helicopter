"""Observable clock-gap scaling with an exactly equivalent raw squared loss."""
import following_groups_tune as shared
import schema_discovery
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

ROOT, common, risk, ID, TARGET = shared.ROOT, shared.common, shared.risk, shared.ID, shared.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/scaled_finite_tune_v1')
ORIGINAL = risk.feature_sources
GAP = 'conv_nm_minus_sched_sec'
MISSING = 'conv_nm_minus_sched_missing'


def scale_of(gap, missing):
    gap, missing = np.asarray(gap, float), np.asarray(missing, float)
    valid = np.isfinite(gap) & (gap != -999999.) & (missing == 0)
    return np.hypot(900., np.where(valid, gap, 0.))


def declare():
    controls = {f: common.read_json(risk.union_folder(f) / 'manifest.json') for f in ('F1', 'F3')}
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    columns = controls['F1']['feature_columns']
    assert GAP in columns and MISSING in columns
    fresh = {f: ROOT / f'private_runs/tail240_20260916/models/following_groups_tune_{"v1" if f == "F1" else "v2"}/{f}/control387' for f in controls}
    for path in fresh.values():
        assert common.read_json(path / 'manifest.json')['original_control_max_abs_delta'] == 0.
    record = dict(name='fullfinite_observable_clockgap_scaled_residual', source_sha256=common.sha256(__file__),
        source_hashes={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in [shared.__file__, risk.__file__, schema_discovery.__file__]},
        controls={f: dict(original_manifest_sha256=common.sha256(risk.union_folder(f) / 'manifest.json'),
            fresh_manifest_path=str(fresh[f] / 'manifest.json'), fresh_manifest_sha256=common.sha256(fresh[f] / 'manifest.json')) for f in controls},
        columns=columns, params=controls['F1']['fit']['params'],
        scale='s=hypot(900, NMoffblock-minus-airport-schedule); absent clock/sentinel =>900. Existing observed fields only; fixed900, no tuned scale.',
        target='Z=(rawY-P)/s, weight=s^2/mean_fit(s^2); raw prediction=P+sZ. Weighted transformed squared loss is original raw squared loss divided by fit scale normalizer.',
        hypothesis='Relative source disagreement corrections may transfer across clock-gap magnitudes. No new information; output function class and effective regularization differ.',
        training='Original full finiteNM fit/tune, original purges and387 fields/category vocab. Sameleaf63/2500cap/150patience/2CPU. Tune weightedRMSE stopping is monotone in rawRMSE.',
        evaluation='Matched raw387 plus fixed25 ordinary global9 blend; frozen oldcomponent replacement diagnostic reported separately, not alternative advancement path.',
        gate='Both months matched and fixed25 ordinary ensemble gains, every day removal positive, >=2 seasonal ordinary ensemble seconds. Gate failure => no score refit.',
        risk='Report fit/tune scale quantiles, weight ESS and max share. Extreme observed scales can amplify error; original labels and all finite rows retained.',
        score_labels_used_for_selection=False, routing_uses_score_targets=False,
        resources='2CPU,10GiB sampledRSS and8GiBhostreserve; start18GiB available with parent aggregate resource allocation. No GPU.',
        exposure='June/October tune already exposed; no July/November score predictions or ranking reads.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def run(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 18 * 1024**3
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    def sources(columns):
        result, receipt = schema_discovery.cached_discovery(ORIGINAL, columns)
        common.write_json(dest / 'discovery.json', receipt)
        return result
    risk.feature_sources, risk.guard = sources, shared.guard
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    parts = {s: meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy() for s, p in idx.items() if s in ('fit', 'tune')}
    ids = pd.Index(pd.concat([parts['fit'][ID], parts['tune'][ID]], ignore_index=True))
    nfit = len(parts['fit'])
    matrix, vocab, receipts = risk.load_matrix(ids, nfit, protocol['columns'], dest)
    scales = scale_of(matrix[:, protocol['columns'].index(GAP)], matrix[:, protocol['columns'].index(MISSING)])
    scale = {'fit': scales[:nfit], 'tune': scales[nfit:]}
    normalizer = float(np.mean(scale['fit']**2))
    weights = {s: a**2 / normalizer for s, a in scale.items()}
    target = {s: (f[TARGET].to_numpy(float) - f.proxy_sec.to_numpy(float)) / scale[s] for s, f in parts.items()}
    diagnostics = {s: dict(scale_quantiles=np.quantile(scale[s], [0,.5,.9,.99,.999,1]).tolist(),
        weight_ess=float(w.sum()**2/(w@w)), max_weight_share=float(w.max()/w.sum()), rows=len(w)) for s, w in weights.items()}
    common.write_json(dest / 'scale_diagnostics.json', diagnostics)
    print('SCALE', fold, diagnostics, flush=True)
    model = lgb.LGBMRegressor(**protocol['params'])
    model.fit(matrix[:nfit], target['fit'], sample_weight=weights['fit'],
        categorical_feature=[protocol['columns'].index(c) for c in vocab], feature_name=protocol['columns'],
        eval_X=matrix[nfit:], eval_y=target['tune'], eval_sample_weight=[weights['tune']], eval_metric='rmse',
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
    steps = int(model.best_iteration_ or model.n_estimators_)
    z = model.predict(matrix[nfit:], num_iteration=steps)
    prediction = parts['tune'].proxy_sec.to_numpy(float) + scale['tune'] * z
    y = parts['tune'][TARGET].to_numpy(float)
    np.testing.assert_allclose(weights['tune']*(z-target['tune'])**2, (prediction-y)**2/normalizer, rtol=1e-8, atol=1e-6)
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest / 'model.txt'), num_iteration=steps)
    replay = parts['tune'].proxy_sec.to_numpy(float) + scale['tune'] * lgb.Booster(model_file=str(dest / 'model.txt')).predict(matrix[nfit:])
    np.testing.assert_array_equal(prediction, replay)
    pd.DataFrame({ID: parts['tune'][ID], 'prediction_sec': prediction, 'scale': scale['tune']}).to_parquet(dest / 'tune.parquet', index=False)
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab))
    control_folder = risk.union_folder(fold)
    old = common.read_json(control_folder / 'manifest.json')
    assert common.sha256(control_folder / 'manifest.json') == protocol['controls'][fold]['original_manifest_sha256']
    assert common.sha256(control_folder / 'tune_predictions.parquet') == old['outputs']['tune_predictions.parquet']
    control = pd.read_parquet(control_folder / 'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID], parts['tune'][ID])
    dates = parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    matched = shared.metric.comparison(y, prediction, control.prediction_sec.to_numpy(), dates)
    base = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep = common.read_json(base / 'preparation.json')['folds'][fold]
    path = base / f'{fold}_aligned_tune.parquet'
    assert common.sha256(path) == prep['aligned_tune_sha256']
    blend_weights = common.read_json(base / f'{fold}_weights.json')
    assert blend_weights == prep['weights']
    aligned = pd.read_parquet(path)
    ordinary = parts['tune'].proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID], parts['tune'].loc[ordinary,ID])
    np.testing.assert_array_equal(aligned[TARGET], y[ordinary])
    baseline = aligned[blend_weights['experts']].to_numpy(float) @ np.asarray(blend_weights['global'])
    blended = .75*baseline + .25*prediction[ordinary]
    ensemble = shared.metric.comparison(y[ordinary], blended, baseline, dates[ordinary])
    ensemble.update(reference_rmse=float(np.sqrt(np.mean((baseline-y[ordinary])**2))), rmse=float(np.sqrt(np.mean((blended-y[ordinary])**2))))
    common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold, steps=steps, split=split,
        source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
        feature_receipts=receipts, ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s,f in parts.items()},
        normalizer=normalizer, scale_diagnostics=diagnostics, native_replay_max_abs_delta=0.,
        rmse=float(np.sqrt(np.mean((prediction-y)**2))), matched=matched, ordinary_fixed25=ensemble,
        no_score_prediction=True, outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.name in ['model.txt','tune.parquet','encoder.json','discovery.json','scale_diagnostics.json']}))
    del matrix
    (dest / 'matrix.float32').unlink()
    print('COMPLETE', fold, matched, ensemble, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1','F3'])
    args = parser.parse_args()
    if args.declare_only:
        declare()
    else:
        assert args.fold
        run(args.fold)
