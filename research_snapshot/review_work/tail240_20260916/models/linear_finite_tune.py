"""Matched full387 constant and linear leaves, with bounded fit-only scaling."""
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
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/linear_finite_tune_v1')
ORIGINAL = risk.feature_sources
ARMS = ['constant', 'linear']
FOLD = None
NEURAL = {f: ROOT / 'private_runs/tail240_20260916/state/neural_context' / v / f
          for f, v in [('F1', 'v1'), ('F3', 'v3')]}


def guard(_environment=None):
    info = psutil.Process().memory_info()
    limit = 12 if FOLD == 'F1' else 16
    peak = getattr(info, 'peak_wset', info.rss)
    assert max(info.rss, peak) < limit * 1024**3, (limit, info)
    assert psutil.virtual_memory().available >= 8 * 1024**3
    return int(peak)


def scale_column(values, nfit, filled):
    values = np.asarray(values, dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    if filled:
        values[values == -999999.] = np.nan
    valid = np.isfinite(values[:nfit])
    fit = values[:nfit][valid]
    mean = float(fit.mean()) if len(fit) else 0.
    scale = float(fit.std()) if len(fit) else 1.
    scale = scale if scale > 1e-12 else 1.
    result = ((values-mean)/scale).astype(np.float32)
    assert np.array_equal(np.isfinite(result), np.isfinite(values))
    return result, dict(mean=mean, scale=scale, observed_fit=int(valid.sum()), sentinel_to_nan=bool(filled))


def parameters(base, arm):
    assert arm in ARMS
    return dict(base, linear_tree=arm == 'linear', linear_lambda=5., device_type='cpu', tree_learner='serial')


def declaration():
    controls = {f: common.read_json(risk.union_folder(f) / 'manifest.json') for f in ('F1', 'F3')}
    assert controls['F1']['feature_columns'] == controls['F3']['feature_columns']
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    assert len(controls['F1']['feature_columns']) == 387
    record = dict(source_sha256=common.sha256(__file__),
        sources={str(Path(p).relative_to(ROOT)): common.sha256(p)
                 for p in [shared.__file__, risk.__file__, schema_discovery.__file__, Path(__file__).with_name('test_linear_finite.py')]},
        columns=controls['F1']['feature_columns'], arms=ARMS,
        params={arm: parameters(controls['F1']['fit']['params'], arm) for arm in ARMS},
        controls={f: common.sha256(risk.union_folder(f) / 'manifest.json') for f in controls},
        neural_controls={f: common.sha256(p / 'manifest.json') for f, p in NEURAL.items()},
        preprocessing='Same shared fit-only categorycodes; eachnumeric column finitefitmean/populationSTD (constant/allmissing scale1). Restore -999999 toNaN ONLY for sources whose frozenloader fill=True. Base rawclock legitimate -999999 retained. NaNs preserved, noimputation.',
        contrast='Botharms identical387/cohorts/scaling/leaf63/2500cap/150patience/seed/linear_lambda5; onlylinear_tree differs. Frozen originalglobal387 is additional comparator, not matchedscaledcontrol.',
        target='RawY-originalfiniteNMproxy; add sameNMproxy. All originalfinitefit/tune afterflightpurges; no weights/sampling/clipping/score/ranking.',
        primary='LINEAR fixed25 blend vs current387neuralglobal9 ordinary originalNM[0,7200]: oldglobal9+wPLE*(savedPLE387-oldPLE225).',
        gate='Primarylinear bothmonths positive/everydayremoval positive and>=2seasonalordinaryfixed25seconds. Matchedlinear-vsconstant diagnostic; complementaryexpert neednotwinstandalone. Constantarm diagnostic only, noalternatepromotion.',
        rationale='Priorbase30/leaf15/depth6/200cap linear304.400460 vsconstant308.529326 complete-seasonal, but bothweakagainstthenreference; normalizedmissinglineartestdoesnotcoverfinite387.',
        resources='2CPU, F1 current+OShistoricalpeak<12GiB/start20; F3<16GiB/start24;>=8GiBhostreserve. Once-loadedmemmap,onecolumnscaling,onearmatatime, everyboostroundguard; samplednotOShardlimit.',
        limitation='Alltune development-exposed; same-tune global9weights; noautomaticrefit. This changes preprocessing against originalglobal, but onlyleaftype withinmatchedpair.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def sources(columns):
    result, receipt = schema_discovery.cached_discovery(ORIGINAL, columns)
    common.write_json(OUT / FOLD / 'discovery.json', receipt)
    return result


def compare(y, prediction, reference, dates):
    record = shared.metric.comparison(y, prediction, reference, dates)
    record.update(rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                  reference_rmse=float(np.sqrt(np.mean((reference-y)**2))))
    record['gain'] = record['reference_rmse']-record['rmse']
    return record


def baseline(fold, tune, protocol):
    base = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep = common.read_json(base / 'preparation.json')['folds'][fold]
    assert common.sha256(base / f'{fold}_aligned_tune.parquet') == prep['aligned_tune_sha256']
    weights = common.read_json(base / f'{fold}_weights.json')
    assert weights == prep['weights']
    aligned = pd.read_parquet(base / f'{fold}_aligned_tune.parquet')
    ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID], tune.loc[ordinary, ID])
    np.testing.assert_array_equal(aligned[TARGET], tune.loc[ordinary, TARGET])
    assert common.sha256(NEURAL[fold] / 'manifest.json') == protocol['neural_controls'][fold]
    marker = common.read_json(NEURAL[fold] / 'manifest.json')
    path = NEURAL[fold] / 'tune_predictions.parquet'
    assert common.sha256(path) == marker['outputs'][path.name]
    neural = pd.read_parquet(path).set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET], aligned[TARGET])
    current = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    current += weights['global'][weights['experts'].index('tabm_ple8')] * (neural.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    return ordinary, current


def run(fold):
    global FOLD
    FOLD = fold
    protocol = declaration()
    assert psutil.virtual_memory().available >= (20 if fold == 'F1' else 24) * 1024**3
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    risk.feature_sources, risk.guard = sources, guard
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta = pd.read_parquet(path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    parts = {s: meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy()
             for s, p in idx.items() if s in ('fit', 'tune')}
    del meta
    ids = pd.Index(pd.concat([parts['fit'][ID], parts['tune'][ID]], ignore_index=True))
    nfit = len(parts['fit'])
    matrix, vocab, receipts = risk.load_matrix(ids, nfit, protocol['columns'], dest)
    source_tuples, _ = schema_discovery.cached_discovery(ORIGINAL, protocol['columns'])
    filled = {name for _, _, fill, names in source_tuples if fill for name in names}
    scaler = {}
    for j, name in enumerate(protocol['columns']):
        if name not in vocab:
            values, scaler[name] = scale_column(matrix[:, j], nfit, name in filled)
            matrix[:, j] = values
            del values
        guard()
    matrix.flush()
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab, scaler=scaler))
    target = {s: f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s, f in parts.items()}
    y = parts['tune'][TARGET].to_numpy(float)
    dates = parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    ordinary, current = baseline(fold, parts['tune'], protocol)
    oldfolder = risk.union_folder(fold)
    assert common.sha256(oldfolder / 'manifest.json') == protocol['controls'][fold]
    oldmarker = common.read_json(oldfolder / 'manifest.json')
    assert common.sha256(oldfolder / 'tune_predictions.parquet') == oldmarker['outputs']['tune_predictions.parquet']
    control = pd.read_parquet(oldfolder / 'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID], parts['tune'][ID])
    predictions, summaries = {}, {}
    for arm in ARMS:
        folder = dest / arm
        folder.mkdir()
        model = lgb.LGBMRegressor(**protocol['params'][arm])
        model.fit(matrix[:nfit], target['fit'], categorical_feature=[protocol['columns'].index(c) for c in vocab],
            feature_name=protocol['columns'], eval_X=matrix[nfit:], eval_y=target['tune'], eval_metric='rmse',
            callbacks=[guard, lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
        steps = int(model.best_iteration_ or model.n_estimators_)
        pred = model.predict(matrix[nfit:], num_iteration=steps) + parts['tune'].proxy_sec.to_numpy(float)
        assert np.isfinite(pred).all()
        model.booster_.save_model(str(folder / 'model.txt'), num_iteration=steps)
        native = lgb.Booster(model_file=str(folder / 'model.txt'))
        replay = native.predict(matrix[nfit:], num_threads=2) + parts['tune'].proxy_sec.to_numpy(float)
        np.testing.assert_array_equal(pred, replay)
        pd.DataFrame({ID: parts['tune'][ID], 'prediction_sec': pred}).to_parquet(folder / 'tune.parquet', index=False)
        summaries[arm] = dict(original_global=compare(y, pred, control.prediction_sec.to_numpy(), dates),
            ordinary_fixed25=compare(y[ordinary], .75*current+.25*pred[ordinary], current, dates[ordinary]))
        common.write_json(folder / 'manifest.json', dict(status='complete', arm=arm, fold=fold, steps=steps,
            params=protocol['params'][arm], protocol_sha256=common.sha256(OUT / 'protocol.json'), split=split,
            encoder_sha256=common.sha256(dest / 'encoder.json'), feature_receipts=receipts,
            ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s, f in parts.items()},
            results=summaries[arm], native_replay_max_abs_delta=0., peak_bytes=guard(), no_score_prediction=True,
            outputs={name: common.sha256(folder / name) for name in ['model.txt', 'tune.parquet']}))
        predictions[arm] = pred
        del model, native, replay
        gc.collect()
        print('ARM_COMPLETE', fold, arm, summaries[arm], flush=True)
    matched = compare(y, predictions['linear'], predictions['constant'], dates)
    common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold,
        source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
        matched_linear_vs_constant=matched, arms=summaries,
        primary='linear/ordinary_fixed25 only', no_score_prediction=True, peak_bytes=guard(),
        outputs={name: common.sha256(dest / name) for name in ['encoder.json', 'discovery.json', 'constant/manifest.json', 'linear/manifest.json']}))
    del matrix
    gc.collect()
    (dest / 'matrix.float32').unlink()
    print('COMPLETE', fold, matched, summaries, flush=True)


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
