"""Observed-scale target parameterization with exactly equivalent raw MSE."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import lightgbm as lgb
import argparse
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state'))
import run_missing as shared
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
from encoders import FrameEncoder
common = shared.base.common
ID, TARGET, TIME = shared.state.ID, shared.state.TARGET, shared.state.TIME
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/normalized_missing_tune_v1')
PARAMS = dict(objective='regression', metric='rmse', learning_rate=.05, num_leaves=31,
    max_depth=8, min_data_in_leaf=30, lambda_l2=5., num_threads=2, seed=20260916,
    deterministic=True, force_col_wise=True, verbosity=-1)


def scale_of(x):
    schedule = x.schedule_proxy_sec.to_numpy(float)
    valid = np.isfinite(schedule) & (schedule != -999999)
    return np.sqrt(3600.**2 + np.where(valid, schedule - 900., 0.) ** 2)


def transformed(y, scale):
    y = np.asarray(y, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Finite targets and positive scales required')
    return (y - 900.) / scale


def reconstruct(value, scale):
    return 900. + np.asarray(scale, dtype=float) * np.asarray(value, dtype=float)


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    record = {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'sources': {str(Path(p).relative_to(ROOT)): common.sha256(p) for p in
                    [shared.__file__, shared.base.__file__, shared.identity.__file__,
                     ROOT / 'review_work/breakthrough_20260916/models/encoders.py']},
        'arms': ['unscaled', 'observed_schedule_scale'], 'params': PARAMS, 'round_cap': 600,
        'stopping': 'Tune weightedRMSE60patience; weightedscaled loss exactly rawMSE up a positiveconstant within eacharm.',
        'features': 'Exact existing76airport+ID features; both arms identical. No new information claim.',
        'training_scope': 'All original purged missingNM fit and tune only. No scoring prediction evaluated.',
        'target_identity': 'RawY=900+s(X)*Z; Z=(Y-900)/s(X); weights=s(X)^2 divided byfitmean. Therefore weightedscaled squaredloss equals unmodified raw squaredloss dividedbyfitmean.',
        'scale': 'Unscaled s=1; observed_schedule_scale sqrt(3600^2+(observed_schedule_proxy-900)^2), missing schedule s=3600. No threshold selected from labels.',
        'hypothesis': 'Multiplicative response to observed schedule magnitude improves conditionalmean extrapolation in trees withconstantleaves; objective remainsrawMSE.',
        'gate': 'Advance normalizedarm onlyif bothJune/October rawRMSE improve unscaledand frozenHistoricalTemplate76 tunescores, day-removalstable; no score fit/inspection here.'}
    path = OUT / 'protocol.json'
    if path.exists():
        old = common.read_json(path)
        record['created_utc'] = old['created_utc']
        assert record == old
    else:
        common.write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print('DECLARED exact rawMSE normalized target; no fitting', flush=True)
        return
    x, meta = shared.load_missing()
    y = meta[TARGET].to_numpy(float)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    summary = {}
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        rows = {s: idx[s][missing[idx[s]]] for s in ('fit', 'tune')}
        frames = {s: x.loc[meta.iloc[p][ID]] for s, p in rows.items()}
        encoder = FrameEncoder().fit(frames['fit'])
        matrices = {s: encoder.transform(f) for s, f in frames.items()}
        summary[fold] = {}
        for arm in protocol['arms']:
            dest = OUT / fold / arm
            dest.mkdir(parents=True, exist_ok=False)
            start = time.monotonic()
            scale = {s: (np.ones(len(f)) if arm == 'unscaled' else scale_of(f)) for s, f in frames.items()}
            normalizer = float(np.mean(scale['fit'] ** 2))
            target = {s: transformed(y[rows[s]], scale[s]) for s in rows}
            weights = {s: scale[s] ** 2 / normalizer for s in rows}
            assert all(np.isfinite(w).all() and np.all(w > 0) for w in weights.values())
            train = lgb.Dataset(matrices['fit'], label=target['fit'], weight=weights['fit'], categorical_feature=list(encoder.categories))
            tune = lgb.Dataset(matrices['tune'], label=target['tune'], weight=weights['tune'], reference=train,
                               categorical_feature=list(encoder.categories))
            model = lgb.train(PARAMS, train, num_boost_round=600, valid_sets=[tune], callbacks=[lgb.early_stopping(60, verbose=False)])
            z = model.predict(matrices['tune'], num_iteration=model.best_iteration)
            pred = reconstruct(z, scale['tune'])
            actual = y[rows['tune']]
            np.testing.assert_allclose(weights['tune'] * (z-target['tune']) ** 2,
                                       (pred-actual) ** 2 / normalizer, rtol=1e-9, atol=1e-7)
            joblib.dump({'model': model, 'encoder': encoder, 'arm': arm}, dest / 'model.joblib')
            saved = joblib.load(dest / 'model.joblib')
            repeated = reconstruct(saved['model'].predict(saved['encoder'].transform(frames['tune']), num_iteration=model.best_iteration), scale['tune'])
            np.testing.assert_array_equal(pred, repeated)
            output = pd.DataFrame({ID: frames['tune'].index, 'prediction_sec': pred, 'raw_target_sec': actual,
                'scale': scale['tune'], 'weight': weights['tune'], 'transformed_target': target['tune']})
            output.to_parquet(dest / 'tune.parquet', index=False)
            record = {'status': 'complete', 'fold': fold, 'arm': arm, 'steps': model.best_iteration,
                'rmse': float(np.sqrt(np.mean((pred-actual) ** 2))), 'mae': float(np.mean(np.abs(pred-actual))),
                'bias': float(np.mean(pred-actual)), 'params': PARAMS, 'normalizer': normalizer,
                'fit_weight_ess': float(weights['fit'].sum()**2 / np.sum(weights['fit']**2)),
                'fit_ids': {s: {'n': len(p), 'hash': common.object_hash(meta.iloc[p][ID].tolist())} for s, p in rows.items()},
                'split': split, 'reload_max_abs_delta': 0., 'runtime_sec': time.monotonic()-start,
                'source_sha256': common.sha256(__file__), 'protocol_sha256': common.sha256(OUT / 'protocol.json'),
                'outputs': {p.name: common.sha256(p) for p in dest.iterdir() if p.is_file()}}
            common.write_json(dest / 'manifest.json', record)
            summary[fold][arm] = record['rmse']
            print('RESULT', fold, arm, record['rmse'], 'steps', model.best_iteration, 'ESS', record['fit_weight_ess'], flush=True)
        control = pd.read_parquet(ROOT / f'private_runs/tail240_20260916/state/models_v2/control/{fold}/tune_predictions.parquet')
        np.testing.assert_array_equal(control[ID], frames['tune'].index)
        summary[fold]['historical_template'] = float(np.sqrt(np.mean((control.prediction_sec.to_numpy()-y[rows['tune']])**2)))
    summary['both_month_gate'] = all(summary[f]['observed_schedule_scale'] < min(summary[f]['unscaled'], summary[f]['historical_template']) for f in ('F1','F3'))
    common.write_json(OUT / 'summary.json', {'status':'complete', 'results':summary, 'no_score_prediction':True})
    print('SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
