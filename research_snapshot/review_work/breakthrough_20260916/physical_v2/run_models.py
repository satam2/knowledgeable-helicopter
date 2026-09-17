"""Frozen full-data physical controls; no competition submission path."""
import lightgbm
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from common import ID, TARGET, MOVEMENT, read_json, write_json, sha256, object_hash, utc_now
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path
import adapter

OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/physical_v2')
ARMS = ['basephysicalfull', 'surfacegeometry', 'weather']
BASE = ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt',
    'AIRCRAFT_OPERATOR_flt', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt',
    'airport_stand', 'airport_runway', 'utc_hour', 'weekday', 'month', 'minute',
    'dep_prior_15m', 'dep_prior_60m', 'arr_prior_15m', 'arr_prior_60m',
    'observed_window_15m_sec', 'observed_window_60m_sec', 'month_edge', 'proxy_missing']


def append_block(x, frame, columns):
    if ID in frame:
        frame = frame.set_index(ID)
    if not x.index.is_unique or not frame.index.is_unique or not np.array_equal(x.index, frame.index):
        raise ValueError('Feature IDs/order differ')
    if not columns or set(columns).intersection(x.columns):
        raise ValueError('Empty or duplicate feature block')
    return pd.concat([x, frame[columns]], axis=1)


def read_block(root, filename, *, verification=False, verification_path=None):
    manifest = read_json(root / 'manifest.json')
    if manifest.get('status', 'complete') != 'complete':
        raise ValueError('Incomplete feature cache')
    expected = manifest.get('outputs', {}).get(filename) or manifest.get('feature_sha256')
    if not expected or sha256(root / filename) != expected:
        raise ValueError('Feature hash mismatch')
    receipt = dict(manifest=str(root / 'manifest.json'), manifest_sha256=sha256(root / 'manifest.json'),
        feature_sha256=expected)
    if verification:
        verification_path = verification_path or root / 'verification.json'
        check = read_json(verification_path)
        if check.get('status') != 'passed' or check.get('manifest_sha256') != receipt['manifest_sha256']:
            raise ValueError('Feature independent verification missing or stale')
        receipt['verification_sha256'] = sha256(verification_path)
        receipt['verification_path'] = str(verification_path)
    return pd.read_parquet(root / filename), receipt


def load_features(arms):
    x, meta = common.load_data()
    x = x[BASE].copy()
    receipts = []
    extra, receipt = read_block(ROOT / 'private_runs/campaign_20260916/aviation', 'features.parquet')
    columns = [c for c in extra if c.startswith(('arr_', 'surface_', 'seq_'))]
    x = append_block(x, extra, columns)
    receipts.append(dict(receipt, columns=columns))
    del extra
    if any(arm in arms for arm in ['surfacegeometry', 'weather']):
        for directory, prefix in [('batch_context', 'batch_surface_T_'), ('geometry_v2', 'histgeom_')]:
            extra, receipt = read_block(ROOT / 'private_runs/breakthrough_20260916' / directory,
                'training_features.parquet', verification=True)
            columns = [c for c in extra if c.startswith(prefix)]
            x = append_block(x, extra, columns)
            receipts.append(dict(receipt, columns=columns))
            del extra
    if 'weather' in arms:
        extra, receipt = read_block(ROOT / 'private_runs/breakthrough_20260916/weather', 'training_features.parquet',
            verification=True, verification_path=ROOT/'private_runs/breakthrough_20260916/geometry/weather_review/verification.json')
        columns = [c for c in extra if c.startswith('weather_T_')]
        x = append_block(x, extra, columns)
        receipts.append(dict(receipt, columns=columns))
        del extra
    x['__movement_ns'] = pd.to_datetime(meta[MOVEMENT], utc=True).dt.as_unit('ns').astype('int64').to_numpy()
    return x, meta, receipts


def arm_columns(x, arm):
    excluded = ('weather_',) if arm == 'surfacegeometry' else ()
    if arm == 'basephysicalfull':
        excluded = ('weather_', 'batch_', 'histgeom_')
    columns = [c for c in x if not c.startswith(excluded)]
    forbidden = [c for c in columns if c.startswith(('takeoff_minus_', 'precision_', 'conv_', 'batch_source_', 'weather_N_', 'batch_surface_N_'))
        or c in ['nm_actual_minus_estimated', 'last_minus_initial', TARGET, 'BLOCK_TIME_UTC_mvt']]
    if forbidden:
        raise ValueError(f'Forbidden physical predictors: {forbidden}')
    return columns


def declare(args, x, receipts):
    dependencies = [Path(__file__), HERE/'adapter.py', Path(common.__file__),
        ROOT/'review_work/campaign_20260916/domain_adapter.py',
        ROOT/'review_work/campaign_20260916/lgbm_adapter.py',
        ROOT/'review_work/campaign_20260916/aviation/arrival_features.py']
    declaration = dict(arms=args.arms, folds=args.folds, seed=args.seed, params=adapter.parameters(args.seed, args.threads),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in dependencies}, receipts=receipts,
        columns={arm:arm_columns(x, arm) for arm in args.arms}, full=True,
        target='Raw Y minus chronological earlier-month q10; predictions add q10 back; every original raw label retained.',
        validation='Original fit/tune/refit/score, tune stopping60 rounds, fresh refit; all evaluation cohorts development-exposed.',
        comparison='Matched full-data physical controls; differs from sampled earlier screen in full data and source-missing flag; additions isolate feature blocks.',
        availability='Base causal arrival/sequence features; surface_T explicitly RETROSPECTIVE supplied batch using OTHER flights NMoffblocks. No OWN direct NMclock predictor. Not globally NM-source-independent.',
        geometry='Frozen pre2025 OSM undirected map distance to physical runway graph, not true taxi route or threshold distance.',
        weather='Latest strictly prior observed valid time at takeoff, max3h; archive publication/correction time unverified.',
        variants=['candidate','blend25','missing_only','missing_blend25'], blend_weight=.25,
        resource_policy='CPU only, two threads, >=8GiB available RAM; parent schedules launch; no GPU initialized.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        if read_json(path)['declaration'] != declaration:
            raise ValueError('Frozen physical declaration changed')
    else:
        write_json(path, dict(created_utc=utc_now(), declaration=declaration))
    return path


def run_one(args, fold, arm, x, meta, protocol):
    dest = external_path(OUT/'models'/f'{arm}_{fold}_s{args.seed}')
    if dest.exists():
        rec = read_json(dest/'manifest.json')
        if rec['status'] != 'complete' or rec['protocol_sha256'] != sha256(protocol):
            raise ValueError('Existing incomplete or different run retained')
        for name, digest in rec['outputs'].items():
            assert sha256(dest/name) == digest
        print('REUSED', dest.name, flush=True)
        return
    if psutil.virtual_memory().available < 8*1024**3:
        raise MemoryError('Need8GiB available RAM')
    idx, split, ids = common.fold_data(meta, fold, full=True)
    reference, refrec = common.reference(fold)
    assert object_hash(split) == object_hash(refrec['split'])
    np.testing.assert_array_equal(meta.iloc[idx['score']][ID], reference[ID])
    np.testing.assert_array_equal(meta.iloc[idx['score']][TARGET], reference[TARGET])
    columns = arm_columns(x, arm)
    dest.mkdir(parents=True)
    rec = dict(status='running', created_utc=utc_now(), arm=arm, fold=fold, full=True,
        split=split, fit_ids=ids, features_used=columns, protocol_sha256=sha256(protocol))
    write_json(dest/'manifest.json',rec)
    started = time.monotonic()
    try:
        frames = {stage:x.iloc[rows][columns].copy() for stage,rows in idx.items()}
        y = meta[TARGET].to_numpy(float)
        tuned, tune_info = adapter.fit(frames['fit'], y[idx['fit']],
            (frames['tune'], y[idx['tune']]), seed=args.seed, threads=args.threads)
        joblib.dump(tuned,dest/'fit_model.joblib',compress=0)
        pd.DataFrame({ID:meta.iloc[idx['tune']][ID].to_numpy(),
            'prediction_sec':adapter.predict(tuned,frames['tune'])}).to_parquet(dest/'tune_predictions.parquet',index=False)
        del tuned
        gc.collect()
        model, refit_info = adapter.fit(frames['refit'],y[idx['refit']],steps=tune_info['steps'],seed=args.seed,threads=args.threads)
        joblib.dump(model,dest/'model.joblib',compress=0)
        prediction = adapter.predict(model,frames['score'])
        replay = float(np.max(np.abs(prediction-adapter.predict(joblib.load(dest/'model.joblib'),frames['score']))))
        assert replay <= 1e-9 and np.isfinite(prediction).all()
        ref = reference.prediction_sec.to_numpy(float)
        missing = ~np.isfinite(meta.iloc[idx['score']].proxy_sec.to_numpy(float))
        special = ref.copy()
        special[missing] = prediction[missing]
        variants = dict(candidate=prediction,blend25=ref+.25*(prediction-ref),
            missing_only=special,missing_blend25=ref+.25*(special-ref))
        reports = {}
        for name, values in variants.items():
            if name.startswith('missing_'):
                assert np.array_equal(values[~missing],ref[~missing])
            frame = reference.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
            frame['prediction_sec'] = values
            metrics, errors = evaluate(frame,meta.iloc[idx['score']][[ID,TARGET]])
            errors.to_parquet(dest/f'{name}.parquet',index=False)
            reports[name] = dict(metrics=metrics,stability=paired_stability(reference,errors,repetitions=500))
        rec.update(status='complete',completed_utc=utc_now(),tune=tune_info,refit=refit_info,reports=reports,
            runtime_sec=time.monotonic()-started,reload_max_abs_delta=replay,
            peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss))
        rec['outputs'] = {p.name:sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(dest/'manifest.json',rec)
        print('RESULT',dest.name,{name:report['metrics']['overall']['rmse_sec'] for name,report in reports.items()},flush=True)
    except Exception as exc:
        rec.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),runtime_sec=time.monotonic()-started)
        write_json(dest/'manifest.json',rec)
        raise
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--declare-only',action='store_true')
    args = parser.parse_args()
    x, meta, receipts = load_features(args.arms)
    protocol = declare(args,x,receipts)
    if args.declare_only:
        print('DECLARED',x.shape,protocol,flush=True)
        return
    for arm in args.arms:
        for fold in args.folds:
            run_one(args,fold,arm,x,meta,protocol)


if __name__ == '__main__':
    main()
