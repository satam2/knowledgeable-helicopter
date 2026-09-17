"""Predeclared full-finite shared-token relation-attention tune experiment."""
import torch
import lightgbm
import argparse
import gc
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/neural_context'))
import run as context
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import schema_discovery
import attention_adapter as adapter

common, risk = context.common, context.risk
ID, TARGET, TIME = context.ID, context.TARGET, common.MOVEMENT
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/neural_attention/v1')
ENSEMBLE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
CONTROLS = {f:ROOT / 'private_runs/tail240_20260916/state/neural_context' / v / f
            for f, v in [('F1', 'v1'), ('F3', 'v3')]}


def guard():
    info = psutil.Process().memory_info()
    assert max(info.rss, info.peak_wset) < 20 * 1024**3, f'20GiB process budget {info}'
    assert psutil.virtual_memory().available >= 8 * 1024**3, 'Host reserve below8GiB'
    return dict(rss_bytes=info.rss, peak_wset_bytes=info.peak_wset,
                available_bytes=psutil.virtual_memory().available)


def declare():
    sources = [Path(__file__), Path(adapter.__file__), Path(context.__file__), Path(risk.__file__),
               Path(adapter.frozen.__file__), Path(adapter.ple.__file__),
               Path(sys.modules['encoders'].__file__), Path(schema_discovery.__file__),
               Path(__file__).with_name('test_attention.py'), Path(__file__).with_name('DESIGN.md')]
    controls = {}
    columns = None
    for fold, folder in CONTROLS.items():
        manifest = common.read_json(folder / 'manifest.json')
        assert manifest['status'] == 'complete'
        if columns is None:
            columns = manifest['feature_columns']
        assert columns == manifest['feature_columns'] and len(columns) == 387
        controls[fold] = dict(folder=str(folder), manifest_sha256=common.sha256(folder / 'manifest.json'),
            fit_model_sha256=manifest['outputs']['fit_model.joblib'],
            tune_predictions_sha256=manifest['outputs']['tune_predictions.parquet'],
            fit_rows=manifest['full_finite_fit_rows'], tune_rows=manifest['full_finite_tune_rows'],
            fit_ids_hash=manifest['fit_ids_hash'], tune_ids_hash=manifest['tune_ids_hash'],
            fit_label_hash=manifest['fit_label_hash'], tune_label_hash=manifest['tune_label_hash'],
            global9_weights_sha256=common.sha256(ENSEMBLE / f'{fold}_weights.json'),
            global9_tune_sha256=common.sha256(ENSEMBLE / f'{fold}_aligned_tune.parquet'))
    value = dict(source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in sources},
        feature_columns=columns, controls=controls, folds=['F1', 'F3'], stages=['fit', 'tune'],
        primary='Frozen global9 original PLE225 component replacement with candidate, all ordinary tune rows',
        secondary=['Matched saved PLE387 full finite and ordinary', 'Fixed25 candidate/global9 ordinary blend'],
        gate='Primary only: both months positive, every leave-one-day-out positive, >=2 seasonal ordinary RMSE seconds; no score on negative',
        season_weights=[192122 / 344841, 152719 / 344841],
        training='Same full finite purged cohorts, rawY-P, PLE387 static path, shared24values+24masks+phase, 64latent2heads, zero64to8head; freshseed20260916, batch4096, AdamW.001/.0001, max40/patience5, gradient10',
        relations='DEPonly duration=own_offset-peer_offset; timestamp=own_offset-peer_offset-age. Both clocks+age finite; ARRmasked. Original8tokens14fields unchanged. No new source or rank expansion.',
        preprocessing='Frozen fit-only FrameEncoder; grouped16 exact official48PLE bins; raw reconstruction then shared fit-only present-token semantic mean/std, masks; frozen trainer unchanged',
        interpretation='Relation representation plus query attention plus capacity package, not isolated attention. Same-tune global9 weights; repeatedly exposed development evidence.',
        control='Saved387 native complete tune parity <=1e-6 seconds before candidate canary/fit; native candidate replay exact',
        resources='ExclusiveGPU only after central allocation and independent preflight;2CPU;20GiB process/8GiB hostreserve/>=28GiB startup;4096 backward8192 inference canary before fitting <8GiB GPU; no optimizer canary update',
        authority='No score/refit/ranking stage; no alternative branch/grid or adaptive endpoint selection')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value, 'Frozen source/protocol changed'
    else:
        common.write_json(path, value)
    return value


def run(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 28 * 1024**3
    folder = common.external_path(OUT / fold)
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    common.write_json(folder / 'launch.json', dict(created_utc=common.utc_now(), resources=guard(),
        protocol_sha256=common.sha256(OUT / 'protocol.json')))
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path) == '503df9cb08f3ed7260fd7d969a3505d7b58434e20bccc80f1526d517a292e976'
    meta = pd.read_parquet(path, columns=[ID, 'FLIGHT_ID_mvt', TIME, 'ADEP_mvt', TARGET, 'proxy_sec'])
    indices, split, _ = common.fold_data(meta, fold, full=True)
    parts = {s:meta.iloc[indices[s][np.isfinite(meta.iloc[indices[s]].proxy_sec)]].copy() for s in ['fit', 'tune']}
    fit, tune = parts['fit'], parts['tune']
    binding = protocol['controls'][fold]
    for stage, frame in parts.items():
        assert len(frame) == binding[stage + '_rows']
        assert common.object_hash(frame[ID].tolist()) == binding[stage + '_ids_hash']
        assert common.object_hash(frame[TARGET].tolist()) == binding[stage + '_label_hash']
    del meta, parts
    gc.collect()
    ids = pd.Index(pd.concat([fit[ID], tune[ID]], ignore_index=True))
    original = risk.feature_sources
    def cached(columns):
        values, receipt = schema_discovery.cached_discovery(original, columns)
        common.write_json(folder / 'discovery.json', receipt)
        guard()
        return values
    risk.feature_sources = cached
    risk.guard = guard
    try:
        matrix, vocab, receipts = risk.load_matrix(ids, len(fit), protocol['feature_columns'], folder)
    finally:
        risk.feature_sources = original
    frame = context.decode_frame(matrix, vocab, protocol['feature_columns'])
    frame.index = ids
    xf, xt = frame.iloc[:len(fit)], frame.iloc[len(fit):]
    control_folder = CONTROLS[fold]
    for name, field in [('manifest.json', 'manifest_sha256'), ('fit_model.joblib', 'fit_model_sha256'), ('tune_predictions.parquet', 'tune_predictions_sha256')]:
        assert common.sha256(control_folder / name) == binding[field]
    control = joblib.load(control_folder / 'fit_model.joblib')
    old = pd.read_parquet(control_folder / 'tune_predictions.parquet').set_index(ID).loc[tune[ID]]
    assert np.array_equal(old[TARGET], tune[TARGET])
    replay = adapter.frozen.predict(control, xt) + tune.proxy_sec.to_numpy(float)
    delta = float(np.max(np.abs(replay - old.prediction_sec.to_numpy())))
    assert delta <= 1e-6, f'Saved387 native parity failed {delta}'
    common.write_json(folder / 'control_replay.json', dict(rows=len(tune), max_abs_delta_sec=delta,
        model_sha256=binding['fit_model_sha256'], before_candidate_fit=True))
    del control, replay
    gc.collect()
    guard()
    original_infer = adapter.frozen.infer
    def guarded_infer(*args, **kwargs):
        guard()
        values = original_infer(*args, **kwargs)
        guard()
        return values
    adapter.frozen.infer = guarded_infer
    try:
        model, evidence = adapter.fit(xf, (fit[TARGET] - fit.proxy_sec).to_numpy(float),
            (xt, (tune[TARGET] - tune.proxy_sec).to_numpy(float)), seed=20260916, threads=2)
        pred = adapter.predict(model, xt) + tune.proxy_sec.to_numpy(float)
        joblib.dump(model, folder / 'fit_model.joblib')
        reload = joblib.load(folder / 'fit_model.joblib')
        replay = adapter.predict(reload, xt) + tune.proxy_sec.to_numpy(float)
        np.testing.assert_array_equal(pred, replay)
    finally:
        adapter.frozen.infer = original_infer
    output = tune[[ID, TIME, 'ADEP_mvt', TARGET, 'proxy_sec']].copy()
    output['prediction_sec'] = pred
    output['control387_prediction_sec'] = old.prediction_sec.to_numpy()
    output.to_parquet(folder / 'tune_predictions.parquet', index=False)
    ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
    dates = tune[TIME].dt.floor('D').to_numpy()
    metrics = {name:context.paired(tune[TARGET].to_numpy()[mask], pred[mask], old.prediction_sec.to_numpy()[mask], dates[mask])
        for name, mask in [('all_finite', np.ones(len(tune), bool)), ('ordinary', ordinary)]}
    for name, field in [(f'{fold}_weights.json', 'global9_weights_sha256'), (f'{fold}_aligned_tune.parquet', 'global9_tune_sha256')]:
        assert common.sha256(ENSEMBLE / name) == binding[field]
    weights = common.read_json(ENSEMBLE / f'{fold}_weights.json')
    aligned = pd.read_parquet(ENSEMBLE / f'{fold}_aligned_tune.parquet').set_index(ID)
    assert set(aligned.index) == set(tune.loc[ordinary, ID])
    own = output.set_index(ID).loc[aligned.index]
    np.testing.assert_array_equal(own[TARGET], aligned[TARGET])
    baseline = aligned[weights['experts']].to_numpy() @ np.asarray(weights['global'])
    replacement = baseline + weights['global'][weights['experts'].index('tabm_ple8')] * (own.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
    blend = .75 * baseline + .25 * own.prediction_sec.to_numpy()
    for key, candidate in [('primary_replacement', replacement), ('secondary_blend25', blend)]:
        metrics[key] = context.paired(aligned[TARGET].to_numpy(), candidate, baseline, aligned[TIME].dt.floor('D').to_numpy())
    common.write_json(folder / 'metrics.json', metrics)
    common.write_json(folder / 'fit_evidence.json', evidence)
    resources = guard()
    common.write_json(folder / 'manifest.json', dict(status='complete', fold=fold, split=split,
        protocol_sha256=common.sha256(OUT / 'protocol.json'), feature_receipts=receipts,
        fit_ids_hash=binding['fit_ids_hash'], tune_ids_hash=binding['tune_ids_hash'],
        fit_label_hash=binding['fit_label_hash'], tune_label_hash=binding['tune_label_hash'],
        fit_rows=len(fit), tune_rows=len(tune), selected_epochs=model['steps'],
        native_reload_max_delta_sec=0., resources=resources, runtime_sec=time.monotonic() - started,
        outputs={n:common.sha256(folder / n) for n in ['control_replay.json', 'fit_model.joblib', 'tune_predictions.parquet', 'metrics.json', 'fit_evidence.json']}))
    del frame, xf, xt, matrix, model, reload
    gc.collect()
    (folder / 'matrix.float32').unlink()
    print('COMPLETE', fold, {k:v['gain'] for k,v in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare_only:
            declare()
            print(common.sha256(OUT / 'protocol.json'), flush=True)
        else:
            assert args.fold
            run(args.fold)
