"""Isolate CatBoost's omitted162 fields with the frozen225 training recipe."""
import torch
import lightgbm
import argparse
import gc
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits
import schema_discovery
import following_groups_tune as shared

ROOT, common, risk, ID, TARGET = shared.ROOT, shared.common, shared.risk, shared.ID, shared.TARGET
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/neural_context'))
import run as context
decode_frame = context.decode_frame
ADAPTER_PATH = ROOT / 'review_work/breakthrough_20260916/models_retrieval/combined_catboost/adapter.py'
spec = importlib.util.spec_from_file_location('union387_frozen_catboost_adapter', ADAPTER_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/catboost_union387_v1')
ENSEMBLE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
NEURAL = {fold: ROOT / 'private_runs/tail240_20260916/state/neural_context' / version / fold
          for fold, version in [('F1', 'v1'), ('F3', 'v3')]}


def control_folder(fold):
    return ROOT / 'private_runs/breakthrough_20260916/combined_catboost' / f'catboost_combined_aobt_allfinite_{fold}_s20260916'


def compose(aligned, weights, current_neural, new_catboost):
    coefficients = np.asarray(weights['global_weights'], dtype=float)
    experts = weights['experts']
    current = aligned[experts].to_numpy(float) @ coefficients
    current += coefficients[experts.index('tabm_ple8')] * (current_neural-aligned.tabm_ple8.to_numpy(float))
    proposed = current + coefficients[experts.index('catboost_combined')] * (new_catboost-aligned.catboost_combined.to_numpy(float))
    return current, proposed


def compare(y, candidate, reference, days):
    value = shared.metric.comparison(y, candidate, reference, days)
    value.update(rmse=float(np.sqrt(np.mean((y-candidate)**2))), reference_rmse=float(np.sqrt(np.mean((y-reference)**2))))
    value['gain'] = value['reference_rmse']-value['rmse']
    return value


def declare():
    controls = {f: common.read_json(control_folder(f) / 'manifest.json') for f in ['F1', 'F3']}
    columns = common.read_json(risk.union_folder('F1') / 'manifest.json')['feature_columns']
    assert len(columns) == 387
    for fold, marker in controls.items():
        assert marker['status'] == 'complete' and marker['feature_columns'] == columns[:225]
        assert marker['fit']['params'] == adapter.parameters()
    paths = [Path(__file__), ADAPTER_PATH, Path(sys.modules['encoders'].__file__), Path(context.__file__),
             Path(risk.__file__), Path(schema_discovery.__file__), Path(__file__).with_name('test_catboost_union387.py')]
    bindings = {}
    for fold, marker in controls.items():
        bindings[fold] = dict(manifest_sha256=common.sha256(control_folder(fold) / 'manifest.json'),
            fit_ids=marker['fit_ids'], outputs=marker['outputs'],
            neural_manifest_sha256=common.sha256(NEURAL[fold] / 'manifest.json'),
            weights_sha256=common.sha256(ENSEMBLE / f'{fold}_weights.json'),
            aligned_sha256=common.sha256(ENSEMBLE / f'{fold}_aligned_tune.parquet'))
    record = dict(status='declared', source_hashes={str(p.relative_to(ROOT)): common.sha256(p) for p in paths},
        columns=columns, parameters=adapter.parameters(), controls=bindings,
        contrast='ONLY input225to387:112 ordered event fields plus50 retrospective aggregate fields. Exact original adapter/fit-only encoder/rawY-NM/5000cap/depth8/lr.05/L2=8/100patience/seed20260916/2CPU.',
        cohort='All original finite fit/tune rows after full-flight purges; original labels; no weights/clipping/sampling/score/ranking.',
        replay='Beforefit, reconstruct first225 and replay old saved fit model on every tune row, absolute tolerance1e-7 seconds. Candidate saved nativeCBM and joblib CPU replay <=1e-7.',
        primary='Replace original catboost_combined component at original global9 coefficient inside CURRENT PLE387 ensemble. No coefficient refit. Matched CB225 standalone secondary.',
        gate='Both primary months positive and all day removals positive, >=2seconds seasonal ordinary RMSE gain. F1failure holdsF3. No automatic refit or score authorization.',
        weights=[192122/344841, 152719/344841],
        resources='One exclusive GPU, native frozen gpu_ram_part.65;2CPU;20GiBOSpeak/current process cap sampled every2seconds with ownprocess exit87 and failure receipt;8GiBhostreserve;startup28GiB. GPU synthetic canary and independent preflight required before parent allocation.',
        limitation='Exposed development folds and original weights fitted on same tune labels. GPU training stochastic. This is an information test, not a promised10sec gain or new algorithm.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record, 'Frozen protocol or source changed'
    else:
        common.write_json(path, record)
    return record


def memory():
    info = psutil.Process().memory_info()
    return dict(rss=info.rss, peak=getattr(info, 'peak_wset', info.rss), available=psutil.virtual_memory().available)


def guard():
    state = memory()
    assert max(state['rss'], state['peak']) < 20*1024**3, state
    assert state['available'] >= 8*1024**3, state
    return state


def resource_monitor(stop, dest):
    while not stop.wait(2):
        state = memory()
        if max(state['rss'], state['peak']) >= 20*1024**3 or state['available'] < 8*1024**3:
            common.write_json(dest / 'resource_failure.json', dict(status='resource_failed', resources=state, utc=common.utc_now()))
            os._exit(87)


def fit_fold(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 28*1024**3
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    stop = threading.Event()
    monitor = threading.Thread(target=resource_monitor, args=(stop, dest), daemon=True)
    monitor.start()
    started = time.monotonic()
    try:
        original = risk.feature_sources
        def sources(columns):
            result, receipt = schema_discovery.cached_discovery(original, columns)
            common.write_json(dest / 'discovery.json', receipt)
            return result
        risk.feature_sources, risk.guard = sources, guard
        metadata = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
        assert common.sha256(metadata) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
        meta = pd.read_parquet(metadata, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
        indices, split, _ = common.fold_data(meta, fold, full=True)
        parts = {stage: meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy()
                 for stage, p in indices.items() if stage in ['fit', 'tune']}
        del meta
        fit, tune = parts['fit'], parts['tune']
        binding = protocol['controls'][fold]
        for stage, frame in parts.items():
            assert dict(n=len(frame), hash=common.object_hash(frame[ID].tolist())) == binding['fit_ids'][stage]
        ids = pd.Index(pd.concat([fit[ID], tune[ID]], ignore_index=True))
        matrix, vocab, receipts = risk.load_matrix(ids, len(fit), protocol['columns'], dest)
        frame = decode_frame(matrix, vocab, protocol['columns'])
        frame.index = ids
        xf, xt = frame.iloc[:len(fit)], frame.iloc[len(fit):]
        reference = control_folder(fold)
        assert common.sha256(reference / 'manifest.json') == binding['manifest_sha256']
        for name in ['fit_model.joblib', 'tune_predictions.parquet']:
            assert common.sha256(reference / name) == binding['outputs'][name]
        old_model = joblib.load(reference / 'fit_model.joblib')
        stored = pd.read_parquet(reference / 'tune_predictions.parquet').set_index(ID).loc[tune[ID]]
        np.testing.assert_array_equal(stored[TARGET], tune[TARGET])
        proxy = tune.proxy_sec.to_numpy(float)
        control_prediction = stored.prediction_sec.to_numpy(float)
        replay = adapter.predict(old_model, xt[protocol['columns'][:225]]) + proxy
        delta = float(np.max(np.abs(replay-control_prediction)))
        assert delta <= 1e-7, delta
        common.write_json(dest / 'control_replay.json', dict(status='passed', rows=len(tune), max_abs_delta_sec=delta))
        del old_model, stored, replay
        gc.collect()
        guard()
        model, evidence = adapter.fit(xf, (fit[TARGET]-fit.proxy_sec).to_numpy(float),
            (xt, (tune[TARGET]-tune.proxy_sec).to_numpy(float)), seed=20260916, threads=2)
        prediction = adapter.predict(model, xt) + proxy
        joblib.dump(model, dest / 'fit_model.joblib')
        model['estimator'].save_model(str(dest / 'model.cbm'))
        replay = adapter.predict(joblib.load(dest / 'fit_model.joblib'), xt) + proxy
        np.testing.assert_allclose(replay, prediction, atol=1e-7, rtol=0)
        native = adapter.catboost.CatBoostRegressor()
        native.load_model(str(dest / 'model.cbm'))
        replay_native = native.predict(model['encoder'].transform(xt), thread_count=2) + proxy
        np.testing.assert_allclose(replay_native, prediction, atol=1e-7, rtol=0)
        output = tune[[ID, common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec']].assign(prediction_sec=prediction)
        output.to_parquet(dest / 'tune_predictions.parquet', index=False)
        for name, key in [(f'{fold}_weights.json', 'weights_sha256'), (f'{fold}_aligned_tune.parquet', 'aligned_sha256')]:
            assert common.sha256(ENSEMBLE / name) == binding[key]
        aligned = pd.read_parquet(ENSEMBLE / f'{fold}_aligned_tune.parquet')
        ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
        np.testing.assert_array_equal(aligned[ID], tune.loc[ordinary, ID])
        np.testing.assert_array_equal(aligned[TARGET], tune.loc[ordinary, TARGET])
        assert common.sha256(NEURAL[fold] / 'manifest.json') == binding['neural_manifest_sha256']
        neural_marker = common.read_json(NEURAL[fold] / 'manifest.json')
        neural_path = NEURAL[fold] / 'tune_predictions.parquet'
        assert common.sha256(neural_path) == neural_marker['outputs'][neural_path.name]
        neural = pd.read_parquet(neural_path).set_index(ID).loc[aligned[ID]]
        np.testing.assert_array_equal(neural[TARGET], aligned[TARGET])
        weights = common.read_json(ENSEMBLE / f'{fold}_weights.json')
        current, proposed = compose(aligned, dict(experts=weights['experts'], global_weights=weights['global']),
            neural.prediction_sec.to_numpy(float), prediction[ordinary])
        y, days = tune[TARGET].to_numpy(float), tune[common.MOVEMENT].dt.floor('D').to_numpy()
        metrics = dict(matched_all_finite=compare(y, prediction, control_prediction, days),
            primary_current387=compare(y[ordinary], proposed, current, days[ordinary]))
        common.write_json(dest / 'fit_evidence.json', evidence)
        common.write_json(dest / 'metrics.json', metrics)
        common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold,
            protocol_sha256=common.sha256(OUT / 'protocol.json'), split=split, columns=protocol['columns'],
            ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s, f in parts.items()},
            source_receipts=receipts, metrics=metrics, resources=guard(), runtime_sec=time.monotonic()-started,
            native_max_abs_delta=float(np.max(np.abs(replay_native-prediction))),
            no_score_prediction=True, outputs={name: common.sha256(dest / name) for name in
                ['fit_model.joblib', 'model.cbm', 'tune_predictions.parquet', 'fit_evidence.json', 'metrics.json', 'control_replay.json']}))
        print('COMPLETE', fold, metrics, flush=True)
    finally:
        stop.set()
        monitor.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    with threadpool_limits(2):
        pa.set_cpu_count(2)
        pa.set_io_thread_count(1)
        if args.declare_only:
            declare()
        else:
            assert args.fold
            fit_fold(args.fold)
