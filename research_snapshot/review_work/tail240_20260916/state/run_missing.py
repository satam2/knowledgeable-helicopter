"""Matched missing-route historical template with or without airport state."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import airport_state as state
sys.path.insert(0, str(state.ROOT / 'review_work/breakthrough_20260916/missing'))
import run_missing_models as base
import run_id_context as identity
import joblib
import numpy as np
import pandas as pd
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path

OUT = external_path(state.ROOT / 'private_runs/tail240_20260916/state/models_v1')
CACHE = state.ROOT / 'private_runs/tail240_20260916/state/cache_v1'
GLOBAL = state.ROOT / 'private_runs/breakthrough_20260916/missing/route_composition_v4/global9'
ARMS = ('control', 'both12')
SEED = 20260916


def declaration():
    sources = [Path(__file__), Path(state.__file__), Path(base.__file__), Path(identity.__file__),
               state.ROOT / 'review_work/campaign_20260916/common.py']
    value = {'created_utc': utc_now(), 'arms': ARMS, 'model': 'Original missing HistoricalTemplate + CatBoost residual',
             'parameters': base.PARAMS, 'seed': SEED, 'threads': 2,
             'state': 'control no state; both12 appends frozen separate taxi6/source6 stage caches',
             'fit': 'Full original missing-NM fit/refit rows; same stopping rule; new control fit. Stage state joined by exact ID from original whole-DEP purged caches.',
             'scope': 'No-state versus both12 first screen. Taxi-only/source-only not launched. Adaptive exposed development.',
             'prediction_variants': {'candidate': 'Replace global9 missing route with raw new missing prediction; every finite-NM row preserved exactly',
                                     'blend25': 'Fixed25percent candidate plus75percent completeglobal9; every finite-NM row preserved exactly'},
             'reference': {fold: sha256(GLOBAL / fold / 'manifest.json') for fold in ('F1', 'F3')},
             'state_protocol_sha256': sha256(CACHE.parent / 'protocol.json'),
             'source_hashes': {str(path.relative_to(state.ROOT)): sha256(path) for path in sources},
             'failure_policy': 'No clipping/floor/capping/outlierexclusion; preserve errors and all scoring rows. No score-selected weights or route thresholds.'}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        old = read_json(path)
        comparison = dict(value)
        comparison['created_utc'] = old['created_utc']
        assert comparison == old
        return old
    write_json(path, value)
    return value


def load_missing():
    x, meta = base.load_data()
    cache_marker = read_json(identity.CACHE / 'audit.json')
    assert sha256(identity.CACHE / 'features.parquet') == cache_marker['feature_sha256']
    peer = pd.read_parquet(identity.CACHE / 'features.parquet').set_index(state.ID)
    assert np.array_equal(peer.index, meta[state.ID])
    peer = peer.loc[x.index]
    times = meta.set_index(state.ID).loc[x.index, state.TIME]
    x = pd.concat([x, identity.id_context_features(x, peer, times)], axis=1)
    del peer
    state.guard()
    return x, meta


def run(arm, fold, x, meta, cache_receipt):
    dest = OUT / arm / fold
    dest.mkdir(parents=True, exist_ok=False)
    idx, split, _ = base.common.fold_data(meta, fold, full=True)
    assert object_hash(cache_receipt['folds'][fold]['split']) == object_hash(split)
    original_reference, original_marker = base.common.reference(fold)
    assert object_hash(original_marker['split']) == object_hash(split)
    global_marker = read_json(GLOBAL / fold / 'manifest.json')
    reference_path = GLOBAL / fold / 'candidate.parquet'
    assert global_marker['status'] == 'complete' and sha256(reference_path) == global_marker['outputs']['candidate.parquet']
    reference = pd.read_parquet(reference_path)
    np.testing.assert_array_equal(reference[state.ID], meta.iloc[idx['score']][state.ID])
    np.testing.assert_array_equal(reference[state.TARGET], meta.iloc[idx['score']][state.TARGET])
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    selected = {stage: rows[missing[rows]] for stage, rows in idx.items()}
    frames, clocks = {}, {}
    for stage, rows in selected.items():
        ids = meta.iloc[rows][state.ID]
        frames[stage] = x.loc[ids].copy()
        if arm == 'both12':
            record = cache_receipt['folds'][fold]['stages'][stage]
            path = state.ROOT / record['path']
            assert sha256(path) == record['sha256']
            added = pd.read_parquet(path).set_index(state.ID)
            np.testing.assert_array_equal(added.index, meta.iloc[idx[stage]][state.ID])
            added = added.loc[ids]
            assert list(added) == state.COLUMNS
            frames[stage] = pd.concat([frames[stage], added], axis=1)
        clocks[stage] = pd.Series(pd.to_datetime(meta.iloc[rows][state.TIME], utc=True).to_numpy(), index=frames[stage].index)
    y = meta[state.TARGET].to_numpy(float)
    record = {'status': 'running', 'created_utc': utc_now(), 'arm': arm, 'fold': fold, 'split': split,
              'feature_columns': list(frames['fit']), 'seed': SEED, 'threads': 2,
              'protocol_sha256': sha256(OUT / 'protocol.json'), 'cache_manifest_sha256': sha256(CACHE / 'manifest.json'),
              'reference_manifest_sha256': sha256(GLOBAL / fold / 'manifest.json'),
              'fit_ids': {stage: {'n': len(rows), 'hash': object_hash(meta.iloc[rows][state.ID].tolist())} for stage, rows in selected.items()}}
    write_json(dest / 'manifest.json', record)
    start = time.perf_counter()
    try:
        state.guard()
        model, tuning = base.fit_arm('historical_template', frames['fit'], y[selected['fit']], clocks['fit'],
                                     tune=(frames['tune'], y[selected['tune']], clocks['tune']), seed=SEED, threads=2)
        joblib.dump(model, dest / 'fit_model.joblib')
        tune_pred = base.predict_arm(model, frames['tune'], clocks['tune'])
        pd.DataFrame({state.ID: frames['tune'].index, 'prediction_sec': tune_pred}).to_parquet(dest / 'tune_predictions.parquet', index=False)
        del model
        gc.collect()
        state.guard()
        model, refit = base.fit_arm('historical_template', frames['refit'], y[selected['refit']], clocks['refit'], selected=tuning, seed=SEED, threads=2)
        joblib.dump(model, dest / 'model.joblib')
        pred = base.predict_arm(model, frames['score'], clocks['score'])
        repeated = base.predict_arm(joblib.load(dest / 'model.joblib'), frames['score'], clocks['score'])
        np.testing.assert_allclose(pred, repeated, rtol=0, atol=1e-9)
        assert np.isfinite(pred).all()
        values = reference.prediction_sec.to_numpy(float).copy()
        changed = missing[idx['score']]
        values[changed] = pred
        reports = {}
        for variant, output in {'candidate': values, 'blend25': reference.prediction_sec.to_numpy(float) + .25 * (values-reference.prediction_sec.to_numpy(float))}.items():
            np.testing.assert_array_equal(output[~changed], reference.prediction_sec.to_numpy(float)[~changed])
            frame = reference.drop(columns=[state.TARGET, 'error_sec', 'squared_error', 'label_bin'], errors='ignore').copy()
            frame['prediction_sec'] = output
            metrics, errors = evaluate(frame, meta.iloc[idx['score']][[state.ID, state.TARGET]])
            errors.to_parquet(dest / f'{variant}.parquet', index=False)
            reports[variant] = {'metrics': metrics, 'stability': paired_stability(reference, errors, repetitions=500)}
        record.update(status='complete', completed_utc=utc_now(), tune=tuning, refit=refit, reports=reports,
                      changed_rows=int(changed.sum()), complete_score_rows=len(reference), protected_routes_equal=True,
                      reload_max_abs_delta=float(np.max(np.abs(pred-repeated))), runtime_sec=time.perf_counter()-start,
                      source_hashes=read_json(OUT/'protocol.json')['source_hashes'])
        record['outputs'] = {p.name: sha256(p) for p in dest.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(dest / 'manifest.json', record)
        print('RESULT', arm, fold, {key: value['metrics']['overall']['rmse_sec'] for key,value in reports.items()}, flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), runtime_sec=time.perf_counter()-start)
        write_json(dest/'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declaration()
    if args.declare_only:
        print('Declared matched control and both12; no fit')
        return
    cache_receipt = read_json(CACHE / 'manifest.json')
    assert cache_receipt['status'] == 'complete'
    assert sha256(state.__file__) == cache_receipt['adapter_sha256']
    x, meta = load_missing()
    for arm in ARMS:
        for fold in ('F1', 'F3'):
            run(arm, fold, x, meta, cache_receipt)
    result = {}
    for arm in ARMS:
        manifests = {fold: read_json(OUT/arm/fold/'manifest.json') for fold in ('F1', 'F3')}
        result[arm] = {variant: (192122/344841*manifests['F1']['reports'][variant]['metrics']['overall']['rmse_sec']**2 +
                                 152719/344841*manifests['F3']['reports'][variant]['metrics']['overall']['rmse_sec']**2)**.5
                       for variant in ('candidate','blend25')}
    write_json(OUT/'summary.json', {'status':'complete', 'seasonal_rmse': result, 'no_variant_selected':True})
    print('SUMMARY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
