"""Matched historical-template model with observable public date hypotheses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
from pathlib import Path
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state'))
import run_missing as shared
base = shared.base
common = base.common
ID, TARGET, TIME = shared.state.ID, shared.state.TARGET, shared.state.TIME
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.metrics import evaluate, paired_stability
from taxiout.paths import external_path

OUT = external_path(ROOT / 'private_runs/tail240_20260916/models/opdi_dates_missing_v1')
CACHE = ROOT / 'private_runs/tail240_20260916/source_distinctions/opdi_dates_v1'
CONTROL = ROOT / 'private_runs/tail240_20260916/state/models_v2/control'
BASELINE = ROOT / 'private_runs/breakthrough_20260916/missing/route_composition_v4/global9'
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'


def sources():
    return {str(path.relative_to(ROOT)): sha256(path) for path in
            [Path(__file__), Path(shared.__file__), Path(base.__file__), Path(shared.identity.__file__),
             ROOT / 'review_work/campaign_20260916/common.py',
             ROOT / 'review_work/tail240_20260916/source_distinctions/opdi_alias_dates.py']}


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    value = {'created_utc': utc_now(), 'name': 'historical_template_plus_public_alias_date45',
        'source_hashes': sources(), 'baseline_binding_sha256': sha256(BINDING),
        'score_labels_used_for_selection': False, 'routing_uses_score_targets': False,
        'selection_periods': ['fit', 'tune'], 'prediction_transform': 'raw_unclipped',
        'tail_definition': 'No label-derived tail route. All original missingNM records remain eligible.',
        'routing_definition': 'Missing proxy only. All finite-NM values remain completeglobal9 exactly.',
        'selection_rule': 'Original HistoricalTemplate+CatBoost600depth5L2=30; originaltune stopping80; refitselectedtrees. Two predeclared fixed weights1and.25, no score choice.',
        'parameters': base.PARAMS, 'threads': 2, 'seed': 20260916,
        'features': 'Exact existing76airport+IDfeatures plus45observable OPDIflightlist date/mapping support and offsets. Sameclock/targethistories asoldcontrol.',
        'availability': 'Retrospective supplied/public batch context. No target/block inputs; publicfirst_seen is not offblock or guaranteed flightidentity.',
        'inference_columns': ['airport_record_fields', 'observed_record_order_context', 'public_flight_code_route_first_last_seen', 'earlier_month_prefix_mapping'],
        'cache_protocol_sha256': sha256(CACHE / 'protocol.json'),
        'controls': {fold: sha256(CONTROL / fold / 'manifest.json') for fold in ('F1', 'F3')},
        'variants': ['candidate', 'blend25']}
    path = OUT / 'protocol.json'
    if path.exists():
        old = read_json(path)
        value['created_utc'] = old['created_utc']
        assert value == old
    else:
        write_json(path, value)
    return value


def run(fold, x, meta, protocol, cache_manifest):
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    idx, split, _ = common.fold_data(meta, fold, full=True)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    rows = {stage: positions[missing[positions]] for stage, positions in idx.items()}
    ids = {stage: {'n': len(pos), 'hash': object_hash(meta.iloc[pos][ID].tolist())} for stage, pos in rows.items()}
    marker = read_json(CONTROL / fold / 'manifest.json')
    assert marker['status'] == 'complete' and ids == marker['fit_ids']
    assert object_hash(split) == object_hash(marker['split'])
    frames = {stage: x.loc[meta.iloc[pos][ID]] for stage, pos in rows.items()}
    times = {stage: pd.Series(pd.to_datetime(meta.iloc[pos][TIME], utc=True).to_numpy(), index=frames[stage].index)
             for stage, pos in rows.items()}
    record = {'status': 'running', 'created_utc': utc_now(), 'fold': fold, 'fit_ids': ids, 'split': split,
              'feature_columns': list(x), 'source_hashes': sources(), 'protocol_sha256': sha256(OUT / 'protocol.json'),
              'cache_manifest_sha256': sha256(CACHE / 'manifest.json'), 'control_manifest_sha256': sha256(CONTROL / fold / 'manifest.json')}
    write_json(dest / 'manifest.json', record)
    started = time.monotonic()
    try:
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise MemoryError('Host reserve')
        y = meta[TARGET].to_numpy(float)
        fit, tuned = base.fit_arm('historical_template', frames['fit'], y[rows['fit']], times['fit'],
            tune=(frames['tune'], y[rows['tune']], times['tune']), seed=20260916, threads=2)
        joblib.dump(fit, dest / 'fit_model.joblib')
        pd.DataFrame({ID: frames['tune'].index, 'prediction_sec': base.predict_arm(fit, frames['tune'], times['tune'])}).to_parquet(dest / 'tune_predictions.parquet', index=False)
        del fit
        gc.collect()
        model, refitted = base.fit_arm('historical_template', frames['refit'], y[rows['refit']], times['refit'],
                                      selected=tuned, seed=20260916, threads=2)
        joblib.dump(model, dest / 'model.joblib')
        pred = base.predict_arm(model, frames['score'], times['score'])
        replay = base.predict_arm(joblib.load(dest / 'model.joblib'), frames['score'], times['score'])
        np.testing.assert_allclose(pred, replay, rtol=0, atol=1e-9)
        baseline_marker = read_json(BASELINE / fold / 'manifest.json')
        path = BASELINE / fold / 'candidate.parquet'
        assert sha256(path) == baseline_marker['outputs']['candidate.parquet']
        baseline = pd.read_parquet(path)
        np.testing.assert_array_equal(baseline[ID], meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(baseline[TARGET], y[idx['score']])
        mask = missing[idx['score']]
        candidate = baseline.prediction_sec.to_numpy(float).copy()
        candidate[mask] = pred
        reports = {}
        for variant, weight in [('candidate', 1.), ('blend25', .25)]:
            values = baseline.prediction_sec.to_numpy(float) + weight * (candidate - baseline.prediction_sec.to_numpy(float))
            np.testing.assert_array_equal(values[~mask], baseline.prediction_sec.to_numpy(float)[~mask])
            frame = baseline.drop(columns=[TARGET, 'error_sec', 'squared_error', 'label_bin'], errors='ignore').copy()
            frame['prediction_sec'] = values
            metrics, errors = evaluate(frame, baseline[[ID, TARGET]])
            errors.to_parquet(dest / f'{variant}.parquet', index=False)
            reports[variant] = {'metrics': metrics, 'stability': paired_stability(baseline, errors, repetitions=500)}
        assert sources() == record['source_hashes']
        record.update(status='complete', completed_utc=utc_now(), tune=tuned, refit=refitted,
                      reports=reports, runtime_sec=time.monotonic() - started, reload_max_abs_delta=float(np.max(np.abs(pred-replay))))
        record['outputs'] = {p.name: sha256(p) for p in dest.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(dest / 'manifest.json', record)
        print('RESULT', fold, {variant: r['metrics']['overall']['rmse_sec'] for variant, r in reports.items()}, flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write_json(dest / 'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print('Declared publicdate45 missing experiment; no fitting', flush=True)
        return
    cache = read_json(CACHE / 'manifest.json')
    assert cache['status'] == 'complete'
    assert sha256(CACHE / 'protocol.json') == cache['protocol_sha256']
    assert sha256(CACHE / 'features.parquet') == cache['outputs']['features.parquet']
    extension = pd.read_parquet(CACHE / 'features.parquet').set_index(ID)
    x, meta = shared.load_missing()
    assert len(x.columns) == 76
    assert list(x) == read_json(CONTROL / 'F1/manifest.json')['feature_columns']
    assert x.index.isin(extension.index).all() and extension.index.is_unique
    assert len(extension.columns) == 45 and not set(extension).intersection(x)
    x = pd.concat([x, extension.loc[x.index]], axis=1)
    for fold in ('F1', 'F3'):
        run(fold, x, meta, protocol, cache)
    scores = {}
    for variant in ('candidate', 'blend25'):
        score = {fold: read_json(OUT / fold / 'manifest.json')['reports'][variant]['metrics']['overall']['rmse_sec'] for fold in ('F1', 'F3')}
        scores[variant] = (192122/344841*score['F1']**2 + 152719/344841*score['F3']**2)**.5
    write_json(OUT / 'summary.json', {'status': 'complete', 'seasonal_rmse': scores, 'variants_predeclared': True})
    print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
