"""Matched full-cohort finite-NM ATFM pilot; immutable tune-only experiment."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import lightgbm as lgb
import argparse
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/risk'))
import run_risk as risk
import build_features as addition
common, ID, TARGET = risk.common, risk.ID, risk.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/forensics/atfm/tune_f1_v2')
CONTROL = ROOT / 'private_runs/tail240_20260916/models/following_groups_tune_v1'
BASE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
ORIGINAL_SOURCES = risk.feature_sources


def sources(desired):
    result = ORIGINAL_SOURCES(desired)
    chosen = [name for name in desired if name in addition.FEATURES]
    if chosen:
        marker = common.read_json(addition.OUT / 'manifest.json')
        assert marker['status'] == 'complete'
        assert marker['source_sha256'] == common.sha256(addition.__file__)
        path = addition.OUT / 'training_features.parquet'
        expected = marker['outputs'][path.name]
        assert common.sha256(path) == expected
        result.append((path, expected, False, chosen))
    return result


def guard(_env=None):
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert memory.rss < 18 * 1024**3 and peak < 18 * 1024**3, f'ATFM F1 18 GiB process ceiling exceeded: RSS={memory.rss}, peak={peak}'
    assert psutil.virtual_memory().available >= 8 * 1024**3, 'Host reserve below 8 GiB'
    return memory.rss


def declare():
    controls = {fold: common.read_json(risk.union_folder(fold) / 'manifest.json') for fold in ('F1', 'F3')}
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    columns = controls['F1']['feature_columns']
    assert columns == controls['F3']['feature_columns'] and len(columns) == 387
    assert len(addition.FEATURES) == 29 and not set(columns).intersection(addition.FEATURES)
    record = dict(source_sha256=common.sha256(__file__),
        imported_sources={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in [risk.__file__, addition.__file__]},
        params=controls['F1']['fit']['params'], columns=columns + addition.FEATURES,
        cache_manifest_sha256=common.sha256(addition.OUT / 'manifest.json'),
        original_manifests={fold: common.sha256(risk.union_folder(fold) / 'manifest.json') for fold in controls},
        global9_preparation_sha256=common.sha256(BASE / 'preparation.json'),
        target='Raw Y minus finite NM proxy, every original finite fit/tune row after original flight-ID purges. No clipping or sampling.',
        fitting='Unchanged leaf63 control parameters and 150-round early stopping; fit-only categorical vocabulary; 29 numeric ATFM fields retain NaN.',
        control='Reuse freshly fitted following_groups_tune_v1/control387 only after split, IDs, params, feature columns, source integrity and original prediction parity <=1e-7 pass. Missing control stops run.',
        scope='F1 June tune only under this resource declaration; F3 requires a new declaration after observed F1 memory. No score or refit. No ranking labels read.',
        availability='Whole-day public aggregates published in arrears; retrospective supplied-batch covariates. UTC movement date join is a declared assumption; source timezone is unspecified.',
        gate='Both months: candidate improves matched union387 on all finite and ordinary proxy[0,7200], with every one-day deletion positive; fixed25 complement to global9 ordinary improves by >2 seconds and every one-day deletion stays positive. Independent replay required before advancement.',
        diagnostics='Report fixed replacement of existing lgb63_union weighted component, without fitting new weights. Diagnostic only; no alternate promotion rule.',
        caution='Global9 weights were learned on the same tune month; this is adaptive development, not fresh validation.',
        resources='F1 only: 2 CPU, current and OS peak process working set below 18 GiB, host available reserve >=8 GiB. Launch requires >=28 GiB available and parent serialized allocation. F3 not authorized by this declaration.',
        licensing='EUROCONTROL PRU attribution and local noncommercial research only; competition prize eligibility unresolved.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def compare(y, prediction, control, dates):
    gain = (y - control)**2 - (y - prediction)**2
    days, codes = np.unique(dates, return_inverse=True)
    counts = np.bincount(codes)
    removal = (gain.sum() - np.bincount(codes, weights=gain)) / (len(y) - counts)
    reference_rmse = float(np.sqrt(np.mean((y - control)**2)))
    rmse = float(np.sqrt(np.mean((y - prediction)**2)))
    return dict(n=len(y), rmse=rmse, reference_rmse=reference_rmse,
        rmse_improvement=reference_rmse-rmse, mse_gain=float(gain.mean()),
        day_removal_min_gain=float(removal.min()), all_day_removals_improve=bool((removal > 0).all()), days=len(days))


def run(fold):
    protocol = declare()
    assert fold == 'F1', 'F3 requires a separate memory declaration after F1 evidence'
    assert psutil.virtual_memory().available >= 28 * 1024**3, 'Launch requires 28 GiB host available'
    guard()
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    frames = {stage: meta.iloc[positions[np.isfinite(meta.iloc[positions].proxy_sec.to_numpy())]].copy()
              for stage, positions in idx.items() if stage in ('fit', 'tune')}
    cohort = {stage: dict(n=len(frame), hash=common.object_hash(frame[ID].tolist())) for stage, frame in frames.items()}
    ids = pd.Index(pd.concat([frames['fit'][ID], frames['tune'][ID]], ignore_index=True))
    nfit = len(frames['fit'])
    dest = OUT / fold
    assert not dest.exists(), 'Immutable fold destination already exists'
    control_dir = CONTROL / fold / 'control387'
    marker = common.read_json(control_dir / 'manifest.json')
    control_protocol = common.read_json(CONTROL / 'protocol.json')
    assert marker['status'] == 'complete' and marker['params'] == protocol['params']
    assert common.sha256(ROOT / 'review_work/tail240_20260916/models/following_groups_tune.py') == marker['source_sha256']
    assert common.sha256(CONTROL / 'protocol.json') == marker['protocol_sha256']
    assert control_protocol['full_columns'] == protocol['columns'][:387]
    assert marker['ids'] == cohort and common.object_hash(marker['split']) == common.object_hash(split)
    for filename, expected in marker['outputs'].items():
        assert common.sha256(control_dir / filename) == expected
    control = pd.read_parquet(control_dir / 'tune.parquet')
    np.testing.assert_array_equal(control[ID], frames['tune'][ID])
    original_dir = risk.union_folder(fold)
    original_marker = common.read_json(original_dir / 'manifest.json')
    assert common.sha256(original_dir / 'tune_predictions.parquet') == original_marker['outputs']['tune_predictions.parquet']
    original = pd.read_parquet(original_dir / 'tune_predictions.parquet')
    np.testing.assert_array_equal(original[ID], control[ID])
    parity = float(np.max(np.abs(original.prediction_sec.to_numpy() - control.prediction_sec.to_numpy())))
    assert parity <= 1e-7 and marker['original_control_max_abs_delta'] <= 1e-7
    control_encoder = common.read_json(control_dir / 'encoder.json')
    assert control_encoder['columns'] == protocol['columns'][:387]
    dest.mkdir()
    common.write_json(dest / 'control_receipt.json', dict(manifest_sha256=common.sha256(control_dir / 'manifest.json'),
        control_path=str(control_dir), original_max_abs_delta=parity, ids=cohort, split=split))
    risk.feature_sources = sources
    risk.guard = guard
    matrix, vocab, receipts = risk.load_matrix(ids, nfit, protocol['columns'], dest)
    assert vocab == control_encoder['vocab']
    target_fit = frames['fit'][TARGET].to_numpy(float) - frames['fit'].proxy_sec.to_numpy(float)
    target_tune = frames['tune'][TARGET].to_numpy(float) - frames['tune'].proxy_sec.to_numpy(float)
    guard()
    model = lgb.LGBMRegressor(**protocol['params'])
    model.fit(matrix[:nfit], target_fit, categorical_feature=[protocol['columns'].index(name) for name in vocab],
        feature_name=protocol['columns'], eval_X=matrix[nfit:], eval_y=target_tune, eval_metric='rmse',
        callbacks=[guard, lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
    steps = int(model.best_iteration_ or model.n_estimators_)
    proxy = frames['tune'].proxy_sec.to_numpy(float)
    prediction = model.predict(matrix[nfit:], num_iteration=steps) + proxy
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest / 'model.txt'), num_iteration=steps)
    replay_model = lgb.Booster(model_file=str(dest / 'model.txt'))
    replay = replay_model.predict(matrix[nfit:]) + proxy
    np.testing.assert_array_equal(prediction, replay)
    pd.DataFrame({ID: frames['tune'][ID], 'prediction_sec': prediction}).to_parquet(dest / 'tune.parquet', index=False)
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab))
    peak = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
    guard()
    common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold, steps=steps,
        params=protocol['params'], split=split, ids=cohort, source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT / 'protocol.json'), feature_receipts=receipts,
        peak_bytes=peak, native_replay_max_abs_delta=0.,
        outputs={name: common.sha256(dest / name) for name in ['model.txt', 'tune.parquet', 'encoder.json', 'control_receipt.json']}))
    del model, replay_model, matrix
    gc.collect()
    (dest / 'matrix.float32').unlink()
    y = frames['tune'][TARGET].to_numpy(float)
    dates = frames['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    ordinary = (proxy >= 0) & (proxy <= 7200)
    matched = compare(y, prediction, control.prediction_sec.to_numpy(), dates)
    matched_ordinary = compare(y[ordinary], prediction[ordinary], control.prediction_sec.to_numpy()[ordinary], dates[ordinary])
    prep = common.read_json(BASE / 'preparation.json')['folds'][fold]
    aligned_path = BASE / f'{fold}_aligned_tune.parquet'
    assert common.sha256(aligned_path) == prep['aligned_tune_sha256']
    weights = common.read_json(BASE / f'{fold}_weights.json')
    assert weights == prep['weights']
    aligned = pd.read_parquet(aligned_path)
    np.testing.assert_array_equal(aligned[ID], frames['tune'].loc[ordinary, ID])
    np.testing.assert_array_equal(aligned[TARGET], y[ordinary])
    np.testing.assert_allclose(aligned['lgb63_union'], control.prediction_sec.to_numpy()[ordinary], atol=1e-7, rtol=0)
    reference = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    blended = .75 * reference + .25 * prediction[ordinary]
    fixed25 = compare(y[ordinary], blended, reference, dates[ordinary])
    union_weight = weights['global'][weights['experts'].index('lgb63_union')]
    replaced = reference + union_weight * (prediction[ordinary] - aligned['lgb63_union'].to_numpy())
    replacement = compare(y[ordinary], replaced, reference, dates[ordinary])
    fold_gate = all(record['mse_gain'] > 0 and record['all_day_removals_improve'] for record in [matched, matched_ordinary, fixed25]) and fixed25['rmse_improvement'] > 2
    report = dict(status='complete', fold=fold, matched_all_finite=matched, matched_ordinary=matched_ordinary,
        ordinary_fixed25=fixed25, ordinary_union_replacement=replacement, replacement_weight=union_weight,
        fold_gate=fold_gate, no_score_prediction=True, caveat=protocol['caution'])
    common.write_json(dest / 'assessment.json', report)
    print('RESULT', fold, report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1'])
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print('Declared ATFM416 F1 v2 with 18 GiB process ceiling; no fit launched', flush=True)
    else:
        assert args.fold
        run(args.fold)
