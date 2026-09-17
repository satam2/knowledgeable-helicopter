"""Matched 436-field test of observed off-block peer membership."""
import following_groups_tune as shared
import schema_discovery
import argparse
import gc
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

ROOT, common, risk, ID, TARGET = shared.ROOT, shared.common, shared.risk, shared.ID, shared.TARGET
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/forensics/nm_clock_peers'))
import build_v1 as peers

OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/nm_clock_peers_tune_v1')
ORIGINAL = risk.feature_sources
FOLD = None
NEURAL = {f: ROOT / 'private_runs/tail240_20260916/state/neural_context' / v / f
          for f, v in [('F1', 'v1'), ('F3', 'v3')]}


def declaration():
    controls = {f: common.read_json(risk.union_folder(f) / 'manifest.json') for f in ('F1', 'F3')}
    assert controls['F1']['feature_columns'] == controls['F3']['feature_columns']
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    assert len(peers.COLUMNS) == 49
    columns = controls['F1']['feature_columns'] + peers.COLUMNS
    assert len(columns) == len(set(columns)) == 436
    paths = [__file__, shared.__file__, schema_discovery.__file__, risk.__file__, peers.__file__]
    record = dict(source_sha256=common.sha256(__file__),
        sources={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in paths},
        peer_protocol_sha256=common.sha256(peers.OUT / 'protocol.json'),
        columns=columns, params=controls['F1']['fit']['params'],
        controls={f: common.sha256(risk.union_folder(f) / 'manifest.json') for f in controls},
        neural_controls={f: common.sha256(p / 'manifest.json') for f, p in NEURAL.items()},
        difference='Same frozen union387 leaf63 plus49 observedNM-selected peer moments. Prior fresh387 native controls reproduced exactly; no new control fit.',
        training='All original finite fit/tune, original flight-ID purges, raw Y-P, fit-only native categories, same2500cap/150patience/seed/2CPU.',
        primary='Fixed25 candidate blend with current387-neural global9 ordinary tune baseline. Baseline=oldglobal9+wPLE*(savedPLE387-oldPLE225).',
        gate='Both months matched436 beats387 and primary improves, all leave-one-day-out gains positive, seasonal primary >=2seconds. No alternate promotion path.',
        resources='2CPU, sampled10GiB processRSS,8GiB hostreserve,18GiB startup; central allocation required.',
        limitation='Repeatedly exposed tune with global9 weights fitted on tune. No independent generalization claim, no score or ranking reads.')
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
    marker = common.read_json(peers.OUT / 'manifest.json')
    assert marker['status'] == 'complete'
    assert marker['source_sha256'] == common.sha256(peers.__file__)
    assert marker['protocol_sha256'] == common.sha256(peers.OUT / 'protocol.json')
    assert marker['feature_columns'] == peers.COLUMNS and marker['rows'] == 2085047
    path = peers.OUT / 'features.parquet'
    digest = marker['outputs']['features.parquet']
    assert common.sha256(path) == digest
    result.append((path, digest, True, peers.COLUMNS))
    return result


def compare(y, candidate, reference, dates):
    record = shared.metric.comparison(y, candidate, reference, dates)
    record.update(rmse=float(np.sqrt(np.mean((candidate-y)**2))),
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
    target = {s: f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s, f in parts.items()}
    model = lgb.LGBMRegressor(**protocol['params'])
    model.fit(matrix[:nfit], target['fit'], categorical_feature=[protocol['columns'].index(c) for c in vocab],
        feature_name=protocol['columns'], eval_X=matrix[nfit:], eval_y=target['tune'], eval_metric='rmse',
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
    steps = int(model.best_iteration_ or model.n_estimators_)
    prediction = model.predict(matrix[nfit:], num_iteration=steps) + parts['tune'].proxy_sec.to_numpy(float)
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest / 'model.txt'), num_iteration=steps)
    replay = lgb.Booster(model_file=str(dest / 'model.txt')).predict(matrix[nfit:]) + parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(prediction, replay)
    pd.DataFrame({ID: parts['tune'][ID], 'prediction_sec': prediction}).to_parquet(dest / 'tune.parquet', index=False)
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab))
    control_folder = risk.union_folder(fold)
    control_manifest = common.read_json(control_folder / 'manifest.json')
    assert common.sha256(control_folder / 'tune_predictions.parquet') == control_manifest['outputs']['tune_predictions.parquet']
    control = pd.read_parquet(control_folder / 'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID], parts['tune'][ID])
    y = parts['tune'][TARGET].to_numpy(float)
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
    neural_manifest = common.read_json(NEURAL[fold] / 'manifest.json')
    assert common.sha256(NEURAL[fold] / 'tune_predictions.parquet') == neural_manifest['outputs']['tune_predictions.parquet']
    neural = pd.read_parquet(NEURAL[fold] / 'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET], aligned[TARGET])
    oldbase = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    pleweight = weights['global'][weights['experts'].index('tabm_ple8')]
    baseline = oldbase + pleweight * (neural.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
    primary = compare(y[ordinary], .75*baseline+.25*prediction[ordinary], baseline, dates[ordinary])
    common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold, steps=steps, split=split,
        source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
        feature_receipts=receipts, params=protocol['params'],
        ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s, f in parts.items()},
        matched=matched, ordinary_fixed25=primary, native_replay_max_abs_delta=0., no_score_prediction=True,
        outputs={p.name: common.sha256(p) for p in dest.iterdir()
                 if p.name in ['model.txt', 'encoder.json', 'tune.parquet', 'discovery.json']}))
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
