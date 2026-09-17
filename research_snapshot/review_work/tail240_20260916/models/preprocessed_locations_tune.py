"""Matched union387 preprocessing: unknown locations are not physical groups."""
import following_groups_tune as shared
import schema_discovery
import argparse
import gc
from pathlib import Path
import sys
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

ROOT, common, risk, ID, TARGET = shared.ROOT, shared.common, shared.risk, shared.ID, shared.TARGET
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/preprocessing_semantics'))
import semantic_locations as semantic
import linear_finite_tune as comparison
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/preprocessed_locations_v1')
ORIGINAL = risk.feature_sources
NATIVE = {f: ROOT / 'private_runs/tail240_20260916/models' / version / f / 'control387'
          for f, version in [('F1', 'following_groups_tune_v1'), ('F3', 'following_groups_tune_v2')]}


def guard(_env=None):
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    assert max(info.rss, peak) < 10*1024**3 and psutil.virtual_memory().available >= 8*1024**3
    return int(peak)


def preprocess(matrix, columns, stand, runway):
    if len(matrix) != len(stand) or len(matrix) != len(runway):
        raise ValueError('Location and feature rows differ')
    valid = dict(stand=semantic.known_location(stand, encoded=True), runway=semantic.known_location(runway, encoded=True))
    groups = semantic.feature_groups(columns)
    changed = {}
    for group, key, value in [('stand', 'stand', -999999.), ('query_runway', 'runway', -999999.),
                             ('arrival_runway', 'runway', -999999.), ('stand_equality', 'stand', 0.),
                             ('runway_equality', 'runway', 0.)]:
        rows = np.flatnonzero(~valid[key])
        count = 0
        for name in groups[group]:
            j = columns.index(name)
            count += int(np.sum(matrix[rows, j] != value))
            matrix[rows, j] = value
        changed[group] = dict(query_rows=len(rows), columns=groups[group], changed_cells=count)
    return changed


def declare():
    controls = {f: common.read_json(risk.union_folder(f) / 'manifest.json') for f in ['F1', 'F3']}
    columns = controls['F1']['feature_columns']
    assert len(columns) == 387 and controls['F3']['feature_columns'] == columns
    assert controls['F1']['fit']['params'] == controls['F3']['fit']['params']
    paths = [Path(__file__), Path(semantic.__file__), Path(risk.__file__), Path(schema_discovery.__file__),
             Path(comparison.__file__), Path(__file__).with_name('test_preprocessed_locations.py')]
    record = dict(source_hashes={str(p.relative_to(ROOT)): common.sha256(p) for p in paths},
        columns=columns, groups=semantic.feature_groups(columns), parameters=controls['F1']['fit']['params'],
        controls={f: common.sha256(risk.union_folder(f) / 'manifest.json') for f in controls},
        native_controls={f: common.sha256(NATIVE[f] / 'manifest.json') for f in controls},
        neural_controls={f: common.sha256(comparison.NEURAL[f] / 'manifest.json') for f in controls},
        helper_evidence_sha256=common.sha256(ROOT / 'private_runs/tail240_20260916/state/preprocessing_semantics/v2/receipt.json'),
        change='Only unknownlocation query context preprocessing. Encoded rawtokens m:/s0:/s7:UNKNOWN/s2:NA are unavailable locations. Original owncategory and all otherfeatures retained.36 groupfields become originalLGBmissing sentinel;16 known-equality fields become0. No globalNaN change or numeric scaling.',
        scope='All original finite fit/tune rows and rawY-NM labels. Original387/leaf63/2500cap/150patience/seed/2CPU; originalsavedcontrol native replay before transforming inputs. No target cleanup or row filtering.',
        rationale='Observed earlyfit literals UNKNOWNstand622 and NArunway4; rawnullstand16. Cachedquery physical-group context exists forunknownstand. Interpretation test, not proven source corruption. Small footprint; no10secforecast.',
        exclusions='Airport-wide context and active-runway counts stay unchanged; complete event-level activecount correction is separate. Source-category signal preserved. No correction to clock meaning or targets.',
        primary='CurrentPLE387 ensemble + original unionLGB coefficient*(preprocessedUnion-originalUnion). No weights refit. Matched union standalone secondary.',
        gate='Primary bothmonths/all-dayremovals positive and>=2seasonalordinaryseconds before any scoreconsideration. NegativeF1holdsF3. Matchedgainreported butnotalternatepromotion.',
        weights=[192122/344841, 152719/344841],
        resources='2CPU,10GiB current/OShistoricalpeak sampled eachtree andloadstage; startup18GiB;8GiBhostreserve;noGPU.',
        limitation='Developmentexposedtune; upstreamweights fittedontune; modelstopping usesoriginaltune. No score/refit/ranking access.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def raw_location_tokens(ids, source_tuples):
    values = {name: np.empty(len(ids), dtype=object) for name in ['STAND_mvt', 'RUNWAY_mvt']}
    seen = np.zeros(len(ids), dtype=bool)
    for path, _, _, names in source_tuples:
        if not all(name in names for name in values):
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *values], use_threads=False):
            frame = batch.to_pandas()
            pos = ids.get_indexer(frame[ID])
            keep = pos >= 0
            pos = pos[keep]
            assert len(np.unique(pos)) == len(pos) and not seen[pos].any()
            seen[pos] = True
            for name in values:
                values[name][pos] = frame.loc[keep, name].astype('string').to_numpy(dtype=object)
    assert seen.all()
    return values


def run(fold):
    protocol = declare()
    assert psutil.virtual_memory().available >= 18*1024**3
    dest = OUT / fold
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    tuples, discovery = schema_discovery.cached_discovery(ORIGINAL, protocol['columns'])
    risk.feature_sources = lambda columns: tuples if columns == protocol['columns'] else (_ for _ in ()).throw(ValueError('Changed schema'))
    risk.guard = guard
    metadata = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(metadata) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta = pd.read_parquet(metadata, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    parts = {stage: meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy()
             for stage, p in idx.items() if stage in ['fit', 'tune']}
    del meta
    ids = pd.Index(pd.concat([parts['fit'][ID], parts['tune'][ID]], ignore_index=True))
    nfit = len(parts['fit'])
    matrix, vocab, receipts = risk.load_matrix(ids, nfit, protocol['columns'], dest)
    prior = risk.union_folder(fold)
    assert common.sha256(prior / 'manifest.json') == protocol['controls'][fold]
    marker = common.read_json(prior / 'manifest.json')
    for name in ['tune_predictions.parquet']:
        assert common.sha256(prior / name) == marker['outputs'][name]
    control = pd.read_parquet(prior / 'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID], parts['tune'][ID])
    assert common.sha256(NATIVE[fold] / 'manifest.json') == protocol['native_controls'][fold]
    native_marker = common.read_json(NATIVE[fold] / 'manifest.json')
    for name in ['model.txt', 'encoder.json']:
        assert common.sha256(NATIVE[fold] / name) == native_marker['outputs'][name]
    assert common.read_json(NATIVE[fold] / 'encoder.json') == dict(columns=protocol['columns'], vocab=vocab)
    native = lgb.Booster(model_file=str(NATIVE[fold] / 'model.txt'))
    replay = native.predict(matrix[nfit:], num_threads=2) + parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(replay, control.prediction_sec.to_numpy())
    del native, replay
    tokens = raw_location_tokens(ids, tuples)
    impacts = {}
    for stage, section in [('fit', slice(0, nfit)), ('tune', slice(nfit, None))]:
        impacts[stage] = preprocess(matrix[section], protocol['columns'],
            tokens['STAND_mvt'][section], tokens['RUNWAY_mvt'][section])
    del tokens
    matrix.flush()
    common.write_json(dest / 'preprocessing.json', dict(impacts=impacts, discovery=discovery))
    common.write_json(dest / 'encoder.json', dict(columns=protocol['columns'], vocab=vocab))
    guard()
    target = {s: f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s, f in parts.items()}
    model = lgb.LGBMRegressor(**protocol['parameters'])
    model.fit(matrix[:nfit], target['fit'], categorical_feature=[protocol['columns'].index(c) for c in vocab],
        feature_name=protocol['columns'], eval_X=matrix[nfit:], eval_y=target['tune'], eval_metric='rmse',
        callbacks=[guard, lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
    steps = int(model.best_iteration_ or model.n_estimators_)
    prediction = model.predict(matrix[nfit:], num_iteration=steps) + parts['tune'].proxy_sec.to_numpy(float)
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(dest / 'model.txt'), num_iteration=steps)
    replay = lgb.Booster(model_file=str(dest / 'model.txt')).predict(matrix[nfit:], num_threads=2)
    replay += parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(replay, prediction)
    ordinary, current = comparison.baseline(fold, parts['tune'], protocol)
    ensemble = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    weights = common.read_json(ensemble / f'{fold}_weights.json')
    coefficient = weights['global'][weights['experts'].index('lgb63_union')]
    candidate = current + coefficient * (prediction[ordinary]-control.prediction_sec.to_numpy()[ordinary])
    y = parts['tune'][TARGET].to_numpy(float)
    dates = parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    results = dict(matched=comparison.compare(y, prediction, control.prediction_sec.to_numpy(), dates),
        primary=comparison.compare(y[ordinary], candidate, current, dates[ordinary]))
    pd.DataFrame({ID: parts['tune'][ID], 'prediction_sec': prediction}).to_parquet(dest / 'tune.parquet', index=False)
    record = dict(status='complete', fold=fold, protocol_sha256=common.sha256(OUT / 'protocol.json'),
        source_sha256=common.sha256(__file__), split=split, parameters=protocol['parameters'], steps=steps,
        ids={s: dict(n=len(f), hash=common.object_hash(f[ID].tolist())) for s, f in parts.items()},
        feature_receipts=receipts, original_control_max_abs_delta=0., native_replay_max_abs_delta=0.,
        coefficient=coefficient, results=results, peak_bytes=guard(), elapsed_seconds=time.monotonic()-started,
        no_score_prediction=True, outputs={name: common.sha256(dest / name)
            for name in ['model.txt', 'tune.parquet', 'encoder.json', 'preprocessing.json']})
    common.write_json(dest / 'manifest.json', record)
    del matrix, model
    gc.collect()
    print('COMPLETE', fold, results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    declare() if args.declare_only else run(args.fold)
