"""Missing-source normalized raw-MSE specialist with matched context information."""
import lightgbm as lgb
import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as base
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/state/risk'))
import run_risk as risk
common, ID, TARGET, TIME = base.common, base.ID, base.TARGET, base.TIME
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/normalized_context/v1')
DUPLICATES = {'ADEP_mvt':'airport', 'RUNWAY_mvt':'runway', 'STAND_mvt':'stand',
    'ADES_mvt':'destination', 'AIRCRAFT_TYPE_mvt':'aircraft', 'airport_stand':'airport_stand',
    'airport_runway':'airport_runway', 'utc_hour':'movement_hour', 'weekday':'movement_dayofweek',
    'month':'movement_month', 'minute':'movement_minute', 'takeoff_minus_SCHED_TIME_UTC_mvt':'schedule_proxy_sec'}


def selected_frame(path, columns, ids):
    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *columns], use_threads=False):
        frame = batch.to_pandas()
        keep = frame[ID].isin(ids)
        if keep.any():
            pieces.append(frame.loc[keep])
    return pd.concat(pieces, ignore_index=True).set_index(ID) if pieces else pd.DataFrame(columns=columns, index=pd.Index([], name=ID))


def context_frame(ids, desired):
    grouped = defaultdict(list)
    sources = risk.feature_sources(desired)
    for path, digest, fill, names in sources:
        grouped[tuple(names)].append((path, digest, fill))
    result = pd.DataFrame(index=ids)
    receipts = []
    for names, paths in grouped.items():
        frames = []
        for path, digest, fill in paths:
            part = selected_frame(path, list(names), ids)
            frames.append(part)
            receipts.append(dict(path=str(path), sha256=digest, selected_rows=len(part), columns=list(names)))
        combined = pd.concat(frames)
        assert combined.index.is_unique and set(combined.index) == set(ids)
        combined = combined.loc[ids]
        assert not set(names).intersection(result)
        for name in names:
            if pd.api.types.is_numeric_dtype(combined[name]):
                combined[name] = pd.to_numeric(combined[name]).replace([np.inf, -np.inf], np.nan).astype('float32')
            else:
                combined[name] = combined[name].astype('string').astype('category')
        result = pd.concat([result, combined], axis=1)
        print('CONTEXT', paths[0][0].parent.name, len(combined), len(names), 'RSS', risk.guard(), flush=True)
        del combined, frames
        gc.collect()
    assert set(result) == set(desired) and len(result.columns) == len(desired)
    return result[desired], receipts


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    x, meta = base.shared.load_missing()
    expected = common.read_json(ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models/historical_template_F1_s20260916/manifest.json')['features_used']
    assert list(x) == expected and len(x.columns) == 76
    assert len(x) == 22470 and x.index.is_unique
    full = common.read_json(risk.union_folder('F1')/'manifest.json')['feature_columns']
    desired = [name for name in full if name not in DUPLICATES]
    assert len(full) == 387 and len(desired) == 375 and not set(desired).intersection(x)
    sources = [Path(__file__), Path(base.__file__), Path(risk.__file__),
               ROOT/'review_work/breakthrough_20260916/models/encoders.py', Path(base.shared.__file__)]
    protocol = dict(source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in sources},
        baseline_protocol_sha256=common.sha256(base.OUT/'protocol.json'),
        arms=['control76', 'context451'], base_columns=expected, context_columns=desired,
        removed_semantic_duplicates=DUPLICATES, params=base.PARAMS, round_cap=600, patience=60,
        target='Exact frozen normalized target900+s(X)*Z, schedule-observed scale and s^2/fitmean weights; equivalent rawMSE; raw labels unchanged.',
        scope='Only full original purged missingNM fit/tune June/October; all22470missing rows feature coverage checked. No score/refit/ranking predictions.',
        features='Control exact76 and same76+375 nonduplicate union387 fields; final supplied retrospective batch including neighboring clocks/futuredaycontext, not realtime.',
        encoding='Exact frozen FrameEncoder fit-only categories; numeric infinities/sentinel become NaN; missing context remainsmissing; no targetderivedfeatures.',
        novelty='Union225/337/387 models trained finiteNM and preserved missingV2; prior missingonly blocks ID76/ARR15/OPDI8/state12 do not test fullunion context.',
        control='Fresh control76 tune predictions must reproduce frozen normalized_missing_tune observed_schedule_scale exactly beforecontextfit.',
        advancement='No automatic scoreadvance. Require rawRMSE improvementbothmonths versusfreshcontrol; allsingle-day-removal gainspositive; report bootstrap/topbeneficialrow sensitivity. Parent owns furtherdeclaration.',
        resource='2CPU,6GiBsampledRSS,8GiBhostreserve; keepfullcohorts.')
    p = OUT/'protocol.json'
    if p.exists():
        assert common.read_json(p) == protocol
    else:
        common.write_json(p, protocol)
    cache = OUT/'context.parquet'
    receipt_path = OUT/'cache_manifest.json'
    if cache.exists():
        receipt = common.read_json(receipt_path)
        assert receipt['protocol_sha256'] == common.sha256(p)
        assert common.sha256(cache) == receipt['feature_sha256']
        context = pd.read_parquet(cache).set_index(ID)
    else:
        context, receipts = context_frame(x.index, desired)
        context.reset_index().to_parquet(cache, index=False)
        common.write_json(receipt_path, dict(status='complete', rows=len(context), columns=desired,
            original_missing_ids_hash=common.object_hash(x.index.tolist()), protocol_sha256=common.sha256(p),
            feature_sha256=common.sha256(cache), sources=receipts))
    assert np.array_equal(context.index, x.index) and list(context) == desired
    return x, context, meta, protocol


def compare(y, pred, reference, timestamps):
    loss = (y-pred)**2
    old = (y-reference)**2
    gain = old-loss
    codes, days = pd.factorize(timestamps.dt.floor('D'), sort=True)
    weights = np.random.default_rng(20260916).multinomial(len(days), np.full(len(days), 1/len(days)), size=300)
    removal = []
    for i, day in enumerate(days):
        keep = codes != i
        removal.append(dict(day=str(day), rmse_improvement=float(np.sqrt(old[keep].mean())-np.sqrt(loss[keep].mean()))))
    sensitivity = {}
    ranked = np.argsort(-gain, kind='stable')
    for count in [1, 5, 10]:
        keep = np.ones(len(y), bool)
        keep[ranked[:count]] = False
        sensitivity[str(count)] = float(np.sqrt(old[keep].mean())-np.sqrt(loss[keep].mean()))
    return dict(rmse=float(np.sqrt(loss.mean())), control_rmse=float(np.sqrt(old.mean())),
        rmse_improvement=float(np.sqrt(old.mean())-np.sqrt(loss.mean())),
        mse_improvement_ci95=risk.bootstrap_interval(gain, np.ones(len(y)), codes, weights),
        all_day_removals_improve=all(row['rmse_improvement'] > 0 for row in removal),
        day_removals=removal, remove_top_beneficial_rows_rmse_improvement=sensitivity)


def run(fold, x, context, meta, protocol):
    idx, split, _ = common.fold_data(meta, fold, full=True)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy())
    rows = {stage:idx[stage][missing[idx[stage]]] for stage in ['fit', 'tune']}
    y = meta[TARGET].to_numpy(float)
    frames = {stage:x.loc[meta.iloc[positions][ID]] for stage, positions in rows.items()}
    scales = {stage:base.scale_of(frame) for stage, frame in frames.items()}
    norm = float(np.mean(scales['fit']**2))
    targets = {stage:base.transformed(y[rows[stage]], scales[stage]) for stage in rows}
    weights = {stage:scales[stage]**2/norm for stage in rows}
    assert all(np.isfinite(value).all() and (value > 0).all() for value in weights.values())
    saved_baseline = base.OUT/fold/'observed_schedule_scale'
    old = common.read_json(saved_baseline/'manifest.json')
    assert old['status'] == 'complete'
    assert common.sha256(saved_baseline/'tune.parquet') == old['outputs']['tune.parquet']
    reference = pd.read_parquet(saved_baseline/'tune.parquet')
    assert np.array_equal(reference[ID], frames['tune'].index)
    assert np.array_equal(reference.raw_target_sec, y[rows['tune']])
    for arm in protocol['arms']:
        dest = common.external_path(OUT/fold/arm)
        dest.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        data = {stage:(frame if arm == 'control76' else pd.concat([frame, context.loc[frame.index]], axis=1)) for stage, frame in frames.items()}
        encoder = base.FrameEncoder().fit(data['fit'])
        matrix = {stage:encoder.transform(frame) for stage, frame in data.items()}
        train = lgb.Dataset(matrix['fit'], label=targets['fit'], weight=weights['fit'], categorical_feature=list(encoder.categories))
        tune = lgb.Dataset(matrix['tune'], label=targets['tune'], weight=weights['tune'], reference=train, categorical_feature=list(encoder.categories))
        model = lgb.train(base.PARAMS, train, num_boost_round=600, valid_sets=[tune], callbacks=[lgb.early_stopping(60, verbose=False)])
        z = model.predict(matrix['tune'], num_iteration=model.best_iteration)
        pred = base.reconstruct(z, scales['tune'])
        np.testing.assert_allclose(weights['tune']*(z-targets['tune'])**2, (pred-y[rows['tune']])**2/norm, rtol=1e-9, atol=1e-7)
        if arm == 'control76':
            np.testing.assert_array_equal(pred, reference.prediction_sec.to_numpy())
            assert model.best_iteration == old['steps']
        joblib.dump(dict(model=model, encoder=encoder), dest/'model.joblib')
        saved = joblib.load(dest/'model.joblib')
        replay = base.reconstruct(saved['model'].predict(saved['encoder'].transform(data['tune']), num_iteration=model.best_iteration), scales['tune'])
        np.testing.assert_array_equal(pred, replay)
        result = compare(y[rows['tune']], pred, reference.prediction_sec.to_numpy(), meta.iloc[rows['tune']][TIME])
        pd.DataFrame({ID:frames['tune'].index, 'prediction_sec':pred, 'raw_target_sec':y[rows['tune']],
            'scale':scales['tune'], 'weight':weights['tune'], 'transformed_target':targets['tune']}).to_parquet(dest/'tune.parquet', index=False)
        common.write_json(dest/'metrics.json', result)
        common.write_json(dest/'manifest.json', dict(status='complete', fold=fold, arm=arm, steps=model.best_iteration,
            params=base.PARAMS, feature_columns=list(data['fit']), split=split, normalizer=norm,
            fit_ids={stage:dict(n=len(positions), hash=common.object_hash(meta.iloc[positions][ID].tolist())) for stage, positions in rows.items()},
            label_hashes={stage:common.object_hash(y[positions].tolist()) for stage, positions in rows.items()},
            protocol_sha256=common.sha256(OUT/'protocol.json'), cache_manifest_sha256=common.sha256(OUT/'cache_manifest.json'),
            source_sha256=common.sha256(__file__), baseline_manifest_sha256=common.sha256(saved_baseline/'manifest.json'),
            reload_max_abs_delta=0., runtime_sec=time.monotonic()-started, sampled_rss_bytes=risk.guard(),
            outputs={name:common.sha256(dest/name) for name in ['model.joblib', 'tune.parquet', 'metrics.json']}))
        print('RESULT', fold, arm, result, flush=True)
        del train, tune, model, saved, data, matrix, encoder
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        x, context, meta, protocol = prepare()
        if not args.prepare_only:
            assert args.fold
            run(args.fold, x, context, meta, protocol)
