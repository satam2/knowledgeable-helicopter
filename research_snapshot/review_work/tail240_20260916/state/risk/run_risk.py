"""Fixed finite-NM source-gap classifier diagnostic; no taxi-time routing."""
import lightgbm as lgb
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
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common

ID, TARGET = common.ID, common.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/risk/v1')
SOURCE = Path(__file__).resolve()
BASE = ROOT / 'private_runs/breakthrough_20260916'
PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=31,
              min_data_in_leaf=100, lambda_l2=10., feature_fraction=1.,
              bagging_fraction=1., bagging_freq=0, num_threads=2,
              seed=20260916, deterministic=True, force_col_wise=True,
              verbosity=-1, metric='binary_logloss')
BINS = [0., .001, .005, .01, .02, .05, .1, .2, .5, 1.]


def guard():
    rss = psutil.Process().memory_info().rss
    assert rss < 6 * 1024**3, f'RSS limit: {rss}'
    assert psutil.virtual_memory().available >= 8 * 1024**3, 'Host reserve below 8GiB'
    return int(rss)


def union_folder(fold):
    return BASE / 'deeper_context_union' / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'


def own_columns(columns):
    excluded = {'dep_prior_15m', 'dep_prior_60m', 'arr_prior_15m', 'arr_prior_60m',
                'observed_window_15m_sec', 'observed_window_60m_sec', 'month_edge'}
    return [name for i, name in enumerate(columns) if (i < 30 and name not in excluded)
            or name.startswith(('retro_own_', 'source_', 'conv_'))]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    union = common.read_json(union_folder('F1') / 'manifest.json')
    full = union['feature_columns']
    assert len(full) == 387 and len(set(full)) == 387
    record = dict(source_sha256=common.sha256(SOURCE), seed=20260916,
        folds=['F1', 'F3'], stages=['fit', 'tune'], full_columns=full,
        own_columns=own_columns(full), params=PARAMS, rounds=150,
        label='abs(raw_Y - finite_NM_proxy) > 1800 seconds; strict inequality',
        selection='Original make_fold flight-ID purges before finite-NM restriction; every eligible row retained.',
        evaluation='Tune only June/October; no July/November predictions or labels evaluated, no ranking targets.',
        fitting='Fixed 150 rounds, binary logloss, no weighting/resampling/early stopping/grid/calibration fitting.',
        baseline='Constant prevalence in full finite-NM fit cohort.',
        primary_slice='0 <= finite NM proxy <= 7200; all finite reported separately; fit always all finite.',
        calibration_bins=BINS, top_fractions=[.01, .05, .1], bootstrap_replicates=300,
        uncertainty='Fixed day-cluster bootstrap of tune days for Brier improvement and top-risk positive/SSE capture.',
        availability='Final supplied-batch contract: full387 includes future/adjacent events and same-day aggregates; not real-time.',
        purpose='Diagnostic discrimination/calibration and concentration of saved union FIT-model TUNE squared error; no routing or model promotion.',
        resources='2 CPU, sampled RSS <6GiB, >=8GiB host reserve; limits not OS enforced.')
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record, 'Frozen protocol changed'
    else:
        common.write_json(path, record)
    return record


def categories_for(values, fit_mask):
    return sorted(set(values.loc[fit_mask].dropna().astype(str)))


def category_codes(values, known):
    mapping = {value: i + 2 for i, value in enumerate(known)}
    return values.astype('string').map(mapping).fillna(1).where(values.notna(), 0).to_numpy(np.float32)


def feature_sources(desired):
    frozen = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    assert common.source_hashes() == frozen['source_hashes']
    base_dir = ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
    files = sorted(base_dir.glob('training_*.parquet'))
    assert len(files) == 12
    sources = []
    for path in files:
        marker = common.read_json(path.with_suffix('.json'))
        assert marker['identity']['inputs'] == frozen['raw_hashes']
        assert marker['identity']['features'] == frozen['base_config']['features']
        sources.append((path, marker['sha256'], False))
    for folder, filename in [
        (ROOT / 'private_runs/mechanism_20260916/information/retrospective_v2', 'training_features.parquet'),
        (BASE / 'batch_context', 'training_features.parquet'),
        (BASE / 'missing/source_conventions', 'features.parquet'),
        (BASE / 'geometry_v2', 'training_features.parquet'),
        (BASE / 'weather', 'training_features.parquet'),
        (BASE / 'missing/sequence_flatten', 'training_features.parquet'),
        (BASE / 'retrospective_research', 'training_features.parquet')]:
        marker = common.read_json(folder / 'manifest.json')
        expected = marker['feature_sha256'] if filename == 'features.parquet' else marker['outputs'][filename]
        sources.append((folder / filename, expected, 'sequence_flatten' not in str(folder)))
    result = []
    for path, expected, fill in sources:
        columns = [name for name in desired if name in pq.read_schema(path).names]
        if columns:
            assert common.sha256(path) == expected, path
            result.append((path, expected, fill, columns))
    return result


def load_matrix(ids, nfit, desired, folder):
    """Fit-only vocabularies and batch reads bound peak memory independently of year size."""
    assert ids.is_unique
    sources = feature_sources(desired)
    matrix = np.memmap(folder / 'matrix.float32', mode='w+', dtype='float32',
                       shape=(len(ids), len(desired)), order='F')
    column_index = {name: i for i, name in enumerate(desired)}
    vocab_sets = {}
    category_names = set()
    for path, _, _, names in sources:
        schema = pq.read_schema(path)
        for name in names:
            dtype = schema.field(name).type
            if pa.types.is_dictionary(dtype):
                dtype = dtype.value_type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)):
                category_names.add(name)
    for path, _, _, names in sources:
        cat_names = [name for name in names if name in category_names]
        if not cat_names:
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *cat_names], use_threads=False):
            frame = batch.to_pandas()
            positions = ids.get_indexer(frame[ID])
            fit_mask = (positions >= 0) & (positions < nfit)
            for name in cat_names:
                vocab_sets.setdefault(name, set()).update(categories_for(frame[name], fit_mask))
        guard()
    vocab = {name: sorted(values) for name, values in vocab_sets.items()}
    counts = np.zeros(len(desired), dtype=np.int64)
    receipts = []
    for path, expected, fill, names in sources:
        selected_rows = 0
        seen = np.zeros(len(ids), dtype=bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *names], use_threads=False):
            frame = batch.to_pandas()
            positions = ids.get_indexer(frame[ID])
            keep = positions >= 0
            positions = positions[keep]
            assert len(np.unique(positions)) == len(positions) and not seen[positions].any()
            seen[positions] = True
            frame = frame.loc[keep]
            selected_rows += len(frame)
            for name in names:
                if name in vocab:
                    values = category_codes(frame[name], vocab[name])
                else:
                    values = pd.to_numeric(frame[name]).to_numpy(dtype=np.float32, na_value=np.nan)
                    values[~np.isfinite(values)] = np.nan
                    if fill:
                        values[np.isnan(values)] = -999999.
                matrix[positions, column_index[name]] = values
        for name in names:
            counts[column_index[name]] += selected_rows
        receipts.append(dict(path=str(path), sha256=expected, rows=selected_rows, columns=names))
        print('SOURCE', path.parent.name, path.name, selected_rows, 'RSS', guard(), flush=True)
    assert np.all(counts == len(ids)), dict(zip(desired, counts.tolist()))
    matrix.flush()
    return matrix, vocab, receipts


def bootstrap_interval(numerator, denominator, day_codes, weights):
    nday = weights.shape[1]
    num = np.bincount(day_codes, weights=numerator, minlength=nday)
    den = np.bincount(day_codes, weights=denominator, minlength=nday)
    totals = weights @ den
    values = (weights @ num)[totals > 0] / totals[totals > 0]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]


def metric_report(y, probability, fit_prevalence, sse, dates):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    assert len(y) == len(p) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    n = len(y)
    day_codes, days = pd.factorize(dates, sort=True)
    rng = np.random.default_rng(20260916)
    weights = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=300)
    improvement = (y-fit_prevalence)**2 - (y-p)**2
    result = dict(rows=n, positives=int(y.sum()), prevalence=float(y.mean()),
        average_probability=float(p.mean()), roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        pr_auc_average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(np.mean((y-p)**2)), logloss=float(log_loss(y, p, labels=[0, 1])),
        baseline_brier=float(np.mean((y-fit_prevalence)**2)),
        baseline_logloss=float(log_loss(y, np.full(n, fit_prevalence), labels=[0, 1])),
        brier_improvement_ci95=bootstrap_interval(improvement, np.ones(n), day_codes, weights),
        union_sse=float(sse.sum()), calibration=[], top_risk={})
    bin_ids = np.minimum(np.searchsorted(BINS, p, side='right') - 1, len(BINS)-2)
    for i in range(len(BINS)-1):
        mask = bin_ids == i
        result['calibration'].append(dict(lower=BINS[i], upper=BINS[i+1], rows=int(mask.sum()),
            mean_probability=float(p[mask].mean()) if mask.any() else None,
            observed_rate=float(y[mask].mean()) if mask.any() else None))
    order = np.argsort(-p, kind='stable')
    for fraction in [.01, .05, .1]:
        selected = np.zeros(n, bool)
        selected[order[:max(1, int(np.ceil(n * fraction)))]] = True
        result['top_risk'][str(fraction)] = dict(rows=int(selected.sum()),
            threshold=float(p[selected].min()), precision=float(y[selected].mean()),
            positive_capture=float(y[selected].sum()/y.sum()) if y.sum() else None,
            positive_capture_ci95=bootstrap_interval(y*selected, y, day_codes, weights),
            union_sse_capture=float(sse[selected].sum()/sse.sum()),
            union_sse_capture_ci95=bootstrap_interval(sse*selected, sse, day_codes, weights))
    return result


def run(fold, arm):
    protocol = freeze()
    folder = common.external_path(OUT / fold / arm)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    indices, split, _ = common.fold_data(meta, fold, full=True)
    selected = {stage: indices[stage][np.isfinite(meta.iloc[indices[stage]].proxy_sec.to_numpy())]
                for stage in ['fit', 'tune']}
    fit, tune = (meta.iloc[selected[stage]].copy() for stage in ['fit', 'tune'])
    assert np.isfinite(fit[TARGET]).all() and np.isfinite(tune[TARGET]).all()
    ids = pd.Index(pd.concat([fit[ID], tune[ID]], ignore_index=True))
    yfit = (np.abs(fit[TARGET].to_numpy()-fit.proxy_sec.to_numpy()) > 1800).astype(np.int8)
    ytune = (np.abs(tune[TARGET].to_numpy()-tune.proxy_sec.to_numpy()) > 1800).astype(np.int8)
    desired = protocol['full_columns' if arm == 'full387' else 'own_columns']
    union_path = union_folder(fold)
    union_manifest = common.read_json(union_path/'manifest.json')
    assert union_manifest['feature_columns'] == protocol['full_columns']
    tune_path = union_path/'tune_predictions.parquet'
    assert common.sha256(tune_path) == union_manifest['outputs'][tune_path.name]
    predictions = pd.read_parquet(tune_path).set_index(ID)
    assert predictions.index.is_unique and set(predictions.index) == set(tune[ID])
    pred = predictions.loc[tune[ID], 'prediction_sec'].to_numpy(float)
    assert np.isfinite(pred).all()
    sse = (tune[TARGET].to_numpy()-pred)**2
    del meta, predictions
    gc.collect()
    matrix, vocab, receipts = load_matrix(ids, len(fit), desired, folder)
    categories = [desired.index(name) for name in vocab]
    dataset = lgb.Dataset(matrix[:len(fit)], label=yfit, feature_name=desired,
        categorical_feature=categories, free_raw_data=True, params={'max_bin':255})
    model = lgb.train(PARAMS, dataset, num_boost_round=150)
    p = model.predict(matrix[len(fit):], num_threads=2)
    rss = guard()
    model.save_model(str(folder/'model.txt'))
    reloaded = lgb.Booster(model_file=str(folder/'model.txt'))
    replay = reloaded.predict(matrix[len(fit):], num_threads=2)
    assert np.array_equal(p, replay)
    common.write_json(folder/'encoder.json', dict(columns=desired, vocab=vocab))
    save = tune[[ID, 'ADEP_mvt', common.MOVEMENT, TARGET, 'proxy_sec']].copy()
    save['large_gap'] = ytune
    save['probability'] = p
    save['union_fit_prediction_sec'] = pred
    save.to_parquet(folder/'tune_probabilities.parquet', index=False)
    reports = {}
    ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
    dates = tune[common.MOVEMENT].dt.floor('D').to_numpy()
    for name, mask in [('ordinary_proxy', ordinary), ('all_finite', np.ones(len(tune), bool))]:
        reports[name] = metric_report(ytune[mask], p[mask], yfit.mean(), sse[mask], dates[mask])
    slices = []
    for dimension, values in [('airport', tune.ADEP_mvt.astype(str).to_numpy()), ('day', dates)]:
        for value in np.unique(values):
            mask = (values == value) & ordinary
            if mask.any():
                slices.append(dict(dimension=dimension, value=str(value), rows=int(mask.sum()),
                    positives=int(ytune[mask].sum()), mean_probability=float(p[mask].mean()),
                    observed_rate=float(ytune[mask].mean()), brier=float(np.mean((ytune[mask]-p[mask])**2)),
                    union_sse=float(sse[mask].sum())))
    common.write_json(folder/'metrics.json', reports)
    common.write_json(folder/'slices.json', slices)
    common.write_json(folder/'manifest.json', dict(status='complete', fold=fold, arm=arm,
        source_sha256=common.sha256(SOURCE), protocol_sha256=common.sha256(OUT/'protocol.json'),
        split=split, full_unmodified_finite_fit_rows=len(fit), full_unmodified_finite_tune_rows=len(tune),
        fit_ids_hash=common.object_hash(fit[ID].tolist()), tune_ids_hash=common.object_hash(tune[ID].tolist()),
        fit_label_hash=common.object_hash(fit[TARGET].tolist()), tune_label_hash=common.object_hash(tune[TARGET].tolist()),
        fit_positive_count=int(yfit.sum()), fit_prevalence=float(yfit.mean()), feature_columns=desired,
        params=PARAMS, rounds=model.current_iteration(), feature_receipts=receipts,
        union_manifest_sha256=common.sha256(union_path/'manifest.json'),
        union_fit_model_sha256=union_manifest['outputs']['fit_model.joblib'],
        union_tune_sha256=common.sha256(tune_path), reload_max_abs_probability_delta=0.,
        sampled_rss_bytes=rss, runtime_sec=time.monotonic()-started,
        outputs={name:common.sha256(folder/name) for name in ['model.txt', 'encoder.json', 'metrics.json', 'slices.json', 'tune_probabilities.parquet']}))
    del dataset, matrix, model, reloaded
    gc.collect()
    (folder/'matrix.float32').unlink()
    print('COMPLETE', fold, arm, reports, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    parser.add_argument('--arm', choices=['own', 'full387'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare:
            print(freeze())
        else:
            assert args.fold and args.arm
            run(args.fold, args.arm)
