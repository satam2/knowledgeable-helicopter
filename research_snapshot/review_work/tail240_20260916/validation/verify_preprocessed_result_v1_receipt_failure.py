"""Independently reconstruct every masked input, native predictions and endpoint."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import lightgbm as lgb
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT, read, sha, write, object_hash, guard
import validate_candidate as v
from verify_linear_finite import check_result

ID, TIME, TARGET = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'
BASE = ROOT / 'private_runs/tail240_20260916/models/preprocessed_locations_v1'


def main(fold):
    folder = BASE / fold
    out = ROOT / f'private_runs/tail240_20260916/validation/preprocessed_locations_{fold}_v1'
    assert not out.exists()
    protocol, marker = read(BASE / 'protocol.json'), read(folder / 'manifest.json')
    assert marker['status'] == 'complete' and marker['protocol_sha256'] == sha(BASE / 'protocol.json')
    for relative, expected in protocol['source_hashes'].items():
        assert sha(ROOT / relative) == expected
    for name, expected in marker['outputs'].items():
        assert sha(folder / name) == expected
    assert marker['parameters'] == protocol['parameters']
    original_folder = ROOT / 'private_runs/tail240_20260916/models' / ('following_groups_tune_v1' if fold == 'F1' else 'following_groups_tune_v2') / fold / 'control387'
    assert sha(original_folder / 'manifest.json') == protocol['native_controls'][fold]
    original = read(original_folder / 'manifest.json')
    assert marker['feature_receipts'] == original['feature_receipts']
    encoder = read(folder / 'encoder.json')
    assert encoder == read(original_folder / 'encoder.json')
    columns, vocab = encoder['columns'], encoder['vocab']
    assert columns == protocol['columns'] and len(columns) == 387
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(meta_path) == read(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    assert not any(marker['split']['purged_related_departures'].values())
    chunks = []
    for batch in pq.ParquetFile(meta_path).iter_batches(batch_size=8192, columns=[ID, TIME, TARGET, 'proxy_sec'], use_threads=False):
        frame = batch.to_pandas()
        start, stop = [pd.Timestamp(value, tz='UTC') for value in marker['split']['spec']['refit']]
        chunks.append(frame.loc[frame[TIME].ge(start) & frame[TIME].lt(stop) & np.isfinite(frame.proxy_sec)])
    meta = pd.concat(chunks, ignore_index=True)
    boundary = pd.Timestamp(marker['split']['spec']['tune'][0], tz='UTC')
    fit, tune = meta.loc[meta[TIME].lt(boundary)], meta.loc[meta[TIME].ge(boundary)].reset_index(drop=True)
    assert {stage: dict(n=len(part), hash=object_hash(part[ID].tolist())) for stage, part in [('fit', fit), ('tune', tune)]} == marker['ids']
    ids = pd.Index(pd.concat([fit[ID], tune[ID]], ignore_index=True)); nfit = len(fit)
    assert ids.is_unique
    del chunks, meta, fit
    gc.collect()
    stored = np.memmap(folder / 'matrix.float32', mode='r', dtype='float32', shape=(len(ids), len(columns)), order='F')
    original_tune = np.empty((len(tune), len(columns)), dtype='float32', order='F')
    unknown = {name: np.zeros(len(ids), bool) for name in ('stand', 'runway')}
    token_seen = np.zeros(len(ids), bool)
    rebuilt = {name: set() for name in vocab}
    groups = protocol['groups']
    column_groups = {name: group for group, names in groups.items() for name in names}
    counts = {stage: {group: dict(query_rows=0, columns=names, changed_cells=0) for group, names in groups.items()} for stage in ('fit', 'tune')}
    totals = {name: 0 for name in columns}
    for source in marker['feature_receipts']:
        path = Path(source['path']); assert sha(path) == source['sha256']
        fill = 'screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        names = source['columns']; seen = np.zeros(len(ids), bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *names], use_threads=False):
            frame = batch.to_pandas(); positions = ids.get_indexer(frame[ID]); keep = positions >= 0
            positions = positions[keep]; frame = frame.loc[keep]
            assert len(np.unique(positions)) == len(positions) and not seen[positions].any()
            seen[positions] = True
            if 'STAND_mvt' in names:
                assert not token_seen[positions].any(); token_seen[positions] = True
                for key in unknown:
                    token = frame[key.upper()+'_mvt'].astype('string')
                    unknown[key][positions] = token.isna().to_numpy() | token.isin(['m:', 's0:', 's7:UNKNOWN', 's2:NA']).to_numpy()
            for name in names:
                j = columns.index(name)
                if name in vocab:
                    rebuilt[name].update(frame.loc[positions < nfit, name].dropna().astype(str).tolist())
                    values = frame[name].astype('string')
                    mapping = {word: i+2 for i, word in enumerate(vocab[name])}
                    values = values.map(mapping).fillna(1).where(values.notna(), 0).to_numpy('float32')
                else:
                    values = pd.to_numeric(frame[name]).to_numpy(dtype='float32', na_value=np.nan)
                    values[~np.isfinite(values)] = np.nan
                    if fill:
                        values[np.isnan(values)] = -999999.
                tune_rows = positions >= nfit
                original_tune[positions[tune_rows]-nfit, j] = values[tune_rows]
                group = column_groups.get(name)
                if group:
                    assert token_seen.all()
                    key = 'stand' if group in ('stand', 'stand_equality') else 'runway'
                    value = 0. if group.endswith('equality') else -999999.
                    change = unknown[key][positions]
                    for stage, stage_rows in [('fit', positions < nfit), ('tune', positions >= nfit)]:
                        counts[stage][group]['changed_cells'] += int((change & stage_rows & (values != value)).sum())
                    values[change] = value
                np.testing.assert_array_equal(values, stored[positions, j], err_msg=name)
                totals[name] += len(positions)
            guard()
        assert int(seen.sum()) == source['rows']
    assert all(value == len(ids) for value in totals.values()) and token_seen.all()
    assert {name: sorted(values) for name, values in rebuilt.items()} == vocab
    for stage, section in [('fit', slice(0, nfit)), ('tune', slice(nfit, None))]:
        for group in groups:
            key = 'stand' if group in ('stand', 'stand_equality') else 'runway'
            counts[stage][group]['query_rows'] = int(unknown[key][section].sum())
    assert counts == read(folder / 'preprocessing.json')['impacts']
    del stored
    gc.collect(); guard()
    for name in ('model.txt', 'encoder.json'):
        assert sha(original_folder / name) == original['outputs'][name]
    oldmodel = lgb.Booster(model_file=str(original_folder / 'model.txt'))
    old = oldmodel.predict(original_tune, num_threads=1) + tune.proxy_sec.to_numpy(float)
    prior = ROOT / f'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    assert sha(prior / 'manifest.json') == protocol['controls'][fold]
    assert sha(prior / 'tune_predictions.parquet') == read(prior / 'manifest.json')['outputs']['tune_predictions.parquet']
    old_saved = pd.read_parquet(prior / 'tune_predictions.parquet')
    np.testing.assert_array_equal(old_saved[ID], tune[ID])
    np.testing.assert_array_equal(old, old_saved.prediction_sec)
    del oldmodel
    for group, names in groups.items():
        mask = unknown['stand' if group in ('stand', 'stand_equality') else 'runway'][nfit:]
        for name in names:
            original_tune[mask, columns.index(name)] = 0. if group.endswith('equality') else -999999.
    model = lgb.Booster(model_file=str(folder / 'model.txt'))
    assert model.feature_name() == columns and model.current_iteration() == marker['steps']
    predicted = model.predict(original_tune, num_threads=1) + tune.proxy_sec.to_numpy(float)
    saved = pd.read_parquet(folder / 'tune.parquet')
    np.testing.assert_array_equal(saved[ID], tune[ID])
    np.testing.assert_array_equal(predicted, saved.prediction_sec)
    del model, original_tune
    gc.collect(); guard()
    ensemble = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep = read(ensemble / 'preparation.json')['folds'][fold]
    assert sha(ensemble / f'{fold}_aligned_tune.parquet') == prep['aligned_tune_sha256']
    weights = read(ensemble / f'{fold}_weights.json'); assert weights == prep['weights']
    aligned = pd.read_parquet(ensemble / f'{fold}_aligned_tune.parquet')
    ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID], tune.loc[ordinary, ID])
    np.testing.assert_array_equal(aligned[TARGET], tune.loc[ordinary, TARGET])
    np.testing.assert_array_equal(aligned.lgb63_union, old[ordinary])
    neural = ROOT / 'private_runs/tail240_20260916/state/neural_context' / ('v1' if fold == 'F1' else 'v3') / fold
    assert sha(neural / 'manifest.json') == protocol['neural_controls'][fold]
    assert sha(neural / 'tune_predictions.parquet') == read(neural / 'manifest.json')['outputs']['tune_predictions.parquet']
    ple = pd.read_parquet(neural / 'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(ple[TARGET], aligned[TARGET])
    current = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    current += weights['global'][weights['experts'].index('tabm_ple8')] * (ple.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    coefficient = weights['global'][weights['experts'].index('lgb63_union')]
    assert coefficient == marker['coefficient']
    candidate = current + coefficient*(predicted[ordinary]-old[ordinary])
    y = tune[TARGET].to_numpy(float); days = tune[TIME].dt.floor('D').to_numpy()
    reports = dict(matched=check_result(y, predicted, old, days, marker['results']['matched']),
                   primary=check_result(y[ordinary], candidate, current, days[ordinary], marker['results']['primary']))
    affected = (unknown['stand'][nfit:] | unknown['runway'][nfit:])[ordinary]
    improvement = (y[ordinary]-current)**2 - (y[ordinary]-candidate)**2
    attribution = {name: dict(n=int(mask.sum()), sse_gain=float(improvement[mask].sum())) for name, mask in [('masked_queries', affected), ('unmasked_queries', ~affected)]}
    out.mkdir(parents=True)
    result = dict(status='passed', source_sha256=sha(Path(__file__)), producer_manifest_sha256=sha(folder / 'manifest.json'),
                  all387_original_fit_tune_inputs_and_masks_exact=True, all15_fit_vocabularies_exact=True,
                  all_unaffected_cells_exact=True, original_native_max_abs_delta=0., candidate_native_max_abs_delta=0.,
                  cohorts=marker['ids'], preprocessing=counts, metrics=reports, attribution=attribution,
                  peak_bytes=guard(), no_GPU_or_fitting=True, no_score_evaluation=True)
    write(out / 'receipt.json', result)
    print('PASSED', fold, marker['results'], attribution, 'peak', result['peak_bytes'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--fold', choices=['F1', 'F3'], required=True)
    args = parser.parse_args(); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    with threadpool_limits(1):
        main(args.fold)
