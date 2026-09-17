"""Separate CPU tensor reconstruction and centrally allocated small-batch GPU replay."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import torch
import lightgbm
import argparse
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT, read, sha, write, guard, object_hash

sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
import tabm_ple_gpu as ple
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/neural_capacity'))
import run_capacity as producer

BASE = ROOT / 'private_runs/tail240_20260916/state/neural_capacity'
MODEL = BASE / 'v2/models/F1'
OUT = ROOT / 'private_runs/tail240_20260916/validation/neural_capacity_replay_F1_v1'
ID, TIME, TARGET = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'
BATCH, ATOL = 1024, .001


def preparation():
    canary = BASE / 'v1/canary/F1'
    receipt = read(canary / 'receipt.json')
    previous = ROOT / 'private_runs/tail240_20260916/validation/neural_capacity_canary_F1/receipt.json'
    verified = read(previous)
    assert verified['status'] == 'passed' and verified['producer_receipt_sha256'] == sha(canary / 'receipt.json')
    assert receipt['preparation_sha256'] == sha(canary / 'preparation.joblib')
    protocol = read(BASE / 'v1/protocol.json')
    assert sha(BASE / 'v1/protocol.json') == receipt['protocol_sha256']
    for relative, expected in protocol['source_hashes'].items():
        assert sha(ROOT / relative) == expected
    return joblib.load(canary / 'preparation.joblib'), protocol, receipt, sha(previous)


def cohorts(protocol):
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(path) == '503df9cb08f3ed7260fd7d969a3505d7b58434e20bccc80f1526d517a292e976'
    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, TIME, TARGET, 'proxy_sec'], use_threads=False):
        data = batch.to_pandas()
        keep = data[TIME].ge(pd.Timestamp('2025-01-01', tz='UTC')) & data[TIME].lt(pd.Timestamp('2025-07-01', tz='UTC')) & np.isfinite(data.proxy_sec)
        pieces.append(data.loc[keep])
    data = pd.concat(pieces, ignore_index=True)
    fit = data.loc[data[TIME].lt(pd.Timestamp('2025-06-01', tz='UTC'))]
    tune = data.loc[data[TIME].ge(pd.Timestamp('2025-06-01', tz='UTC'))].reset_index(drop=True)
    binding = protocol['controls']['F1']
    for stage, frame in [('fit', fit), ('tune', tune)]:
        assert len(frame) == binding[stage + '_rows']
        assert object_hash(frame[ID].tolist()) == binding[stage + '_ids_hash']
        assert object_hash(frame[TARGET].tolist()) == binding[stage + '_label_hash']
    residual = (fit[TARGET] - fit.proxy_sec).to_numpy(float)
    return tune, dict(y_mean=float(residual.mean()), y_scale=max(float(residual.std()), 1.))


def prepare_cpu():
    assert not OUT.exists()
    prep, protocol, canary, prior_hash = preparation()
    encoder = prep['encoder']
    tune, scaling = cohorts(protocol)
    binding = protocol['controls']['F1']
    ids = pd.Index(tune[ID])
    columns, nums, cats = encoder.columns, encoder.numeric, list(encoder.categories)
    original = read(ROOT / 'private_runs/tail240_20260916/models/linear_finite_tune_v1/F1/encoder.json')
    assert original['columns'] == columns
    source_audit = ROOT / 'private_runs/tail240_20260916/validation/union387_source_audit_v1/receipt.json'
    assert read(source_audit)['all387_columns_exact_once_per_requested_id']
    assert canary['feature_receipts'] == read(ROOT / 'private_runs/tail240_20260916/models/linear_finite_tune_v1/F1/constant/manifest.json')['feature_receipts']
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', dict(source_sha256=sha(Path(__file__)), batch_size=BATCH,
          smaller_batch_absolute_tolerance_sec=ATOL, relative_tolerance=0., independent_network_members=32,
          GPU_allocated_cap_bytes=4*1024**3, host_peak_cap_bytes=4*1024**3,
          prior_canary_validation_sha256=prior_hash, source_audit_sha256=sha(source_audit),
          no_model_changes=True, no_optimizer=True,
          limitation='Smaller-batch replay is numerical equivalence within a declared tolerance, not bitwise native replay. Producer native reload uses original batch8192.'))
    numbers = np.lib.format.open_memmap(OUT / 'numbers.npy', mode='w+', dtype='float32', shape=(len(ids), 2*len(nums)))
    categories = np.lib.format.open_memmap(OUT / 'categories.npy', mode='w+', dtype='int64', shape=(len(ids), len(cats)))
    retained = np.memmap(MODEL / 'matrix.float32', mode='r', dtype='float32', shape=(binding['fit_rows']+len(ids), len(columns)), order='F')
    seen = {name: np.zeros(len(ids), bool) for name in columns}
    base_categories = set(columns[:11])
    for receipt in canary['feature_receipts']:
        path = Path(receipt['path'])
        assert sha(path) == receipt['sha256']
        fill = 'screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *receipt['columns']], use_threads=False):
            data = batch.to_pandas()
            positions = ids.get_indexer(data[ID]); keep = positions >= 0; positions = positions[keep]
            data = data.loc[keep]
            if not len(data):
                continue
            for name in receipt['columns']:
                assert len(np.unique(positions)) == len(positions) and not seen[name][positions].any()
                seen[name][positions] = True
                values = data[name]
                if name in nums:
                    j = nums.index(name)
                    raw = pd.to_numeric(values).to_numpy(dtype='float32', na_value=np.nan)
                    raw[~np.isfinite(raw)] = np.nan
                    if fill:
                        raw[np.isnan(raw)] = -999999.
                    np.testing.assert_array_equal(raw, retained[binding['fit_rows']+positions, columns.index(name)])
                    missing = ~np.isfinite(raw) | (raw == -999999.)
                    numbers[positions, j] = (np.where(missing, encoder.medians[j], raw) - encoder.means[j]) / encoder.scales[j]
                    numbers[positions, len(nums)+j] = missing.astype('float32')
                else:
                    raw = values.astype('string')
                    mapping = {word: i+2 for i, word in enumerate(original['vocab'][name])}
                    loader_codes = raw.map(mapping).fillna(1).where(raw.notna(), 0).to_numpy('float32')
                    np.testing.assert_array_equal(loader_codes, retained[binding['fit_rows']+positions, columns.index(name)])
                    if name not in base_categories:
                        raw = raw.fillna('MISSING')
                    codes = raw.map(encoder.categories[name]).fillna(1).where(raw.notna(), 0).to_numpy('int64')
                    categories[positions, cats.index(name)] = codes
            guard()
    assert all(value.all() for value in seen.values())
    numbers.flush(); categories.flush()
    del numbers, categories, retained, seen
    gc.collect()
    tune.to_parquet(OUT / 'tune_metadata.parquet', index=False)
    result = dict(status='cpu_inputs_passed', source_sha256=sha(Path(__file__)), protocol_sha256=sha(OUT / 'protocol.json'),
                  scientific_protocol_sha256=sha(BASE / 'v1/protocol.json'), resource_protocol_sha256=sha(BASE / 'v2/protocol.json'),
                  preparation_sha256=canary['preparation_sha256'], prior_canary_validation_sha256=prior_hash,
                  rows=len(tune), tune_ids_hash=binding['tune_ids_hash'], tune_label_hash=binding['tune_label_hash'],
                  raw_all387_tune_inputs_equal_retained_matrix=True, independent_tensor_encoding=True,
                  target_scaling=scaling, peak_bytes=guard(), no_GPU=True,
                  outputs={name: sha(OUT / name) for name in ('numbers.npy', 'categories.npy', 'tune_metadata.parquet')})
    write(OUT / 'cpu_receipt.json', result)
    print(result, flush=True)


def replay_gpu():
    assert not (OUT / 'gpu_receipt.json').exists()
    cpu = read(OUT / 'cpu_receipt.json')
    assert cpu['source_sha256'] == sha(Path(__file__)) and cpu['protocol_sha256'] == sha(OUT / 'protocol.json')
    for name, expected in cpu['outputs'].items():
        assert sha(OUT / name) == expected
    prep, protocol, canary, prior_hash = preparation()
    marker = read(MODEL / 'manifest.json')
    assert marker['status'] == 'complete' and marker['protocol_sha256'] == cpu['scientific_protocol_sha256']
    for name, expected in marker['outputs'].items():
        assert sha(MODEL / name) == expected
    saved = joblib.load(MODEL / 'fit_model.joblib')
    encoder = saved['encoder']
    for name in ('numeric', 'columns', 'categories'):
        assert getattr(encoder, name) == getattr(prep['encoder'], name)
    for name in ('medians', 'means', 'scales'):
        np.testing.assert_array_equal(getattr(encoder, name), getattr(prep['encoder'], name))
    for name, expected in cpu['target_scaling'].items():
        assert saved[name] == expected
    net = ple.Network(2*len(encoder.numeric), [len(m)+2 for m in encoder.categories.values()], prep['bins'], prep['embedded'], prep['passthrough'], 32)
    net.load_state_dict(saved['estimator'].state_dict(), strict=True)
    state = net.state_dict()
    assert all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point())
    assert sum(p.numel() for p in net.parameters()) == read(MODEL / 'fit_evidence.json')['parameter_count']
    np.testing.assert_array_equal(net.embedded.numpy(), prep['embedded'].numpy())
    del saved
    gc.collect(); guard()
    values = np.load(OUT / 'numbers.npy', mmap_mode='r')
    categories = np.load(OUT / 'categories.npy', mmap_mode='r')
    tune = pd.read_parquet(OUT / 'tune_metadata.parquet')
    recorded = pd.read_parquet(MODEL / 'tune_predictions.parquet')
    for name in (ID, TIME, TARGET, 'proxy_sec'):
        np.testing.assert_array_equal(recorded[name], tune[name])
    torch.cuda.reset_peak_memory_stats()
    net = net.cuda().eval()
    predictions = np.empty(len(tune), dtype=float)
    with torch.inference_mode():
        for start in range(0, len(tune), BATCH):
            stop = min(start+BATCH, len(tune))
            x = torch.from_numpy(np.array(values[start:stop], copy=True)).cuda()
            c = torch.from_numpy(np.array(categories[start:stop], copy=True)).cuda()
            result = net(x, c)
            assert result.shape == (stop-start, 32) and torch.isfinite(result).all()
            predictions[start:stop] = result.mean(dim=1).cpu().numpy().astype(float)
            assert torch.cuda.max_memory_allocated() < 4*1024**3
            if start % (BATCH*30) == 0:
                print('GPU_ROWS', stop, 'host_peak', guard(), flush=True)
    predictions = predictions*cpu['target_scaling']['y_scale']+cpu['target_scaling']['y_mean']+tune.proxy_sec.to_numpy(float)
    delta = np.abs(predictions-recorded.prediction_sec.to_numpy(float))
    result = dict(status='passed' if delta.max() <= ATOL else 'numerical_tolerance_failed',
                  source_sha256=sha(Path(__file__)), cpu_receipt_sha256=sha(OUT / 'cpu_receipt.json'),
                  producer_manifest_sha256=sha(MODEL / 'manifest.json'), model_sha256=marker['outputs']['fit_model.joblib'],
                  rows=len(tune), batch_size=BATCH, atol_sec=ATOL, rtol=0., max_abs_delta_sec=float(delta.max()),
                  mean_abs_delta_sec=float(delta.mean()), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_bytes=guard(), independent_architecture_strict_state_load=True, encoder_exact_to_verified_canary=True,
                  parameters_unchanged=True, selected_epochs=marker['selected_epochs'])
    net.cpu(); del net, x, c
    gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / 'independent_predictions.npy', predictions)
    result['predictions_sha256'] = sha(OUT / 'independent_predictions.npy')
    write(OUT / 'gpu_receipt.json', result)
    print(result, flush=True)
    assert result['status'] == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['cpu', 'gpu'], required=True)
    args = parser.parse_args()
    pa.set_cpu_count(1); pa.set_io_thread_count(1); torch.set_num_threads(1)
    with threadpool_limits(1):
        (prepare_cpu if args.stage == 'cpu' else replay_gpu)()
