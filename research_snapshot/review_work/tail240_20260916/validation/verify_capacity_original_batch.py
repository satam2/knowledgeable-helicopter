"""Declared original batch8192 replay after preserved smaller-batch failure."""
import verify_capacity_replay as frozen
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

OUT = frozen.OUT / 'original_batch8192_v1'
BATCH, ATOL, GPU_CAP = 8192, .001, 12*1024**3


def declare():
    failed = frozen.read(frozen.OUT / 'gpu_receipt.json')
    assert failed['status'] == 'numerical_tolerance_failed'
    diagnosis = frozen.read(frozen.OUT / 'tolerance_diagnosis.json')
    assert diagnosis['native_encoder_numeric_and_category_tensors_exact_all_rows']
    value = dict(source_sha256=frozen.sha(Path(__file__)), original_helper_sha256=frozen.sha(Path(frozen.__file__)),
        batch_size=BATCH, atol_sec=ATOL, rtol=0., GPU_allocated_cap_bytes=GPU_CAP, host_peak_cap_bytes=4*1024**3,
        original_failure_sha256=frozen.sha(frozen.OUT / 'gpu_receipt.json'),
        diagnosis_sha256=frozen.sha(frozen.OUT / 'tolerance_diagnosis.json'),
        cpu_inputs_sha256=frozen.sha(frozen.OUT / 'cpu_receipt.json'),
        producer_manifest_sha256=frozen.sha(frozen.MODEL / 'manifest.json'),
        purpose='Reproduce frozen producer inference geometry after exact input/bin/state checks. First8192batch must remain below12GiB GPU before continuing. Stop on failure; no retries, no tolerance change or model edits.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert frozen.read(path) == value
    else:
        frozen.write(path, value)
    return value


def replay():
    protocol = declare()
    assert not (OUT / 'receipt.json').exists() and not (OUT / 'failure.json').exists()
    cpu = frozen.read(frozen.OUT / 'cpu_receipt.json')
    for name, expected in cpu['outputs'].items():
        assert frozen.sha(frozen.OUT / name) == expected
    prep, _, _, _ = frozen.preparation()
    marker = frozen.read(frozen.MODEL / 'manifest.json')
    assert frozen.sha(frozen.MODEL / 'fit_model.joblib') == marker['outputs']['fit_model.joblib']
    saved = joblib.load(frozen.MODEL / 'fit_model.joblib')
    encoder = prep['encoder']
    for key in ('numeric', 'categories', 'columns'):
        assert getattr(encoder, key) == getattr(saved['encoder'], key)
    for key in ('means', 'medians', 'scales'):
        np.testing.assert_array_equal(getattr(encoder, key), getattr(saved['encoder'], key))
    net = frozen.ple.Network(2*len(encoder.numeric), [len(m)+2 for m in encoder.categories.values()],
                            prep['bins'], prep['embedded'], prep['passthrough'], 32)
    left, right = net.numeric_embeddings.impl.state_dict(), saved['estimator'].numeric_embeddings.impl.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])
    net.load_state_dict(saved['estimator'].state_dict(), strict=True)
    assert torch.equal(net.embedded, prep['embedded']) and torch.equal(net.passthrough, prep['passthrough'])
    for key, value in cpu['target_scaling'].items():
        assert saved[key] == value
    del saved, left, right
    frozen.gc.collect(); frozen.guard()
    numeric = np.load(frozen.OUT / 'numbers.npy', mmap_mode='r')
    categories = np.load(frozen.OUT / 'categories.npy', mmap_mode='r')
    tune = pd.read_parquet(frozen.OUT / 'tune_metadata.parquet')
    stored = pd.read_parquet(frozen.MODEL / 'tune_predictions.parquet')
    np.testing.assert_array_equal(stored[frozen.ID], tune[frozen.ID])
    prediction = np.empty(len(tune), float)
    torch.cuda.reset_peak_memory_stats()
    try:
        net = net.cuda().eval()
        with torch.inference_mode():
            for start in range(0, len(tune), BATCH):
                stop = min(start+BATCH, len(tune))
                numbers = torch.from_numpy(np.array(numeric[start:stop], copy=True)).cuda()
                cats = torch.from_numpy(np.array(categories[start:stop], copy=True)).cuda()
                result = net(numbers, cats)
                assert result.shape == (stop-start, 32) and torch.isfinite(result).all()
                prediction[start:stop] = result.mean(dim=1).cpu().numpy()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                assert peak < GPU_CAP, f'GPU allocated cap exceeded: {peak}'
                host = frozen.guard()
                if start == 0:
                    frozen.write(OUT / 'first_batch.json', dict(status='passed', rows=stop,
                        GPU_peak_allocated_bytes=peak, GPU_reserved_bytes=torch.cuda.memory_reserved(), host_peak_bytes=host,
                        protocol_sha256=frozen.sha(OUT / 'protocol.json')))
                print('ORIGINAL_BATCH_ROWS', stop, 'GPU_PEAK', peak, 'HOST_PEAK', host, flush=True)
        prediction = prediction*cpu['target_scaling']['y_scale']+cpu['target_scaling']['y_mean']+tune.proxy_sec.to_numpy(float)
        delta = np.abs(prediction-stored.prediction_sec.to_numpy(float))
        receipt = dict(status='passed' if delta.max() <= ATOL else 'numerical_tolerance_failed',
            source_sha256=frozen.sha(Path(__file__)), protocol_sha256=frozen.sha(OUT / 'protocol.json'),
            rows=len(tune), batch_size=BATCH, atol_sec=ATOL, rtol=0., max_abs_delta_sec=float(delta.max()),
            exact_prediction_equality=bool(np.array_equal(prediction,stored.prediction_sec.to_numpy(float))),
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), host_peak_bytes=frozen.guard(),
            independent_architecture_strict_state_load=True, tensors_and_bins_exact=True,
            model_sha256=marker['outputs']['fit_model.joblib'], producer_manifest_sha256=protocol['producer_manifest_sha256'],
            original_batch1024_failure_preserved=True)
        np.save(OUT / 'independent_predictions.npy', prediction)
        receipt['prediction_sha256'] = frozen.sha(OUT / 'independent_predictions.npy')
        frozen.write(OUT / 'receipt.json', receipt)
        print(receipt, flush=True)
        assert receipt['status'] == 'passed'
    except BaseException as exc:
        frozen.write(OUT / 'failure.json', dict(status='failed_no_retry', exception=type(exc).__name__, message=str(exc),
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), protocol_sha256=frozen.sha(OUT / 'protocol.json')))
        raise
    finally:
        net.cpu()
        frozen.gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1)
    with frozen.threadpool_limits(1):
        declare() if args.declare_only else replay()
