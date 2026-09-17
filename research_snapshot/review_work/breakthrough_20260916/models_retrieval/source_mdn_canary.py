"""CPU-only synthetic timing of the frozen source-mixture forward/backward path."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'review_work/breakthrough_20260916/models/source_mdn/adapter.py'
OUT = ROOT/'private_runs/breakthrough_20260916/models_retrieval/source_mdn_canary'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_adapter():
    spec = importlib.util.spec_from_file_location('source_mdn_adapter_canary', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(adapter, rows, seed):
    rng = np.random.default_rng(seed)
    values = {}
    for clock in adapter.CLOCKS:
        a = rng.normal(900., 300., size=rows)
        a[rng.random(rows) < .1] = np.nan
        values['takeoff_minus_'+clock] = a.astype('float32')
    for j in range(99):
        a = rng.normal(size=rows).astype('float32')
        a[rng.random(rows) < .05] = np.nan
        values[f'numeric_{j}'] = a
    for j in range(11):
        values[f'category_{j}'] = pd.Categorical(rng.integers(0, 128, size=rows).astype(str))
    frame = pd.DataFrame(values)
    assert frame.shape == (rows, 115)
    # Every clock unavailable and extreme raw outcomes are both represented.
    frame.loc[0, ['takeoff_minus_'+c for c in adapter.CLOCKS]] = np.nan
    labels = rng.normal(1000., 400., size=rows)
    labels[:4] = [-12., 131167., -1e12, 1e12]
    return frame, labels


def benchmark(adapter, frame, raw_labels, batch_size, repeats, objective, threads, seed):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    encoder = adapter.FrameEncoder().fit(frame, neural=True)
    # Fixed diagnostic affine units keep extreme synthetic raw values in the test.
    center, scale = 1000., 400.
    source = adapter.tensors(encoder, frame, center, scale)
    ids = torch.arange(batch_size) % len(frame)
    values = tuple(value[ids] for value in source)
    labels = torch.as_tensor((raw_labels-center)/scale, dtype=torch.float64)[ids]
    cardinalities = [len(mapping)+2 for mapping in encoder.categories.values()]
    network = adapter.SourceMixture(source[0].shape[1], cardinalities, scale).cpu()
    assert all(p.device.type == 'cpu' for p in network.parameters())
    assert all(v.device.type == 'cpu' for v in values)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.0001)
    samples = []
    for step in range(repeats+1):
        if psutil.virtual_memory().available < 8*1024**3:
            raise MemoryError('8GiB reserve required throughout synthetic benchmark')
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        parameters = network(*values, return_distribution=True)
        forward = time.perf_counter()-began
        loss_started = time.perf_counter()
        if objective == 'student_t_nll':
            loss = adapter.member_negative_log_likelihood(parameters, labels).mean()
        else:
            means = adapter.analytical_mean(parameters[0], parameters[1])
            loss = ((means-labels[:, None])**2).mean()
        loss_seconds = time.perf_counter()-loss_started
        assert torch.isfinite(loss)
        backward_started = time.perf_counter()
        loss.backward()
        grads = [p.grad for p in network.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        gradient_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 10.)
        assert torch.isfinite(gradient_norm)
        # PyTorch2.14 AdamW.step probes the accelerator stream even for CPU tensors.
        # This diagnostic intentionally times forward/backward only to leave CUDA untouched.
        assert all(torch.isfinite(p).all() for p in network.parameters())
        backward = time.perf_counter()-backward_started
        total = time.perf_counter()-began
        record = {'step': step, 'warmup': step == 0, 'forward_sec': forward,
            'loss_sec': loss_seconds, 'backward_sec': backward, 'total_sec': total,
            'loss': float(loss.detach()), 'gradient_norm_before_cap': float(gradient_norm)}
        samples.append(record)
        print('STEP', objective, batch_size, step, round(total, 4), flush=True)
    with torch.inference_mode():
        network.eval()
        mean = network(*values).mean(dim=1)
        assert torch.isfinite(mean).all()
        logw, means, scales = network(*values, return_distribution=True)
        assert torch.isfinite(scales).all() and torch.all(scales > 0)
        assert torch.all(logw[0, :, :5].exp() == 0) and torch.all(logw[0, :, 5].exp() == 1)
    median = float(np.median([v['total_sec'] for v in samples[1:]]))
    return {'objective': objective, 'batch_size': batch_size, 'unique_synthetic_rows': len(frame),
        'raw_features': len(frame.columns), 'encoded_numeric': source[0].shape[1],
        'categorical_columns': len(cardinalities), 'cardinalities': cardinalities,
        'parameters': sum(p.numel() for p in network.parameters()),
        'input_tensor_bytes': sum(v.nelement()*v.element_size() for v in values)+labels.nelement()*labels.element_size(),
        'distribution_each_tensor_shape': [batch_size, 8, 6],
        'three_distribution_tensors_float64_equivalent_bytes': batch_size*8*6*8*3,
        'median_step_sec': median, 'samples': samples,
        'finite_loss_gradients_parameters_mean': True, 'all_missing_clock_direct_only': True,
        'optimizer_step_executed': False,
        'cpu_only_per_million_rows_epoch_seconds': math.ceil(1000000/batch_size)*median,
        'caution': 'CPU synthetic forward/backward timing only; optimizerstep excluded becausePyTorch2.14 probesCUDAstream evenforCPU. Repeating2Krows tobatch4096testsallocation/compute, not diversity. NoGPUtiming/peakVRAM.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    path = OUT/f'cpu_s{args.seed}.json'
    if path.exists():
        raise ValueError('Preserve prior receipt; select newseed')
    if psutil.virtual_memory().available < 10*1024**3:
        raise MemoryError('Need10GiBfree before importingMDN toretain8GiBreserve plusdiagnosticmemory')
    adapter = load_adapter()
    source_hashes = {'canary': sha(__file__), 'frozen_adapter': sha(SOURCE),
                     'encoder': sha(ROOT/'review_work/breakthrough_20260916/models/encoders.py')}
    record = {'status': 'running', 'source_hashes': source_hashes, 'private_data_read': False,
        'network_calls': False, 'gpu_launched': False, 'cuda_visible_devices': '', 'threads': args.threads, 'benchmarks': []}
    write(path, record)
    began = time.perf_counter()
    try:
        assert not torch.cuda.is_initialized()
        frame, labels = synthetic(adapter, 2048, args.seed)
        for objective in ['student_t_nll', 'same_head_mean_mse']:
            result = benchmark(adapter, frame, labels, 4096, args.repeats, objective, args.threads, args.seed)
            record['benchmarks'].append(result)
            write(path, record)
        assert not torch.cuda.is_initialized()
        assert sha(SOURCE) == source_hashes['frozen_adapter']
        record.update(status='passed', runtime_sec=time.perf_counter()-began,
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            nll_vs_same_head_mse_step_ratio=record['benchmarks'][0]['median_step_sec']/record['benchmarks'][1]['median_step_sec'],
            gpu_estimate='NotinferablefromCPU: consumerGPUdoubleprecisionratioandkernelsdiffer. ScheduleactualGPUcanarybefore64epochbudget.')
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        write(path, record)
        raise
    write(path, record)
    print('PASSED', path, record['runtime_sec'], flush=True)


if __name__ == '__main__':
    main()
