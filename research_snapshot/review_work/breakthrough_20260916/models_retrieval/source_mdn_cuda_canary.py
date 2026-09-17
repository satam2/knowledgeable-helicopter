"""Centrally launched CUDA-only synthetic source-MDN optimizer timing."""
import torch
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import tabm

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT/'review_work/breakthrough_20260916/models/source_mdn/adapter.py'
ENCODER = ROOT/'review_work/breakthrough_20260916/models/encoders.py'
OUT = ROOT/'private_runs/breakthrough_20260916/models_retrieval/source_mdn_cuda_canary'
EXPECTED = {
    'source_mdn_adapter.py': '9287d11ae86ce99fdcaeb006393279bce70268f0c82d46a8846d939dd78fb13d',
    'encoders.py': '821d6b1324d17f39443f084b1e6c2e4b319e8e47023f812a78e03fa914ad9c2c',
}
WARMUP = 3
TIMED = 10
BATCH = 4096


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def source_files():
    return {'source_mdn_cuda_canary.py': Path(__file__), 'source_mdn_adapter.py': ADAPTER,
            'encoders.py': ENCODER, 'installed_tabm.py': Path(tabm.__file__)}


def prepare(args):
    directory = OUT/f'cuda_s{args.seed}'
    sources = source_files()
    hashes = {name: sha(path) for name, path in sources.items()}
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise ValueError('Frozen source changed: '+name)
    if (directory/'manifest.json').exists():
        raise ValueError('Preserve previous attemptedcanary; select a newseed')
    payload = {'device': args.device, 'seed': args.seed, 'threads': args.threads,
        'source_hashes': hashes, 'versions': {name: version(name) for name in ['torch', 'tabm', 'numpy', 'pandas']},
        'warmup_steps': WARMUP, 'timed_steps': TIMED, 'batch_size': BATCH,
        'synthetic_rows': 2048, 'raw_features': 115, 'cpu_fallback_allowed': False,
        'private_data_read': False, 'network_calls': False,
        'timing': 'SynchronizedGPU forward/StudentTdoubleNLL/backward/gradclip/AdamWstep; excludesfeatureencoding,H2D,fulltune andcheckpoints',
        'target': 'Synthetic rawsecondrange[-1e12,1e12], negative and131167; affinecenter1000scale400; no clipping',
        'initial_min_host_free_gib': 10, 'continued_min_host_free_gib': 8}
    path = directory/'prepared.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('PreparedCUDAcanary changed')
    write(path, payload)
    snapshot = directory/'source'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = snapshot/name
        if destination.exists() and sha(destination) != hashes[name]:
            raise ValueError('Snapshotchanged: '+name)
        if not destination.exists():
            shutil.copyfile(source, destination)
    return directory, payload


def load_adapter():
    spec = importlib.util.spec_from_file_location('frozen_mdn_cuda_timing_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(module, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for clock in module.CLOCKS:
        values = rng.normal(900., 300., 2048).astype('float32')
        values[rng.random(2048) < .1] = np.nan
        data['takeoff_minus_'+clock] = values
    for i in range(99):
        values = rng.normal(size=2048).astype('float32')
        values[rng.random(2048) < .05] = np.nan
        data[f'numeric_{i}'] = values
    for i in range(11):
        data[f'category_{i}'] = pd.Categorical(rng.integers(0, 128, 2048).astype(str))
    frame = pd.DataFrame(data)
    frame.loc[0, ['takeoff_minus_'+c for c in module.CLOCKS]] = np.nan
    target = rng.normal(1000., 400., 2048)
    target[:4] = [-12., 131167., -1e12, 1e12]
    assert frame.shape == (2048, 115)
    return frame, target


def launch(args, directory, declaration):
    record = {'status': 'running', 'declaration': declaration, 'samples': [], 'gpu_launched': False}
    write(directory/'manifest.json', record)
    began = time.perf_counter()
    try:
        if args.device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly required; noCPUfallback')
        if psutil.virtual_memory().available < 10*1024**3:
            raise MemoryError('Need10GiBavailable beforeCUDAcanary')
        module = load_adapter()
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        frame, raw = synthetic(module, args.seed)
        encoder = module.FrameEncoder().fit(frame, neural=True)
        source = module.tensors(encoder, frame, 1000., 400.)
        ids = torch.arange(BATCH) % len(frame)
        values = tuple(value[ids].to('cuda') for value in source)
        labels = torch.as_tensor((raw-1000.)/400., dtype=torch.float64)[ids].to('cuda')
        network = module.SourceMixture(source[0].shape[1], [len(v)+2 for v in encoder.categories.values()], 400.).to('cuda')
        optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.0001)
        assert all(p.device.type == 'cuda' for p in network.parameters())
        assert all(v.device.type == 'cuda' for v in values)
        record.update(gpu_launched=True, gpu_name=torch.cuda.get_device_name(),
            gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            encoded_numeric=source[0].shape[1], categorical_columns=len(encoder.categories),
            parameters=sum(p.numel() for p in network.parameters()))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        network.train()
        for step in range(WARMUP+TIMED):
            if psutil.virtual_memory().available < 8*1024**3:
                raise MemoryError('8GiBavailable reserve violated')
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            distribution = network(*values, return_distribution=True)
            loss = module.member_negative_log_likelihood(distribution, labels).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 10.)
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            assert torch.isfinite(loss) and torch.isfinite(norm)
            assert gradients and all(torch.isfinite(v).all() for v in gradients)
            assert all(torch.isfinite(p).all() for p in network.parameters())
            item = {'step': step, 'warmup': step < WARMUP, 'synchronized_training_step_sec': seconds,
                    'loss': float(loss.detach().cpu()), 'gradient_norm_before_cap': float(norm.detach().cpu())}
            record['samples'].append(item)
            print('CUDA_STEP', step, 'warmup' if step < WARMUP else 'timed', round(seconds, 6), flush=True)
        network.eval()
        with torch.inference_mode():
            logw, means, scales = network(*values, return_distribution=True)
            raw_mean = module.analytical_mean(logw, means).mean(dim=1)*400.+1000.
            assert torch.isfinite(raw_mean).all()
            assert torch.isfinite(scales).all() and torch.all(scales > 0)
            assert torch.all(logw[0, :, :5].exp() == 0) and torch.all(logw[0, :, 5].exp() == 1)
        torch.cuda.synchronize()
        timed = [v['synchronized_training_step_sec'] for v in record['samples'] if not v['warmup']]
        median = float(np.median(timed))
        assert len(timed) == TIMED
        assert {k: sha(p) for k, p in source_files().items()} == declaration['source_hashes']
        record.update(status='passed', median_step_sec=median, mean_step_sec=float(np.mean(timed)),
            p90_step_sec=float(np.quantile(timed, .9)), projected_1M_training_epoch_sec=math.ceil(1000000/BATCH)*median,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
            runtime_sec=time.perf_counter()-began, finite_original_mean_gradients=True,
            all_missing_clock_direct_only=True, optimizer_steps=WARMUP+TIMED,
            limitations='Syntheticcategorymix andfixedbatch; extrapolationexcludesfullfeaturesencoding,H2D,fulltune,IO,checkpoints,anddata-specificstopping. Noaccuracyclaim.')
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc(), runtime_sec=time.perf_counter()-began)
        write(directory/'manifest.json', record)
        raise
    write(directory/'manifest.json', record)
    print('CUDA_CANARY_PASSED', directory, 'projected1Mepochsec', record['projected_1M_training_epoch_sec'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda'], required=True)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 2:
        raise ValueError('CanaryCPUthreadbudget is1or2')
    directory, declaration = prepare(args)
    if args.prepare_only:
        assert not torch.cuda.is_initialized()
        print('PREPARED_ONLY', directory, 'CUDAuninitialized', flush=True)
        return
    launch(args, directory, declaration)


if __name__ == '__main__':
    main()
