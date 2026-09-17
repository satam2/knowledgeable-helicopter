"""Original PLE static network with only ensemble member count increased to32."""
import torch
import lightgbm
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
import tabm_gpu as frozen
import tabm_ple_gpu as ple
from encoders import FrameEncoder

MEMBERS = 32


def grouped_bins(numbers, n_original):
    varying = (numbers[:, :n_original] != numbers[0, :n_original]).any(dim=0)
    embedded = torch.where(varying)[0]
    chosen = set(embedded.tolist())
    passthrough = torch.tensor([i for i in range(numbers.shape[1]) if i not in chosen], dtype=torch.long)
    bins = []
    for columns in embedded.split(16):
        bins.extend(ple.rtdl_num_embeddings.compute_bins(numbers[:, columns], n_bins=min(48, len(numbers)-1)))
    return bins, embedded, passthrough


def prepare(x):
    encoder = FrameEncoder().fit(x, neural=True)
    numbers, _ = frozen.tensors(encoder, x)
    bins, embedded, passthrough = grouped_bins(numbers, len(encoder.numeric))
    del numbers
    gc.collect()
    return dict(encoder=encoder, bins=bins, embedded=embedded, passthrough=passthrough)


def network(preparation):
    encoder = preparation['encoder']
    return ple.Network(2*len(encoder.numeric), [len(m)+2 for m in encoder.categories.values()],
        preparation['bins'], preparation['embedded'], preparation['passthrough'], MEMBERS)


def canary(x, preparation, seed=20260916):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    estimator = network(preparation).cuda()
    values = tuple(v.cuda() for v in frozen.tensors(preparation['encoder'], x.iloc[:8192]))
    torch.cuda.reset_peak_memory_stats()
    estimator.eval()
    with torch.no_grad():
        outputs = estimator(*values)
        assert outputs.shape == (len(values[0]), 32)
        assert torch.isfinite(outputs).all()
        assert torch.isfinite(outputs.mean(dim=1)).all()
    estimator.train()
    predicted = estimator(*(v[:4096] for v in values))
    predicted.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in estimator.parameters())
    peak = torch.cuda.max_memory_allocated()
    assert peak < 12*1024**3, f'Canary12GiB allocated GPU budget: {peak}'
    result = dict(members=MEMBERS, inference_rows=len(values[0]), backward_rows=min(len(values[0]),4096),
        finite_outputs=True, finite_gradients=True, optimizer_updates=0, full_training_started=False,
        gpu_peak_allocated_bytes=peak, parameter_count=sum(p.numel() for p in estimator.parameters()))
    del estimator, values, outputs, predicted
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fit(x, y, tuning=None, *, seed=20260916, threads=2, steps=None):
    torch.set_num_threads(threads)
    started = time.monotonic()
    preparation = prepare(x)
    preparation_time = time.monotonic()-started
    original = frozen.Network
    frozen.Network = lambda n_numeric, cardinalities: ple.Network(n_numeric, cardinalities,
        preparation['bins'], preparation['embedded'], preparation['passthrough'], MEMBERS)
    try:
        model, evidence = frozen.fit(x, y, tuning, steps=steps, seed=seed, threads=threads)
    finally:
        frozen.Network = original
    evidence['params'].update(members=MEMBERS, numeric_embedding='PiecewiseLinearEmbeddings version B',
        numeric_embedding_dim=16, numeric_quantile_bins=48, start_scaling_init='normal', feature_grouped_scaling=True)
    evidence.update(bin_fit_sec=preparation_time, bin_source='Every original fit row only',
        embedded_numeric_indices=preparation['embedded'].tolist(), passthrough_indices=preparation['passthrough'].tolist(),
        actual_bin_edges=[len(b) for b in preparation['bins']],
        parameter_count=sum(p.numel() for p in model['estimator'].parameters()),
        only_model_change='PLE ensemble members8to32; no attention or new inputs; frozen trainer unchanged')
    return model, evidence


predict = frozen.predict
