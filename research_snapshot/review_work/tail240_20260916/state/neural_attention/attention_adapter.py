"""PLE387 plus shared query-conditioned original-event relation pooling."""
import gc
import time
from pathlib import Path
import sys
import torch
import lightgbm
from torch import nn
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
import tabm_gpu as frozen
import tabm_ple_gpu as ple
from encoders import FrameEncoder

CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
FIELDS = ['offset_' + name for name in CLOCKS] + ['aobt_second', 'movement_second',
          'arrival_duration', 'age_sec', 'landing_age_sec', 'same_runway',
          'same_stand', 'same_operator', 'padding_missing']
PREFIXES = [f'flat_{phase}{rank}_' for phase in ['dep', 'arr'] for rank in range(1, 5)]
OWN_CLOCKS = ['takeoff_minus_' + name for name in CLOCKS]


def clean(values):
    result = np.asarray(values, dtype=np.float64).copy()
    result[(result == -999999) | ~np.isfinite(result)] = np.nan
    return result


def token_statistics(frame, encoder):
    """Pool only present fit tokens per semantic field; no rank-specific token scale."""
    count = np.zeros(24, dtype=np.int64)
    sums = np.zeros(24)
    squares = np.zeros(24)
    for start in range(0, len(frame), 65536):
        part = frame.iloc[start:start + 65536]
        own = clean(part[OWN_CLOCKS].to_numpy())
        for slot, prefix in enumerate(PREFIXES):
            raw = clean(part[[prefix + field for field in FIELDS]].to_numpy())
            present = raw[:, 13] == 0
            duration = own - raw[:, :5]
            timestamp = duration - raw[:, 8:9]
            valid = present[:, None] & np.isfinite(own) & np.isfinite(raw[:, :5]) & np.isfinite(raw[:, 8:9])
            if slot >= 4:
                valid[:] = False
            duration[~valid] = np.nan
            timestamp[~valid] = np.nan
            values = np.column_stack([raw, duration, timestamp])
            valid = np.isfinite(values) & present[:, None]
            values = np.where(valid, values, 0.)
            count += valid.sum(axis=0)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
    means = sums / np.maximum(count, 1)
    variance = np.maximum(squares / np.maximum(count, 1) - means**2, 0.)
    scales = np.where(variance > 1e-12, np.sqrt(variance), 1.)
    return dict(means=means.astype(np.float32).tolist(), scales=scales.astype(np.float32).tolist(),
                count=count.tolist(), numeric_columns=list(encoder.numeric),
                fields=FIELDS + ['duration_difference_' + n for n in CLOCKS] + ['timestamp_difference_' + n for n in CLOCKS])


class Network(nn.Module):
    def __init__(self, n_numeric, cardinalities, bins, embedded, passthrough, members, encoder, stats):
        super().__init__()
        # Initialize the unchanged static network first to preserve its seeded initialization.
        self.base = ple.Network(n_numeric, cardinalities, bins, embedded, passthrough, members)
        numeric = encoder.numeric
        self.n_original = len(numeric)
        self.register_buffer('token_columns', torch.tensor([[numeric.index(prefix + field) for field in FIELDS] for prefix in PREFIXES]))
        self.register_buffer('own_columns', torch.tensor([numeric.index(name) for name in OWN_CLOCKS]))
        query = [j for j, name in enumerate(numeric) if not name.startswith('flat_')]
        self.register_buffer('query_columns', torch.tensor(query + [j + len(numeric) for j in query]))
        self.register_buffer('raw_means', torch.from_numpy(encoder.means.copy()))
        self.register_buffer('raw_scales', torch.from_numpy(encoder.scales.copy()))
        self.register_buffer('shared_means', torch.tensor(stats['means']))
        self.register_buffer('shared_scales', torch.tensor(stats['scales']))
        self.register_buffer('phase', torch.tensor([0.] * 4 + [1.] * 4))
        self.token_mlp = nn.Sequential(nn.Linear(49, 64), nn.ReLU(), nn.Linear(64, 64))
        query_width = len(self.query_columns) + sum(e.embedding_dim for e in self.base.embeddings)
        self.query_mlp = nn.Sequential(nn.Linear(query_width, 64), nn.ReLU(), nn.Linear(64, 64))
        self.attention = nn.MultiheadAttention(64, 2, dropout=0., batch_first=True)
        self.correction_head = nn.Linear(64, members)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def token_inputs(self, numbers):
        columns = self.token_columns
        raw = numbers[:, columns] * self.raw_scales[columns] + self.raw_means[columns]
        missing = numbers[:, columns + self.n_original] > .5
        present = (~missing[:, :, 13]) & (raw[:, :, 13] < .5)
        own = numbers[:, self.own_columns] * self.raw_scales[self.own_columns] + self.raw_means[self.own_columns]
        own_missing = numbers[:, self.own_columns + self.n_original] > .5
        duration = own[:, None, :] - raw[:, :, :5]
        timestamp = duration - raw[:, :, 8:9]
        valid = (~own_missing[:, None, :]) & (~missing[:, :, :5]) & (~missing[:, :, 8:9])
        valid = valid & (self.phase[None, :, None] == 0) & present[:, :, None]
        values = torch.cat([raw, duration, timestamp], dim=2)
        masks = torch.cat([missing, ~valid, ~valid], dim=2)
        normalized = (values - self.shared_means) / self.shared_scales
        normalized = torch.where(masks | ~present[:, :, None], 0., normalized)
        tokens = torch.cat([normalized, masks.to(numbers.dtype), self.phase[None, :, None].expand(len(numbers), -1, -1)], dim=2)
        return tokens, present

    def correction(self, numbers, categories):
        tokens, present = self.token_inputs(numbers)
        keys = self.token_mlp(tokens)
        query_values = [numbers[:, self.query_columns]]
        query_values.extend(embedding(categories[:, i]) for i, embedding in enumerate(self.base.embeddings))
        query = self.query_mlp(torch.cat(query_values, dim=1))[:, None, :]
        absent = ~present.any(dim=1)
        padding = ~present
        # Open one zeroed dummy key for all-absent rows to avoid all-masked softmax NaNs.
        padding = padding.clone()
        padding[absent, 0] = False
        keys = torch.where(present[:, :, None], keys, 0.)
        pooled, _ = self.attention(query, keys, keys, key_padding_mask=padding, need_weights=False)
        correction = self.correction_head(pooled[:, 0])
        return torch.where(absent[:, None], 0., correction)

    def forward(self, numbers, categories):
        return self.base(numbers, categories) + self.correction(numbers, categories)


def grouped_bins(numbers, n_original):
    varying = (numbers[:, :n_original] != numbers[0, :n_original]).any(dim=0)
    embedded = torch.where(varying)[0]
    chosen = set(embedded.tolist())
    passthrough = torch.tensor([i for i in range(numbers.shape[1]) if i not in chosen], dtype=torch.long)
    bins = []
    for columns in embedded.split(16):
        bins.extend(ple.rtdl_num_embeddings.compute_bins(numbers[:, columns], n_bins=min(48, len(numbers) - 1)))
    return bins, embedded, passthrough


def canary(x, encoder, bins, embedded, passthrough, stats, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n = 2 * len(encoder.numeric)
    cards = [len(mapping) + 2 for mapping in encoder.categories.values()]
    estimator = Network(n, cards, bins, embedded, passthrough, 8, encoder, stats).cuda()
    values = tuple(value.cuda() for value in frozen.tensors(encoder, x.iloc[:8192]))
    torch.cuda.reset_peak_memory_stats()
    estimator.eval()
    with torch.no_grad():
        prediction = estimator(*values)
        baseline = estimator.base(*values)
        assert torch.equal(prediction, baseline), 'Zero correction initial static parity'
        assert torch.isfinite(prediction).all()
    estimator.train()
    prediction = estimator(*(value[:4096] for value in values))
    prediction.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in estimator.parameters())
    peak = torch.cuda.max_memory_allocated()
    assert peak < 8 * 1024**3, f'Canary GPU budget: {peak}'
    result = dict(fit_rows=min(4096, len(x)), inference_rows=len(values[0]),
                  zero_head_static_exact=True, finite_gradients=True, gpu_peak_bytes=peak,
                  optimizer_updates=0, predictive_evaluation=False)
    del estimator, values, prediction, baseline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
    torch.set_num_threads(threads)
    started = time.monotonic()
    encoder = FrameEncoder().fit(x, neural=True)
    stats = token_statistics(x, encoder)
    numbers, _ = frozen.tensors(encoder, x)
    bins, embedded, passthrough = grouped_bins(numbers, len(encoder.numeric))
    del numbers
    gc.collect()
    preparation = time.monotonic() - started
    canary_receipt = canary(x, encoder, bins, embedded, passthrough, stats, seed)
    original = frozen.Network
    frozen.Network = lambda n_numeric, cardinalities: Network(n_numeric, cardinalities, bins, embedded, passthrough, 8, encoder, stats)
    try:
        model, evidence = frozen.fit(x, y, tuning, steps=steps, seed=seed, threads=threads)
    finally:
        frozen.Network = original
    evidence.update(preparation_sec=preparation, token_statistics=stats, canary=canary_receipt,
                    architecture='Unchanged PLE387 plus shared original8 token/relation MLP64, query64, 2head attention, zero initialized correction8',
                    parameter_count=sum(p.numel() for p in model['estimator'].parameters()),
                    package_effect='Derived relation encoding plus shared query attention plus capacity; not isolated attention effect')
    return model, evidence


predict = frozen.predict
