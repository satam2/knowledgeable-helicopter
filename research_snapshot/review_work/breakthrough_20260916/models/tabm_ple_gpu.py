"""Official piecewise-linear numerical embeddings with the frozen TabM trainer."""
import math
import time
from importlib.metadata import version
import torch
from torch import nn
import tabm
import rtdl_num_embeddings
import tabm_gpu as frozen
from encoders import FrameEncoder

MEMBERS = 8
N_BINS = 48
D_EMBEDDING = 16


def fit_bins(numbers, n_original_numeric):
    varying = (numbers[:, :n_original_numeric] != numbers[0, :n_original_numeric]).any(dim=0)
    embedded = torch.where(varying)[0]
    passthrough = torch.tensor([i for i in range(numbers.shape[1]) if i not in set(embedded.tolist())], dtype=torch.long)
    bins = rtdl_num_embeddings.compute_bins(numbers[:, embedded], n_bins=min(N_BINS, len(numbers) - 1)) if len(embedded) else []
    return bins, embedded, passthrough


class Network(nn.Module):
    def __init__(self, n_numeric, cardinalities, bins, embedded, passthrough, members):
        super().__init__()
        self.register_buffer("embedded", embedded)
        self.register_buffer("passthrough", passthrough)
        self.numeric_embeddings = (rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding=D_EMBEDDING, activation=False, version="B") if bins else None)
        dimensions = [min(32, max(4, math.ceil(math.sqrt(count)))) for count in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(count, dimension, padding_idx=0)
                                         for count, dimension in zip(cardinalities, dimensions)])
        for embedding in self.embeddings:
            with torch.no_grad():
                embedding.weight[1].zero_()
        chunks = [D_EMBEDDING] * len(embedded) + [1] * len(passthrough) + dimensions
        self.ensemble_view = tabm.EnsembleView(k=members)
        self.backbone = tabm.make_tabm_backbone(d_in=sum(chunks), n_blocks=3, d_block=256, dropout=.1,
                                               k=members, arch_type="tabm", start_scaling_init="normal",
                                               start_scaling_init_chunks=chunks)
        self.output = tabm.LinearEnsemble(256, 1, k=members)

    def forward(self, numbers, categories):
        values = []
        if self.numeric_embeddings is not None:
            values.append(self.numeric_embeddings(numbers[:, self.embedded]).flatten(1))
        values.append(numbers[:, self.passthrough])
        values.extend(embedding(categories[:, i]) for i, embedding in enumerate(self.embeddings))
        return self.output(self.backbone(self.ensemble_view(torch.cat(values, dim=1)))).squeeze(-1)


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    torch.set_num_threads(threads)
    started = time.monotonic()
    encoder = FrameEncoder().fit(x, neural=True)
    numbers, _ = frozen.tensors(encoder, x)
    bins, embedded, passthrough = fit_bins(numbers, len(encoder.numeric))
    preparation = time.monotonic() - started
    del numbers, encoder
    original = frozen.Network
    frozen.Network = lambda n_numeric, cardinalities: Network(n_numeric, cardinalities, bins, embedded, passthrough, MEMBERS)
    try:
        model, evidence = frozen.fit(x, y, tuning, steps=steps, seed=seed, threads=threads)
    finally:
        frozen.Network = original
    evidence["params"].update(members=MEMBERS, numeric_embedding="PiecewiseLinearEmbeddings version B",
                               numeric_embedding_dim=D_EMBEDDING, numeric_quantile_bins=N_BINS,
                               start_scaling_init="normal", feature_grouped_scaling=True)
    evidence.update(bin_fit_sec=preparation, bin_source="Entire fit context only; no score or tune values",
                    embedded_numeric_indices=embedded.tolist(), passthrough_indices=passthrough.tolist(),
                    actual_bin_edges=[len(edges) for edges in bins],
                    numeric_embedding_version=version("rtdl_num_embeddings"),
                    encoding="Frozen fit-only category/median/standardization/missing indicators plus official piecewise numerical embeddings; raw labels and squared loss unchanged")
    return model, evidence


predict = frozen.predict
