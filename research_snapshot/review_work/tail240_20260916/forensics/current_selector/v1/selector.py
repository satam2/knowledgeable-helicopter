"""Small CPU nonlinear convex selector; fit accepts early data only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import torch
import lightgbm
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models/context_gate'))
from encoders import FrameEncoder
import gate

SEED = 20260916
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


class Network(torch.nn.Module):
    def __init__(self, numeric_dimensions, category_sizes, initial_weights):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(size, 4, padding_idx=1) for size in category_sizes])
        for embedding in self.embeddings:
            torch.nn.init.normal_(embedding.weight, 0., .01)
            with torch.no_grad():
                embedding.weight[1].zero_()
        self.hidden = torch.nn.Linear(numeric_dimensions + 4*len(category_sizes), 32)
        self.output = torch.nn.Linear(32, 9)
        torch.nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor(np.log(initial_weights), dtype=torch.float32))

    def forward(self, numeric, categories):
        parts = [numeric] + [embedding(categories[:, j]) for j, embedding in enumerate(self.embeddings)]
        return torch.softmax(self.output(torch.relu(self.hidden(torch.cat(parts, dim=1)))), dim=1)


def fit(early_features, early_predictions, early_labels, *, epochs=20, batch_size=4096, guard=lambda: None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    p = np.asarray(early_predictions, dtype=np.float64)
    y = np.asarray(early_labels, dtype=np.float64)
    assert p.shape == (len(y), 9) and len(early_features) == len(y) > 0
    assert np.isfinite(p).all() and np.isfinite(y).all()
    weights = gate.constant_weights(y, p)
    initial = np.maximum(weights, 1e-6)
    initial /= initial.sum()
    encoder = FrameEncoder().fit(early_features, neural=True)
    numeric, categories = encoder.transform(early_features)
    dimensions = dict(numeric_dimensions=numeric.shape[1],
                      category_sizes=[len(v)+2 for v in encoder.categories.values()])
    network = Network(**dimensions, initial_weights=initial)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.0001)
    scale = max(float(np.mean((y-p@weights)**2)), 1.)
    row_mean = p.mean(axis=1)
    relative = torch.from_numpy((p-row_mean[:, None]).astype(np.float32))
    target = torch.from_numpy((y-row_mean).astype(np.float32))
    xn, xc = torch.from_numpy(numeric), torch.from_numpy(categories)
    generator = torch.Generator().manual_seed(SEED)
    history, hidden_gradient, embedding_gradient = [], 0., 0.
    started = time.monotonic()
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.
        for begin in range(0, len(y), batch_size):
            guard()
            index = order[begin:begin+batch_size]
            optimizer.zero_grad(set_to_none=True)
            w = network(xn[index], xc[index])
            error = torch.sum(w*relative[index], dim=1)-target[index]
            loss = error.square().mean()/scale
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite selector loss')
            loss.backward()
            hidden_gradient = max(hidden_gradient, float(network.hidden.weight.grad.abs().max()))
            embedding_gradient = max([embedding_gradient] + [float(e.weight.grad.abs().max()) for e in network.embeddings])
            optimizer.step()
            total += float(loss.detach())*len(index)
        history.append(dict(epoch=epoch+1, scaled_training_batch_loss=total/len(y)))
    network.eval()
    return dict(network=network, encoder=encoder, weights=weights, initial_weights=initial,
                dimensions=dimensions, evidence=dict(epochs=epochs, batch_size=batch_size,
                    hidden_max_gradient=hidden_gradient, embedding_max_gradient=embedding_gradient,
                    initialization_max_abs_delta=float(np.max(np.abs(p@initial-p@weights))),
                    target_scale=scale, history=history, runtime_sec=time.monotonic()-started,
                    torch_version=torch.__version__, numpy_version=np.__version__))


def predict(model, features, predictions, *, batch_size=8192, guard=lambda: None):
    p = np.asarray(predictions, dtype=np.float64)
    assert p.shape == (len(features), 9) and np.isfinite(p).all()
    result, all_weights = np.empty(len(p)), np.empty_like(p)
    model['network'].eval()
    with torch.inference_mode():
        for begin in range(0, len(p), batch_size):
            guard()
            end = min(begin+batch_size, len(p))
            numeric, categories = model['encoder'].transform(features.iloc[begin:end])
            weights = model['network'](torch.from_numpy(numeric), torch.from_numpy(categories)).numpy().astype(np.float64)
            weights /= weights.sum(axis=1, keepdims=True)
            mean = p[begin:end].mean(axis=1)
            result[begin:end] = mean + np.sum(weights*(p[begin:end]-mean[:, None]), axis=1)
            all_weights[begin:end] = weights
    assert np.isfinite(result).all() and np.isfinite(all_weights).all()
    assert (all_weights >= 0).all() and np.max(np.abs(all_weights.sum(1)-1)) < 1e-6
    assert (result >= p.min(1)-1e-8).all() and (result <= p.max(1)+1e-8).all()
    return result, all_weights


def save(model, folder):
    torch.save(model['network'].state_dict(), folder / 'state.pt')
    joblib.dump({k: v for k, v in model.items() if k != 'network'}, folder / 'model.joblib')


def load(folder):
    model = joblib.load(folder / 'model.joblib')
    network = Network(**model['dimensions'], initial_weights=model['initial_weights'])
    network.load_state_dict(torch.load(folder / 'state.pt', map_location='cpu', weights_only=True))
    network.eval()
    model['network'] = network
    return model
