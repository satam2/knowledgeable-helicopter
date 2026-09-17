"""Small CPU TabM screen with train-only preprocessing and raw-MSE loss."""
import copy
from importlib.metadata import version
import math
import time

# Windows torch DLL initialization must precede pandas/numpy in the entry point too.
import torch
from torch import nn
import numpy as np
import pandas as pd
import tabm


class FrameEncoder:
    def fit(self, x):
        self.columns = list(x.columns)
        self.categories = {}
        self.numeric = []
        for col in self.columns:
            if isinstance(x[col].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(x[col]):
                values = x[col].astype("string")
                self.categories[col] = {v: i + 2 for i, v in enumerate(sorted(values.dropna().unique()))}
            else:
                self.numeric.append(col)
        a = self._numbers(x)
        self.medians = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0 for v in a.T], dtype=np.float32)
        filled = np.where(np.isfinite(a), a, self.medians)
        self.means = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = filled.std(axis=0, dtype=np.float64)
        self.scales = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        return self

    def _numbers(self, x):
        if not self.numeric:
            return np.empty((len(x), 0), dtype=np.float32)
        a = x[self.numeric].to_numpy(dtype=np.float32, na_value=np.nan).copy()
        a[(a == -999999) | ~np.isfinite(a)] = np.nan
        return a

    def transform(self, x):
        if list(x.columns) != self.columns:
            raise ValueError("Feature schema/order differs from fitted encoder")
        a = self._numbers(x)
        missing = ~np.isfinite(a)
        a = (np.where(missing, self.medians, a) - self.means) / self.scales
        numbers = np.concatenate([a, missing.astype(np.float32)], axis=1)
        codes = []
        for col, mapping in self.categories.items():
            values = x[col].astype("string")
            code = values.map(mapping).fillna(1).to_numpy(dtype=np.int64)
            code[values.isna().to_numpy()] = 0
            codes.append(code)
        cats = np.column_stack(codes) if codes else np.empty((len(x), 0), dtype=np.int64)
        return torch.from_numpy(np.ascontiguousarray(numbers)), torch.from_numpy(np.ascontiguousarray(cats))


class EmbeddedTabM(nn.Module):
    def __init__(self, n_num, cardinalities):
        super().__init__()
        dims = [min(8, max(2, math.ceil(math.sqrt(c)))) for c in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(c, d, padding_idx=0) for c, d in zip(cardinalities, dims)])
        for embedding in self.embeddings:
            with torch.no_grad():
                embedding.weight[1].zero_()
        self.backbone = tabm.TabM.make(n_num_features=n_num + sum(dims), d_out=1,
            n_blocks=2, d_block=128, k=8, dropout=0.1, arch_type="tabm")

    def forward(self, numbers, cats):
        features = [numbers] + [embedding(cats[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.backbone(torch.cat(features, dim=1)).squeeze(-1)


def _pred_scaled(estimator, tensors, batch_size=2048):
    estimator.eval()
    numbers, cats = tensors
    result = np.empty(len(numbers), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(numbers), batch_size):
            end = min(start + batch_size, len(numbers))
            result[start:end] = estimator(numbers[start:end], cats[start:end]).mean(dim=1).numpy()
    return result


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or not len(y) or not np.isfinite(y).all():
        raise ValueError("Fit requires nonempty aligned finite labels")
    if steps is not None and (steps < 1 or tuning is not None):
        raise ValueError("Refit steps must be positive and exclude tuning")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    encoder = FrameEncoder().fit(x)
    train = encoder.transform(x)
    y_mean = float(y.mean())
    y_scale = max(float(y.std()), 1.0)
    labels = torch.from_numpy(((y - y_mean) / y_scale).astype(np.float32))
    estimator = EmbeddedTabM(train[0].shape[1], [len(m) + 2 for m in encoder.categories.values()])
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=0.001, weight_decay=0.01)
    tune_tensors = encoder.transform(tuning[0]) if tuning is not None else None
    tune_y = np.asarray(tuning[1], dtype=np.float64) if tuning is not None else None
    if tune_y is not None and (not len(tune_y) or len(tune_y) != len(tuning[0]) or not np.isfinite(tune_y).all()):
        raise ValueError("Tuning requires nonempty aligned finite labels")
    epochs = 15 if steps is None else int(steps)
    best_epoch, best_loss, best_state = 1, float("inf"), None
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        estimator.train()
        order = torch.randperm(len(labels), generator=generator)
        train_sum = 0.0
        for ids in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            predictions = estimator(train[0][ids], train[1][ids])
            # TabM members each optimize squared error, then are averaged only at inference.
            loss = (predictions - labels[ids, None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TabM training loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(ids)
        record = {"epoch": epoch, "train_mse_sec2": train_sum / len(labels) * y_scale ** 2}
        if tuning is not None:
            predicted = _pred_scaled(estimator, tune_tensors) * y_scale + y_mean
            validation_loss = float(np.mean((predicted - tune_y) ** 2))
            record["tune_mse_sec2"] = validation_loss
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = copy.deepcopy(estimator.state_dict())
        else:
            best_epoch = epoch
        record["seconds"] = time.monotonic() - epoch_start
        history.append(record)
        print("TABM_EPOCH", record, flush=True)
        if tuning is not None and epoch - best_epoch >= 3:
            break
    if best_state is not None:
        estimator.load_state_dict(best_state)
    estimator.eval()
    model = dict(estimator=estimator, encoder=encoder, y_mean=y_mean, y_scale=y_scale, steps=best_epoch, threads=threads)
    evidence = dict(steps=best_epoch, fit_seconds=time.monotonic() - start, rows=len(x),
        features=len(x.columns), seed=seed, threads=threads, library="tabm", version=version("tabm"),
        torch_version=torch.__version__, device="cpu", history=history,
        categorical_encoding="fit-only vocab; missing=0 unknown=1; learned embeddings dimension2..8",
        numeric_encoding="fit-only median imputation and standardization; nonfinite/-999999 missing; per-numeric missing indicators",
        target_encoding="fit-only affine mean/std; no label clipping/log transform; mean per-member squared loss",
        params=dict(n_blocks=2, d_block=128, k=8, dropout=0.1, batch_size=1024,
                    max_epochs=epochs, patience=3, learning_rate=0.001, weight_decay=0.01))
    return model, evidence


def predict(model, x):
    torch.set_num_threads(model["threads"])
    result = _pred_scaled(model["estimator"], model["encoder"].transform(x)) * model["y_scale"] + model["y_mean"]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite TabM prediction")
    return result
