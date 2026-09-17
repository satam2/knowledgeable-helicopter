"""Fit-reference-only encoders and a bounded GRU/static matched network."""
import torch
from torch import nn
import copy
import time
import numpy as np
import pandas as pd
import cache


def category_values(values):
    return values.astype('string').replace({'<missing>': pd.NA, 'MISSING': pd.NA, 'm:': pd.NA})


class NumericEncoder:
    def fit(self, values):
        a = self.raw(values)
        self.mean = np.nanmean(a, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.).astype('float32')
        self.std = np.nanstd(a, axis=0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-6), self.std, 1.).astype('float32')
        return self

    def raw(self, values):
        a = np.asarray(values, dtype='float32').copy()
        a[(a == -999999) | ~np.isfinite(a)] = np.nan
        return np.sign(a) * np.log1p(np.abs(a))

    def transform(self, values):
        a = self.raw(values)
        missing = ~np.isfinite(a)
        a = (np.where(missing, self.mean, a) - self.mean) / self.std
        return np.concatenate([a, missing.astype('float32')], axis=-1).astype('float32')


class CategoryEncoder:
    def fit(self, frame):
        self.columns = list(frame)
        self.maps = {}
        for c in self.columns:
            counts = category_values(frame[c]).dropna().value_counts()
            vocabulary = sorted(counts.iloc[:126].index.tolist())
            self.maps[c] = {str(v): i + 2 for i, v in enumerate(vocabulary)}
        self.sizes = [len(self.maps[c]) + 2 for c in self.columns]
        return self

    def transform(self, frame):
        out = []
        for c in self.columns:
            s = category_values(frame[c])
            a = s.map(self.maps[c]).fillna(1).to_numpy(dtype='int64')
            a[s.isna().to_numpy()] = 0
            out.append(a)
        return np.column_stack(out) if out else np.empty((len(frame), 0), dtype='int64')


class QueryEncoder:
    def fit(self, x):
        self.columns = list(x)
        self.numeric = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        self.categories = [c for c in x if c not in self.numeric]
        self.numbers = NumericEncoder().fit(x[self.numeric].to_numpy('float32'))
        self.cats = CategoryEncoder().fit(x[self.categories])
        return self

    def transform(self, x):
        if list(x) != self.columns:
            raise ValueError('Query schema changed')
        return self.numbers.transform(x[self.numeric].to_numpy('float32')), self.cats.transform(x[self.categories])


class EventStore:
    def __init__(self, events, queries, neighbors):
        self.events, self.queries, self.neighbors = events, queries, neighbors
        self.numeric = events[cache.NUMS].to_numpy('float32')
        self.times = events.time_ns.to_numpy('int64')
        self.movement = events.movement_ns.to_numpy('int64')
        self.phase = events.phase.eq('DEP').to_numpy()
        self.query_times = queries.time_ns.to_numpy('int64')
        self.equality_events, self.equality_queries = {}, {}
        for c in ('runway', 'stand', 'operator'):
            # Codes are used only for equality, never embeddings or frequency-based selection.
            codes, vocabulary = pd.factorize(category_values(events[c]), sort=False)
            self.equality_events[c] = codes.astype('int32')
            self.equality_queries[c] = pd.Index(vocabulary).get_indexer(category_values(queries[c])).astype('int32')

    def training_events(self, rows):
        idx = np.asarray(self.neighbors[rows])
        return np.unique(idx[idx >= 0])

    def fit_encoder(self, rows):
        selected = self.training_events(rows)
        if not len(selected):
            raise ValueError('No context events in fit references')
        numeric = NumericEncoder().fit(self.numeric[selected])
        cats = CategoryEncoder().fit(self.events.iloc[selected][cache.CATS])
        return {'numeric': numeric, 'cats': cats, 'fit_event_count': len(selected)}

    def encoded(self, encoder):
        return encoder['numeric'].transform(self.numeric), encoder['cats'].transform(self.events[cache.CATS])

    def batch(self, rows, encoded):
        idx = np.asarray(self.neighbors[rows], dtype='int64')
        mask = idx >= 0
        safe = np.maximum(idx, 0)
        event_num, event_cat = encoded
        age = (self.query_times[rows, None] - self.times[safe]) / 1e9
        gap = np.zeros_like(age)
        gap[:, 1:] = (self.times[safe[:, 1:]] - self.times[safe[:, :-1]]) / 1e9
        landing_age = (self.query_times[rows, None] - self.movement[safe]) / 1e9
        landing_age[self.phase[safe]] = np.nan
        dynamic = np.stack([age, gap, landing_age], axis=-1).astype('float32')
        missing = ~np.isfinite(dynamic)
        dynamic = np.concatenate([np.where(missing, 0., dynamic / 3600.), missing.astype('float32')], axis=-1)
        numeric = np.concatenate([event_num[safe], dynamic], axis=-1)
        equality = np.stack([(self.equality_events[c][safe] == self.equality_queries[c][rows, None])
                             & (self.equality_events[c][safe] >= 0) & (self.equality_queries[c][rows, None] >= 0)
                             for c in ('runway', 'stand', 'operator')], axis=-1).astype('int64')
        categories = np.concatenate([event_cat[safe], equality], axis=-1)
        numeric[~mask], categories[~mask] = 0., 0
        support = np.column_stack([mask.sum(axis=1)/32.,
            (mask & self.phase[safe]).sum(axis=1)/16.,
            (mask & ~self.phase[safe]).sum(axis=1)/16.]).astype('float32')
        return numeric, categories, mask, support


class Network(nn.Module):
    def __init__(self, query_numeric, query_sizes, event_sizes):
        super().__init__()
        self.query_embedding = nn.ModuleList([nn.Embedding(n, 4) for n in query_sizes])
        self.event_embedding = nn.ModuleList([nn.Embedding(n, 4) for n in event_sizes])
        self.query = nn.Sequential(nn.Linear(query_numeric + 4*len(query_sizes) + 3, 64), nn.ReLU())
        self.gru = nn.GRU(22 + 4*len(event_sizes), 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, qn, qc, en, ec, mask, support, contextual):
        qcat = [layer(qc[:, j]) for j, layer in enumerate(self.query_embedding)]
        query = self.query(torch.cat([qn, *qcat, support], dim=1))
        context = torch.zeros_like(query)
        if contextual:
            ecat = [layer(ec[:, :, j]) for j, layer in enumerate(self.event_embedding)]
            token = torch.cat([en, *ecat], dim=-1)
            lengths = mask.sum(dim=1)
            packed = nn.utils.rnn.pack_padded_sequence(token, lengths.clamp(min=1).cpu(),
                                                       batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            context = hidden[0] * (lengths > 0)[:, None]
        return self.head(torch.cat([query, context], dim=1)).squeeze(1)


def tensors(q, event, device):
    qn, qc = q
    en, ec, mask, support = event
    return [torch.as_tensor(a, device=device) for a in (qn, qc, en, ec, mask, support)]


def predict(model, x, rows, store, *, device='cpu', batch_size=1024, event_arrays=None):
    network = Network(**model['architecture']).to(device)
    network.load_state_dict(model['state'])
    network.eval()
    encoded = event_arrays if event_arrays is not None else store.encoded(model['event_encoder'])
    predictions = np.empty(len(rows), dtype='float64')
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chosen = rows[start:start+batch_size]
            q = model['query_encoder'].transform(x.iloc[chosen])
            event = store.batch(chosen, encoded)
            value = network(*tensors(q, event, device), contextual=model['contextual'])
            predictions[start:start+len(chosen)] = value.cpu().numpy() * model['target_scale'] + model['target_mean']
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite residual prediction')
    return predictions


def fit(x, target, rows, store, *, tune_rows=None, steps=None, contextual=True,
        seed=20260916, threads=2, device='cpu', epochs=12, batch_size=256, patience=3):
    if (steps is None) == (tune_rows is None):
        raise ValueError('Require full tune rows for fit or explicit steps for fresh refit')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    cache.memory_guard()
    start = time.monotonic()
    qencoder = QueryEncoder().fit(x.iloc[rows])
    eencoder = store.fit_encoder(rows)
    encoded = store.encoded(eencoder)
    qfit = qencoder.transform(x.iloc[rows])
    target = np.asarray(target, dtype='float64')
    if not np.isfinite(target[rows]).all() or (tune_rows is not None and not np.isfinite(target[tune_rows]).all()):
        raise ValueError('Nonfinite training/tune labels')
    mean = float(np.mean(target[rows]))
    scale = max(float(np.std(target[rows])), 1.)
    architecture = {'query_numeric': qfit[0].shape[1], 'query_sizes': qencoder.cats.sizes,
                    'event_sizes': eencoder['cats'].sizes + [2, 2, 2]}
    network = Network(**architecture).to(device)
    parameters = sum(p.numel() for p in network.parameters())
    if parameters >= 1000000:
        raise ValueError('Prototype exceeded1M parameters')
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=.01)
    model = {'architecture': architecture, 'query_encoder': qencoder, 'event_encoder': eencoder,
             'contextual': contextual, 'target_mean': mean, 'target_scale': scale}
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    rng = np.random.default_rng(seed)
    max_epochs = int(steps) if steps is not None else epochs
    for epoch in range(1, max_epochs + 1):
        cache.memory_guard()
        network.train()
        order = rng.permutation(len(rows))
        epoch_start = time.monotonic()
        train_sum = 0.
        for start_pos in range(0, len(rows), batch_size):
            local = order[start_pos:start_pos+batch_size]
            chosen = rows[local]
            q = (qfit[0][local], qfit[1][local])
            event = store.batch(chosen, encoded)
            y = torch.as_tensor((target[chosen]-mean)/scale, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = network(*tensors(q, event, device), contextual=contextual)
            loss = torch.mean((prediction-y)**2)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite raw squared loss')
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(local)
        model['state'] = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        item = {'epoch': epoch, 'training_raw_mse': train_sum / len(rows) * scale**2,
                'train_seconds': time.monotonic()-epoch_start}
        if tune_rows is not None:
            eval_start = time.monotonic()
            values = predict(model, x, tune_rows, store, device=device, event_arrays=encoded)
            mse = float(np.mean((values-target[tune_rows])**2))
            item.update(tune_raw_mse=mse, tune_rows=len(tune_rows), tune_seconds=time.monotonic()-eval_start)
            if mse < best_loss:
                best_loss, best_epoch, stale = mse, epoch, 0
                best_state = copy.deepcopy(model['state'])
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(item)
        print('EPOCH', 'context' if contextual else 'static', item, flush=True)
        if tune_rows is not None and stale >= patience:
            break
    if tune_rows is not None:
        model['state'] = best_state
    evidence = {'steps': best_epoch, 'epochs_run': len(history), 'history': history,
        'fit_rows': len(rows), 'tune_rows': 0 if tune_rows is None else len(tune_rows),
        'fit_context_events': eencoder['fit_event_count'], 'parameters': parameters,
        'runtime_sec': time.monotonic()-start, 'contextual': contextual, 'device': device,
        'target': 'rawY-minusP; affine mean/std fromfit only; no clipping',
        'category_policy': 'max126 observed categories ranked byfit-reference frequency only; missing0unknown1',
        'token_numeric': '8stored signedlog normalized using unique fit-referenceevents;3dynamic age/gap/landingage scaled3600;11missingbits',
        'query_numeric': 'signedlog thenfitmean/std plusmissingbits', 'torch': torch.__version__}
    return model, evidence
