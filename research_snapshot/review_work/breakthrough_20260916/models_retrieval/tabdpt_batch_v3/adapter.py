"""V3 changes query batch size only; v2 checkpoint/reference/attention frozen."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location('tabdpt_math_v2_frozen', HERE.parent / 'tabdpt_v2/adapter.py')
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)
BATCH = 16
REFERENCE_LIMIT = v2.REFERENCE_LIMIT
CONTEXT = v2.CONTEXT
ENSEMBLES = v2.ENSEMBLES


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2, device='cuda', max_epochs=None):
    model, evidence = v2.fit(x, y, tuning, steps=steps, seed=seed, threads=threads, device=device, max_epochs=max_epochs)
    model['query_batch_size'] = BATCH
    evidence['family'] = 'TabDPT1.3_reference32K_mathSDPA_batch_v3'
    evidence['prediction_params']['batch_size'] = BATCH
    evidence['batch_only_change'] = 'Same frozenv2 fit/preprocessor/checkpoint/context/ensemble/seed/rawlabels; querybatch selected by timing+equivalence canary'
    return model, evidence


def predict_batch(model, x, batch_size):
    if batch_size not in (16, 64, 128):
        raise ValueError('Undeclared query batch')
    if model['estimator'].use_flash or model['estimator'].model.use_flash:
        raise AssertionError('V3 retains v2 official nonflash path')
    torch.set_num_threads(model['threads'])
    result = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(x), 2048):
            encoded = model['processor'].transform(x.iloc[start:start + 2048])
            result.append(np.asarray(model['estimator'].predict(encoded, context_size=model['context_size'],
                batch_size=batch_size, n_ensembles=ENSEMBLES, seed=model['seed']), dtype=float).reshape(-1))
    prediction = np.concatenate(result) if result else np.empty(0, dtype=float)
    if len(prediction) != len(x) or not np.isfinite(prediction).all():
        raise ValueError('Invalid V3 predictions')
    return prediction


def predict(model, x):
    return predict_batch(model, x, model['query_batch_size'])
