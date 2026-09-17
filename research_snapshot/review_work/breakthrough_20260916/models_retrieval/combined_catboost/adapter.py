"""Matched combined-information CatBoost with fit-only category identities."""
import time
from pathlib import Path
import sys
import catboost
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'models'))
from encoders import FrameEncoder, json_safe


def parameters(seed=20260916, threads=2, device='GPU', iterations=5000):
    options = dict(iterations=iterations, depth=8, learning_rate=.05, l2_leaf_reg=8,
        loss_function='RMSE', eval_metric='RMSE', random_seed=seed,
        task_type=device, thread_count=threads, allow_writing_files=False,
        verbose=500 if device == 'GPU' else False)
    if device == 'GPU':
        options.update(devices='0', gpu_ram_part=.65)
    return options


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2, device='GPU', max_iterations=5000):
    y = np.asarray(y, float)
    if not len(x) or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError('Training requires nonempty aligned finite targets')
    if steps is not None and (steps < 1 or tuning is not None):
        raise ValueError('Refit requires positive steps without tuning')
    if steps is None and tuning is None:
        raise ValueError('Initial fit requires original tune data')
    if device not in ('CPU', 'GPU') or not 1 <= threads <= 2:
        raise ValueError('Invalid device or thread budget')
    if device == 'GPU' and max_iterations != 5000:
        raise ValueError('GPU experiment freezes maximum5000 iterations')
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('At least8GiB host reserve required before fit')
    started = time.monotonic()
    encoder = FrameEncoder().fit(x)
    train = encoder.transform(x)
    kwargs = {'cat_features': list(encoder.categories)}
    if tuning is not None:
        tx, ty = tuning
        ty = np.asarray(ty, float)
        if not len(tx) or len(tx) != len(ty) or not np.isfinite(ty).all():
            raise ValueError('Tuning requires nonempty aligned finite targets')
        kwargs.update(eval_set=(encoder.transform(tx), ty), early_stopping_rounds=100, use_best_model=True)
    options = parameters(seed, threads, device, max_iterations if steps is None else int(steps))
    model = catboost.CatBoostRegressor(**options)
    model.fit(train, y, **kwargs)
    selected = int(model.tree_count_)
    if steps is not None and selected != int(steps):
        raise AssertionError('Refit did not retain selected tree count')
    evidence = {'steps': selected, 'rows': len(x), 'tune_rows': len(tuning[0]) if tuning is not None else 0,
        'features': len(x.columns), 'version': catboost.__version__, 'device': device,
        'fit_runtime_sec': time.monotonic() - started, 'params': json_safe(model.get_params()),
        'best_iteration': int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
        'encoding': 'Frozen FrameEncoder fitted on fit/refit rows only; categorical identities missing0 unknown1; native CatBoost categories; numeric nonfinite/-999999 become NaN.',
        'objective': 'Raw unmodified source residual with squared loss; no target clipping, sampling or transforms',
        'selection': 'Original full tune RMSE with100-round patience; fresh full eligible refit atselected tree count'}
    return {'estimator': model, 'encoder': encoder, 'steps': selected, 'threads': threads}, evidence


def predict(model, x):
    values = model['estimator'].predict(model['encoder'].transform(x),
        ntree_end=model['steps'], thread_count=model['threads'])
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite CatBoost predictions')
    return values
