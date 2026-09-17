"""Sequential global or airport-specific CatBoost residual model banks."""

import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def fit_bank(x, y, airports, folder, params, *, per_airport=False, tuning=None, selected_steps=None):
    y, airports = np.asarray(y, float), np.asarray(airports).astype(str)
    if len(x) != len(y) or len(x) != len(airports) or not np.isfinite(y).all():
        raise ValueError('Residual fit rows/targets/airports must be aligned and finite')
    if tuning is not None and selected_steps is not None:
        raise ValueError('Refit steps cannot use tuning data')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    keys = sorted(np.unique(airports)) if per_airport else ['global']
    bank = {'per_airport': per_airport, 'models': {}, 'columns': list(x)}
    evidence = {'groups': {}, 'selected_steps': {}}
    if tuning is not None:
        tx, ty, ta = tuning
        ty, ta = np.asarray(ty, float), np.asarray(ta).astype(str)
        if len(tx) != len(ty) or len(tx) != len(ta) or not np.isfinite(ty).all():
            raise ValueError('Residual tuning rows/targets/airports must align')
    for key in keys:
        choose = airports == key if per_airport else np.ones(len(x), bool)
        options = dict(params)
        if selected_steps is not None:
            if key not in selected_steps:
                raise ValueError('No tune-selected count for an airport in refit')
            options['iterations'] = selected_steps[key]
        model = CatBoostRegressor(**options)
        fit_x = x.loc[choose]
        kwargs = {'cat_features': list(fit_x.select_dtypes(['category', 'object', 'string']).columns)}
        tune_n = 0
        if tuning is not None:
            tune_choose = ta == key if per_airport else np.ones(len(tx), bool)
            tune_n = int(tune_choose.sum())
            if not tune_n:
                raise ValueError(f'No original tune rows for airport {key}')
            kwargs.update(eval_set=(tx.loc[tune_choose], ty[tune_choose]), early_stopping_rounds=100, use_best_model=True)
        started = time.monotonic()
        print('BANK_FIT', key, 'rows', len(fit_x), 'tune', tune_n, 'max_trees', options['iterations'], 'device', options['task_type'], flush=True)
        model.fit(fit_x, y[choose], **kwargs)
        path = folder / f'{key}.cbm'
        model.save_model(str(path))
        steps = int(model.tree_count_)
        bank['models'][key] = str(path.resolve())
        evidence['selected_steps'][key] = steps
        evidence['groups'][key] = {'fit_rows': len(fit_x), 'tune_rows': tune_n, 'steps': steps,
            'fit_seconds': time.monotonic() - started, 'best_iteration': model.get_best_iteration()}
        print('BANK_DONE', key, evidence['groups'][key], flush=True)
        del model, fit_x
        gc.collect()
    return bank, evidence


def predict_bank(bank, x, airports, threads=2):
    if 'columns' in bank and list(x) != bank['columns']:
        raise ValueError('Residual bank prediction schema changed')
    airports = np.asarray(airports).astype(str)
    if len(airports) != len(x):
        raise ValueError('Prediction airport rows differ')
    output = np.full(len(x), np.nan)
    keys = sorted(np.unique(airports)) if bank['per_airport'] else ['global']
    for key in keys:
        if key not in bank['models']:
            raise ValueError(f'Unseen airport has no frozen residual bank: {key}')
        choose = airports == key if bank['per_airport'] else np.ones(len(x), bool)
        model = CatBoostRegressor().load_model(bank['models'][key])
        output[choose] = model.predict(x.loc[choose], thread_count=threads)
        del model
    if not np.isfinite(output).all():
        raise ValueError('Residual bank did not produce finite predictions for every row')
    return output
