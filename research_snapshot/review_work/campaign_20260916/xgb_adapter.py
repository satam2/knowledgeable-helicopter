"""CPU histogram XGBoost screen on the same native categorical input contract."""

import time

import numpy as np
import xgboost as xgb

from lgbm_adapter import NativeFrameEncoder


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Fit requires aligned finite labels")
    if steps is not None and (steps < 1 or tuning is not None):
        raise ValueError("Refit steps must be positive and exclude tuning")
    start = time.monotonic()
    encoder = NativeFrameEncoder().fit(x)
    train = encoder.transform(x)
    estimator = xgb.XGBRegressor(
        n_estimators=600 if steps is None else int(steps), learning_rate=0.05,
        max_depth=6, objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", device="cpu", enable_categorical=True,
        max_cat_to_onehot=4, max_cat_threshold=64, n_jobs=threads,
        random_state=seed, reg_lambda=5.0, min_child_weight=30,
        subsample=1.0, colsample_bytree=1.0,
        early_stopping_rounds=60 if tuning is not None else None)
    options = {}
    if tuning is not None:
        tx, ty = tuning
        options["eval_set"] = [(encoder.transform(tx), np.asarray(ty, dtype=np.float64))]
    estimator.fit(train, y, verbose=False, **options)
    selected = int(estimator.best_iteration + 1) if tuning is not None else int(estimator.n_estimators)
    model = {"estimator": estimator, "encoder": encoder, "steps": selected}
    return model, {"steps": selected, "fit_seconds": time.monotonic() - start,
                   "rows": len(x), "features": len(x.columns), "seed": seed,
                   "threads": threads, "library": "xgboost", "version": xgb.__version__,
                   "categorical_encoding": "native; fit-only sorted strings; missing=0 unknown=1 known>=2",
                   "numeric_encoding": "float32; nonfinite becomes missing; no label clipping",
                   "params": {key: ("NaN (missing value)" if isinstance(value, float) and np.isnan(value) else value)
                              for key, value in estimator.get_params().items()}}


def predict(model, x):
    return np.asarray(model["estimator"].predict(model["encoder"].transform(x),
                      iteration_range=(0, model["steps"])), dtype=np.float64)
