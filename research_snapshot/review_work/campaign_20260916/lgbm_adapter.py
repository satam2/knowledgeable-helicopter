"""CPU LightGBM screen with fit-only native categorical encoding."""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd


class NativeFrameEncoder:
    """Share explicit fit-only category identities across the two tree adapters."""

    def fit(self, x):
        self.columns = list(x.columns)
        self.maps = {}
        for col in self.columns:
            if isinstance(x[col].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(x[col]):
                values = x[col].astype("string")
                self.maps[col] = {value: index + 2 for index, value in enumerate(sorted(values.dropna().unique()))}
        return self

    def transform(self, x):
        if list(x.columns) != self.columns:
            raise ValueError("Feature schema/order differs from fitted encoder")
        out = {}
        for col in self.columns:
            if col in self.maps:
                values = x[col].astype("string")
                codes = values.map(self.maps[col]).fillna(1).to_numpy(dtype=np.int32)
                codes[values.isna().to_numpy()] = 0
                out[col] = pd.Categorical(codes, categories=range(len(self.maps[col]) + 2))
            else:
                values = pd.to_numeric(x[col], errors="raise").to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
                values[~np.isfinite(values)] = np.nan
                out[col] = values
        return pd.DataFrame(out, index=x.index)


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Fit requires aligned finite labels")
    if steps is not None and (steps < 1 or tuning is not None):
        raise ValueError("Refit steps must be positive and exclude tuning")
    start = time.monotonic()
    encoder = NativeFrameEncoder().fit(x)
    train = encoder.transform(x)
    estimator = lgb.LGBMRegressor(
        n_estimators=600 if steps is None else int(steps), learning_rate=0.05,
        num_leaves=31, max_depth=8, objective="regression", n_jobs=threads,
        random_state=seed, verbosity=-1, deterministic=True, force_col_wise=True,
        reg_lambda=5.0, min_child_samples=30, subsample=1.0, colsample_bytree=1.0)
    options = {}
    if tuning is not None:
        tx, ty = tuning
        options = {"eval_X": encoder.transform(tx), "eval_y": np.asarray(ty, dtype=np.float64),
                   "eval_metric": "rmse", "callbacks": [lgb.early_stopping(60, verbose=False)]}
    estimator.fit(train, y, categorical_feature=list(encoder.maps), **options)
    selected = int(estimator.best_iteration_ or estimator.n_estimators_)
    model = {"estimator": estimator, "encoder": encoder, "steps": selected}
    return model, {"steps": selected, "fit_seconds": time.monotonic() - start,
                   "rows": len(x), "features": len(x.columns), "seed": seed,
                   "threads": threads, "library": "lightgbm", "version": lgb.__version__,
                   "categorical_encoding": "native; fit-only sorted strings; missing=0 unknown=1 known>=2",
                   "numeric_encoding": "float32; nonfinite becomes missing; no label clipping",
                   "params": estimator.get_params()}


def predict(model, x):
    return np.asarray(model["estimator"].predict(model["encoder"].transform(x),
                      num_iteration=model["steps"]), dtype=np.float64)
