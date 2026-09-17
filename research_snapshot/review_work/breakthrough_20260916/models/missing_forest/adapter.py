"""Mean-regression forests with chronological template features and one-hot categories."""
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/breakthrough_20260916/missing"))
import run_missing_models as missing

FAMILY = "extratrees"
LEAVES = (1, 5, 20)
TIME_COLUMN = "_forest_movement_time"


def input_frame(x):
    x = x.drop(columns=TIME_COLUMN).copy()
    columns = list(x.select_dtypes("number"))
    x[columns] = x[columns].replace([-999999, np.inf, -np.inf], np.nan)
    return x


def encoder(x):
    categories = list(x.select_dtypes(["object", "string", "category"]))
    numeric = [column for column in x if column not in categories]
    category_pipeline = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))])
    return ColumnTransformer([("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
                               ("category", category_pipeline, categories)], sparse_threshold=1.)


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
    if steps is not None and (steps not in LEAVES or tuning is not None):
        raise ValueError("Refit requires the tune-selected minimum leaf size")
    if steps is None and tuning is None:
        raise ValueError("Original tune data required for leaf-size selection")
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or len(y) != len(x):
        raise ValueError("Unmodified original labels must be finite and aligned")
    started = time.monotonic()
    times = pd.Series(pd.to_datetime(x[TIME_COLUMN], utc=True).to_numpy(), index=x.index)
    raw = input_frame(x)
    prior = missing.HistoricalTemplate().fit(raw, y, times)
    cross = missing.crossfit_templates(raw, y, times)
    frame = pd.concat([raw, cross], axis=1)
    transform = encoder(frame)
    values = transform.fit_transform(frame)
    tune_values = None
    if tuning is not None:
        tx, ty = tuning
        tt = pd.Series(pd.to_datetime(tx[TIME_COLUMN], utc=True).to_numpy(), index=tx.index)
        traw = input_frame(tx)
        tune_values = transform.transform(pd.concat([traw, prior.transform(traw, tt)], axis=1))
    estimator_class = ExtraTreesRegressor if FAMILY == "extratrees" else RandomForestRegressor
    models = []
    best = None
    best_mse = float("inf")
    selected = steps
    for leaf in LEAVES if steps is None else [steps]:
        model = estimator_class(n_estimators=300, min_samples_leaf=leaf, max_features=.7,
                                 criterion="squared_error", n_jobs=threads, random_state=seed,
                                 bootstrap=FAMILY == "randomforest")
        began = time.monotonic()
        model.fit(values, y)
        mse = float(np.mean(np.square(model.predict(tune_values) - np.asarray(tuning[1], float)))) if tuning is not None else None
        evidence = {"min_samples_leaf": leaf, "tune_mse_sec2": mse, "seconds": time.monotonic() - began}
        models.append(evidence)
        print("FOREST_TUNE", FAMILY, evidence, flush=True)
        if steps is not None or mse < best_mse:
            best, best_mse, selected = model, mse, leaf
    bundle = {"estimator": best, "encoder": transform, "prior": prior, "columns": list(x),
              "feature_columns": list(frame), "family": FAMILY, "steps": selected}
    return bundle, {"steps": selected, "steps_semantics": "Tune-selected minimum leaf size; every forest has300trees",
                    "rows": len(x), "features_before_onehot": len(frame.columns), "encoded_features": values.shape[1],
                    "runtime_sec": time.monotonic() - started, "tune_candidates": models,
                    "params": {"n_estimators": 300, "max_features": .7, "min_samples_leaf": selected,
                               "criterion": "squared_error", "n_jobs": threads, "bootstrap": FAMILY == "randomforest"},
                    "target": "Direct unmodified raw-second labels, no clipping or transform",
                    "templates": "Earlier-month crossfit training priors; tune/score priors fit strictly before query",
                    "encoding": "Fit-only sparse one-hot categories, numeric median imputation and missing indicators"}


def predict(model, x):
    if list(x) != model["columns"]:
        raise ValueError("Forest input schema differs")
    times = pd.Series(pd.to_datetime(x[TIME_COLUMN], utc=True).to_numpy(), index=x.index)
    raw = input_frame(x)
    frame = pd.concat([raw, model["prior"].transform(raw, times)], axis=1)
    if list(frame) != model["feature_columns"]:
        raise ValueError("Forest constructed feature schema differs")
    return np.asarray(model["estimator"].predict(model["encoder"].transform(frame)), float)
