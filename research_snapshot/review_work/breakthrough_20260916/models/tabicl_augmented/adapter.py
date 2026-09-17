"""Offline TabICL with chronology-safe training context and historical features."""
import torch
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "review_work/breakthrough_20260916/missing"))
import tabicl_gpu as frozen
import run_missing_models as missing

TIME_COLUMN = "_forest_movement_time"


def unpack(x):
    times = pd.Series(pd.to_datetime(x[TIME_COLUMN], utc=True).to_numpy(), index=x.index)
    return x.drop(columns=TIME_COLUMN), times


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    raw, times = unpack(x)
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or len(y) != len(x):
        raise ValueError("Original labels must be finite and aligned")
    prior = missing.HistoricalTemplate().fit(raw, y, times)
    cross = missing.crossfit_templates(raw, y, times)
    features = pd.concat([raw, cross], axis=1)
    tune = None
    if tuning is not None:
        tx, ty = tuning
        traw, tt = unpack(tx)
        tune = (pd.concat([traw, prior.transform(traw, tt)], axis=1), ty)
    model, evidence = frozen.fit(features, y, tune, steps=steps, seed=seed, threads=threads)
    evidence.update(template_features="Earlier-month training crossfit; tune/score prior from permitted preceding context only",
                    input_features=list(raw), model_features=list(features), prior_rows=prior.history_n,
                    family_comparison="Same raw airport+IDcontext+template features as existing CatBoosthistoricaltemplate; different model and directtarget")
    return {"tabicl": model, "prior": prior, "columns": list(x), "feature_columns": list(features)}, evidence


def predict(model, x):
    if list(x) != model["columns"]:
        raise ValueError("Augmented TabICL input schema changed")
    raw, times = unpack(x)
    frame = pd.concat([raw, model["prior"].transform(raw, times)], axis=1)
    if list(frame) != model["feature_columns"]:
        raise ValueError("Augmented TabICL model features changed")
    return frozen.predict(model["tabicl"], frame)
