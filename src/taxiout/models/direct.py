import time

import numpy as np
import psutil
from catboost import CatBoostRegressor

from taxiout.artifacts import peak_rss


class ResourceLimit:
    def __init__(self, config):
        self.start = time.monotonic()
        self.config = config
        self.failure = None

    def after_iteration(self, info):
        if info.iteration % 50 == 0:
            print(f"  iteration {info.iteration}, elapsed {time.monotonic() - self.start:.1f}s", flush=True)
        if psutil.virtual_memory().available < self.config["memory_reserve_gb"] * 1024 ** 3:
            self.failure = "Training stopped to preserve 2 GB host memory reserve"
        if time.monotonic() - self.start > self.config["max_run_minutes"] * 60:
            self.failure = "Training exceeded configured time budget"
        return self.failure is None


def fit_model(x, y, config, iterations=None, tuning=None):
    if not np.isfinite(y).all():
        raise ValueError("Model fit labels must be finite")
    if psutil.virtual_memory().available < config["memory_reserve_gb"] * 1024 ** 3:
        raise MemoryError("Insufficient available RAM for configured reserve")
    params = dict(iterations=iterations or config["iterations"], depth=config["depth"], learning_rate=config["learning_rate"],
                  l2_leaf_reg=config["l2_leaf_reg"], loss_function=config["loss"], eval_metric="RMSE",
                  random_seed=config["seed"], thread_count=config["threads"], verbose=False, allow_writing_files=False)
    if "leaf_estimation_method" in config:
        params["leaf_estimation_method"] = config["leaf_estimation_method"]
    model = CatBoostRegressor(**params)
    callback = ResourceLimit(config)
    options = {"cat_features": list(x.select_dtypes("category").columns), "callbacks": [callback]}
    if tuning is not None:
        options.update(eval_set=tuning, early_stopping_rounds=config["early_stopping_rounds"], use_best_model=True)
    start = time.monotonic()
    model.fit(x, y, **options)
    if callback.failure:
        raise RuntimeError(callback.failure)
    elapsed = time.monotonic() - start
    evidence = {"rows": len(x), "features": len(x.columns), "iterations": model.tree_count_, "runtime_sec": elapsed,
                "peak_rss_bytes": peak_rss(), "available_ram_after": psutil.virtual_memory().available}
    print(f"Fit {len(x):,} rows, {model.tree_count_} trees in {elapsed:.1f}s", flush=True)
    return model, evidence
