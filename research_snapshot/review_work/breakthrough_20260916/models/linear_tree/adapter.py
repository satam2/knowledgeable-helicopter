"""Matched constant/linear LightGBM leaves with fit-only native-safe scaling."""
import lightgbm as lgb
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
from lgbm_adapter import NativeFrameEncoder

LINEAR_TREE = True
MAX_TREES = 200
MAX_MEMORY_GIB = 16.


class ScaledNativeEncoder:
    def fit(self, x):
        self.native = NativeFrameEncoder().fit(x)
        self.numeric = [name for name in x if name not in self.native.maps]
        self.means, self.scales = {}, {}
        for name in self.numeric:
            values = pd.to_numeric(x[name], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
            valid = np.isfinite(values) & (values != -999999)
            self.means[name] = float(values[valid].mean()) if valid.any() else 0.
            scale = float(values[valid].std()) if valid.any() else 1.
            self.scales[name] = scale if scale > 1e-12 else 1.
        return self

    def transform(self, x):
        frame = self.native.transform(x)
        for name in self.numeric:
            values = frame[name].to_numpy(dtype=np.float64)
            values[(values == -999999) | ~np.isfinite(values)] = np.nan
            frame[name] = ((values - self.means[name]) / self.scales[name]).astype("float32")
        return frame


def parameters(seed, threads, steps):
    return {"n_estimators": steps, "learning_rate": .05, "num_leaves": 15, "max_depth": 6,
            "objective": "regression", "n_jobs": threads, "random_state": seed,
            "verbosity": -1, "deterministic": True, "force_col_wise": True,
            "reg_lambda": 5., "min_child_samples": 30, "subsample": 1., "colsample_bytree": 1.,
            "linear_tree": LINEAR_TREE, "linear_lambda": 0., "device_type": "cpu", "tree_learner": "serial"}


def structural_memory(rows, numeric, total, threads=2):
    matrix_features = min(15, numeric)
    normal_equations = (threads + 1) * 15 * (((matrix_features + 1) * (matrix_features + 2) // 2 + 8) + matrix_features + 9) * 8
    raw_numeric = rows * numeric * 4
    leaf_map = rows * 4
    vectors_and_bins = rows * (total * 2 + 8 * 4)
    dataframe_copies = rows * total * 8 * 3
    subtotal = raw_numeric + leaf_map + normal_equations + vectors_and_bins + dataframe_copies
    conservative = 2 * 1024 ** 3 + 4 * subtotal
    return {"rows": rows, "numeric_features": numeric, "total_features": total,
            "raw_numeric_bytes": raw_numeric, "linear_leaf_map_bytes": leaf_map,
            "normal_equations_bytes": normal_equations, "dataframe_copy_allowance_bytes": dataframe_copies,
            "vector_and_bin_allowance_bytes": vectors_and_bins,
            "conservative_process_bytes": conservative,
            "conservative_process_gib": conservative / 1024 ** 3,
            "method": "Official raw-float numeric storage+row leafmap+perleaf/thread matrices;3float64framecopies+vectors/bins;4xallocationmargin+2GiBprocess/datareserve",
            "limitation": "Conservative planning estimate, not allocator-guaranteed upper bound"}


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
    y = np.asarray(y, dtype=np.float64)
    if len(y) != len(x) or not len(y) or not np.isfinite(y).all():
        raise ValueError("All original raw residuals must be aligned, nonempty and finite")
    if steps is not None and (steps < 1 or steps > MAX_TREES or tuning is not None):
        raise ValueError("Refit requires original-tune selected tree count only")
    started = time.monotonic()
    encoder = ScaledNativeEncoder().fit(x)
    planned_rows = len(x) + (len(tuning[0]) if tuning is not None else 0)
    memory = structural_memory(planned_rows, len(encoder.numeric), len(x.columns), threads)
    if memory["conservative_process_gib"] > MAX_MEMORY_GIB:
        raise MemoryError("Linear-tree planning estimate exceeds16GiB; full pilot rejected")
    if psutil.virtual_memory().available < memory["conservative_process_bytes"] + 4 * 1024 ** 3:
        raise MemoryError("Insufficient current host reserve for declared linear-tree pilot")
    train = encoder.transform(x)
    model = lgb.LGBMRegressor(**parameters(seed, threads, MAX_TREES if steps is None else int(steps)))
    options = {}
    if tuning is not None:
        tx, ty = tuning
        ty = np.asarray(ty, dtype=np.float64)
        if len(tx) != len(ty) or not np.isfinite(ty).all():
            raise ValueError("Original tune labels must be aligned and finite")
        options = {"eval_X": encoder.transform(tx), "eval_y": ty, "eval_metric": "rmse",
                   "callbacks": [lgb.early_stopping(40, verbose=False)]}
    model.fit(train, y, categorical_feature=list(encoder.native.maps), **options)
    selected = int(model.best_iteration_ or model.n_estimators_)
    bundle = {"estimator": model, "encoder": encoder, "steps": selected, "threads": threads}
    return bundle, {"steps": selected, "rows": len(y), "features": len(x.columns), "runtime_sec": time.monotonic() - started,
                    "parameters": model.get_params(), "library_version": lgb.__version__, "memory_estimate": memory,
                    "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
                    "numeric_scaling": "Fit-only finite non-sentinel mean/populationstd; nativeNaN preserved, no imputation or clipping",
                    "category_contract": "Native categorical split features, fit-only identities; excluded from linear leaf regressions",
                    "target": "Unmodified rawY minus everyfiniteNMproxy; squaredloss; no residual clipping",
                    "tuning": "Originaltune RMSE stopping40;200max;independentfullrefit"}


def predict(model, x):
    prediction = model["estimator"].predict(model["encoder"].transform(x), num_iteration=model["steps"], num_threads=model["threads"])
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Linear-tree predictions must remain finite without clipping")
    return prediction
