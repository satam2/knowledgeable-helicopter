"""Exact three-moment residual decomposition selected by reconstructed raw MSE."""
import lightgbm as lgb
import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
from lgbm_adapter import NativeFrameEncoder

Q_SECONDS = 300.
MAX_TREES = 600
HEADS = ("center", "upper", "lower")


def decompose(residual):
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1 or not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Decomposition requires aligned finite raw residuals")
    parts = {"center": np.clip(residual, -Q_SECONDS, Q_SECONDS),
             "upper": np.maximum(residual - Q_SECONDS, 0.),
             "lower": np.minimum(residual + Q_SECONDS, 0.)}
    reconstructed = sum(parts.values())
    if not np.allclose(reconstructed, residual, rtol=1e-14, atol=1e-9):
        raise AssertionError("Raw residual decomposition identity failed")
    return parts


def parameters(seed, threads):
    return {"objective": "regression", "metric": "None", "learning_rate": .05,
            "num_leaves": 31, "max_depth": 8, "lambda_l2": 5., "min_data_in_leaf": 30,
            "feature_fraction": 1., "bagging_fraction": 1., "num_threads": threads,
            "seed": seed, "deterministic": True, "force_col_wise": True, "verbosity": -1}


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("Training features and original residuals must align")
    if steps is not None and (steps < 1 or steps > MAX_TREES or tuning is not None):
        raise ValueError("Refit uses the tune-selected common tree prefix only")
    if steps is None and tuning is None:
        raise ValueError("Original tune rows are required to select common prefix")
    started = time.monotonic()
    parts = decompose(y)
    encoder = NativeFrameEncoder().fit(x)
    train = encoder.transform(x)
    rounds = MAX_TREES if steps is None else int(steps)
    tune = None
    summed_tune = None
    if tuning is not None:
        tx, ty = tuning
        ty = np.asarray(ty, dtype=np.float64)
        if len(tx) != len(ty):
            raise ValueError("Tune features and raw residuals must align")
        tune_parts = decompose(ty)
        tune = encoder.transform(tx)
        summed_tune = np.zeros((rounds, len(ty)), dtype=np.float64)
    boosters = {}
    diagnostics = {}
    for head in HEADS:
        dataset = lgb.Dataset(train, label=parts[head], categorical_feature=list(encoder.maps), free_raw_data=True)
        head_mse = []
        options = {}
        if tune is not None:
            validation = lgb.Dataset(tune, label=tune_parts[head], reference=dataset, free_raw_data=True)

            def evaluate_component(predicted, _dataset):
                iteration = len(head_mse)
                summed_tune[iteration] += np.asarray(predicted, dtype=np.float64)
                loss = float(np.square(predicted - tune_parts[head]).mean())
                head_mse.append(loss)
                return "component_mse", loss, False

            options = {"valid_sets": [validation], "valid_names": ["original_tune"], "feval": evaluate_component}
        booster = lgb.train(parameters(seed, threads), dataset, num_boost_round=rounds, **options)
        boosters[head] = booster
        if tuning is not None and len(head_mse) != rounds:
            raise AssertionError("Component trajectory has unexpected prefix count")
        diagnostics[head] = {"target_min": float(parts[head].min()), "target_max": float(parts[head].max()),
                             "nonzero_rows": int(np.count_nonzero(parts[head])), "mean": float(parts[head].mean()),
                             "trained_trees": booster.num_trees(),
                             "component_best_prefix": int(np.argmin(head_mse) + 1) if head_mse else None,
                             "component_best_mse": min(head_mse) if head_mse else None}
        print("RESIDUAL_MOMENT_HEAD", head, diagnostics[head], flush=True)
    if tuning is not None:
        raw_curve = np.mean(np.square(summed_tune - ty[None, :]), axis=1)
        selected = int(np.argmin(raw_curve) + 1)
        direct = sum(booster.predict(tune, num_iteration=selected, num_threads=threads) for booster in boosters.values())
        replay_delta = float(np.max(np.abs(direct - summed_tune[selected - 1])))
        if replay_delta > 1e-7:
            raise AssertionError("Reconstructed tune trajectory disagrees with selected model prefix")
        selection = {"raw_mse_by_common_prefix": raw_curve.tolist(), "selected_raw_mse": float(raw_curve[selected - 1]),
                     "tune_rows": len(ty), "trajectory_replay_max_abs_delta": replay_delta}
    else:
        selected = int(steps)
        selection = {"refit_common_prefix_from_original_tune": selected}
    model = {"boosters": boosters, "encoder": encoder, "steps": selected, "threads": threads, "q_seconds": Q_SECONDS}
    evidence = {"steps": selected, "steps_semantics": "Common prefix minimizing reconstructed raw residual tuneMSE",
                "rows": len(y), "features": len(x.columns), "runtime_sec": time.monotonic() - started,
                "parameters": parameters(seed, threads), "q_seconds": Q_SECONDS, "maximum_trees": MAX_TREES,
                "decomposition_max_abs_error": float(np.max(np.abs(sum(parts.values()) - y))),
                "component_diagnostics": diagnostics, "selection": selection,
                "target_contract": "center=clip(r,-300,300);upper=max(r-300,0);lower=min(r+300,0);sum exactly r. All raw labels retained.",
                "mean_contract": "Three squared-loss regression means summed; no prediction clipping, event routing, or probability calibration",
                "library": "lightgbm", "version": lgb.__version__}
    return model, evidence


def predict(model, x):
    values = model["encoder"].transform(x)
    return sum(np.asarray(booster.predict(values, num_iteration=model["steps"], num_threads=model["threads"]), dtype=np.float64)
               for booster in model["boosters"].values())
