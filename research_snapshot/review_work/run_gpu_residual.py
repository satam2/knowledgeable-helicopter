"""Full-data GPU residual experiment; preserve B+D outside residual routes."""

import argparse
import gc
import time

import numpy as np
from catboost import CatBoostRegressor

from run_screening import OUT, config_for
from taxiout.artifacts import Run, object_hash, read_json, sha256, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.residual import proxy_status, residual_target
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID, TARGET, align
from taxiout.splits import make_fold


def fit_gpu(x, y, config, trees=None, tuning=None):
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite training labels")
    model = CatBoostRegressor(iterations=trees or config["iterations"], depth=config["depth"],
                              learning_rate=config["learning_rate"], l2_leaf_reg=config["l2_leaf_reg"],
                              loss_function="RMSE", eval_metric="RMSE", random_seed=config["seed"],
                              thread_count=4, task_type="GPU", devices="0", gpu_ram_part=.25,
                              verbose=100, allow_writing_files=False)
    options = {"cat_features": list(x.select_dtypes("category").columns)}
    if tuning is not None:
        options.update(eval_set=tuning, early_stopping_rounds=100, use_best_model=True)
    start = time.monotonic()
    model.fit(x, y, **options)
    return model, {"rows": len(x), "trees": model.tree_count_, "runtime_sec": time.monotonic() - start,
                   "device": "RTX 5080 / CatBoost GPU", "gpu_ram_part": .25}


def execute(fold):
    name = "H_gpu_full"
    config = {**config_for("baseline"), "candidate": name, "iterations": 2500, "depth": 8,
              "learning_rate": .05, "train_sample": None, "early_stopping_rounds": 100,
              "task_type": "GPU", "gpu_ram_part": .25, "rome_schedule_residual": True}
    base = read_json(OUT / "combinations" / f"B_plus_D_{fold}.json")
    manifest = verified_manifest(config)
    run = Run(f"{name}-{fold}", config, manifest)
    write_json(run.path / "gpu_recipe.json", {"config": config, "base_run": base["run_id"],
               "runner_sha256": sha256(__file__), "note": "GPU and scale change together; adaptive candidate, not a one-factor screen."})
    print(f"RUN {run.id}", flush=True)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    if split["split_hash"] != base["split"]["split_hash"]:
        raise ValueError("Baseline split mismatch")
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(float)
    mask = proxy_status(meta.proxy_sec, config) == "present"
    sf, st, sr = [p[mask[p]] for p in [idx["fit"], idx["tune"], idx["refit"]]]
    target = residual_target(y, meta.proxy_sec)
    print(f"ALL ELIGIBLE RESIDUAL ROWS: fit={len(sf)} tune={len(st)} refit={len(sr)}", flush=True)
    tuned, tuning = fit_gpu(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
    trees = tuned.tree_count_
    del tuned
    gc.collect()
    model, refit = fit_gpu(x.iloc[sr], target[sr], config, trees=trees)
    pipeline, models, _ = load_bundle(OUT / "models" / base["run_id"])
    pipeline.config = config
    models["residual"] = model
    output = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    output["candidate"], output["fold"] = name, fold
    metrics, errors = evaluate(output, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(output.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("GPU model serialization parity failure")
    result = run.complete(fold=fold, split=split, training_rows=len(sr),
                          training_id_hash=object_hash(x.index[sr].tolist()),
                          training={"tuning": tuning, "refit": refit},
                          iterations={**base["iterations"], "residual": trees},
                          components={"other_routes": base["run_id"], "residual": "full-data GPU"},
                          component_configs={"other_routes": base["config"], "residual": config},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__),
                          selection="Adaptive full-data GPU residual; nondeterministic GPU training, original score cohorts")
    write_json(OUT / "refinements" / f"{name}_{fold}.json", result)
    print(f"RESULT {name} {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["F1", "F2", "F3", "G1"], required=True)
    execute(p.parse_args().fold)
