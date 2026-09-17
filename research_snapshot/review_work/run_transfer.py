"""Movement-only pretraining and missing-clock fine-tuning, with original time splits."""

import argparse
import gc
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from run_screening import OUT, config_for
from taxiout.artifacts import Run, object_hash, read_json, sha256, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.direct import ResourceLimit, fit_model
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID, TARGET, align
from taxiout.splits import make_fold
from taxiout.train import sample_positions


def movement_only(frame):
    masked = frame.copy()
    for col in masked:
        if col.endswith("_flt"):
            if isinstance(masked[col].dtype, pd.CategoricalDtype):
                masked[col] = pd.Categorical(["m:"] * len(masked))
            else:
                masked[col] = np.float32(-999999)
        elif col in ["nm_actual_minus_estimated", "last_minus_initial"]:
            masked[col] = np.float32(-999999)
        elif col == "proxy_missing":
            masked[col] = np.float32(1)
    return masked


def adapt(base, x, y, config, iterations, tuning=None):
    params = base.get_params()
    params.update(iterations=iterations)
    model = CatBoostRegressor(**params)
    callback = ResourceLimit(config)
    options = {"cat_features": list(x.select_dtypes("category").columns),
               "init_model": base, "callbacks": [callback]}
    if tuning is not None:
        options.update(eval_set=tuning, early_stopping_rounds=config["early_stopping_rounds"], use_best_model=True)
    model.fit(x, y, **options)
    if callback.failure:
        raise RuntimeError(callback.failure)
    return model


def execute(fold):
    config = {**config_for("baseline"), "candidate": "G_shared_missing", "train_sample": 500000,
              "iterations": 400, "depth": 6, "learning_rate": .05}
    base_record = read_json(OUT / "results" / f"baseline_{fold}.json")
    manifest = verified_manifest(config)
    run = Run(f"G_shared_missing-{fold}", config, manifest)
    print(f"RUN {run.id}", flush=True)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    if split["split_hash"] != base_record["split"]["split_hash"]:
        raise ValueError("Baseline split mismatch")
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(float)
    masked = movement_only(x)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy())
    # The bundle can use its normal missing route because every masked input is already missing there.
    for col in x:
        if not np.array_equal(x.loc[missing, col].astype(str).to_numpy(), masked.loc[missing, col].astype(str).to_numpy()):
            raise ValueError(f"Inference masking would differ for missing rows: {col}")
    fit = sample_positions(x, idx["fit"], config["train_sample"], config["seed"])
    refit = sample_positions(x, idx["refit"], config["train_sample"], config["seed"])
    mf, mt, mr = [p[missing[p]] for p in [idx["fit"], idx["tune"], idx["refit"]]]
    pretrain, pre_evidence = fit_model(masked.iloc[fit], y[fit], config, tuning=(masked.iloc[idx["tune"]], y[idx["tune"]]))
    pre_trees = pretrain.tree_count_
    tuned = adapt(pretrain, masked.iloc[mf], y[mf], config, 400, (masked.iloc[mt], y[mt]))
    fine_trees = tuned.tree_count_ - pre_trees
    print(f"Selected pretrain={pre_trees}, additional missing trees={fine_trees}", flush=True)
    del pretrain, tuned
    gc.collect()
    shared, refit_evidence = fit_model(masked.iloc[refit], y[refit], config, iterations=pre_trees)
    missing_model = adapt(shared, masked.iloc[mr], y[mr], config, fine_trees) if fine_trees > 0 else shared
    pipeline, models, _ = load_bundle(OUT / "models" / base_record["run_id"])
    pipeline.config = config
    models["missing"] = missing_model
    output = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    output["candidate"], output["fold"] = config["candidate"], fold
    metrics, errors = evaluate(output, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(output.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("Serialization parity failure")
    result = run.complete(fold=fold, split=split, training_rows=len(refit),
                          iterations={**base_record["iterations"], "missing": missing_model.tree_count_},
                          training={"pretrain": pre_evidence, "refit": refit_evidence, "fine_trees": fine_trees,
                                    "fine_rows": len(mr), "training_id_hash": object_hash(x.index[refit].tolist())},
                          base_run=base_record["run_id"], metrics=metrics, reload_max_abs_delta=delta,
                          runner_sha256=sha256(__file__), selection="Adaptive missing-clock transfer screen")
    write_json(OUT / "refinements" / f"G_shared_missing_{fold}.json", result)
    print(f"RESULT G_shared_missing {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["F1", "F3", "F2", "G1"], required=True)
    execute(parser.parse_args().fold)
