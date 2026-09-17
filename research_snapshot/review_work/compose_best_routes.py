"""Upgrade B+D's residual component while preserving its other qualified routes."""

import argparse
import numpy as np
import pandas as pd

from run_screening import OUT
from taxiout.artifacts import Run, read_json, sha256, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID
from taxiout.splits import make_fold


def execute(fold):
    name = "FB_plus_D"
    base = read_json(OUT / "combinations" / f"B_plus_D_{fold}.json")
    residual = read_json(OUT / "refinements" / f"F_stronger_residual_{fold}.json")
    if base["split"]["split_hash"] != residual["split"]["split_hash"] or base["input_manifest_hash"] != residual["input_manifest_hash"]:
        raise ValueError("Component input/split mismatch")
    pipeline, models, config = load_bundle(OUT / "models" / base["run_id"])
    _, residual_models, _ = load_bundle(OUT / "models" / residual["run_id"])
    models["residual"] = residual_models["residual"]
    config = {**config, "candidate": name}
    pipeline.config = config
    manifest = verified_manifest(config)
    run = Run(f"{name}-{fold}", config, manifest)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    output = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    a = pd.read_parquet(OUT / "models" / base["run_id"] / "score_predictions.parquet")
    b = pd.read_parquet(OUT / "models" / residual["run_id"] / "score_predictions.parquet")
    if not np.array_equal(output[ID], a[ID]) or not np.array_equal(output[ID], b[ID]):
        raise ValueError("Score cohort mismatch")
    mask = output.route.isin(["residual", "residual_long_proxy"])
    expected = np.where(mask, b.prediction_sec, a.prediction_sec)
    if not np.allclose(expected, output.prediction_sec, rtol=0, atol=1e-9):
        raise ValueError("Route composition parity failed")
    output["candidate"], output["fold"] = name, fold
    metrics, errors = evaluate(output, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(output.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("Serialization parity failed")
    result = run.complete(fold=fold, split=split, training_rows=residual["training_rows"],
                          iterations={**base["iterations"], "residual": residual["iterations"]["residual"]},
                          components={"other_routes": base["run_id"], "residual": residual["run_id"]},
                          component_configs={"other_routes": base["config"], "residual": residual["config"]},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__),
                          selection="Adaptive disjoint-route composition; no learned score-set weights or thresholds")
    write_json(OUT / "combinations" / f"{name}_{fold}.json", result)
    print(f"RESULT {name} {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["F1", "F2", "F3", "G1"], required=True)
    execute(p.parse_args().fold)
