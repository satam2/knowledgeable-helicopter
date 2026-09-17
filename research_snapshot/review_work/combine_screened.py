"""Compose independently trained routes; no weights are learned on scoring labels."""

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


def combine(name, fold):
    component = OUT / ("results" if name == "B_plus_D" else "refinements") / f"{'B_trees' if name == 'B_plus_D' else 'F_stronger_residual'}_{fold}.json"
    first = read_json(component)
    second = read_json(OUT / "results" / f"D_rome_{fold}.json")
    if first["split"]["split_hash"] != second["split"]["split_hash"] or first["input_manifest_hash"] != second["input_manifest_hash"]:
        raise ValueError("Components have different splits or raw inputs")
    pipeline, models, config = load_bundle(OUT / "models" / first["run_id"])
    _, other_models, _ = load_bundle(OUT / "models" / second["run_id"])
    models["rome_schedule"] = other_models["rome_schedule"]
    config = {**config, "candidate": name, "rome_schedule_residual": True}
    pipeline.config = config
    manifest = verified_manifest(config)
    run = Run(f"{name}-{fold}", config, manifest)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    output = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    expected = pd.read_parquet(OUT / "models" / first["run_id"] / "score_predictions.parquet")
    replacement = pd.read_parquet(OUT / "models" / second["run_id"] / "score_predictions.parquet")
    if not np.array_equal(output[ID], expected[ID]) or not np.array_equal(output[ID], replacement[ID]):
        raise ValueError("Combined score IDs differ")
    mask = output.route.eq("rome_schedule_residual")
    values = np.where(mask, replacement.prediction_sec, expected.prediction_sec)
    if not np.allclose(values, output.prediction_sec, rtol=0, atol=1e-9):
        raise ValueError("Combined model does not reproduce its declared route composition")
    output["candidate"], output["fold"] = name, fold
    metrics, errors = evaluate(output, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(output.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("Combined model serialization mismatch")
    result = run.complete(fold=fold, split=split, training_rows=first["training_rows"],
                          components={"ordinary_routes": first["run_id"], "rome_missing": second["run_id"]},
                          iterations={**first["iterations"], "rome_schedule": second["iterations"]["rome_schedule"]},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__),
                          selection="Development-only composition; no score-fitted blend weights")
    write_json(OUT / "combinations" / f"{name}_{fold}.json", result)
    print(f"RESULT {name} {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=["B_plus_D", "F_plus_D"], required=True)
    p.add_argument("--fold", choices=["F1", "F2", "F3", "G1"], required=True)
    args = p.parse_args()
    combine(args.candidate, args.fold)
