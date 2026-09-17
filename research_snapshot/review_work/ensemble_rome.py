"""Equal-weight Rome-head ensemble with no score-fitted weights."""

import argparse
import numpy as np
import pandas as pd
from catboost import sum_models

from run_screening import OUT
from taxiout.artifacts import Run, read_json, sha256, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID
from taxiout.splits import make_fold


def execute(fold, ordinary):
    name = f"{ordinary}_plus_D3"
    first = read_json(OUT / ("results" if ordinary == "B" else "refinements") / f"{'B_trees' if ordinary == 'B' else 'F_stronger_residual'}_{fold}.json")
    heads = [read_json(OUT / "results" / f"D_rome_{fold}.json")]
    heads += [read_json(OUT / "refinements" / f"D_rome_seed_{fold}_{seed}.json") for seed in [20260911, 20260912]]
    if any(h["split"]["split_hash"] != first["split"]["split_hash"] for h in heads):
        raise ValueError("Component split mismatch")
    pipeline, models, config = load_bundle(OUT / "models" / first["run_id"])
    head_models = [load_bundle(OUT / "models" / h["run_id"])[1]["rome_schedule"] for h in heads]
    models["rome_schedule"] = sum_models(head_models, weights=[1 / 3] * 3)
    config = {**config, "candidate": name, "rome_schedule_residual": True}
    pipeline.config = config
    manifest = verified_manifest(config)
    run = Run(f"{name}-{fold}", config, manifest)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    output = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    base = pd.read_parquet(OUT / "models" / first["run_id"] / "score_predictions.parquet")
    component_predictions = [pd.read_parquet(OUT / "models" / h["run_id"] / "score_predictions.parquet") for h in heads]
    if any(not np.array_equal(output[ID], frame[ID]) for frame in [base, *component_predictions]):
        raise ValueError("Score ID mismatch")
    mask = output.route.eq("rome_schedule_residual")
    expected = np.where(mask, np.mean([frame.prediction_sec.to_numpy() for frame in component_predictions], axis=0), base.prediction_sec)
    if not np.allclose(output.prediction_sec, expected, rtol=0, atol=1e-8):
        raise ValueError("CatBoost summed model differs from the equal-weight prediction ensemble")
    output["candidate"], output["fold"] = name, fold
    metrics, errors = evaluate(output, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(output.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("Serialized ensemble mismatch")
    result = run.complete(fold=fold, split=split, training_rows=first["training_rows"],
                          iterations={**first["iterations"], "rome_schedule": models["rome_schedule"].tree_count_},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__),
                          components={"ordinary": first["run_id"], "rome_heads": [h["run_id"] for h in heads]},
                          weights=[1 / 3] * 3, selection="Equal-weight seed ensemble only on Rome route; other components fixed")
    write_json(OUT / "combinations" / f"{name}_{fold}.json", result)
    print(f"RESULT {name} {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["F1", "F3"], required=True)
    parser.add_argument("--ordinary", choices=["B", "F"], default="B")
    args = parser.parse_args()
    execute(args.fold, args.ordinary)
