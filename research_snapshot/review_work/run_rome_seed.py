"""Repeat the rare Rome route at fixed seeds; all other routes stay fixed."""

import argparse
import gc

import numpy as np

from run_screening import OUT, config_for
from taxiout.artifacts import Run, object_hash, read_json, sha256, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.direct import fit_model
from taxiout.models.residual import residual_target
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID, TARGET, align
from taxiout.splits import make_fold


def execute(fold, seed):
    config = {**config_for("D_rome"), "candidate": "D_rome_seed", "seed": seed}
    base = read_json(OUT / "results" / f"baseline_{fold}.json")
    manifest = verified_manifest(config)
    run = Run(f"D_rome_seed-{fold}-{seed}", config, manifest)
    print(f"RUN {run.id}", flush=True)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    if split["split_hash"] != base["split"]["split_hash"]:
        raise ValueError("Baseline split mismatch")
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(float)
    mask = ~np.isfinite(meta.proxy_sec.to_numpy()) & np.isfinite(meta.schedule_sec.to_numpy())
    sf, st, sr = [p[mask[p]] for p in [idx["fit"], idx["tune"], idx["refit"]]]
    target = residual_target(y, meta.schedule_sec)
    tuned, tuning = fit_model(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
    trees = tuned.tree_count_
    del tuned
    gc.collect()
    model, refit = fit_model(x.iloc[sr], target[sr], config, iterations=trees)
    pipeline, models, _ = load_bundle(OUT / "models" / base["run_id"])
    pipeline.config = config
    models["rome_schedule"] = model
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
        raise ValueError("Serialization mismatch")
    result = run.complete(fold=fold, split=split, training_rows=len(sr),
                          training_id_hash=object_hash(x.index[sr].tolist()),
                          training={"tuning": tuning, "refit": refit},
                          iterations={**base["iterations"], "rome_schedule": trees},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__),
                          base_run=base["run_id"], selection="Fixed additional seeds 20260911 and 20260912; only Rome head varies")
    write_json(OUT / "refinements" / f"D_rome_seed_{fold}_{seed}.json", result)
    print(f"RESULT D_rome_seed {fold} {seed}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["F1", "F3"], required=True)
    args = p.parse_args()
    for seed in [20260911, 20260912]:
        execute(args.fold, seed)
