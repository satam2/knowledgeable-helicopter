"""Follow-up component experiments; reuses the exact baseline outside changed routes."""

import argparse
import gc

import numpy as np

from run_screening import OUT, config_for
from taxiout.artifacts import Run, object_hash, read_json, sha256, source_hashes, write_json
from taxiout.cache import load_training
from taxiout.config import load_config
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.direct import fit_model
from taxiout.models.residual import proxy_status, residual_target
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID, TARGET, align
from taxiout.splits import make_fold
from taxiout.train import sample_positions

RECIPES = {
    "E_rome_local": {"iterations": 600, "depth": 4, "learning_rate": .05, "l2_leaf_reg": 30,
                     "rome_schedule_residual": True},
    "F_stronger_residual": {"iterations": 1200, "depth": 8, "learning_rate": .05,
                            "train_sample": 500000},
}


def execute(name, fold, seed=None):
    result_file = OUT / "refinements" / f"{name}_{fold}{'_' + str(seed) if seed is not None else ''}.json"
    if result_file.exists():
        old = read_json(result_file)
        if old["status"] == "complete" and old["runner_sha256"] == sha256(__file__):
            print(f"REUSED {name} {fold}: {old['metrics']['overall']['rmse_sec']:.6f}s", flush=True)
            return old
        raise ValueError("Existing refinement artifact is incompatible")
    base = read_json(OUT / "results" / f"baseline_{fold}.json")
    protocol = read_json(OUT / "protocol.json")
    if base["source_hashes"] != source_hashes() or protocol["source_hashes"] != source_hashes():
        raise ValueError("Baseline source is not current")
    config = {**config_for("baseline"), **RECIPES[name], "candidate": name}
    if seed is not None:
        config["seed"] = seed
    manifest = verified_manifest(config)
    run = Run(f"{name}-{fold}", config, manifest)
    run.manifest.update(fold=fold, runner_sha256=sha256(__file__), base_run=base["run_id"],
                        status="incomplete", selection="adaptive follow-up after initial screening results")
    write_json(run.path / "manifest.json", run.manifest)
    print(f"RUN {run.id}", flush=True)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    if split["split_hash"] != base["split"]["split_hash"]:
        raise ValueError("Baseline split mismatch")
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(float)
    pipeline, models, _ = load_bundle(OUT / "models" / base["run_id"])
    pipeline.config = config
    status = proxy_status(meta.proxy_sec, config)
    if name == "E_rome_local":
        mask = (status == "missing") & meta.ADEP_mvt.eq("LIRF").to_numpy() & np.isfinite(meta.schedule_sec)
        sf, st, sr = [p[mask[p]] for p in [idx["fit"], idx["tune"], idx["refit"]]]
        target = residual_target(y, meta.schedule_sec)
        component = "rome_schedule"
    else:
        fit = sample_positions(x, idx["fit"], config["train_sample"], config["seed"])
        refit = sample_positions(x, idx["refit"], config["train_sample"], config["seed"])
        mask = status == "present"
        sf, st, sr = [p[mask[p]] for p in [fit, idx["tune"], refit]]
        target = residual_target(y, meta.proxy_sec)
        component = "residual"
    if min(len(sf), len(st)) < 20:
        raise ValueError("Insufficient fit/tune support for refinement")
    print(f"COMPONENT {component}: fit={len(sf)} tune={len(st)} refit={len(sr)}", flush=True)
    tuning_model, tune_evidence = fit_model(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
    trees = tuning_model.tree_count_
    del tuning_model
    gc.collect()
    models[component], refit_evidence = fit_model(x.iloc[sr], target[sr], config, iterations=trees)
    predictions = predict_features(pipeline, models, config, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    predictions["candidate"], predictions["fold"] = name, fold
    metrics, errors = evaluate(predictions, labels.iloc[idx["score"]])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    rp, rm, rc = load_bundle(run.path, allow_incomplete=True)
    repeated = predict_features(rp, rm, rc, x.iloc[idx["score"]], meta.iloc[idx["score"]])
    delta = float(np.max(np.abs(predictions.prediction_sec - repeated.prediction_sec)))
    if delta > 1e-9:
        raise ValueError("Serialization parity failure")
    iterations = {**base["iterations"], component: trees}
    result = run.complete(fold=fold, split=split, training_rows=len(sr),
                          training_id_hash=object_hash(x.index[sr].tolist()), iterations=iterations,
                          training={"component": component, "tuning": tune_evidence, "refit": refit_evidence,
                                    "fit_id_hash": object_hash(x.index[sf].tolist()),
                                    "tune_id_hash": object_hash(x.index[st].tolist())},
                          metrics=metrics, reload_max_abs_delta=delta, runner_sha256=sha256(__file__))
    write_json(result_file, result)
    print(f"RESULT {name} {fold}: {metrics['overall']['rmse_sec']:.6f}s", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, choices=list(RECIPES))
    p.add_argument("--fold", required=True, choices=["F1", "F2", "F3", "G1"])
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    execute(args.candidate, args.fold, args.seed)
