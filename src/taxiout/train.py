import gc
import time

import numpy as np
import pandas as pd
import psutil

from taxiout.artifacts import Run, object_hash, read_json, source_hashes, write_json
from taxiout.availability import POLICY_VERSION
from taxiout.cache import load_training
from taxiout.config import ROOT, load_config
from taxiout.paths import artifact_path
from taxiout.features.pipeline import FeaturePipeline
from taxiout.io import verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.baselines import HierarchicalMean
from taxiout.models.direct import fit_model
from taxiout.models.residual import proxy_status, residual_target
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import ID, TARGET, align
from taxiout.splits import make_fold


def sample_positions(x, positions, size, seed):
    if size is None or len(positions) <= size:
        return positions
    # Sorting IDs first makes membership independent of file/DataFrame ordering.
    sorted_positions = positions[np.argsort(x.index.to_numpy()[positions], kind="stable")]
    choice = np.random.default_rng(seed).choice(len(sorted_positions), size, replace=False)
    return np.sort(sorted_positions[choice])


def find_run(config, fold, input_manifest):
    config_hash = object_hash(config)
    current_sources = source_hashes()
    for path in sorted(artifact_path("models").glob("*/manifest.json"), reverse=True):
        record = read_json(path)
        if record.get("status") == "complete" and record.get("fold") == fold and record.get("config_hash") == config_hash and record.get("input_manifest_hash") == object_hash(input_manifest):
            if record.get("source_hashes") == current_sources:
                return path.parent, record
    return None


def append_experiment(manifest, metrics):
    record = {"run_id": manifest["run_id"], "candidate": manifest["config"]["candidate"], "fold": manifest.get("fold"),
                 "config_hash": manifest["config_hash"], "training_rows": manifest["training_rows"],
                 "rmse_sec": metrics["overall"]["rmse_sec"], "serialized_rmse_sec": metrics["serialized"]["rmse_sec"],
                 "missing_rmse_sec": metrics["slices"].get("proxy_status", {}).get("missing", {}).get("rmse_sec"),
                 "n": metrics["overall"]["n"], "sse": metrics["overall"]["sse"], "runtime_sec": manifest["runtime_sec"],
                 "peak_rss_bytes": manifest["peak_rss_bytes"]}
    write_json(artifact_path("reports/experiments", manifest["run_id"] + ".json"), record)


def run_baselines(x, meta, labels, idx, config, run):
    fit, score = idx["refit"], idx["score"]
    estimator = HierarchicalMean(config["hierarchy_alpha"]).fit(x.iloc[fit], labels.iloc[fit])
    airport = estimator.predict(x.iloc[score], depth=1)
    proxy = meta.iloc[score]["proxy_sec"].to_numpy()
    predictions = {"global_mean": np.full(len(score), estimator.mean), "airport_mean": airport,
                   "hierarchical_mean": estimator.predict(x.iloc[score]), "raw_proxy": np.where(np.isfinite(proxy), proxy, airport)}
    result = {}
    for name, values in predictions.items():
        frame = meta.iloc[score].copy()
        frame["prediction_sec"] = values
        frame["route"] = name
        frame["proxy_status"] = proxy_status(proxy, {"proxy_min": 0, "proxy_max": 7200})
        frame["candidate"], frame["fold"] = name, run.manifest.get("fold", "benchmark")
        result[name], errors = evaluate(frame, labels.iloc[score])
        errors.to_parquet(run.path / (name + "_scores.parquet"), index=False)
    write_json(run.path / "baseline_metrics.json", result)
    return result


def evaluate_candidate(config, fold, confirm=False):
    if fold == "C1" and not confirm:
        raise ValueError("C1 requires confirm and a frozen release config")
    manifest = verified_manifest(config)
    if confirm:
        freeze = read_json(artifact_path("reports/freeze.json"))
        if freeze["config_hash"] != object_hash(config):
            raise ValueError("Configuration differs from pre-confirmation freeze")
        if find_run(config, "C1", manifest):
            raise ValueError("C1 already scored for this frozen candidate; reuse existing evidence")
    run = Run(f"{config['candidate']}-{fold}", config, manifest)
    run.manifest["fold"] = fold
    write_json(run.path / "manifest.json", run.manifest)
    print(f"RUN {run.id}", flush=True)
    x, meta, labels = load_training(config, manifest)
    idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(dtype=float)
    fit = sample_positions(x, idx["fit"], config["train_sample"], config["seed"])
    refit = sample_positions(x, idx["refit"], config["train_sample"], config["seed"])
    tune, score = idx["tune"], idx["score"]
    baseline = run_baselines(x, meta, labels, idx, config, run)
    models, training = {}, {}
    is_residual_architecture = config["candidate"].startswith("residual")
    fallback_config = {**config, "candidate": "direct", "proxy_min": 0, "proxy_max": 7200}
    fallback_config.pop("missing_specialist", None)
    fallback_config.pop("long_proxy_correction", None)
    fallback_config.pop("rome_schedule_residual", None)
    base_config = dict(config)
    base_config.pop("rome_schedule_residual", None)
    base_run = find_run(base_config, fold, manifest) if config.get("rome_schedule_residual") else None
    if config.get("rome_schedule_residual") and base_run is None:
        raise ValueError("Rome screen requires a completed compatible baseline on this fold")
    existing = find_run(fallback_config, fold, manifest) if is_residual_architecture or config["candidate"] == "direct_specialist" else None
    existing = base_run or existing
    if existing:
        pipeline, models, _ = load_bundle(existing[0])
        training["direct_reused_from"] = existing[1]["run_id"]
        direct_iterations = existing[1]["iterations"]["direct"]
    else:
        tuning_model, tuning_evidence = fit_model(x.iloc[fit], y[fit], config, tuning=(x.iloc[tune], y[tune]))
        direct_iterations = tuning_model.tree_count_
        del tuning_model
        gc.collect()
        models["direct"], training["direct_refit"] = fit_model(x.iloc[refit], y[refit], config, iterations=direct_iterations)
        training["direct_tuning"] = tuning_evidence
        pipeline = FeaturePipeline(config).fit_features(x.iloc[refit])
    pipeline.config = config
    iterations = {"direct": direct_iterations}
    requested_components = ["residual"] if is_residual_architecture else []
    if config["candidate"] == "direct_specialist" or config.get("missing_specialist", False):
        requested_components.append("missing")
    if base_run:
        if base_run[1]["split"]["split_hash"] != split["split_hash"]:
            raise ValueError("Baseline reuse split mismatch")
        iterations = dict(base_run[1]["iterations"])
        training["base_reused_from"] = base_run[1]["run_id"]
        requested_components = []
    for route in requested_components:
        status = proxy_status(meta["proxy_sec"], config)
        is_residual = route == "residual"
        subset = status == ("present" if is_residual else "missing")
        component_periods = [fit, tune, refit] if is_residual else [idx["fit"], tune, idx["refit"]]
        sf, st, sr = [positions[subset[positions]] for positions in component_periods]
        if min(len(sf), len(st)) < 50:
            training[route + "_unavailable"] = "insufficient training/tuning support; shared direct fallback"
        else:
            component_config = dict(config)
            component_config.pop("missing_specialist", None)
            component_config.pop("long_proxy_correction", None)
            if is_residual:
                component_config["candidate"] = "residual" if config.get("long_proxy_correction", False) else config["candidate"].removesuffix("_specialist")
            else:
                component_config.update(candidate="direct_specialist", proxy_min=0, proxy_max=7200)
            component_run = find_run(component_config, fold, manifest) if config.get("missing_specialist", False) else None
            if component_run is None and config.get("missing_specialist", False):
                compatible_route = not is_residual or (config["proxy_min"] == 0 and config["proxy_max"] == 7200)
                if compatible_route:
                    reference_config = {**fallback_config, "candidate": "residual_specialist", "missing_specialist": True}
                    component_run = find_run(reference_config, fold, manifest)
            if component_run:
                _, component_models, _ = load_bundle(component_run[0])
                models[route] = component_models[route]
                iterations[route] = component_run[1]["iterations"][route]
                training[route + "_reused_from"] = component_run[1]["run_id"]
                training[route + "_refit"] = component_run[1]["training"][route + "_refit"]
                if not is_residual:
                    pipeline.fit_features(x.iloc[np.union1d(refit, sr)])
                continue
            target = residual_target(y, meta["proxy_sec"]) if is_residual else y
            tuning_model, training[route + "_tuning"] = fit_model(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
            iterations[route] = tuning_model.tree_count_
            del tuning_model
            gc.collect()
            models[route], training[route + "_refit"] = fit_model(x.iloc[sr], target[sr], config, iterations=iterations[route])
            training[route + "_refit"]["label_id_hash"] = object_hash(x.index[sr].tolist())
            if not is_residual:
                pipeline.fit_features(x.iloc[np.union1d(refit, sr)])
    if config.get("rome_schedule_residual"):
        # Learn on eligible missing-clock rows; route only by observed airport and clocks.
        subset = (proxy_status(meta["proxy_sec"], config) == "missing") & np.isfinite(meta["schedule_sec"].to_numpy())
        sf, st, sr = [positions[subset[positions]] for positions in [idx["fit"], tune, idx["refit"]]]
        if min(len(sf), len(st)) < 50:
            raise ValueError("Insufficient schedule-head fit/tune support")
        target = residual_target(y, meta["schedule_sec"])
        tuned, training["rome_schedule_tuning"] = fit_model(x.iloc[sf], target[sf], config, tuning=(x.iloc[st], target[st]))
        iterations["rome_schedule"] = tuned.tree_count_
        del tuned
        gc.collect()
        models["rome_schedule"], training["rome_schedule_refit"] = fit_model(x.iloc[sr], target[sr], config, iterations=iterations["rome_schedule"])
        training["rome_schedule_refit"]["label_id_hash"] = object_hash(x.index[sr].tolist())
        training["rome_schedule_refit"]["rome_rows"] = int(meta.iloc[sr]["ADEP_mvt"].eq("LIRF").sum())
    predictions = predict_features(pipeline, models, config, x.iloc[score], meta.iloc[score])
    predictions["candidate"], predictions["fold"] = config["candidate"], fold
    metrics, errors = evaluate(predictions, labels.iloc[score])
    errors.to_parquet(run.path / "score_predictions.parquet", index=False)
    errors.nlargest(100, "squared_error").to_parquet(run.path / "largest_errors.parquet", index=False)
    errors.groupby(["ADEP_mvt", "day"], observed=True).agg(n=(ID, "size"), sse=("squared_error", "sum")).sort_values("sse", ascending=False).to_csv(run.path / "error_days.csv")
    write_json(run.path / "metrics.json", metrics)
    save_bundle(run.path, pipeline, models, config)
    reloaded_pipeline, reloaded_models, reloaded_config = load_bundle(run.path, allow_incomplete=True)
    reloaded = predict_features(reloaded_pipeline, reloaded_models, reloaded_config, x.iloc[score], meta.iloc[score])
    delta = float(np.max(np.abs(reloaded.prediction_sec.to_numpy() - predictions.prediction_sec.to_numpy())))
    if delta > 1e-9:
        raise ValueError("Serialized model prediction parity failed")
    complete = run.complete(fold=fold, split=split, policy=POLICY_VERSION, training_rows=len(refit),
                            training_id_hash=object_hash(x.index[refit].tolist()), iterations=iterations,
                            training=training, metrics=metrics, baseline_metrics=baseline,
                            feature_names=pipeline.columns, feature_dtypes=pipeline.dtypes, reload_max_abs_delta=delta)
    append_experiment(complete, metrics)
    if confirm:
        acceptance = freeze["confirmation_acceptance"]
        baseline_limit = min(baseline[name]["overall"]["rmse_sec"] for name in acceptance["beat_best_baseline"])
        missing_limit = baseline["airport_mean"]["slices"]["proxy_status"]["missing"]["rmse_sec"] * acceptance["missing_proxy_rmse_max_relative_to_airport_mean"]
        accepted = metrics["overall"]["rmse_sec"] < baseline_limit and metrics["slices"]["proxy_status"]["missing"]["rmse_sec"] <= missing_limit
        write_json(artifact_path("reports/confirmation.json"), {"run_id": complete["run_id"], "config_hash": object_hash(config),
                   "input_manifest_hash": object_hash(manifest), "status": "complete", "metrics": metrics,
                   "accepted": accepted, "baseline_rmse_limit": baseline_limit, "missing_proxy_rmse_limit": missing_limit})
        if not accepted:
            raise RuntimeError("Frozen confirmation acceptance failed; investigate before final release")
    print(f"COMPLETE {run.id}: RMSE {metrics['overall']['rmse_sec']:.3f}s", flush=True)
    return complete


def train_final(config, output_name=None):
    if config.get("rome_schedule_residual"):
        raise ValueError("Rome schedule head is screening-only until F2/G1 and seed validation")
    freeze = read_json(artifact_path("reports/freeze.json"))
    if freeze["config_hash"] != object_hash(config):
        raise ValueError("Final config differs from freeze")
    from taxiout.artifacts import sha256
    for file, digest in freeze["source_hashes"].items():
        if file.startswith("src/") and sha256(ROOT / file) != digest:
            raise ValueError(f"Source changed after freeze: {file}")
    manifest = verified_manifest(config)
    confirmation = read_json(artifact_path("reports/confirmation.json"))
    if not confirmation.get("accepted") or confirmation["status"] != "complete" or confirmation["config_hash"] != object_hash(config) or confirmation["input_manifest_hash"] != object_hash(manifest):
        raise ValueError("Frozen candidate has no completed confirmation")
    run = Run(output_name or "release", config, manifest)
    x, meta, labels = load_training(config, manifest)
    positions = sample_positions(x, np.arange(len(x)), config["train_sample"], config["seed"])
    y = align(x.reset_index()[[ID]], labels, [TARGET])[TARGET].to_numpy(dtype=float)
    pipeline = FeaturePipeline(config).fit_features(x.iloc[positions])
    models, training = {}, {}
    iterations = freeze["final_iterations"]
    models["direct"], training["direct"] = fit_model(x.iloc[positions], y[positions], config, iterations=iterations["direct"])
    status = proxy_status(meta["proxy_sec"], config)
    for name in ["residual", "missing"]:
        if name in iterations:
            component_positions = positions if name == "residual" else np.arange(len(x))
            eligible = component_positions[status[component_positions] == ("present" if name == "residual" else "missing")]
            target = residual_target(y, meta["proxy_sec"]) if name == "residual" else y
            models[name], training[name] = fit_model(x.iloc[eligible], target[eligible], config, iterations=iterations[name])
            training[name]["label_id_hash"] = object_hash(x.index[eligible].tolist())
            if name == "missing":
                pipeline.fit_features(x.iloc[np.union1d(positions, eligible)])
    save_bundle(run.path, pipeline, models, config)
    complete = run.complete(fold="final", training_rows=len(positions), training_id_hash=object_hash(x.index[positions].tolist()),
                            iterations=iterations, training=training, policy=POLICY_VERSION,
                            feature_names=pipeline.columns, feature_dtypes=pipeline.dtypes, freeze=freeze)
    print(f"FINAL {run.id}", flush=True)
    return complete
