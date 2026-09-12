import gc

import numpy as np
import pandas as pd
import psutil

from taxiout.artifacts import Run, object_hash, peak_rss, write_json
from taxiout.cache import load_training, ranking_features
from taxiout.config import ROOT
from taxiout.features.pipeline import FeaturePipeline
from taxiout.io import audited_departures, read_raw, verified_manifest
from taxiout.metrics import evaluate
from taxiout.models.baselines import HierarchicalMean
from taxiout.models.direct import fit_model
from taxiout.models.residual import proxy_status
from taxiout.schema import ID, MOVEMENT, PHASE, TARGET
from taxiout.submission import build_submission
from taxiout.train import run_baselines


def benchmark(config):
    if not config["features"]["legacy"] or config["coverage"] != "continuous-audit-only":
        raise ValueError("Diagnostic reproduction requires explicit legacy/continuous policy")
    manifest = verified_manifest(config)
    run = Run("benchmark", config, manifest)
    x, meta, labels = load_training(config, manifest)
    train = np.flatnonzero(meta[MOVEMENT].lt(pd.Timestamp("2025-11-01", tz="UTC")))
    score = np.flatnonzero(meta[MOVEMENT].ge(pd.Timestamp("2025-11-01", tz="UTC")))
    fit = np.sort(np.random.default_rng(config["seed"]).choice(train, min(100000, len(train)), replace=False))
    baseline = run_baselines(x, meta, labels, {"refit": train, "score": score}, config, run)
    clock_cols = [c for c in x if c.startswith("takeoff_minus_")] + ["nm_actual_minus_estimated", "last_minus_initial", "proxy_missing"]
    reports, training = {}, {}
    for name, columns in [("operational_context", [c for c in x if c not in clock_cols]), ("with_clocks", list(x))]:
        rss_before = psutil.Process().memory_info().rss
        model, training[name] = fit_model(x.iloc[fit][columns], labels.iloc[fit][TARGET].to_numpy(dtype=float), config)
        frame = meta.iloc[score].copy()
        frame["prediction_sec"] = model.predict(x.iloc[score][columns], thread_count=config["threads"])
        frame["route"] = name
        frame["proxy_status"] = proxy_status(frame["proxy_sec"], config)
        frame["candidate"], frame["fold"] = name, "legacy_Nov_Dec"
        reports[name], errors = evaluate(frame, labels.iloc[score])
        errors.to_parquet(run.path / (name + "_scores.parquet"), index=False)
        model.save_model(str(run.path / (name + ".cbm")))
        training[name]["rss_before_fit"] = rss_before
        training[name]["rss_after_fit_predict"] = psutil.Process().memory_info().rss
        print(f"BENCHMARK {name}: {reports[name]['overall']['rmse_sec']:.5f}s", flush=True)
        del model
        gc.collect()
    write_json(run.path / "metrics.json", {**baseline, **reports})
    pilot = training["with_clocks"]
    measured_increment = max(0, pilot["rss_after_fit_predict"] - pilot["rss_before_fit"])
    # Categorical dictionaries and library allocations are fixed costs; numeric pool storage scales.
    projected_pool = len(x) * len(x.columns) * 4 * 8
    estimate = psutil.Process().memory_info().rss + projected_pool + max(measured_increment, 256 * 1024 ** 2)
    pilot_report = {"rows": len(fit), "full_training_rows": len(x), "measurements": training,
                    "estimated_full_peak_rss_bytes": estimate, "available_ram_bytes": psutil.virtual_memory().available,
                    "current_rss_bytes": psutil.Process().memory_info().rss, "reserve_bytes": 2 * 1024 ** 3,
                    "estimated_full_fit_seconds": pilot["runtime_sec"] * len(x) / len(fit) * 240 / 180,
                    "estimator": "current RSS + 8 float32 feature-pool equivalents + measured pilot incremental/fixed allocation",
                    "all_rows_fit_feasible": projected_pool + max(measured_increment, 256 * 1024 ** 2) < psutil.virtual_memory().available - 2 * 1024 ** 3}
    write_json(ROOT / "reports/resource_pilot.json", pilot_report)
    if not pilot_report["all_rows_fit_feasible"]:
        print("Resource estimate requires reducing fitting sample before full-data training", flush=True)
    complete = run.complete(fold="legacy_Nov_Dec", training_rows=len(fit), available_training_rows=len(train),
                            sample_ids_hash=object_hash(x.index[fit].tolist()), metrics={**baseline, **reports}, training=training,
                            coverage="continuous-audit-only; original monthly ordering/RNG sample", pilot=pilot_report)
    write_json(ROOT / "reports/benchmark.json", complete)
    return complete


def anytime(config):
    manifest = verified_manifest(config)
    run = Run("anytime-airport-mean", config, manifest)
    training = audited_departures(manifest, columns=[ID, "ADEP_mvt", TARGET])
    means = training.groupby("ADEP_mvt", observed=True)[TARGET].mean()
    global_mean = float(training[TARGET].mean())
    ranking = read_raw(ROOT / config["raw_dir"] / "ranking.parquet", [ID, PHASE, "ADEP_mvt"])
    prediction = ranking.loc[ranking[PHASE].eq("DEP"), [ID, "ADEP_mvt"]].copy()
    prediction["prediction_sec"] = prediction["ADEP_mvt"].map(means).astype(float).fillna(global_mean)
    prediction["route"] = "airport_mean"
    prediction.to_parquet(run.path / "ranking_predictions.parquet", index=False)
    output = ROOT / "submissions/anytime_candidate.parquet"
    validation = build_submission(ROOT / config["raw_dir"] / "submitting.parquet", prediction, output)
    write_json(run.path / "means.json", {"global": global_mean, "airports": means.to_dict()})
    complete = run.complete(training_rows=len(training), validation=validation, method="full-2025 airport mean, global fallback")
    write_json(ROOT / "reports/anytime_candidate.json", complete)
    return complete
