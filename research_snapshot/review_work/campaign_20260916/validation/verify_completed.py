"""Independent integrity, cohort, metric and seasonal day-block verification."""

import argparse
import gc
import importlib
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
if "tabm" in sys.argv:
    import torch
import lightgbm
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import OUT, WORKSPACE, ID, TARGET, read_json, write_json, sha256, object_hash, utc_now, reference
from taxiout.metrics import scores

SCRIPT_SHA256_AT_START = sha256(__file__)
WEIGHTS = {"F1": 192122 / 344841, "F3": 152719 / 344841}


def stability(first, second):
    frame = first[[ID, "day", "ADEP_mvt", "squared_error"]].copy()
    frame["candidate_sse"] = second.squared_error.to_numpy()
    daily = frame.groupby("day").agg(a=("squared_error", "sum"), b=("candidate_sse", "sum"), n=(ID, "size"))
    totals = daily.sum().to_numpy(float)
    leave = totals - daily.to_numpy(float)
    delta = np.sqrt(leave[:, 1] / leave[:, 2]) - np.sqrt(leave[:, 0] / leave[:, 2])
    blocks = frame.groupby(["ADEP_mvt", "day"], observed=True).agg(a=("squared_error", "sum"), b=("candidate_sse", "sum"), n=(ID, "size")).to_numpy(float)
    return {"leave_one_day_out_delta_range": [float(delta.min()), float(delta.max())],
            "blocks": blocks, "daily": daily.to_numpy(float), "totals": totals}


def compare_metric(actual, expected):
    for key in ("n", "sse", "rmse_sec", "mae_sec", "bias_sec", "p95_abs_error_sec"):
        if not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-7):
            raise AssertionError(f"Metric differs: {key}")


def seasonal_paired(first_blocks, repetitions=1000):
    rng = np.random.default_rng(20260916)
    delta = []
    for _ in range(repetitions):
        a = b = 0.0
        for fold, blocks in first_blocks.items():
            sampled = blocks[rng.integers(0, len(blocks), len(blocks))].sum(axis=0)
            a += WEIGHTS[fold] * sampled[0] / sampled[2]
            b += WEIGHTS[fold] * sampled[1] / sampled[2]
        delta.append(np.sqrt(b) - np.sqrt(a))
    return {"seed": 20260916, "repetitions": repetitions,
            "candidate_minus_reference_rmse_95pct": np.quantile(delta, [.025, .975]).tolist(),
            "method": "Paired airport-day blocks sampled separately in July/November, seasonal weighted MSE reconstructed each resample.",
            "caveat": "Development stability only; adaptive selection and extreme events limit interpretation."}


def seasonal_remove_day(stats):
    deltas = []
    for removed_fold, values in stats.items():
        for day in values["daily"]:
            a = b = 0.0
            for fold, item in stats.items():
                total = item["totals"] - day if fold == removed_fold else item["totals"]
                a += WEIGHTS[fold] * total[0] / total[2]
                b += WEIGHTS[fold] * total[1] / total[2]
            deltas.append(float(np.sqrt(b) - np.sqrt(a)))
    return {"removed_dates": len(deltas), "candidate_minus_reference_delta_range": [min(deltas), max(deltas)],
            "method": "Remove each UTC day from its season; retain July/November population weights."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", choices=["catboost", "ridge", "lightgbm", "xgboost", "tabm", "missing_mixture"])
    args = parser.parse_args()
    refs = {fold: reference(fold)[0] for fold in WEIGHTS}
    observed_statuses = {fold: sorted(frame.proxy_status.dropna().unique().tolist()) for fold, frame in refs.items()}
    verified, pending, groups = [], [], {}
    paths = [*(OUT / "models").glob("*/manifest.json"), *(OUT / "missing_mixture/models").glob("*/manifest.json")]
    for path in sorted(paths):
        record = read_json(path)
        if record["status"] != "complete":
            pending.append({"name": path.parent.name, "status": record["status"]})
            continue
        fold = record["name"].split("_")[-2]
        if fold not in WEIGHTS:
            continue
        for name, digest in record["outputs"].items():
            if sha256(path.parent / name) != digest:
                raise AssertionError(f"Output hash mismatch {path.parent.name}/{name}")
        ref = refs[fold]
        block_variants = {}
        for variant, report in record["reports"].items():
            frame = pd.read_parquet(path.parent / f"{variant}.parquet")
            if not np.array_equal(frame[ID], ref[ID]) or not np.array_equal(frame[TARGET], ref[TARGET]):
                raise AssertionError("Score cohort or original labels differ")
            metric = scores(frame[TARGET], frame.prediction_sec)
            compare_metric(metric, report["metrics"]["overall"])
            err = frame.prediction_sec.to_numpy() - frame[TARGET].to_numpy()
            if not np.allclose(err**2, frame.squared_error, rtol=1e-12, atol=1e-8):
                raise AssertionError("Stored squared error differs")
            s = stability(ref, frame)
            if not np.allclose(s["leave_one_day_out_delta_range"], report["stability"]["leave_one_day_out_delta_range"], atol=1e-8):
                raise AssertionError("Day-removal sensitivity differs")
            for column in ("ADEP_mvt", "proxy_status", "route"):
                for category, group in frame.groupby(column, observed=True, dropna=False):
                    compare_metric(scores(group[TARGET], group.prediction_sec), report["metrics"]["slices"][column][str(category)])
            if variant == "blend25":
                raw = pd.read_parquet(path.parent / "candidate.parquet", columns=["prediction_sec"])
                if not np.allclose(frame.prediction_sec.to_numpy(), .75 * ref.prediction_sec.to_numpy() + .25 * raw.prediction_sec.to_numpy(), rtol=0, atol=1e-9):
                    raise AssertionError("Fixed blend differs")
            block_variants[variant] = s
        if record["target"] == "correction":
            frame = pd.read_parquet(path.parent / "candidate.parquet")
            proxy = ref.proxy_sec.to_numpy()
            protected = ~(np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
            if not np.array_equal(frame.loc[protected, "prediction_sec"], ref.loc[protected, "prediction_sec"]):
                raise AssertionError("Correction changed protected missing/exceptional routes")
        if record["family"] == "lightgbm_soft_missing":
            archived_protocol = OUT / "missing_mixture/preflight_protocol_before_exact_protected_blend.json"
            if sha256(archived_protocol) != record["protocol_sha256"]:
                raise AssertionError("Missing-mixture executed protocol no longer matches its manifest")
            declaration = read_json(archived_protocol)["declaration"]
            for filename, digest in declaration["source_hashes"].items():
                source = WORKSPACE / filename
                if source.name == "run_missing_mixture.py":
                    source = source.with_name("run_missing_mixture_original_executed.py")
                if sha256(source) != digest:
                    raise AssertionError("Missing-mixture executed source hash differs")
            frame = pd.read_parquet(path.parent / "candidate.parquet")
            changed = ~np.isfinite(ref.proxy_sec.to_numpy()) & np.isfinite(ref.schedule_sec.to_numpy())
            if not np.array_equal(frame.loc[~changed, "prediction_sec"], ref.loc[~changed, "prediction_sec"]):
                raise AssertionError("Missing mixture changed protected routes")
            blend = pd.read_parquet(path.parent / "blend25.parquet")
            protected_blend_delta = float(np.max(np.abs(blend.loc[~changed, "prediction_sec"].to_numpy() - ref.loc[~changed, "prediction_sec"].to_numpy())))
            if protected_blend_delta > 1e-9:
                raise AssertionError("Missing mixture protected blend changed more than arithmetic tolerance")
        verified.append({"name": record["name"], "manifest_sha256": sha256(path),
                         "relative_directory": path.parent.relative_to(OUT).as_posix(),
                         "score_rows": len(ref), "variants": list(record["reports"]),
                         "recovered_with_unknown_timings": bool(record.get("recovery")),
                         **({"executed_protocol_archive_verified": True, "protected_blend_max_abs_delta": protected_blend_delta,
                             "current_runner_unmeasured": True} if record["family"] == "lightgbm_soft_missing" else {})})
        key = record["name"].replace("_" + fold + "_", "_SCREEN_")
        groups.setdefault(key, {})[fold] = (record, block_variants)
        print("VERIFIED", record["name"], flush=True)
    uncertainty = {}
    for key, pairs in groups.items():
        if set(pairs) != set(WEIGHTS):
            continue
        uncertainty[key] = {variant: {
            "bootstrap": seasonal_paired({fold: pairs[fold][1][variant]["blocks"] for fold in WEIGHTS}),
            "day_removal": seasonal_remove_day({fold: pairs[fold][1][variant] for fold in WEIGHTS})}
            for variant in pairs["F1"][1]}
    replay = []
    if args.replay:
        import joblib
        from common import load_data, fold_data
        x, meta = load_data()
        extension = None
        if args.replay == "missing_mixture":
            module = importlib.import_module("aviation.run_missing_mixture_original_executed")
            x, _ = module.load_schedule_extension(x)
        elif args.replay in ("catboost", "ridge"):
            module = importlib.import_module("baseline_adapters")
        else:
            module = importlib.import_module({"lightgbm": "lgbm_adapter", "xgboost": "xgb_adapter", "tabm": "tabm_adapter"}[args.replay])
        for record in verified:
            path = OUT / record["relative_directory"]
            manifest = read_json(path / "manifest.json")
            requested_family = "lightgbm_soft_missing" if args.replay == "missing_mixture" else args.replay
            if manifest["family"] != requested_family:
                continue
            fold = manifest["name"].split("_")[-2]
            idx, split, sampled = fold_data(meta, fold, manifest["full"])
            if object_hash(split) != object_hash(manifest["split"]) or ("sampled" in manifest and sampled != manifest["sampled"]):
                raise AssertionError("Replay split/sample changed")
            if args.replay == "missing_mixture":
                schedule = meta.schedule_sec.to_numpy(float)
                eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float)) & np.isfinite(schedule)
                selected = {stage: positions[eligible[positions]] for stage, positions in idx.items()}
                fit_ids = {stage: {"n": len(positions), "hash": object_hash(meta.iloc[positions][ID].tolist())}
                           for stage, positions in selected.items()}
                if fit_ids != manifest["fit_ids"]:
                    raise AssertionError("Missing-mixture fit/tune/refit/eligible-score IDs differ")
                for stage, information in [("fit", "tune"), ("refit", "refit")]:
                    positions = selected[stage]
                    actual_good = int((np.abs(meta.iloc[positions][TARGET].to_numpy() - schedule[positions]) <= 60).sum())
                    if actual_good != manifest[information]["consistent_n"] or len(positions) != manifest[information]["fit_rows"]:
                        raise AssertionError("Training regime counts differ from fit-only labels")
                model = joblib.load(path / "model.joblib")
                values = module.predict_mixture(model, x.iloc[selected["score"]], schedule[selected["score"]])
                ref = refs[fold]
                expected = ref.prediction_sec.to_numpy().copy()
                expected[eligible[idx["score"]]] = values
                stored = pd.read_parquet(path / "candidate.parquet", columns=["prediction_sec"]).prediction_sec.to_numpy()
                delta = float(np.max(np.abs(stored - expected)))
                if delta > 1e-9:
                    raise AssertionError("Missing mixture saved-model replay differs")
                replay.append({"name": record["name"], "max_abs_delta": delta,
                               "training_regime_counts_verified": True, "complete_score_rows": len(stored)})
                del model, values, stored, expected
                gc.collect()
                continue
            fitted = {stage: rows for stage, rows in idx.items()}
            if manifest["target"] == "correction":
                proxy_all = meta.proxy_sec.to_numpy()
                eligible = np.isfinite(proxy_all) & (proxy_all >= 0) & (proxy_all <= 7200)
                fitted = {stage: (rows if stage == "score" else rows[eligible[rows]]) for stage, rows in idx.items()}
            fit_ids = {stage: {"n": len(rows), "hash": object_hash(meta.iloc[rows][ID].tolist())} for stage, rows in fitted.items()}
            if fit_ids != manifest["fit_ids"]:
                raise AssertionError("Fit/tune/refit/score eligible row hashes differ")
            features = x.iloc[idx["score"]]
            if manifest["features"] != "base":
                if manifest["features"] != "physical_base":
                    marker = read_json(OUT / "aviation/manifest.json")
                    if sha256(OUT / "aviation/features.parquet") != marker["feature_sha256"]:
                        raise AssertionError("Aviation feature cache hash changed")
                    if extension is None:
                        extension = pd.read_parquet(OUT / "aviation/features.parquet").set_index(ID)
                        if not np.array_equal(extension.index, x.index):
                            raise AssertionError("Aviation feature IDs differ")
                    features = pd.concat([features, extension.iloc[idx["score"]]], axis=1)
                if manifest["features"].startswith("physical"):
                    features = features.copy()
                    features["__movement_ns"] = pd.to_datetime(meta.iloc[idx["score"]]["MVT_TIME_UTC_mvt"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
                features = features[manifest["features_used"]]
            model = joblib.load(path / "model.joblib")
            if isinstance(model, dict) and "estimator" in model:
                estimator = model["estimator"]
                while isinstance(estimator, dict):
                    estimator = estimator["estimator"]
                if hasattr(estimator, "set_params"):
                    estimator.set_params(n_jobs=2)
                if "threads" in model:
                    model["threads"] = 2
            if hasattr(model, "set_params") and args.replay == "catboost":
                values = np.asarray(model.predict(features, thread_count=2))
            elif manifest["features"].startswith("physical"):
                values = importlib.import_module("domain_adapter").predict(model, features)
            else:
                values = module.predict(model, features)
            expected = values
            ref = refs[fold]
            if manifest["target"] == "correction":
                proxy = ref.proxy_sec.to_numpy()
                changed = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
                expected = ref.prediction_sec.to_numpy().copy()
                expected[changed] = proxy[changed] + values[changed]
            stored = pd.read_parquet(path / "candidate.parquet", columns=["prediction_sec"]).prediction_sec.to_numpy()
            delta = float(np.max(np.abs(stored - expected)))
            if delta > 1e-5:
                raise AssertionError("Independent model replay differs")
            replay.append({"name": record["name"], "max_abs_delta": delta})
            del model, values, stored, expected
            gc.collect()
        del x, meta
    stamp = utc_now().replace(":", "-").replace("+", "_")
    receipt = {"status": "verified", "created_utc": utc_now(), "verified": verified,
        "pending_or_failed": pending, "seasonal_paired_uncertainty": uncertainty,
        "reference_proxy_status_values": observed_statuses, "independent_model_replay": replay,
        "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
        "script_sha256": SCRIPT_SHA256_AT_START,
        "script_unchanged_during_run": sha256(__file__) == SCRIPT_SHA256_AT_START,
        "scope": "Complete output hashes, all score rows/labels, all metrics and slices, fixed blend and protected routes, independent day removal, seasonal paired bootstrap; model replay only named family when requested."}
    dest = OUT / "validation" / f"completed_runs_{stamp}.json"
    write_json(dest, receipt)
    print("VERIFICATION RECEIPT", dest, flush=True)


if __name__ == "__main__":
    main()
