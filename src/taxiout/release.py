import numpy as np
import pandas as pd
import yaml

from taxiout.artifacts import object_hash, read_json, sha256, source_hashes, utc_now, write_json
from taxiout.config import ROOT
from taxiout.io import verified_manifest
from taxiout.metrics import paired_stability, pooled, season_score
from taxiout.predict import predict_candidate
from taxiout.submission import build_submission


def run_path(run_id):
    path = ROOT / "models" / run_id
    if not path.is_dir():
        raise ValueError(f"Unknown run {run_id}")
    manifest = read_json(path / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run {run_id}")
    return path, manifest


def compare(run_ids, reference="direct"):
    records, groups = {}, {}
    for run_id in run_ids:
        path, manifest = run_path(run_id)
        candidate, fold = manifest["config"]["candidate"], manifest["fold"]
        prior_hashes = {m["config_hash"] for _, m in groups.get(candidate, {}).values()}
        if prior_hashes and manifest["config_hash"] not in prior_hashes:
            raise ValueError("Candidate configurations differ across folds; compare one frozen configuration at a time")
        if fold in groups.get(candidate, {}):
            raise ValueError("Duplicate candidate/fold in comparison")
        groups.setdefault(candidate, {})[fold] = (path, manifest)
    input_hashes = {m["input_manifest_hash"] for folds in groups.values() for _, m in folds.values()}
    if len(input_hashes) != 1:
        raise ValueError("Cannot compare different input packs")
    for candidate, folds in groups.items():
        record = {"folds": {f: m["metrics"]["overall"] for f, (_, m) in folds.items()},
                  "missing_proxy": {f: m["metrics"]["slices"]["proxy_status"].get("missing") for f, (_, m) in folds.items()},
                  "run_ids": {f: m["run_id"] for f, (_, m) in folds.items()}}
        if "F1" in folds and "F3" in folds:
            record["screening_season_score"] = season_score(record["folds"]["F1"], record["folds"]["F3"], folds["F1"][1]["raw_inputs"]["ranking_departure_months"])
        if all(f in folds for f in ["F1", "F2", "F3"]):
            record["season_score"] = season_score(record["folds"]["F1"], record["folds"]["F3"], folds["F1"][1]["raw_inputs"]["ranking_departure_months"])
            record["pooled_development"] = pooled([record["folds"][f] for f in ["F1", "F2", "F3"]])
        records[candidate] = record
    if reference in groups:
        champion = records[reference]
        for candidate, folds in groups.items():
            if candidate == reference:
                continue
            record = records[candidate]
            paired = {}
            for fold in sorted(set(folds) & set(groups[reference])):
                a_path, a_manifest = groups[reference][fold]
                b_path, b_manifest = folds[fold]
                if a_manifest["split"]["split_hash"] != b_manifest["split"]["split_hash"]:
                    raise ValueError("Fold membership differs between compared models")
                paired[fold] = paired_stability(pd.read_parquet(a_path / "score_predictions.parquet"), pd.read_parquet(b_path / "score_predictions.parquet"))
            record["paired_vs_" + reference] = paired
            if "season_score" in record and "season_score" in champion:
                minimum_gain = max(2., .005 * champion["season_score"])
                seasonal_ok = all(record["folds"][f]["rmse_sec"] - champion["folds"][f]["rmse_sec"] <= max(5., .02 * champion["folds"][f]["rmse_sec"]) for f in ["F1", "F3"])
                missing_ok = all(record["missing_proxy"][f]["rmse_sec"] <= 1.05 * champion["missing_proxy"][f]["rmse_sec"] for f in ["F1", "F2", "F3"])
                record["promotion_gates"] = {"minimum_gain_sec": minimum_gain,
                    "observed_gain_sec": champion["season_score"] - record["season_score"],
                    "gain_pass": champion["season_score"] - record["season_score"] >= minimum_gain,
                    "seasonal_regression_pass": seasonal_ok, "missing_regression_pass": missing_ok,
                    "requires_review": [f for f in ["F2", "G1"] if f in record["folds"] and f in champion["folds"] and record["folds"][f]["rmse_sec"] > champion["folds"][f]["rmse_sec"]]}
    report = {"created_utc": utc_now(), "runs": run_ids, "candidates": records,
              "reference_candidate": reference,
              "selection_policy": "Season score; >= max(2s, 0.5%) gain; seasonal and missing-proxy regression gates; inspect F2/G1 and airport-day stability."}
    write_json(ROOT / "reports/comparison.json", report)
    write_json(ROOT / "reports/comparisons" / (object_hash(run_ids)[:12] + ".json"), report)
    return report


def freeze(config, run_ids):
    path = ROOT / "reports/freeze.json"
    if path.exists():
        raise ValueError("Freeze already exists; retain it and document a new development cycle explicitly")
    manifests = [run_path(run_id)[1] for run_id in run_ids]
    if any(m["config_hash"] != object_hash(config) for m in manifests):
        raise ValueError("Freeze runs must all use the selected configuration")
    by_fold = {m["fold"]: m for m in manifests}
    if not {"F1", "F2", "F3", "G1"}.issubset(by_fold):
        raise ValueError("Freeze requires completed F1/F2/F3/G1 evidence")
    if "C1" in by_fold:
        raise ValueError("Cannot use confirmation to establish the freeze")
    iterations = {name: int(np.median([by_fold[f]["iterations"][name] for f in ["F1", "F2", "F3"]])) for name in by_fold["F1"]["iterations"]}
    report = {"frozen_utc": utc_now(), "config_hash": object_hash(config), "configuration": config,
              "development_runs": run_ids, "final_iterations": iterations,
              "iteration_rule": "Per-component median of F1/F2/F3 chosen iterations; fixed before C1",
              "confirmation_rule": "One C1 run; November tunes iterations under frozen rule, December only scores; reused confirmation previously exposed in diagnostic.",
              "confirmation_acceptance": {"require_complete_reload_parity": True,
                  "beat_best_baseline": ["airport_mean", "hierarchical_mean", "raw_proxy"],
                  "missing_proxy_rmse_max_relative_to_airport_mean": 1.0,
                  "failure_action": "Stop final release; document investigation without silently tuning on December."},
              "source_hashes": source_hashes(), "input_manifest_hash": by_fold["F1"]["input_manifest_hash"]}
    (ROOT / "configs/release.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(path, report)
    return report


def reproduce(config):
    from taxiout.audit import audit
    from taxiout.train import train_final
    verified_manifest(config)
    if not (ROOT / "data/interim/audit/departures.parquet").exists():
        audit(config)
    final = train_final(config, output_name="reproduction")
    bundle = "models/" + final["run_id"]
    predictions = predict_candidate(bundle, config["raw_dir"] + "/ranking.parquet", "data/processed/reproduced_predictions.parquet")
    output = ROOT / "submissions/reproduced_candidate.parquet"
    validation = build_submission(ROOT / config["raw_dir"] / "submitting.parquet", predictions, output)
    record = {"run_id": final["run_id"], "offline": True, "validation": validation, "status": "complete"}
    reference = ROOT / "submissions/local_candidate.parquet"
    if reference.exists():
        a, b = pd.read_parquet(reference), pd.read_parquet(output)
        if not a.equals(b):
            raise ValueError("Offline reproduction differs from released serialized prediction values")
        record["serialized_values_equal_reference"] = True
        record["byte_hash_equal_reference"] = sha256(reference) == sha256(output)
    write_json(ROOT / "reports/reproduction.json", record)
    return record
