"""Recompute public aggregate evidence without accessing private flight records."""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE / "knowledgeable-helicopter"
OUT = BASE / "review_work"


def read(name):
    return json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))


def metric_checks(value, location, results):
    if isinstance(value, dict):
        if {"n", "sse", "rmse_sec"} <= value.keys() and value["n"]:
            computed = math.sqrt(value["sse"] / value["n"])
            results.append({"path": location, "delta": abs(computed - value["rmse_sec"])})
        if "overall" in value and "slices" in value:
            for dimension, groups in value["slices"].items():
                assert sum(v["n"] for v in groups.values()) == value["overall"]["n"], (location, dimension)
                assert math.isclose(sum(v["sse"] for v in groups.values()), value["overall"]["sse"], rel_tol=1e-9), (location, dimension)
        for key, child in value.items():
            metric_checks(child, location + "/" + key, results)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            metric_checks(child, location + "/" + str(i), results)


def main():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    metrics = []
    json_paths = [p for p in tracked if p.startswith("reports/") and p.endswith(".json")]
    for name in json_paths:
        metric_checks(json.loads((REPO / name).read_text(encoding="utf-8")), name, metrics)
    audit, comparison, release = read("data_audit.json"), read("comparison.json"), read("release_manifest.json")
    selected = comparison["candidates"][release["candidate"]]
    training_missing = audit["missing_proxy"] / audit["training_departures"]
    ranking_missing = release["routes"]["specialist_missing"] / release["prediction_rows"]
    folds = {}
    for fold in ["F1", "F2", "F3", "G1", "C1"]:
        overall = selected["folds"][fold] if fold != "C1" else release["confirmation"]["metrics"]["overall"]
        missing = selected["missing_proxy"][fold] if fold != "C1" else release["confirmation"]["metrics"]["slices"]["proxy_status"]["missing"]
        available_mse = (overall["sse"] - missing["sse"]) / (overall["n"] - missing["n"])
        missing_share = missing["n"] / overall["n"]
        folds[fold] = {
            "overall": overall, "missing": missing,
            "missing_row_pct": missing_share * 100,
            "available_rmse": math.sqrt(available_mse),
            "missing_sse_pct": missing["sse"] / overall["sse"] * 100,
            "hypothetical_rmse_after_20pct_missing_rmse_reduction": overall["rmse_sec"] * math.sqrt(1 - .36 * missing["sse"] / overall["sse"]),
            "hypothetical_rmse_at_ranking_missing_mix": math.sqrt((1-ranking_missing) * available_mse + ranking_missing * missing["sse"] / missing["n"]),
        }
    hashes = {}
    for report_name in ["freeze.json", "release_manifest.json"]:
        report = read(report_name)
        source = report["source_hashes"]
        mismatches = [name for name, expected in source.items() if not (REPO / name).exists() or hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected]
        hashes[report_name] = {"count": len(source), "mismatches": mismatches}
    all_candidates = {}
    for name, value in comparison["candidates"].items():
        all_candidates[name] = {key: child for key, child in value.items() if key != "run_ids"}
    experiments = list(csv.DictReader((REPO / "reports/experiments.csv").open(encoding="utf-8", newline="")))
    registry = read("configuration_registry.json")
    c1 = release["confirmation"]["metrics"]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "tracked_files": len(tracked), "source_files": sum(p.endswith(".py") for p in tracked),
        "json_reports": len(json_paths), "metric_checks": len(metrics),
        "maximum_metric_arithmetic_delta": max(item["delta"] for item in metrics),
        "source_hash_checks": hashes,
        "data": {"training_missing_pct": training_missing * 100, "ranking_missing_pct": ranking_missing * 100,
                 "relative_missing_increase_pct": (ranking_missing / training_missing - 1) * 100,
                 "training_direct_fraction_pct": 250000 / audit["training_departures"] * 100,
                 "tail_missing_fraction_pct": audit["missing_proxy_over_two_hours"] / audit["target"]["over_two_hours"] * 100,
                 "tail_probability_missing_pct": audit["missing_proxy_over_two_hours"] / audit["missing_proxy"] * 100,
                 "tail_probability_present_pct": (audit["target"]["over_two_hours"]-audit["missing_proxy_over_two_hours"])/(audit["training_departures"]-audit["missing_proxy"])*100,
                 "routes": release["routes"], "target": audit["target"],
                 "ranking_month_counts": audit["ranking_departure_months"]},
        "folds": folds, "candidates": all_candidates, "confirmation_slices": c1["slices"],
        "registry": {"configurations": len(registry), "runs": sum(len(v["runs"]) for v in registry.values()),
                     "incomplete": [r for v in registry.values() for r in v["runs"] if r["status"] != "complete"]},
        "experiments_rows": len(experiments),
        "benchmark_overall": {k: v["overall"] for k, v in read("benchmark.json")["metrics"].items()},
        "release_status": {k: release.get(k) for k in ["local_technical_status", "uploaded", "accepted", "published", "pending_external_dependencies"]},
    }
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["commit", "tracked_files", "source_files", "json_reports", "metric_checks", "maximum_metric_arithmetic_delta", "source_hash_checks", "data", "registry", "release_status"]}, indent=2))
    print("FOLDS", json.dumps(folds, indent=2))
    print("SELECTED PAIRED", json.dumps({k: v for k,v in selected.items() if k.startswith("paired")}, indent=2))


if __name__ == "__main__":
    main()
