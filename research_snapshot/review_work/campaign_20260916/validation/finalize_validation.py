"""Bind final validation to all current manifests and frozen verifier replays."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import OUT, read_json, write_json, sha256, utc_now


def main():
    verifier_hash = sha256(HERE.with_name("verify_completed.py"))
    manifests = [*(OUT / "models").glob("*/manifest.json"), *(OUT / "missing_mixture/models").glob("*/manifest.json")]
    expected = {read_json(p)["name"]: (p, sha256(p)) for p in manifests if read_json(p)["status"] == "complete"}
    outstanding = [str(p) for p in manifests if read_json(p)["status"] != "complete"]
    receipts = {}
    verified = {}
    replays = {}
    latest = None
    for path in sorted((OUT / "validation").glob("completed_runs_*.json")):
        report = read_json(path)
        if report.get("script_sha256") != verifier_hash or not report.get("script_unchanged_during_run"):
            continue
        receipts[path.name] = sha256(path)
        latest = report
        per_receipt = {}
        for record in report["verified"]:
            if record["name"] in expected and record["manifest_sha256"] == expected[record["name"]][1]:
                verified[record["name"]] = record
                per_receipt[record["name"]] = record
        for replay in report["independent_model_replay"]:
            if replay["name"] in per_receipt:
                replays[replay["name"]] = {**replay, "receipt": path.name}
    absent_verified = sorted(set(expected) - set(verified))
    absent_replay = sorted(set(expected) - set(replays))
    if outstanding or absent_verified or absent_replay:
        raise AssertionError({"incomplete_runs": outstanding, "unverified": absent_verified, "unreplayed": absent_replay})
    for name, (path, digest) in expected.items():
        record = read_json(path)
        for output, expected_hash in record["outputs"].items():
            if sha256(path.parent / output) != expected_hash:
                raise AssertionError(f"Output changed after independent replay: {name}/{output}")
    uncertainties = latest["seasonal_paired_uncertainty"]
    best_keys = [key for key in uncertainties if key.startswith("lightgbm_correction_arrival_full_SCREEN")]
    result = {"status": "verified", "created_utc": utc_now(), "complete_containers": len(expected),
        "independently_replayed_models": len(replays), "all_current_manifest_hashes": {name: digest for name, (_, digest) in expected.items()},
        "model_replays": replays, "verified_containers": verified,
        "verifier_sha256": verifier_hash, "receipt_hashes": receipts,
        "best_candidate_uncertainty": {key: uncertainties[key]["blend25"] for key in best_keys},
        "seasonal_paired_uncertainty": uncertainties,
        "verified_contracts": ["Original scoring IDs and labels", "All output hashes", "All metrics and airport/proxy/route slices", "Complete-cohort saved-model replay", "Fit/tune/refit/score row hashes", "Predeclared fixed blend weights", "Protected exceptional and missing routes", "Independent per-fold and seasonal day-removal", "Paired seasonal airport-day bootstrap"],
        "limitations": ["All scoring folds were previously exposed development data; no new external or January labeled holdout.",
            "Best full-arrival complement has about one second local gain, below the two-second promotion gate.",
            "F1 XGBoost original timing/RSS lost in metadata serialization failure; retained as unknown, predictions/models recovered exactly.",
            "Missing-mixture source was revised after execution; results bind to recovered exact original source and archived original protocol, not the current unmeasured revision.",
            "Final NM event timestamps do not prove real-time publication; arrival duration needs observed completion before query."],
        "earlier_audits": {name: sha256(OUT / "validation" / name) for name in ["existing_evidence_audit.json", "full_cache_identity.json", "aviation_independent_oracle.json", "aviation_test_replay.json", "xgb_f1_recovery.json"]},
        "script_sha256": sha256(__file__)}
    write_json(OUT / "validation/final_validation.json", result)
    print(f"FINAL VERIFIED {len(expected)} containers and {len(replays)} saved-model replays", flush=True)


if __name__ == "__main__":
    main()
