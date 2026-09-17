"""Supplement frozen campaign checks without changing active runner files."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[3]
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(WORKSPACE / "knowledgeable-helicopter-screening/src"))

from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.cache import cache_identity
from taxiout.paths import external_path

OUT = external_path(WORKSPACE / "private_runs/campaign_20260916/validation")


def main():
    protocol = read_json(WORKSPACE / "private_runs/submission_v2/protocol.json")
    old = WORKSPACE / "private_runs/screening_230"
    manifest = read_json(old / "reports/input_manifest.json")
    identity = cache_identity(protocol["base_config"], manifest)
    cache = old / "data/interim/features/a2a101f52a0aa418"
    markers = {}
    for marker in sorted(cache.glob("training_*.json")):
        record = read_json(marker)
        if record["identity"] != identity:
            raise AssertionError(f"Full cache identity differs: {marker.name}")
        actual = sha256(marker.with_suffix(".parquet"))
        if actual != record["sha256"]:
            raise AssertionError(f"Feature cache changed: {marker.name}")
        markers[marker.name] = {"marker_sha256": sha256(marker), "parquet_sha256": actual,
                                "rows": record["rows"]}
    if len(markers) != 12:
        raise AssertionError("Expected twelve feature cache partitions")
    write_json(OUT / "full_cache_identity.json", {
        "status": "verified", "created_utc": utc_now(), "identity": identity,
        "identity_hash": object_hash(identity), "partitions": markers,
        "verification": "Exact current cache_identity equality: raw hashes, coverage, availability policy, libraries, all feature sources, schema, encoder state and fit/split placeholders.",
        "script_sha256": sha256(__file__)})

    sources = {p.relative_to(CAMPAIGN).as_posix(): sha256(p) for p in sorted(CAMPAIGN.rglob("*.py"))}
    models = WORKSPACE / "private_runs/campaign_20260916/models"
    direct = [str(p.relative_to(models)) for p in models.glob("*direct*/manifest.json")]
    write_json(OUT / "runtime_dependency_supplement.json", {
        "created_utc": utc_now(), "source_hashes": sources,
        "dependencies": {"xgb_adapter.py": ["lgbm_adapter.py"],
                         "lgbm_adapter.py": [], "common.py": ["frozen taxiout package"]},
        "all_direct_manifests_at_capture": direct,
        "missing_route_amendment": "Direct missing_only and missing_blend25 routes are explicitly in run.py before any direct run has been launched at this capture. Retain and count them as two additional fixed development variants per direct family, with no score-fitted blend weights. Original PLAN registry listed direct models but omitted these route variants.",
        "prespecification_verified_before_direct_launch": len(direct) == 0,
        "scope": "Supplemental source snapshot hashes; existing model manifests are unchanged."})

    test = CAMPAIGN / "aviation/test_arrival_features.py"
    result = subprocess.run([sys.executable, "-B", "-u", str(test)], capture_output=True, text=True)
    write_json(OUT / "aviation_test_replay.json", {
        "created_utc": utc_now(), "returncode": result.returncode,
        "output": result.stdout + result.stderr,
        "test_sha256": sha256(test), "module_sha256": sha256(test.with_name("arrival_features.py")),
        "scope": "Independent execution of module author's synthetic tests; static review separately recorded."})
    if result.returncode:
        raise AssertionError("Aviation synthetic tests failed")
    print("PASS full twelve-partition cache identity, dependency supplement, and aviation test replay", flush=True)


if __name__ == "__main__":
    main()
