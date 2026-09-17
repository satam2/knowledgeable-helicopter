"""Local-only screening orchestration. All outputs are outside the checkout."""

import argparse
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE / "knowledgeable-helicopter-screening"
RAW = WORKSPACE / "data/09-15-2026-18-55-03_files_list"
OUT = WORKSPACE / "private_runs/screening_230"
os.environ["TAXIOUT_RAW_DIR"] = str(RAW)
os.environ["TAXIOUT_ARTIFACT_ROOT"] = str(OUT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(REPO / "src"))

from taxiout.artifacts import read_json, write_json, sha256, source_hashes, utc_now
from taxiout.config import load_config
from taxiout.paths import artifact_path, raw_root

CONFIGS = {"baseline": "release", "A_rows": "screen_rows", "B_trees": "screen_trees",
           "C_prefix": "screen_prefix", "D_rome": "screen_rome"}


def config_for(name):
    return load_config(f"configs/{CONFIGS[name]}.yaml")


def prepare():
    from taxiout.audit import audit
    from taxiout.cache import prepare as features
    from taxiout.io import verified_manifest
    root = artifact_path()
    root.mkdir(parents=True, exist_ok=True)
    original = read_json(WORKSPACE / "knowledgeable-helicopter/reports/input_manifest.json")
    actual = {p.name: sha256(p) for p in RAW.glob("*.parquet")}
    expected = {r["file"]: r["sha256"] for r in original["files"]}
    if actual != expected:
        raise ValueError("Raw pack does not match the reviewed fourteen files")
    protocol = {"created_utc": utc_now(), "target_sec": 230, "intermediate_target_sec": 280,
                "configs": {k: config_for(k) for k in CONFIGS}, "folds": ["F1", "F3"],
                "source_hashes": source_hashes(), "raw_hashes": actual,
                "raw_root": str(raw_root(config_for("baseline"))), "artifact_root": str(root),
                "evaluations": 8, "baseline_evaluations": 2,
                "status": "prespecified before model scores",
                "further_search": "Choose follow-up experiments after the eight screens; retain every result."}
    protocol_file = root / "protocol.json"
    if protocol_file.exists():
        old = read_json(protocol_file)
        if old["source_hashes"] != protocol["source_hashes"] or old["configs"] != protocol["configs"]:
            raise ValueError("Screening code/config changed after protocol freeze")
    else:
        write_json(protocol_file, protocol)
    if not artifact_path("reports/data_audit.json").exists():
        audit(config_for("baseline"))
    manifest = verified_manifest(config_for("baseline"))
    for name in ["baseline", "C_prefix"]:
        features(config_for(name), manifest)
    print("PREPARE COMPLETE: raw hashes verified; both feature schemas ready", flush=True)


def run(name, fold):
    from taxiout.train import evaluate_candidate, find_run
    from taxiout.io import verified_manifest
    protocol = read_json(artifact_path("protocol.json"))
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Source changed after protocol freeze")
    config = config_for(name)
    manifest = verified_manifest(config)
    prior = find_run(config, fold, manifest)
    result = prior[1] if prior else evaluate_candidate(config, fold)
    write_json(artifact_path("results", f"{name}_{fold}.json"), result)
    print(f"RESULT {name} {fold}: {result['metrics']['overall']['rmse_sec']:.6f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "run"])
    p.add_argument("--candidate", choices=list(CONFIGS))
    p.add_argument("--fold", choices=["F1", "F3", "F2", "G1"])
    args = p.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if not args.candidate or not args.fold:
            p.error("run needs --candidate and --fold")
        run(args.candidate, args.fold)
