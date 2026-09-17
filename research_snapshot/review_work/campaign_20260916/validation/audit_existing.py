"""Read-only verification of prior evidence; write only new external audit receipts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "knowledgeable-helicopter-screening"
sys.path.insert(0, str(REPO / "src"))

from taxiout.artifacts import object_hash, read_json, sha256, source_hashes, utc_now, write_json
from taxiout.config import load_config
from taxiout.metrics import scores, season_score
from taxiout.paths import external_path
from taxiout.schema import FLIGHT_ID, ID, MOVEMENT, TARGET
from taxiout.splits import make_fold

OUT = external_path(WORKSPACE / "private_runs/campaign_20260916/validation")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def synthetic_contracts():
    dates = ["2025-01-02"] * 4 + ["2025-06-02"] * 3 + ["2025-07-02"] * 2
    meta = pd.DataFrame({ID: np.arange(9), FLIGHT_ID: [10, 20, 30, 40, 20, 50, 60, 10, 50],
                         MOVEMENT: pd.to_datetime(dates, utc=True)})
    idx, evidence = make_fold(meta, load_config("configs/folds.yaml")["F1"])
    check(idx["fit"].tolist() == [2, 3], "Synthetic fit flight purge failed")
    check(idx["tune"].tolist() == [4, 6], "Synthetic tune flight purge failed")
    check(idx["refit"].tolist() == [1, 2, 3, 4, 6], "Refit flight purge failed")
    check(idx["score"].tolist() == [7, 8], "Scoring cohort mutated")
    check(not np.array_equal(np.sort(np.r_[idx["fit"], idx["tune"]]), idx["refit"]),
          "Refit must be independently constructed after purges")
    rejected = []
    for repo in (WORKSPACE / "knowledgeable-helicopter", REPO):
        try:
            external_path(repo / "must-not-write-private.parquet")
        except ValueError:
            rejected.append(repo.name)
    check(len(rejected) == 2, "A Git checkout accepted a private artifact path")
    value = season_score(scores([0], [100]), scores([0], [300]))
    expected = np.sqrt((192122 * 100**2 + 152719 * 300**2) / 344841)
    check(abs(value - expected) < 1e-12, "Seasonal weighted-MSE arithmetic differs")
    return {"flight_id_purge": evidence, "checkout_rejection": rejected,
            "weighted_mse_expected": float(expected), "weighted_mse_actual": value}


def main():
    protocol = read_json(WORKSPACE / "private_runs/submission_v2/protocol.json")
    check(source_hashes() == protocol["source_hashes"], "Frozen source hashes changed")
    raw_root = external_path(WORKSPACE / "data/09-15-2026-18-55-03_files_list")
    files = sorted(raw_root.glob("*.parquet"))
    raw = {}
    schemas = {}
    for path in files:
        digest = sha256(path)
        check(digest == protocol["raw_hashes"].get(path.name), f"Raw hash mismatch: {path.name}")
        raw[path.name] = digest
        metadata = pq.ParquetFile(path)
        schemas[path.name] = {"rows": metadata.metadata.num_rows,
                              "schema": str(metadata.schema_arrow)}
        print(f"Verified raw {path.name}", flush=True)
    check(raw == protocol["raw_hashes"], "Raw file inventory changed")

    old_final = read_json(WORKSPACE / "private_runs/next_230/final_verification.json")
    runners = {}
    for name, expected in old_final["sources"].items():
        actual = sha256(WORKSPACE / "review_work" / name)
        check(actual == expected, f"Frozen external runner changed: {name}")
        runners[name] = actual
    evidence_hashes = {}
    for name, expected in old_final["evidence_sha256"].items():
        actual = sha256(WORKSPACE / "private_runs/next_230" / name)
        check(actual == expected, f"Preserved evidence changed: {name}")
        evidence_hashes[name] = actual
    check(sha256(WORKSPACE / "review_work/final_submission_v2.py") == protocol["runner_sha256"],
          "Frozen final submission runner changed")

    audit = read_json(WORKSPACE / "private_runs/screening_230/reports/data_audit.json")
    dep_path = WORKSPACE / "private_runs/screening_230/data/interim/audit/departures.parquet"
    check(sha256(dep_path) == audit["artifacts"][dep_path.name], "Departure audit cache changed")
    meta = pd.read_parquet(dep_path, columns=[ID, FLIGHT_ID, MOVEMENT, TARGET])
    check(len(meta) == protocol["training_rows"], "Departure count differs from submission training")
    check(object_hash(meta[ID].tolist()) == protocol["training_id_hash"], "Training ID order changed")
    selected = {}
    for fold in ("F1", "F2", "F3", "G1"):
        idx, split = make_fold(meta, load_config("configs/folds.yaml")[fold])
        path = WORKSPACE / "private_runs/next_230/models" / f"clock_and_rome_ensemble_{fold}_s20260910"
        record = read_json(path / "manifest.json")
        check(record["status"] == "complete", "Selected reference incomplete")
        check(object_hash(split) == object_hash(record["split"]), "Selected reference split changed")
        for name, expected in record["outputs"].items():
            check(sha256(path / name) == expected, f"Reference output corruption: {fold}/{name}")
        pred = pd.read_parquet(path / "score_predictions.parquet")
        label = meta.iloc[idx["score"]]
        check(np.array_equal(pred[ID], label[ID]), "Scored IDs changed")
        check(np.array_equal(pred[TARGET], label[TARGET]), "Scored labels changed")
        metric = scores(label[TARGET], pred.prediction_sec)
        check(abs(metric["rmse_sec"] - record["metrics"]["overall"]["rmse_sec"]) < 1e-9,
              "Reference RMSE does not replay")
        selected[fold] = {"path": str(path), "manifest_sha256": sha256(path / "manifest.json"),
                          "prediction_sha256": record["outputs"]["score_predictions.parquet"],
                          "split": split, "metrics": metric}
        print(f"Verified selected {fold}: {metric['rmse_sec']:.9f}", flush=True)
    seasonal = season_score(selected["F1"]["metrics"], selected["F3"]["metrics"])
    check(abs(seasonal - protocol["local_seasonal_rmse_sec"]) < 1e-9, "Selected seasonal score changed")
    findings = {"status": "verified", "created_utc": utc_now(),
                "raw_hashes": raw, "raw_schemas": schemas,
                "frozen_source_hashes": source_hashes(), "frozen_external_runner_hashes": runners,
                "preserved_evidence_hashes": evidence_hashes,
                "selected_reference": selected, "selected_seasonal_rmse_sec": seasonal,
                "synthetic_contracts": synthetic_contracts(), "script_sha256": sha256(__file__),
                "scope": "Hashes, metadata, audited labels and complete prediction replay; no training or final submission command."}
    write_json(OUT / "existing_evidence_audit.json", findings)
    print(f"AUDIT VERIFIED: {len(raw)} raw files; {len(source_hashes())} frozen sources; seasonal {seasonal:.9f}", flush=True)


if __name__ == "__main__":
    main()
