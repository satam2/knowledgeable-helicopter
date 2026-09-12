"""Record a completed local final fit and its full prediction/reload evidence."""
import argparse
import json

import numpy as np

from taxiout.artifacts import read_json, sha256, source_hashes, utc_now, write_json
from taxiout.config import ROOT
from taxiout.predict import predict_candidate
from taxiout.submission import build_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    bundle = ROOT / "models" / args.run_id
    run = read_json(bundle / "manifest.json")
    if run["status"] != "complete" or run["fold"] != "final":
        raise ValueError("Finalization requires a complete final fit")
    config = run["config"]
    ranking = config["raw_dir"] + "/ranking.parquet"
    predictions = predict_candidate(bundle, ranking, "data/processed/ranking_predictions.parquet")
    repeat = predict_candidate(bundle, ranking)
    maximum_delta = float(np.max(np.abs(predictions.prediction_sec.to_numpy() - repeat.prediction_sec.to_numpy())))
    if maximum_delta > 1e-9:
        raise ValueError("Repeated inference parity failed")
    validation = build_submission(ROOT / config["raw_dir"] / "submitting.parquet", predictions, ROOT / "submissions/local_candidate.parquet")
    result = {"created_utc": utc_now(), "run_id": args.run_id, "local_technical_status": "awaiting_clean_reproduction",
              "candidate": config["candidate"], "submission": validation, "repeated_inference_max_abs_delta_sec": maximum_delta,
              "prediction_rows": len(predictions), "routes": predictions.route.value_counts().to_dict(),
              "proxy_status": predictions.proxy_status.value_counts().to_dict(), "raw_inputs": run["raw_inputs"],
              "model_outputs": run["outputs"], "source_hashes": source_hashes(),
              "configuration_hash": run["config_hash"], "freeze": read_json(ROOT / "reports/freeze.json"),
              "confirmation": read_json(ROOT / "reports/confirmation.json"),
              "raw_prediction_sha256": sha256(ROOT / "data/processed/ranking_predictions.parquet"),
              "external_data": [], "uploaded": False, "accepted": False, "published": False,
              "pending_external_dependencies": ["verified team/account identity and next version", "authorized upload and acceptance", "authorized public GitHub release"]}
    write_json(ROOT / "reports/release_manifest.json", result)
    print(json.dumps({k: result[k] for k in ["run_id", "candidate", "submission", "routes", "repeated_inference_max_abs_delta_sec"]}, indent=2))


if __name__ == "__main__":
    main()
