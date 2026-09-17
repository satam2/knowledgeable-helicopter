"""Verify fixed lexical ablation and report every declared variant."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_lexical import ROOT, OUT, BASE, prior
from taxiout.artifacts import read_json, write_json, sha256, utc_now


def main():
    destination = ROOT / "private_runs/tail240_20260916/forensics/models/lexical_audit_v1"
    destination.mkdir(parents=True,exist_ok=False)
    old = ROOT / "private_runs/breakthrough_20260916/missing/models"
    meta_path = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(meta_path) == read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][meta_path.name]
    meta = pd.read_parquet(meta_path)
    weights = {"F1":192122/344841,"F3":152719/344841}
    metrics, receipts = [], []
    for arm in ["control","lexical"]:
        for fold in ["F1","F3"]:
            folder = OUT / arm / "models" / f"historical_template_{fold}_s20260916"
            record = read_json(folder / "manifest.json")
            assert record["status"] == "complete"
            for name,digest in record["outputs"].items():
                assert sha256(folder / name) == digest
            baseline = pd.read_parquet(BASE / "global9" / fold / "candidate.parquet")
            indexed = meta.set_index(prior.ID).loc[baseline[prior.ID]]
            missing = ~np.isfinite(indexed.proxy_sec.to_numpy(float))
            for variant in ["candidate","blend25"]:
                pred = pd.read_parquet(folder / f"{variant}.parquet")
                np.testing.assert_array_equal(pred[prior.ID],baseline[prior.ID])
                np.testing.assert_array_equal(pred[prior.TARGET],indexed[prior.TARGET])
                np.testing.assert_array_equal(pred.prediction_sec[~missing],baseline.prediction_sec[~missing])
                y = pred[prior.TARGET].to_numpy(float)
                p = pred.prediction_sec.to_numpy(float)
                assert np.isfinite(p).all()
                metrics.append({"arm":arm,"fold":fold,"variant":variant,"n":len(pred),"missing_n":int(missing.sum()),
                    "mse":float(np.mean((y-p)**2)),"rmse":float(np.sqrt(np.mean((y-p)**2))),
                    "missing_rmse":float(np.sqrt(np.mean((y[missing]-p[missing])**2))),"protected_finite_equal":True})
            if arm == "control":
                previous = old / f"historical_template_{fold}_s20260916"
                old_tune = pd.read_parquet(previous / "tune_predictions.parquet")
                new_tune = pd.read_parquet(folder / "tune_predictions.parquet")
                pd.testing.assert_frame_equal(old_tune,new_tune,check_exact=True)
                old_score = pd.read_parquet(previous / "candidate.parquet")
                new_score = pd.read_parquet(folder / "candidate.parquet")
                np.testing.assert_array_equal(old_score.prediction_sec[missing],new_score.prediction_sec[missing])
                receipts.append({"fold":fold,"control_tune_exact":True,"control_missing_score_exact":True})
    results = pd.DataFrame(metrics)
    results.to_csv(destination / "metrics.csv",index=False)
    seasonal = []
    for (arm,variant), group in results.groupby(["arm","variant"]):
        value = float(np.sqrt(sum(weights[r.fold]*r.mse for r in group.itertuples())))
        seasonal.append({"arm":arm,"variant":variant,"seasonal_rmse":value,"delta_vs_global9":value-272.2639669175})
    write_json(destination / "summary.json",{"created_utc":utc_now(),"status":"passed","source_sha256":sha256(__file__),
        "protocol_sha256":sha256(OUT / "protocol.json"),"completion_sha256":sha256(OUT / "completion.json"),
        "control_receipts":receipts,"seasonal":seasonal,"folds":metrics,
        "interpretation":"Lexical feature ablation only, not independent model selection. Report all variants. Finite routes protected; original control exact on tune and missing score."})
    print(results.to_string(index=False))
    print(pd.DataFrame(seasonal).to_string(index=False))


if __name__ == "__main__":
    main()
