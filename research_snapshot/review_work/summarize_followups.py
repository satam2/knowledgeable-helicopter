"""Verified inventory of all completed local experiments, including adaptive follow-ups."""

import json
import numpy as np
import pandas as pd

from run_screening import OUT, WORKSPACE
from taxiout.artifacts import read_json, sha256, write_json
from taxiout.metrics import paired_stability, scores, season_score
from taxiout.schema import ID, TARGET


def summarize():
    report, frames = {}, {}
    protocol = read_json(OUT / "protocol.json")
    runner_hashes = {sha256(p) for p in (WORKSPACE / "review_work").glob("*.py")}
    for folder in ["results", "refinements", "combinations"]:
        for path in sorted((OUT / folder).glob("*.json")):
            m = read_json(path)
            if m["status"] != "complete":
                raise ValueError("Incomplete published result")
            if m["source_hashes"] != protocol["source_hashes"]:
                raise ValueError("Candidate source differs from the frozen experiment source")
            if "runner_sha256" in m and m["runner_sha256"] not in runner_hashes:
                raise ValueError("Experimental runner changed since the model was trained")
            name = path.stem.rsplit("_" + m["fold"], 1)[0]
            suffix = path.stem.split("_" + m["fold"], 1)[1]
            name += suffix
            run = OUT / "models" / m["run_id"]
            for file, digest in m["outputs"].items():
                if sha256(run / file) != digest:
                    raise ValueError("Artifact checksum mismatch")
            frame = pd.read_parquet(run / "score_predictions.parquet")
            computed = scores(frame[TARGET], frame.prediction_sec)
            if not np.isclose(computed["rmse_sec"], m["metrics"]["overall"]["rmse_sec"], rtol=0, atol=1e-9):
                raise ValueError("Metric mismatch")
            base = read_json(OUT / "results" / f"baseline_{m['fold']}.json")
            if m["split"]["split_hash"] != base["split"]["split_hash"]:
                raise ValueError("Split mismatch")
            b = pd.read_parquet(OUT / "models" / base["run_id"] / "score_predictions.parquet")
            if not np.array_equal(frame[ID], b[ID]) or not np.array_equal(frame[TARGET], b[TARGET]):
                raise ValueError("ID/target mismatch")
            entry = {"run_id": m["run_id"], "overall": computed, "iterations": m["iterations"],
                     "reload_delta": m["reload_max_abs_delta"], "runtime_sec": m["runtime_sec"],
                     "paired_vs_baseline": paired_stability(b, frame)}
            missing = frame.proxy_status.eq("missing")
            rome = missing & frame.ADEP_mvt.eq("LIRF")
            for label, mask in {"missing": missing, "observed": ~missing, "rome_missing": rome,
                                "rome_ordinary": rome & frame[TARGET].le(7200), "rome_tail": rome & frame[TARGET].gt(7200)}.items():
                entry[label] = scores(frame.loc[mask, TARGET], frame.loc[mask, "prediction_sec"]) if mask.any() else None
            report.setdefault(name, {"folds": {}})["folds"][m["fold"]] = entry
            frames[(name, m["fold"])] = frame
    for name, value in report.items():
        folds = value["folds"]
        if all(f in folds for f in ["F1", "F3"]):
            value["seasonal_rmse_sec"] = season_score(folds["F1"]["overall"], folds["F3"]["overall"])
            base_score = season_score(report["baseline"]["folds"]["F1"]["overall"], report["baseline"]["folds"]["F3"]["overall"])
            value["gain_vs_baseline_sec"] = base_score - value["seasonal_rmse_sec"]
            value["mse_reduction_pct"] = 100 * (1 - (value["seasonal_rmse_sec"] / base_score) ** 2)
            observed = []
            for f in ["F1", "F3"]:
                m = folds[f]
                observed.append({"n": m["overall"]["n"], "sse": m["observed"]["sse"]})
            value["perfect_missing_floor_sec"] = season_score(*observed)
            regression = all(folds[f]["overall"]["rmse_sec"] - report["baseline"]["folds"][f]["overall"]["rmse_sec"] <= max(5, .02 * report["baseline"]["folds"][f]["overall"]["rmse_sec"]) for f in ["F1", "F3"])
            missing_ok = all(folds[f]["missing"]["rmse_sec"] <= 1.05 * report["baseline"]["folds"][f]["missing"]["rmse_sec"] for f in ["F1", "F3"])
            value["screening_gates"] = {"useful_gain": value["gain_vs_baseline_sec"] >= max(2, .005 * base_score),
                                        "fold_regression": regression, "missing_regression": missing_ok,
                                        "each_fold_gain_survives_day_removal": all(folds[f]["paired_vs_baseline"]["leave_one_day_out_delta_range"][1] < 0 for f in ["F1", "F3"])}
        value["has_F2_G1"] = all(f in folds for f in ["F2", "G1"])
    output = {"candidates": report, "completed_fold_evaluations": sum(len(v["folds"]) for v in report.values()),
              "target_sec": 230, "caveat": "Adaptive development evidence. No new independent holdout or leaderboard labels."}
    write_json(OUT / "followup_summary.json", output)
    print(json.dumps({k: {"seasonal": round(v["seasonal_rmse_sec"], 3) if "seasonal_rmse_sec" in v else None,
                         "folds": {f: round(m["overall"]["rmse_sec"], 3) for f, m in v["folds"].items()},
                         "gates": v.get("screening_gates")} for k,v in report.items()}, indent=2))
    return output


if __name__ == "__main__":
    summarize()
