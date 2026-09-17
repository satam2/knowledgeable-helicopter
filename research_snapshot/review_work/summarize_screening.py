"""Verify screening artifacts and compare complete, ID-aligned score cohorts."""

import argparse
import json

import numpy as np
import pandas as pd

from run_screening import CONFIGS, OUT, RAW, REPO, WORKSPACE, config_for
from taxiout.artifacts import object_hash, read_json, sha256, source_hashes, utc_now, write_json
from taxiout.metrics import paired_stability, scores, season_score
from taxiout.schema import ID, TARGET


def subset_metrics(frame, mask):
    return scores(frame.loc[mask, TARGET], frame.loc[mask, "prediction_sec"]) if mask.any() else None


def summarize(partial=False):
    protocol = read_json(OUT / "protocol.json")
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Training source changed during the screen")
    records, frames = {}, {}
    for candidate in CONFIGS:
        records[candidate], frames[candidate] = {}, {}
        for fold in ["F1", "F3"]:
            file = OUT / "results" / f"{candidate}_{fold}.json"
            if not file.exists():
                if partial:
                    continue
                raise ValueError(f"Missing evaluation: {candidate} {fold}")
            record = read_json(file)
            if record["status"] != "complete" or record["config_hash"] != object_hash(config_for(candidate)):
                raise ValueError("Incomplete or incompatible result")
            if record["source_hashes"] != protocol["source_hashes"] or record["reload_max_abs_delta"] > 1e-9:
                raise ValueError("Source or serialization parity failure")
            run = OUT / "models" / record["run_id"]
            for name, digest in record["outputs"].items():
                if sha256(run / name) != digest:
                    raise ValueError(f"Output checksum failure: {candidate} {fold} {name}")
            frame = pd.read_parquet(run / "score_predictions.parquet")
            expected_n = {"F1": 190713, "F3": 162332}[fold]
            if len(frame) != expected_n or not frame[ID].is_unique or not np.isfinite(frame.prediction_sec).all():
                raise ValueError("Incomplete, duplicated or nonfinite score cohort")
            recomputed = scores(frame[TARGET], frame.prediction_sec)
            if not np.isclose(recomputed["rmse_sec"], record["metrics"]["overall"]["rmse_sec"], atol=1e-9, rtol=0):
                raise ValueError("Saved predictions do not reproduce RMSE")
            records[candidate][fold], frames[candidate][fold] = record, frame
    report = {"created_utc": utc_now(), "target_sec": 230, "intermediate_target_sec": 280,
              "baseline_folds_complete": len(records["baseline"]),
              "candidate_folds_complete": sum(len(v) for k, v in records.items() if k != "baseline"),
              "status": "complete" if all(len(v) == 2 for v in records.values()) else "partial",
              "candidates": {}, "privacy": {"raw_external": str(RAW), "artifacts_external": str(OUT)},
              "interpretation": "Local July/November development screens; not a 2026 leaderboard score."}
    historical = read_json(WORKSPACE / "knowledgeable-helicopter/reports/comparison.json")["candidates"]["residual_long_proxy_specialist"]
    for candidate, folds in records.items():
        result = {"folds": {}}
        for fold, record in folds.items():
            frame = frames[candidate][fold]
            missing = frame.proxy_status.eq("missing")
            rome = frame.ADEP_mvt.eq("LIRF") & missing
            result["folds"][fold] = {"run_id": record["run_id"], "overall": record["metrics"]["overall"],
                                      "missing": subset_metrics(frame, missing), "observed": subset_metrics(frame, ~missing),
                                      "rome_missing": subset_metrics(frame, rome),
                                      "rome_ordinary": subset_metrics(frame, rome & frame[TARGET].le(7200)),
                                      "rome_tail": subset_metrics(frame, rome & frame[TARGET].gt(7200)),
                                      "unseen": subset_metrics(frame, frame.new_category),
                                      "iterations": record["iterations"], "runtime_sec": record["runtime_sec"],
                                      "peak_rss_gib": record["peak_rss_bytes"] / 2**30,
                                      "serialization_delta": record["reload_max_abs_delta"]}
            if candidate == "baseline":
                result["folds"][fold]["historical_delta_sec"] = record["metrics"]["overall"]["rmse_sec"] - historical["folds"][fold]["rmse_sec"]
            elif fold in records["baseline"]:
                base_record, base = records["baseline"][fold], frames["baseline"][fold]
                if record["split"]["split_hash"] != base_record["split"]["split_hash"]:
                    raise ValueError("Split membership mismatch")
                if not np.array_equal(frame[ID], base[ID]) or not np.array_equal(frame[TARGET], base[TARGET]):
                    raise ValueError("Score IDs or labels differ")
                result["folds"][fold]["paired"] = paired_stability(base, frame)
                result["folds"][fold]["delta_sec"] = record["metrics"]["overall"]["rmse_sec"] - base_record["metrics"]["overall"]["rmse_sec"]
                if candidate == "D_rome":
                    changed_route = rome & frame.schedule_sec.notna()
                    if not np.array_equal(frame.loc[~changed_route, "prediction_sec"], base.loc[~changed_route, "prediction_sec"]):
                        raise ValueError("Rome experiment modified predictions outside its declared route")
                    result["folds"][fold]["unaffected_predictions_exactly_equal"] = True
        if len(folds) == 2:
            result["seasonal_rmse_sec"] = season_score(result["folds"]["F1"]["overall"], result["folds"]["F3"]["overall"])
        report["candidates"][candidate] = result
    baseline = report["candidates"]["baseline"]
    if "seasonal_rmse_sec" in baseline:
        base_score = baseline["seasonal_rmse_sec"]
        report["required_mse_reduction_pct"] = 100 * (1 - (230 / base_score) ** 2)
        for name, result in report["candidates"].items():
            if name == "baseline" or "seasonal_rmse_sec" not in result:
                continue
            gain = base_score - result["seasonal_rmse_sec"]
            min_gain = max(2, .005 * base_score)
            fold_ok = all(result["folds"][f]["delta_sec"] <= max(5, .02 * baseline["folds"][f]["overall"]["rmse_sec"]) for f in ["F1", "F3"])
            missing_ok = all(result["folds"][f]["missing"]["rmse_sec"] <= 1.05 * baseline["folds"][f]["missing"]["rmse_sec"] for f in ["F1", "F3"])
            leave_gains = []
            weights = {"F1": 192122 / 344841, "F3": 152719 / 344841}
            for fold in ["F1", "F3"]:
                b, c = frames["baseline"][fold], frames[name][fold]
                days = pd.DataFrame({"day": b.day, "b": b.squared_error, "c": c.squared_error}).groupby("day").agg(b=("b", "sum"), c=("c", "sum"), n=("b", "size"))
                other = "F3" if fold == "F1" else "F1"
                bm = baseline["folds"][other]["overall"]["rmse_sec"] ** 2
                cm = result["folds"][other]["overall"]["rmse_sec"] ** 2
                for row in days.itertuples():
                    bs = np.sqrt(weights[fold] * (b.squared_error.sum() - row.b) / (len(b) - row.n) + weights[other] * bm)
                    cs = np.sqrt(weights[fold] * (c.squared_error.sum() - row.c) / (len(c) - row.n) + weights[other] * cm)
                    leave_gains.append(float(bs - cs))
            result.update(gain_sec=gain, mse_reduction_pct=100 * (1 - (result["seasonal_rmse_sec"] / base_score) ** 2),
                          seasonal_leave_one_day_out_gain_range=[min(leave_gains), max(leave_gains)],
                          gates={"useful_gain": gain >= min_gain, "fold_regression": fold_ok,
                                 "missing_regression": missing_ok, "gain_survives_every_day_removal": min(leave_gains) > 0})
            result["advance_to_F2_G1"] = all(result["gates"].values())
    if report["status"] == "complete":
        actual = {p.name: sha256(p) for p in RAW.glob("*.parquet")}
        report["privacy"]["all_14_raw_hashes_unchanged"] = actual == protocol["raw_hashes"]
        if not report["privacy"]["all_14_raw_hashes_unchanged"]:
            raise ValueError("Private input pack changed")
        leaks = [str(p) for repo in [REPO, WORKSPACE / "knowledgeable-helicopter"] for suffix in ["*.parquet", "*.cbm"] for p in repo.rglob(suffix)]
        report["privacy"]["data_or_models_in_checkouts"] = leaks
        if leaks:
            raise ValueError("Private artifact discovered inside a checkout")
    write_json(OUT / "screening_summary.json", report)
    print(json.dumps({"status": report["status"], "screenings": report["candidate_folds_complete"],
                      "scores": {k: {"seasonal": v.get("seasonal_rmse_sec"), "F1": v["folds"].get("F1", {}).get("overall", {}).get("rmse_sec"),
                                       "F3": v["folds"].get("F3", {}).get("overall", {}).get("rmse_sec"),
                                       "advance": v.get("advance_to_F2_G1"), "gates": v.get("gates")} for k, v in report["candidates"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    summarize(parser.parse_args().partial)
