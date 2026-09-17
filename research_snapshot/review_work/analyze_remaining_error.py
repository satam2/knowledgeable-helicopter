"""Aggregate remaining error without reproducing flight-level data."""

import json
import numpy as np
import pandas as pd

from run_screening import OUT
from taxiout.artifacts import read_json, write_json
from taxiout.metrics import scores
from taxiout.predict import load_bundle
from taxiout.schema import TARGET


def analyze(name):
    source = read_json(OUT / "followup_summary.json")["candidates"][name]
    budget = {}
    feature_importance = {}
    for fold, weight in [("F1", 192122 / 344841), ("F3", 152719 / 344841)]:
        path = OUT / "models" / source["folds"][fold]["run_id"]
        frame = pd.read_parquet(path / "score_predictions.parquet")
        for key, part in frame.groupby(["ADEP_mvt", "proxy_status"], observed=True):
            label = "/".join(map(str, key))
            record = budget.setdefault(label, {"weighted_mse": 0., "folds": {}})
            record["weighted_mse"] += weight * part.squared_error.sum() / len(frame)
            record["folds"][fold] = scores(part[TARGET], part.prediction_sec)
        _, models, _ = load_bundle(path)
        feature_importance[fold] = {}
        for component in ["residual", "missing", "rome_schedule"]:
            if component in models:
                importance = models[component].get_feature_importance()
                order = np.argsort(importance)[-10:][::-1]
                feature_importance[fold][component] = [{"feature": models[component].feature_names_[i], "importance": float(importance[i])} for i in order]
    total = sum(v["weighted_mse"] for v in budget.values())
    for value in budget.values():
        value["mse_share_pct"] = 100 * value["weighted_mse"] / total
    report = {"candidate": name, "seasonal_rmse_sec": float(np.sqrt(total)),
              "remaining_error": dict(sorted(budget.items(), key=lambda p: p[1]["weighted_mse"], reverse=True)),
              "feature_importance": feature_importance}
    write_json(OUT / "remaining_error.json", report)
    print(json.dumps({k: round(v["mse_share_pct"], 2) for k,v in report["remaining_error"].items()}, indent=2))


if __name__ == "__main__":
    import sys
    analyze(sys.argv[1] if len(sys.argv) > 1 else "B_plus_D")
