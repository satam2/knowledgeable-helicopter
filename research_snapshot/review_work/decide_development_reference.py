"""Compare the strongest route composition to its direct predecessor on every fold."""

import argparse
import json
import numpy as np
import pandas as pd

from run_screening import OUT
from taxiout.artifacts import read_json, write_json
from taxiout.metrics import paired_stability, season_score
from taxiout.schema import ID, TARGET

parser = argparse.ArgumentParser()
parser.add_argument("--reference", default="B_plus_D")
parser.add_argument("--candidate", default="FB_plus_D")
args = parser.parse_args()
summary = read_json(OUT / "followup_summary.json")["candidates"]
reference, candidate = args.reference, args.candidate
comparisons = {}
for fold in ["F1", "F2", "F3", "G1"]:
    first = summary[reference]["folds"][fold]
    second = summary[candidate]["folds"][fold]
    a = pd.read_parquet(OUT / "models" / first["run_id"] / "score_predictions.parquet")
    b = pd.read_parquet(OUT / "models" / second["run_id"] / "score_predictions.parquet")
    if not np.array_equal(a[ID], b[ID]) or not np.array_equal(a[TARGET], b[TARGET]):
        raise ValueError("Incompatible score cohorts")
    outside_residual = ~a.route.isin(["residual", "residual_long_proxy"])
    if not np.array_equal(a.loc[outside_residual, "prediction_sec"], b.loc[outside_residual, "prediction_sec"]):
        raise ValueError("Residual upgrade changed another route")
    comparisons[fold] = paired_stability(a, b)
ref_score = summary[reference]["seasonal_rmse_sec"]
new_score = summary[candidate]["seasonal_rmse_sec"]
gates = {"minimum_gain": ref_score - new_score >= max(2, .005 * ref_score),
         "F1_F3_nonregression": all(comparisons[f]["delta_rmse_second_minus_first"] <= 0 for f in ["F1", "F3"]),
         "F2_G1_nonregression": all(comparisons[f]["delta_rmse_second_minus_first"] <= 0 for f in ["F2", "G1"]),
         "each_screen_gain_survives_day_removal": all(comparisons[f]["leave_one_day_out_delta_range"][1] < 0 for f in ["F1", "F3"]),
         "missing_and_direct_fallback_routes_unchanged": True}
report = {"reference": reference, "candidate": candidate, "gates": gates, "comparisons": comparisons,
          "seasonal_gain_sec": ref_score - new_score, "seasonal_rmse_sec": new_score,
          "recommended_next_development_reference": candidate if all(gates.values()) else reference,
          "status": "development reference only; no final all-year fit or leaderboard score",
          "target_230_reached_locally": new_score <= 230, "intermediate_280_reached_locally": new_score <= 280,
          "open_checks": ["Main residual model seed stability", "Future-year generalization", "Ordinary Rome schedule-head tradeoff"],
          "seed_head_only": "The Rome head was checked at three fixed seeds; other components were held fixed."}
write_json(OUT / f"decision_{candidate}.json", report)
write_json(OUT / "champion_decision.json", report)
print(json.dumps(report, indent=2))
