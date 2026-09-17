"""Export aggregate scores and qualified caveats for review, without flight records."""

import csv

from run_screening import OUT
from taxiout.artifacts import read_json

summary = read_json(OUT / "followup_summary.json")
columns = ["candidate", "fold", "rmse_sec", "baseline_delta_sec", "rows", "missing_rmse_sec",
           "observed_rmse_sec", "rome_ordinary_rmse_sec", "rome_tail_rmse_sec", "runtime_sec",
           "leave_day_delta_min", "leave_day_delta_max", "run_id"]
with (OUT / "all_experiment_scores.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for name, candidate in summary["candidates"].items():
        for fold, m in candidate["folds"].items():
            day = m["paired_vs_baseline"]["leave_one_day_out_delta_range"]
            writer.writerow({"candidate": name, "fold": fold, "rmse_sec": m["overall"]["rmse_sec"],
                             "baseline_delta_sec": m["paired_vs_baseline"]["delta_rmse_second_minus_first"],
                             "rows": m["overall"]["n"], "missing_rmse_sec": m["missing"]["rmse_sec"],
                             "observed_rmse_sec": m["observed"]["rmse_sec"],
                             "rome_ordinary_rmse_sec": m["rome_ordinary"]["rmse_sec"] if m["rome_ordinary"] else None,
                             "rome_tail_rmse_sec": m["rome_tail"]["rmse_sec"] if m["rome_tail"] else None,
                             "runtime_sec": m["runtime_sec"], "leave_day_delta_min": day[0],
                             "leave_day_delta_max": day[1], "run_id": m["run_id"]})
print(OUT / "all_experiment_scores.csv")
