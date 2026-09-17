"""Check whether observed timestamp precision distinguishes Rome schedule regimes."""

import numpy as np
import pandas as pd

from run_screening import RAW, OUT
from taxiout.artifacts import write_json


parts = []
columns = ["PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
for path in sorted(RAW.glob("training_*.parquet")):
    d = pd.read_parquet(path, columns=columns)
    d = d.loc[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()].copy()
    schedule = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
    movement = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    d["precision"] = np.where(schedule.isna(), "absent", np.where(schedule.dt.second.ne(0), "subminute", np.where(schedule.dt.minute.mod(5).ne(0), "whole_minute", "five_minutes")))
    d["offset"] = (movement - schedule).dt.total_seconds()
    d["error"] = d.offset - d.TAXITIME_SEC_mvt
    d["period"] = np.where(movement.dt.month.le(6), "F1_refit", np.where(movement.dt.month.eq(7), "F1_score", np.where(movement.dt.month.le(10), "Aug_Oct", np.where(movement.dt.month.eq(11), "F3_score", "December_exposed"))))
    parts.append(d)
frame = pd.concat(parts)
rows = []
for (period, precision), group in frame.groupby(["period", "precision"], observed=True):
    good = group.error.notna()
    rows.append({"period": period, "precision": precision, "n": len(group),
                 "long_target_pct": float(group.TAXITIME_SEC_mvt.gt(7200).mean() * 100),
                 "schedule_rmse": float(np.sqrt(np.mean(group.loc[good, "error"] ** 2))) if good.any() else None,
                 "within_300_pct": float(group.loc[good, "error"].abs().le(300).mean() * 100) if good.any() else None})
write_json(OUT / "schedule_precision_probe.json", rows)
for row in rows:
    print(row)
