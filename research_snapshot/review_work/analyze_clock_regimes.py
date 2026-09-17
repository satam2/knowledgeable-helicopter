"""Find observable error strata to guide the next training batch."""

import json
import sys
import numpy as np
import pandas as pd

from run_screening import OUT, config_for
from taxiout.artifacts import read_json, write_json
from taxiout.cache import load_training
from taxiout.io import verified_manifest
from taxiout.schema import ID


config = config_for("baseline")
x, _, _ = load_training(config, verified_manifest(config))
rows = []
candidate = sys.argv[1] if len(sys.argv) > 1 else "F_plus_D"
summary = read_json(OUT / "followup_summary.json")["candidates"][candidate]
for fold in ["F1", "F3"]:
    m = summary["folds"][fold]
    frame = pd.read_parquet(OUT / "models" / m["run_id"] / "score_predictions.parquet")
    xf = x.loc[frame[ID]]
    gap = xf.nm_actual_minus_estimated.to_numpy()
    clock_regime = np.select([frame.proxy_status.eq("missing"), gap == -999999,
                              np.abs(gap) <= 300, np.abs(gap) <= 1800],
                             ["NM absent", "estimate absent", "actual-estimated <=5m", "actual-estimated 5-30m"],
                             default="actual-estimated >30m")
    frame["clock_regime"] = clock_regime
    frame["proxy_regime"] = np.select([frame.proxy_status.eq("missing"), frame.proxy_sec.lt(0), frame.proxy_sec.le(1800), frame.proxy_sec.le(3600)],
                                      ["absent", "negative", "0-30m", "30-60m"], default=">60m")
    for feature in ["clock_regime", "proxy_regime"]:
        for value, g in frame.groupby(feature, observed=True):
            rows.append({"fold": fold, "feature": feature, "group": value, "n": len(g),
                         "row_pct": 100 * len(g) / len(frame), "rmse": float(np.sqrt(g.squared_error.mean())),
                         "sse_pct": 100 * float(g.squared_error.sum() / frame.squared_error.sum()),
                         "bias": float(g.error_sec.mean())})
write_json(OUT / "clock_regimes.json", rows)
print(json.dumps(rows, indent=2))
