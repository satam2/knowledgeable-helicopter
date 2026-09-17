"""Small synthetic CPU allocation measurements; never trains on competition rows."""
import adapter
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import psutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    if args.rows > 100000:
        raise ValueError("Synthetic feasibility probe limited to100k rows")
    rng = np.random.default_rng(20260916)
    process = psutil.Process()
    initial = process.memory_info().rss
    x = pd.DataFrame(rng.normal(size=(args.rows, 19)).astype("float32"), columns=[f"numeric{i}" for i in range(19)])
    for i in range(11):
        x[f"category{i}"] = pd.Categorical(rng.integers(0, 100, size=args.rows).astype(str))
    y = x.numeric0.to_numpy() * 300 + np.maximum(x.numeric1.to_numpy(), 0) * 100
    x.loc[x.index[::100], "numeric2"] = np.nan
    adapter.LINEAR_TREE = True
    model, evidence = adapter.fit(x, y, steps=10, threads=2)
    pred = adapter.predict(model, x.iloc[:1000])
    assert np.isfinite(pred).all()
    peak = getattr(process.memory_info(), "peak_wset", process.memory_info().rss)
    record = {"synthetic_only": True, "rows": args.rows, "numeric_features": 19, "categorical_features": 11,
              "trees": 10, "initial_rss_bytes": initial, "peak_rss_bytes": peak,
              "peak_minus_initial_bytes": peak - initial, "fit_seconds": evidence["runtime_sec"],
              "model_bytes": len(model["estimator"].booster_.model_to_string().encode()),
              "planning_estimate": evidence["memory_estimate"], "finite_predictions": True}
    out = adapter.ROOT / "private_runs/breakthrough_20260916/models/linear_tree/feasibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"synthetic_{args.rows}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
