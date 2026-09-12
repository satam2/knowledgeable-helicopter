import numpy as np
import pandas as pd

from taxiout.schema import ID, MOVEMENT, TARGET, align, utc_period_strings


def scores(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape or y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require aligned, finite, nonempty vectors")
    error = prediction - y
    sse = float(np.dot(error, error))
    return {"n": len(y), "sse": sse, "rmse_sec": float(np.sqrt(sse / len(y))),
            "mae_sec": float(np.mean(np.abs(error))), "bias_sec": float(error.mean()),
            "p95_abs_error_sec": float(np.quantile(np.abs(error), .95)), "coverage": 1.0}


def pooled(metrics):
    n, sse = sum(m["n"] for m in metrics), sum(m["sse"] for m in metrics)
    return {"n": n, "sse": sse, "rmse_sec": float(np.sqrt(sse / n))}


def season_score(summer, winter, month_counts=None):
    counts = month_counts or {"2026-01": 152719, "2026-07": 192122}
    if set(counts) != {"2026-01", "2026-07"} or any(n <= 0 for n in counts.values()):
        raise ValueError("Ranking seasons changed; review the seasonal selection protocol")
    total = sum(counts.values())
    return float(np.sqrt(counts["2026-07"] / total * summer["sse"] / summer["n"] + counts["2026-01"] / total * winter["sse"] / winter["n"]))


def evaluate(predictions, labels):
    frame = align(predictions, labels, [TARGET])
    overall = scores(frame[TARGET], frame["prediction_sec"])
    frame["error_sec"] = frame["prediction_sec"].astype(float) - frame[TARGET].astype(float)
    frame["squared_error"] = frame["error_sec"] ** 2
    frame["label_bin"] = pd.cut(frame[TARGET], [-np.inf, 0, 1800, 3600, 7200, np.inf],
                                right=False, labels=["<0", "0-30m", "30-60m", "60-120m", ">120m"]).astype(str)
    # Exact 120 minutes belongs to the 60-120m slice.
    frame.loc[frame[TARGET].eq(7200), "label_bin"] = "60-120m"
    if MOVEMENT in frame:
        frame["month"] = utc_period_strings(frame[MOVEMENT], "M")
        frame["day"] = utc_period_strings(frame[MOVEMENT], "D")
    slices = {}
    for col in ["ADEP_mvt", "month", "proxy_status", "route", "new_category", "label_bin"]:
        if col in frame:
            slices[col] = {}
            for value, group in frame.groupby(col, observed=True, dropna=False):
                metric = scores(group[TARGET], group["prediction_sec"])
                metric["sse_share"] = metric["sse"] / overall["sse"] if overall["sse"] else 0.0
                slices[col][str(value)] = metric
    return {"overall": overall, "serialized": scores(frame[TARGET], np.rint(frame["prediction_sec"])), "slices": slices}, frame


def paired_stability(first, second, seed=20260910, repetitions=1000):
    joined = align(first[[ID, "day", "ADEP_mvt", "squared_error"]], second[[ID, "squared_error"]].rename(columns={"squared_error": "other_sse"}))
    blocks = joined.groupby(["ADEP_mvt", "day"], observed=True).agg(a=("squared_error", "sum"), b=("other_sse", "sum"), n=(ID, "size"))
    values = blocks.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        changes.append(float(np.sqrt(sample[1] / sample[2]) - np.sqrt(sample[0] / sample[2])))
    totals = values.sum(axis=0)
    daily = joined.groupby("day").agg(a=("squared_error", "sum"), b=("other_sse", "sum"), n=(ID, "size"))
    leave = totals - daily.to_numpy(dtype=float)
    leave_delta = np.sqrt(leave[:, 1] / leave[:, 2]) - np.sqrt(leave[:, 0] / leave[:, 2])
    return {"blocks": len(blocks), "seed": seed, "repetitions": repetitions,
            "delta_rmse_second_minus_first": float(np.sqrt(totals[1] / totals[2]) - np.sqrt(totals[0] / totals[2])),
            "bootstrap_95pct_interval": np.quantile(changes, [.025, .975]).tolist(),
            "leave_one_day_out_delta_range": [float(leave_delta.min()), float(leave_delta.max())],
            "largest_error_day": str(daily.a.idxmax()),
            "largest_error_day_sse_share": float(daily.a.max() / totals[0]),
            "caveat": "Airport-day block stability diagnostic; extreme events limit interval reliability."}
