"""Two-expert raw-MSE simplex with predeclared airport shrinkage."""
import numpy as np

SHRINKAGE_ROWS = 1000.


def solve(y, first, second, airports):
    y, first, second = [np.asarray(value, dtype=np.float64) for value in (y, first, second)]
    airports = np.asarray(airports).astype(str)
    if not len(y) or any(value.shape != y.shape for value in (first, second, airports)):
        raise ValueError("Stacking inputs must be aligned nonempty vectors")
    if not all(np.isfinite(value).all() for value in (y, first, second)):
        raise ValueError("Original labels and component predictions must be finite")
    d = second - first
    denominator = float(d @ d)
    global_alpha = float(np.clip(d @ (y - first) / denominator, 0., 1.)) if denominator > 0 else 0.
    variance = denominator / len(y)
    penalty = SHRINKAGE_ROWS * variance
    local = {}
    for airport in sorted(np.unique(airports)):
        choose = airports == airport
        local_d = d[choose]
        local_denominator = float(local_d @ local_d)
        if local_denominator + penalty > 0:
            alpha = float(np.clip((local_d @ (y[choose] - first[choose]) + penalty * global_alpha) / (local_denominator + penalty), 0., 1.))
        else:
            alpha = global_alpha
        local[airport] = {"alpha": alpha, "rows": int(choose.sum()), "disagreement_sse": local_denominator,
                          "effective_disagreement_rows": float(local_denominator ** 2 / np.sum(local_d ** 4)) if local_denominator else 0.}
    return {"global_alpha": global_alpha, "airports": local, "shrinkage_rows": SHRINKAGE_ROWS,
            "global_mean_squared_disagreement": variance, "penalty": penalty, "rows": len(y),
            "effective_disagreement_rows": float(denominator ** 2 / np.sum(d ** 4)) if denominator else 0.,
            "top5_disagreement_share": float(np.sort(d * d)[-5:].sum() / denominator) if denominator else 0.}


def predict(model, first, second, airports, conditioned=False):
    first, second = [np.asarray(value, dtype=np.float64) for value in (first, second)]
    airports = np.asarray(airports).astype(str)
    if first.shape != second.shape or first.shape != airports.shape:
        raise ValueError("Prediction vectors are misaligned")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Component predictions must be finite")
    alpha = np.array([model["airports"].get(airport, {"alpha": model["global_alpha"]})["alpha"] for airport in airports]) if conditioned else np.full(len(first), model["global_alpha"])
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("Frozen simplex coefficients are invalid")
    result = first + alpha * (second - first)
    if np.any(result < np.minimum(first, second) - 1e-8) or np.any(result > np.maximum(first, second) + 1e-8):
        raise AssertionError("Prediction escaped component convex hull")
    return result
