"""Calibration-only convex weights with a predeclared equal-weight shrinkage."""
import numpy as np
from scipy.optimize import minimize


def fit_mixture(predictions, labels):
    p, y = np.asarray(predictions, float), np.asarray(labels, float)
    if p.ndim != 2 or p.shape[1] != 3 or y.shape != (len(p),) or not len(p):
        raise ValueError('Expected nonempty aligned three-expert calibration')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Calibration contains nonfinite values')
    equal = np.full(3, 1/3)
    residual = p - y[:, None]
    gram = residual.T @ residual / len(y)
    scale = max(float(np.trace(gram)), 1.)
    def objective(w):
        return float(w @ gram @ w / scale), 2 * gram @ w / scale
    result = minimize(objective, equal, jac=True, method='SLSQP',
        bounds=[(0., 1.)]*3, constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1,
        'jac': lambda w: np.ones(3)}], options={'ftol': 1e-12, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(result.message)
    simplex = np.clip(result.x, 0., 1.)
    simplex /= simplex.sum()
    return {'equal': equal.tolist(), 'simplex': simplex.tolist(),
            'shrunk': (.5*simplex+.5*equal).tolist(), 'calibration_rows': len(y)}


def predict_mixture(state, predictions, variant):
    p = np.asarray(predictions, float)
    w = np.asarray(state[variant], float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite three-expert predictions')
    if w.shape != (3,) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid convex weights')
    return p @ w
