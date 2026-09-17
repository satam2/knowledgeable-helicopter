"""Small constrained mixtures optimized for the original mixture squared error."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

PENALTY = .001
MAX_ITERATIONS = 150


def constant_weights(y, predictions, center=None, penalty=0.):
    y = np.asarray(y, float)
    p = np.asarray(predictions, float)
    start = np.full(p.shape[1], 1 / p.shape[1]) if center is None else np.asarray(center)
    relative = p - p[:, :1]
    target = y - p[:, 0]
    scale = max(float(np.mean(target ** 2)), 1.)

    def objective(weights):
        errors = relative @ weights - target
        delta = weights - start
        return (float(errors @ errors + penalty * (delta @ delta)) / (len(y) * scale),
                (2 * relative.T @ errors + 2 * penalty * delta) / (len(y) * scale))

    result = minimize(objective, start, jac=True, method='SLSQP', bounds=[(0., 1.)] * p.shape[1],
                      constraints={'type': 'eq', 'fun': lambda w: w.sum()-1., 'jac': lambda w: np.ones_like(w)},
                      options={'maxiter': 200, 'ftol': 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0.)
    return weights / weights.sum()


def airport_weights(y, predictions, airports):
    global_w = constant_weights(y, predictions)
    spread = np.mean(np.sum((predictions - predictions.mean(1, keepdims=True)) ** 2, axis=1))
    penalty = 1000. * spread
    local = {}
    for airport in sorted(set(airports)):
        ids = np.asarray(airports) == airport
        local[airport] = constant_weights(np.asarray(y)[ids], predictions[ids], global_w, penalty)
    return global_w, local, penalty


def objective(parameters, design, predictions, labels, target_scale, penalty=PENALTY):
    coefficients = parameters.reshape(design.shape[1], predictions.shape[1])
    weights = softmax(design @ coefficients, axis=1)
    relative = predictions - predictions[:, :1]
    predicted = np.sum(weights * relative, axis=1)
    errors = predicted - (labels - predictions[:, 0])
    loss = np.mean(errors ** 2) / target_scale + penalty * np.sum(coefficients[1:] ** 2)
    gradient_logits = (2 / len(labels) / target_scale) * errors[:, None] * weights * (relative - predicted[:, None])
    gradient = design.T @ gradient_logits
    gradient[1:] += 2 * penalty * coefficients[1:]
    return float(loss), gradient.ravel()


def frame_with_disagreement(frame, predictions):
    result = frame.copy()
    for i in range(1, predictions.shape[1]):
        result[f'expert_{i}_minus_0_sec'] = predictions[:, i] - predictions[:, 0]
    for name in result:
        if pd.api.types.is_numeric_dtype(result[name]):
            result[name] = result[name].replace([np.inf, -np.inf, -999999], np.nan)
        else:
            result[name] = result[name].astype('string').fillna('MISSING').astype(str)
    return result


def fit(frame, predictions, labels):
    frame = frame_with_disagreement(frame, predictions)
    numeric = [name for name in frame if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in frame if name not in numeric]
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler()), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)], sparse_threshold=0.)
    raw = transform.fit_transform(frame)
    design = np.column_stack([np.ones(len(raw)), raw])
    initial_weights = constant_weights(labels, predictions)
    initial = np.zeros((design.shape[1], predictions.shape[1]))
    initial[0] = np.log(np.maximum(initial_weights, 1e-4))
    scale = max(float(np.mean((predictions @ initial_weights - labels) ** 2)), 1.)
    result = minimize(objective, initial.ravel(), args=(design, predictions, np.asarray(labels), scale), jac=True,
                      method='L-BFGS-B', options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-10, 'gtol': 1e-7})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise FloatingPointError('Nonfinite contextual mixture optimization')
    return {'transform': transform, 'coefficients': result.x.reshape(initial.shape), 'target_scale': scale}, {
        'optimizer_success': bool(result.success), 'optimizer_message': str(result.message), 'iterations': int(result.nit),
        'objective': float(result.fun), 'design_columns': design.shape[1], 'initial_weights': initial_weights.tolist(),
        'l2_penalty': PENALTY, 'max_iterations': MAX_ITERATIONS}


def predict(model, frame, predictions):
    raw = model['transform'].transform(frame_with_disagreement(frame, predictions))
    weights = softmax(np.column_stack([np.ones(len(raw)), raw]) @ model['coefficients'], axis=1)
    result = np.sum(weights * predictions, axis=1)
    if not np.isfinite(result).all() or np.any(result < predictions.min(1)-1e-8) or np.any(result > predictions.max(1)+1e-8):
        raise AssertionError('Mixture must remain inside finite expert convex hull')
    return result, weights
