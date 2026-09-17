"""One pooled ridge coefficient with fixed airport nuisance means and day diagnostics."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SEED = 20260916


def fit_calibration(frame):
    linked = frame.loc[frame.linked].copy()
    if len(linked) < 2:
        raise ValueError('Insufficient linked June calibration rows')
    errors = linked.raw_target_sec.to_numpy(float) - linked.baseline_prediction_sec.to_numpy(float)
    linked['error'] = errors
    means = linked.groupby('ADEP_mvt', observed=True)[['error', 'delta_sec']].mean()
    global_error = float(errors.mean())
    global_delta = float(linked.delta_sec.mean())
    scale = float(linked.delta_sec.std(ddof=0))
    scale = scale if scale > 0 else 1.
    x = (linked.delta_sec.to_numpy(float) - linked.ADEP_mvt.map(means.delta_sec).to_numpy()) / scale
    y = errors - linked.ADEP_mvt.map(means.error).to_numpy()
    model = Ridge(alpha=100., fit_intercept=False, solver='cholesky').fit(x[:, None], y)
    slope = float(model.coef_[0])
    identity = float(np.dot(x, y) / (np.dot(x, x) + 100.))
    np.testing.assert_allclose(slope, identity, rtol=1e-12, atol=1e-12)
    return dict(coefficient=slope, ridge_alpha=100., delta_global_mean=global_delta,
        delta_scale=scale, global_error_mean=global_error,
        airport_error_mean=means.error.to_dict(), airport_delta_mean=means.delta_sec.to_dict(),
        calibration_rows=len(frame), linked_calibration_rows=len(linked),
        no_outcome_clipping=True, coefficient_closed_form=identity)


def predictions(frame, calibration, delta=None):
    baseline = frame.baseline_prediction_sec.to_numpy(float)
    linked = frame.linked.to_numpy(bool)
    airport_error = frame.ADEP_mvt.map(calibration['airport_error_mean']).fillna(calibration['global_error_mean']).to_numpy(float)
    airport_delta = frame.ADEP_mvt.map(calibration['airport_delta_mean']).fillna(calibration['delta_global_mean']).to_numpy(float)
    observed = frame.delta_sec.to_numpy(float) if delta is None else np.asarray(delta, float)
    x = np.where(linked, (observed - airport_delta) / calibration['delta_scale'], 0.)
    bias = baseline + np.where(linked, airport_error, 0.)
    augmented = bias + np.where(linked, calibration['coefficient'] * x, 0.)
    assert np.isfinite(bias).all() and np.isfinite(augmented).all()
    np.testing.assert_array_equal(bias[~linked], baseline[~linked])
    np.testing.assert_array_equal(augmented[~linked], baseline[~linked])
    return bias, augmented


def scores(y, pred):
    e = np.asarray(pred, float) - np.asarray(y, float)
    if not len(e):
        return dict(n=0, sse=None, rmse=None, mae=None, bias=None)
    return dict(n=len(e), sse=float(np.dot(e, e)), rmse=float(np.sqrt(np.mean(e**2))),
                mae=float(np.abs(e).mean()), bias=float(e.mean()))


def paired(y, candidate, reference, days):
    y, candidate, reference = map(lambda x:np.asarray(x, float), [y, candidate, reference])
    a, b = (candidate-y)**2, (reference-y)**2
    codes, unique = pd.factorize(days, sort=True)
    if len(unique) < 2:
        raise ValueError('Day uncertainty needs at least two distinct days')
    n = np.bincount(codes)
    sums = np.bincount(codes, weights=b-a)
    draws = np.random.default_rng(SEED).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=300)
    bootstrap = (draws @ sums) / (draws @ n)
    removals = []
    for k, day in enumerate(unique):
        keep = codes != k
        removals.append(dict(day=str(day), gain=float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))))
    influence = {}
    order = np.argsort(-(b-a), kind='stable')
    for count in [1, 5, 10]:
        keep = np.ones(len(y), bool)
        keep[order[:min(count, len(y)-1)]] = False
        influence[str(count)] = float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))
    return dict(candidate=scores(y, candidate), reference=scores(y, reference),
        gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())), mse_gain=float((b-a).mean()),
        mse_gain_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
        day_removals=removals, all_day_removals_improve=all(r['gain']>0 for r in removals),
        remove_top_beneficial_rows_gain=influence)


def permutation_null(frame, calibration):
    linked = frame.linked.to_numpy(bool)
    indices = np.flatnonzero(linked)
    subset = frame.loc[linked].copy()
    subset['day'] = subset.MVT_TIME_UTC_mvt.dt.floor('D')
    keys = ['ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_OPERATOR_flt', 'day']
    groups = list(subset.groupby(keys, observed=True, dropna=False, sort=True).indices.values())
    rng = np.random.default_rng(SEED)
    delta = frame.delta_sec.to_numpy(float)
    y = frame.raw_target_sec.to_numpy(float)
    bias, actual = predictions(frame, calibration)
    records = []
    for repetition in range(25):
        shuffled = delta.copy()
        for group in groups:
            rows = indices[group]
            shuffled[rows] = delta[rng.permutation(rows)]
        _, candidate = predictions(frame, calibration, shuffled)
        changed = int(np.count_nonzero(shuffled[indices] != delta[indices]))
        records.append(dict(repetition=repetition, changed_linked_rows=changed,
            unchanged_linked_share=float(1-changed/max(len(indices), 1)),
            mse_gain_vs_bias=float(np.mean((bias-y)**2-(candidate-y)**2)),
            rmse_gain_vs_bias=float(np.sqrt(np.mean((bias-y)**2))-np.sqrt(np.mean((candidate-y)**2)))))
    actual_gain = float(np.mean((bias-y)**2-(actual-y)**2))
    return dict(strata=keys, seed=SEED, repetitions=25, linked_rows=len(indices),
        singleton_rows=sum(len(g) for g in groups if len(g)==1),
        singleton_share=sum(len(g) for g in groups if len(g)==1)/max(len(indices), 1),
        actual_mse_gain_vs_bias=actual_gain,
        permutation_gains_ge_actual=sum(r['mse_gain_vs_bias']>=actual_gain for r in records),
        interpretation='Fixed exploratory null; no coefficient refit, model selection or calibrated significance claim', records=records)
