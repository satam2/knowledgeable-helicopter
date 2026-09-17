"""Observation-only extensions; frozen taxiout inference sources stay unchanged."""

import numpy as np
import pandas as pd

from taxiout.availability import assert_observations
from taxiout.features.pipeline import token
from taxiout.features.traffic import nanoseconds
from taxiout.schema import MOVEMENT, PHASE, utc


def runway_features(departures, context):
    assert_observations(departures)
    assert_observations(context)
    event_ns, query_ns = nanoseconds(context[MOVEMENT]), nanoseconds(departures[MOVEMENT])
    event_keys = pd.DataFrame({
        'airport': np.where(context[PHASE].eq('DEP'), context.ADEP_mvt, context.ADES_mvt),
        'runway': context.RUNWAY_mvt.astype('string').to_numpy(),
        'phase': context[PHASE].to_numpy(),
        'month': (context[MOVEMENT].dt.year * 12 + context[MOVEMENT].dt.month).to_numpy()})
    query_keys = pd.DataFrame({
        'airport': departures.ADEP_mvt.to_numpy(),
        'runway': departures.RUNWAY_mvt.astype('string').to_numpy(),
        'month': (departures[MOVEMENT].dt.year * 12 + departures[MOVEMENT].dt.month).to_numpy()})
    runway_groups = {key: np.sort(event_ns[pos]) for key, pos in
                     event_keys.groupby(['airport', 'runway', 'phase', 'month'], observed=True).indices.items()}
    airport_groups = {key: np.sort(event_ns[pos]) for key, pos in
                      event_keys.groupby(['airport', 'phase', 'month'], observed=True).indices.items()}
    queries = query_keys.groupby(['airport', 'runway', 'month'], observed=True).indices
    result = pd.DataFrame(index=departures.index)
    result['rw_missing'] = departures.RUNWAY_mvt.isna().astype('float32')
    empty = np.empty(0, dtype=np.int64)
    for phase in ['DEP', 'ARR']:
        name = phase.lower()
        ages = np.full(len(departures), -1., dtype='float32')
        counts = {w: np.full(len(departures), -1., dtype='float32') for w in [5, 15, 60]}
        shares = {w: np.full(len(departures), -1., dtype='float32') for w in [5, 15, 60]}
        for (airport, runway, month), pos in queries.items():
            times = runway_groups.get((airport, runway, phase, month), empty)
            airport_times = airport_groups.get((airport, phase, month), empty)
            query = query_ns[pos]
            before = np.searchsorted(times, query, side='left')
            has_prior = before > 0
            if has_prior.any():
                ages[pos[has_prior]] = (query[has_prior] - times[before[has_prior] - 1]) / 1e9
            for width in counts:
                start = query - pd.Timedelta(minutes=width).value
                count = before - np.searchsorted(times, start, side='left')
                total = np.searchsorted(airport_times, query, side='left') - np.searchsorted(airport_times, start, side='left')
                counts[width][pos] = count
                shares[width][pos] = np.divide(count, total, out=np.zeros(len(pos), dtype=float), where=total > 0)
        result[f'rw_{name}_age_sec'] = ages
        for width in counts:
            result[f'rw_{name}_prior_{width}m'] = counts[width]
            result[f'rw_{name}_share_{width}m'] = shares[width]
    ts = departures[MOVEMENT]
    month_start = pd.to_datetime(ts.dt.tz_convert(None).to_numpy().astype('datetime64[M]'), utc=True)
    age = (ts - month_start).dt.total_seconds()
    for width in [5, 15, 60]:
        result[f'rw_observed_{width}m_sec'] = np.minimum(age, width * 60).astype('float32')
    return result


def schedule_features(dep):
    assert_observations(dep)
    result = pd.DataFrame(index=dep.index)
    schedule = utc(dep.SCHED_TIME_UTC_mvt)
    movement = utc(dep[MOVEMENT], required=True)
    result['flight_designator'] = token(dep.FLIGHT_mvt).astype('category')
    prefix = dep.FLIGHT_mvt.astype('string').str.extract(r'^([A-Z]{2,3})(?=[0-9])', expand=False)
    result['flight_prefix'] = token(prefix).astype('category')
    for suffix, values in [('flight', dep.FLIGHT_mvt), ('flight_prefix', prefix)]:
        result['airport_' + suffix] = (token(dep.ADEP_mvt) + token(values)).astype('category')
    result['airport_stand_runway'] = (token(dep.ADEP_mvt) + token(dep.STAND_mvt) + token(dep.RUNWAY_mvt)).astype('category')
    result['schedule_hour'] = schedule.dt.hour
    result['schedule_minute'] = schedule.dt.minute
    result['schedule_weekday'] = schedule.dt.dayofweek
    result['schedule_day_offset'] = (movement.dt.normalize() - schedule.dt.normalize()).dt.total_seconds() / 86400
    result['schedule_missing'] = schedule.isna().astype(float)
    for col in result.select_dtypes('number'):
        result[col] = result[col].fillna(-999999).astype('float32')
    return result


def designator_support(x):
    return sorted(set(x.flight_designator.astype(str)) - {'m:'})


def known_designator(x, support):
    return x.flight_designator.astype(str).isin(support).to_numpy()


def fit_blend(y, residual, direct, airports, shrinkage=1000):
    y, residual, direct = [np.asarray(a, dtype=float) for a in [y, residual, direct]]
    if not all(np.isfinite(a).all() for a in [y, residual, direct]) or not len(y):
        raise ValueError('Blend calibration requires finite nonempty tuning predictions')
    delta = direct - residual
    denominator = float(delta @ delta)
    global_weight = float(np.clip(delta @ (y - residual) / denominator, 0, 1)) if denominator else 0.
    airports = np.asarray(airports).astype(str)
    weights, support = {}, {}
    for airport in np.unique(airports):
        use = airports == airport
        local_denominator = float(delta[use] @ delta[use])
        local = float(np.clip(delta[use] @ (y[use] - residual[use]) / local_denominator, 0, 1)) if local_denominator else global_weight
        count = int(use.sum())
        weights[airport] = (count * local + shrinkage * global_weight) / (count + shrinkage)
        support[airport] = count
    return {'global': global_weight, 'airports': weights, 'support': support, 'shrinkage': shrinkage}


def blend_weights(gate, airports, conditioned=True):
    return np.array([gate['airports'].get(str(a), gate['global']) if conditioned else gate['global'] for a in airports])


def apply_blend(original, direct, routes, weights):
    result = np.asarray(original, dtype=float).copy()
    eligible = np.asarray(routes) == 'residual'
    result[eligible] += np.asarray(weights)[eligible] * (np.asarray(direct)[eligible] - result[eligible])
    return result
