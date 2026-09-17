"""Month-isolated aviation features with an explicit arrival-only outcome boundary.

Prediction event is supplied takeoff time. A final NM timestamp is not proof of
real-time publication availability. No departure target or block time enters
the observation feature path. Flight counterparts and tied events are excluded.
"""

import numpy as np
import pandas as pd

from taxiout.availability import assert_observations
from taxiout.schema import BLOCK, CLOCKS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, unique_ids, utc

POLICY = 'completed-arrival-strict-prior-month-isolated-v1'
MISSING = -999999.0


def _ns(values):
    return utc(values, required=True).dt.as_unit('ns').astype('int64').to_numpy()


def _token(values):
    return values.astype('string').fillna('<missing>').astype(str).to_numpy()


def _month(values):
    values = utc(values, required=True)
    return (values.dt.year * 100 + values.dt.month).to_numpy()


def extract_completed_arrivals(raw):
    """Select ARR rows before reading either outcome column; retain incomplete rows.

    Duplicate records of the same flight/airport/landing retain the smallest ID,
    using only landing-time information. Missing flight IDs remain distinct.
    """
    unique_ids(raw)
    arr = raw.loc[raw[PHASE].eq('ARR')].copy()
    landing = utc(arr[MOVEMENT], required=True)
    completion = utc(arr[BLOCK])
    seconds = (completion - landing).dt.total_seconds()
    labels = pd.to_numeric(arr[TARGET], errors='coerce')
    comparable = labels.notna() & seconds.notna()
    if not np.array_equal(labels[comparable].to_numpy(), seconds[comparable].to_numpy()):
        raise ValueError('Arrival taxi-in target identity failed')
    result = pd.DataFrame({ID: arr[ID].to_numpy(), 'flight': _token(arr[FLIGHT_ID]),
        'airport': _token(arr.ADES_mvt), 'runway': _token(arr.RUNWAY_mvt),
        'stand': _token(arr.STAND_mvt), 'month': _month(landing),
        'landing': landing.to_numpy(), 'completion': completion.to_numpy(),
        'duration': seconds.to_numpy()})
    result.loc[result.flight.eq('<missing>'), 'flight'] = 'missing-id:' + result.loc[result.flight.eq('<missing>'), ID].astype(str)
    return result.sort_values(ID, kind='stable').drop_duplicates(['flight', 'airport', 'landing']).reset_index(drop=True)


def _query(dep):
    return pd.DataFrame({ID: dep[ID].to_numpy(), 'flight': _token(dep[FLIGHT_ID]),
        'airport': _token(dep.ADEP_mvt), 'runway': _token(dep.RUNWAY_mvt),
        'stand': _token(dep.STAND_mvt), 'month': _month(dep[MOVEMENT]),
        'time': _ns(dep[MOVEMENT])})


def _window_stats(events, queries, groups, width_minutes):
    """Exact rolling sufficient statistics, subtracting every same-flight event."""
    n = len(queries)
    sums = np.zeros((n, 4), dtype=np.float64)
    if events.empty or queries.empty:
        return sums
    event_groups = events.groupby(groups, observed=True, sort=False).indices
    width = int(width_minutes * 60 * 1e9)
    for key, qpos in queries.groupby(groups, observed=True, sort=False).indices.items():
        epos = event_groups.get(key)
        if epos is None:
            continue
        selected = events.iloc[epos].sort_values('time', kind='stable')
        times = selected.time.to_numpy(dtype=np.int64)
        values = selected.value.to_numpy(dtype=np.float64)
        matrix = np.column_stack([np.ones(len(values)), values, values ** 2, values > 1200])
        prefix = np.vstack([np.zeros(4), np.cumsum(matrix, axis=0)])
        query_time = queries.time.to_numpy()[qpos]
        hi = np.searchsorted(times, query_time, side='left')
        lo = np.searchsorted(times, query_time - width, side='left')
        sums[qpos] = prefix[hi] - prefix[lo]
    valid = queries.flight.ne('<missing>')
    pairs = queries.loc[valid, [*groups, 'flight', 'time']].assign(qpos=np.flatnonzero(valid)).merge(
        events[[*groups, 'flight', 'time', 'value']], on=[*groups, 'flight'], how='inner', suffixes=('_query', '_event'))
    if len(pairs):
        eligible = (pairs.time_event < pairs.time_query) & (pairs.time_event >= pairs.time_query - width)
        pairs = pairs.loc[eligible]
        values = pairs.value.to_numpy(dtype=np.float64)
        correction = np.column_stack([np.ones(len(values)), values, values ** 2, values > 1200])
        np.add.at(sums, pairs.qpos.to_numpy(), -correction)
    sums[:, 0] = np.maximum(sums[:, 0], 0)
    return sums


def _arrival_features(dep, arrivals):
    queries = _query(dep)
    result = {}
    # Invalid/negative taxi-in records are excluded from physical duration sums;
    # no departure labels or score rows are filtered or clipped.
    valid = arrivals.completion.notna() & arrivals.duration.ge(0) & np.isfinite(arrivals.duration)
    events = arrivals.loc[valid].copy()
    events['time'] = _ns(events.completion)
    events['value'] = events.duration.to_numpy()
    for name, keys, widths in [('airport', ['airport', 'month'], [15, 30, 60]),
                                ('runway', ['airport', 'runway', 'month'], [15, 30]),
                                ('stand', ['airport', 'stand', 'month'], [30])]:
        for width in widths:
            sums = _window_stats(events, queries, keys, width)
            count = sums[:, 0]
            mean = np.divide(sums[:, 1], count, out=np.full(len(dep), MISSING), where=count > 0)
            second = np.divide(sums[:, 2], count, out=np.zeros(len(dep)), where=count > 0)
            std = np.where(count > 0, np.sqrt(np.maximum(0, second - np.where(count > 0, mean, 0) ** 2)), MISSING)
            tail = np.divide(sums[:, 3], count, out=np.full(len(dep), MISSING), where=count > 0)
            for stat, values in [('count', count), ('mean_sec', mean), ('std_sec', std), ('over_1200_share', tail)]:
                result[f'arr_{name}_{stat}_{width}m'] = values
    landings = arrivals.copy()
    landings['time'] = _ns(landings.landing)
    landings['value'] = 0.
    # Interval endpoints implement a bounded inventory in O(n log n). Endpoints
    # are queried strictly before takeoff, so future completion values cannot
    # affect a query; expiration is exactly 120 minutes after observed landing.
    opened = _window_stats(landings, queries, ['airport', 'month'], 32 * 24 * 60)[:, 0]
    ended = landings.copy()
    ended['time'] = landings.time.to_numpy() + 7200 * 10**9
    ended.loc[valid, 'time'] = np.minimum(ended.loc[valid, 'time'], _ns(arrivals.loc[valid].completion))
    finished = _window_stats(ended, queries, ['airport', 'month'], 32 * 24 * 60)[:, 0]
    result['surface_arrivals_open_120m'] = np.maximum(0, opened - finished)
    timestamps = utc(dep[MOVEMENT], required=True)
    start = pd.to_datetime(timestamps.dt.strftime('%Y-%m-01'), utc=True)
    result['surface_month_observed_120m_sec'] = np.minimum((timestamps - start).dt.total_seconds(), 7200)
    return result


def _sequence_features(dep, context):
    queries = _query(dep)
    events = pd.DataFrame({ID: context[ID].to_numpy(), 'flight': _token(context[FLIGHT_ID]),
        'airport': np.where(context[PHASE].eq('DEP'), _token(context.ADEP_mvt), _token(context.ADES_mvt)),
        'runway': _token(context.RUNWAY_mvt), 'month': _month(context[MOVEMENT]),
        'time': _ns(context[MOVEMENT]), 'phase': _token(context[PHASE]),
        'value': context.WK_TBL_CAT_flt.astype('string').isin(['H', 'J']).to_numpy(dtype=float)})
    # Deduplication uses only event-time keys; missing flight IDs stay distinct.
    events.loc[events.flight.eq('<missing>'), 'flight'] = 'missing-id:' + events.loc[events.flight.eq('<missing>'), ID].astype(str)
    events = events.sort_values(ID).drop_duplicates(['flight', 'airport', 'time', 'phase'])
    result = {}
    for phase in ['DEP', 'ARR']:
        selected = events.loc[events.phase.eq(phase)]
        totals = _window_stats(selected, queries, ['airport', 'month'], 15)
        runway = _window_stats(selected, queries, ['airport', 'runway', 'month'], 15)
        result[f'seq_{phase.lower()}_prior_15m'] = totals[:, 0]
        result[f'seq_{phase.lower()}_runway_prior_15m'] = runway[:, 0]
        result[f'seq_{phase.lower()}_heavy_share_15m'] = np.divide(totals[:, 1], totals[:, 0], out=np.full(len(dep), MISSING), where=totals[:, 0] > 0)
    return result


def _precision_features(dep):
    result = {}
    for column, name in zip(CLOCKS, ['aobt', 'eobt', 'iobt', 'lobt', 'schedule']):
        values = utc(dep[column])
        known = values.notna()
        result[f'precision_{name}_missing'] = ~known
        result[f'precision_{name}_second'] = values.dt.second.fillna(MISSING)
        result[f'precision_{name}_minute_aligned'] = known & values.dt.second.eq(0) & values.dt.microsecond.eq(0)
        result[f'precision_{name}_five_minute_aligned'] = known & values.dt.minute.mod(5).eq(0) & values.dt.second.eq(0) & values.dt.microsecond.eq(0)
    result['precision_takeoff_second'] = utc(dep[MOVEMENT]).dt.second
    return result


def aviation_features(departures, context, arrivals):
    assert_observations(departures)
    assert_observations(context)
    if not departures[PHASE].eq('DEP').all():
        raise ValueError('Aviation queries must be departures')
    values = {**_arrival_features(departures, arrivals), **_sequence_features(departures, context), **_precision_features(departures)}
    result = pd.DataFrame({key: np.asarray(value, dtype=np.float32) for key, value in values.items()}, index=departures[ID].to_numpy())
    result.index.name = ID
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite aviation features')
    return result


class StandRunwayReference:
    """Earlier-label q10 predictive reference, with hierarchical sparse fallbacks.

    This is not EUROCONTROL's monitoring estimator or geometric route distance.
    Fit inside the permitted fold and cross-fit training features chronologically.
    """
    def __init__(self, min_support=100, shrinkage=50):
        self.min_support, self.shrinkage = min_support, shrinkage

    def fit(self, departures, labels):
        assert_observations(departures)
        unique_ids(labels)
        data = _query(departures).merge(labels[[ID, TARGET]], on=ID, validate='one_to_one')
        if len(data) != len(departures) or not np.isfinite(data[TARGET]).all():
            raise ValueError('Reference fit labels must be complete and finite')
        self.last_fit = data.time.max() if len(data) else -np.inf
        self.global_q10 = float(data[TARGET].quantile(.1)) if len(data) else 900.
        self.n = len(data)
        self.tables = []
        for keys in [['airport'], ['airport', 'runway'], ['airport', 'stand'], ['airport', 'stand', 'runway']]:
            grouped = data.groupby(keys, observed=True)[TARGET]
            table = grouped.agg(n='size').join(grouped.quantile(.1).rename('q10')).reset_index()
            self.tables.append((keys, table))
        return self

    def transform(self, departures):
        assert_observations(departures)
        queries = _query(departures)
        if len(queries) and (queries.time <= self.last_fit).any():
            raise ValueError('Reference queries overlap or precede fitted labels')
        baseline = np.full(len(queries), self.global_q10)
        support = np.zeros(len(queries))
        for keys, table in self.tables:
            matched = queries.merge(table, on=keys, how='left', validate='many_to_one')
            count = matched.n.fillna(0).to_numpy()
            eligible = count >= self.min_support
            weight = count / (count + self.shrinkage)
            baseline = np.where(eligible, weight * matched.q10.fillna(self.global_q10).to_numpy() + (1 - weight) * baseline, baseline)
            support = np.where(eligible, count, support)
        return pd.DataFrame({'physical_reference_q10_sec': baseline.astype('float32'),
                             'physical_reference_group_n': support.astype('float32'),
                             'physical_reference_history_n': np.full(len(queries), self.n, dtype='float32')},
                            index=pd.Index(departures[ID], name=ID))


def chronological_reference(departures, labels, min_support=100):
    months = _month(departures[MOVEMENT])
    chunks = []
    for month in np.unique(months):
        earlier = departures.loc[months < month]
        fit_labels = labels.loc[labels[ID].isin(earlier[ID])]
        reference = StandRunwayReference(min_support=min_support).fit(earlier, fit_labels)
        chunks.append(reference.transform(departures.loc[months == month]))
    return pd.concat(chunks).loc[departures[ID]]
