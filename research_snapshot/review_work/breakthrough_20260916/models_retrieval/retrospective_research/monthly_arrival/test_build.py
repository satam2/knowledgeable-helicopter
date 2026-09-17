"""Synthetic monthly ARR oracle, boundaries, raw duration and Arrow tests."""
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import build as b


def oracle(queries, arrivals):
    q = queries[b.QUERY_COLUMNS].copy()
    a = arrivals[b.ARR_COLUMNS].copy().drop_duplicates()
    if not a[b.PHASE].eq('ARR').all() or not a[b.ID].is_unique:
        raise ValueError('oracle input')
    a = a.sort_values(b.ID)
    keep, seen = [], set()
    for i, row in a.iterrows():
        key = (row[b.FLIGHT_ID], row.ADES_mvt, row[b.MOVEMENT])
        if pd.notna(row[b.FLIGHT_ID]):
            if key in seen:
                continue
            seen.add(key)
        keep.append(i)
    a = a.loc[keep]
    values = (pd.to_datetime(a[b.BLOCK], utc=True) - pd.to_datetime(a[b.MOVEMENT], utc=True)).dt.total_seconds()
    result = []
    for _, row in q.iterrows():
        cells = []
        for scope in b.GROUPS:
            allowed = a.ADES_mvt.eq(row.ADEP_mvt) & a[b.MOVEMENT].dt.strftime('%Y-%m').eq(row[b.MOVEMENT].strftime('%Y-%m'))
            allowed &= a[b.ID].ne(row[b.ID]) & np.isfinite(values)
            if pd.notna(row[b.FLIGHT_ID]):
                allowed &= a[b.FLIGHT_ID].isna() | a[b.FLIGHT_ID].ne(row[b.FLIGHT_ID])
            if scope != 'airport':
                field = 'STAND_mvt' if scope == 'stand' else 'RUNWAY_mvt'
                allowed &= a[field].notna() & a[field].eq(row[field]) if pd.notna(row[field]) else False
            x = values.loc[allowed].to_numpy(float)
            cells.extend([len(x), x.mean(), *np.quantile(x, [.5, .1, .9], method='linear')] if len(x) else [0, np.nan, np.nan, np.nan, np.nan])
        result.append(cells)
    return pd.DataFrame(result, index=pd.Index(q[b.ID], name=b.ID), columns=b.FEATURES, dtype='float32')


def fixture():
    rng = np.random.default_rng(20260916)
    start = pd.Timestamp('2025-07-01', tz='UTC')
    arrivals = pd.DataFrame({b.ID: np.arange(100, 400, dtype=float), b.FLIGHT_ID: rng.integers(0, 60, 300).astype(float),
        b.PHASE: 'ARR', b.MOVEMENT: start + pd.to_timedelta(rng.integers(-24, 800, 300), unit='h'),
        'ADES_mvt': rng.choice(['A', 'B'], 300), 'STAND_mvt': rng.choice(['S1', 'S2', None], 300),
        'RUNWAY_mvt': rng.choice(['R1', 'R2', None], 300)})
    arrivals[b.BLOCK] = arrivals[b.MOVEMENT] + pd.to_timedelta(rng.integers(-900, 20000, 300), unit='s')
    arrivals.loc[:9, b.FLIGHT_ID] = np.nan
    arrivals.loc[10, b.BLOCK] = pd.NaT
    queries = pd.DataFrame({b.ID: np.arange(1, 61, dtype=float), b.FLIGHT_ID: np.arange(60, dtype=float), b.PHASE: 'DEP',
        b.MOVEMENT: start + pd.to_timedelta(rng.integers(0, 30 * 24, 60), unit='h'),
        'ADEP_mvt': rng.choice(['A', 'B', 'C'], 60), 'STAND_mvt': rng.choice(['S1', 'S2', None], 60),
        'RUNWAY_mvt': rng.choice(['R1', 'R2', None], 60)})
    queries.loc[:9, b.FLIGHT_ID] = np.nan
    queries.loc[0, b.ID] = arrivals.loc[0, b.ID]
    duplicate = arrivals.iloc[[20]].copy()
    duplicate[b.ID] = 1000.
    duplicate[b.BLOCK] += pd.Timedelta(days=300)
    arrivals = pd.concat([arrivals, duplicate, arrivals.iloc[[21]]], ignore_index=True)
    return queries, arrivals


def main():
    checks = 0
    for n in range(1, 9):
        values = np.arange(n, dtype=float) ** 2 - 5
        for k in range(n + 1):
            for removed in combinations(range(n), k):
                retained = np.delete(values, removed)
                expected = [len(retained), retained.mean(), *np.quantile(retained, [.5, .1, .9])] if len(retained) else [0, np.nan, np.nan, np.nan, np.nan]
                np.testing.assert_allclose(b.summarize(values, removed), expected, equal_nan=True)
                checks += 1
    queries, arrivals = fixture()
    for kind, column in [('query', b.ID), ('query', b.MOVEMENT), ('arrival', b.ID)]:
        bad_q, bad_a = queries.copy(), arrivals.copy()
        changed = bad_q if kind == 'query' else bad_a
        changed.loc[changed.index[0], column] = pd.NaT if column == b.MOVEMENT else np.nan
        try:
            b.build(bad_q, bad_a)
            raise AssertionError('Null movement identity accepted')
        except ValueError as error:
            assert 'Require' in str(error)
    actual = b.build(queries, arrivals)
    pd.testing.assert_frame_equal(actual, oracle(queries, arrivals), atol=.0005, rtol=1e-6)
    assert np.array_equal(actual.index, queries[b.ID]) and actual.shape == (60, 15)
    poisoned = queries.copy()
    poisoned[b.BLOCK] = pd.Timestamp('2100-01-01', tz='UTC')
    poisoned['TAXITIME_SEC_mvt'] = -1e12
    extra = arrivals.copy()
    extra['TAXITIME_SEC_mvt'] = 1e12
    pd.testing.assert_frame_equal(actual, b.build(poisoned, extra))
    conflict = arrivals.iloc[[0]].copy()
    conflict[b.BLOCK] += pd.Timedelta(seconds=1)
    try:
        b.build(queries, pd.concat([arrivals, conflict]))
        raise AssertionError('Conflicting sameID accepted')
    except ValueError as error:
        assert 'Conflicting' in str(error)
    bad = arrivals.copy()
    bad.loc[0, b.PHASE] = 'DEP'
    try:
        b.build(queries, bad)
        raise AssertionError('DEPblockaccepted')
    except ValueError as error:
        assert 'ARR-only' in str(error)
    q = queries.iloc[[1]].copy()
    q[b.FLIGHT_ID], q['ADEP_mvt'], q['STAND_mvt'], q['RUNWAY_mvt'] = np.nan, 'A', 'S1', 'R1'
    a = arrivals.iloc[:2].copy()
    a[b.FLIGHT_ID], a['ADES_mvt'], a['STAND_mvt'], a['RUNWAY_mvt'] = np.nan, 'A', 'S1', 'R1'
    a[b.MOVEMENT] = pd.Timestamp('2025-07-31T23:59:59Z')
    a[b.BLOCK] = a[b.MOVEMENT] + pd.to_timedelta([-7, 100000], unit='s')
    edge = b.build(q, a)
    assert edge.iloc[0].monthly_arr_airport_count == 2 and edge.iloc[0].monthly_arr_airport_mean_sec == np.float32(49996.5)
    a[b.MOVEMENT] += pd.offsets.MonthBegin(1)
    assert b.build(q, a).iloc[0].monthly_arr_airport_count == 0
    missing_first = arrivals.iloc[[20]].copy()
    missing_first[b.ID] = -10
    missing_first[b.BLOCK] = pd.NaT
    suppressed = b.prepare_arrivals(pd.concat([arrivals, missing_first], ignore_index=True))
    assert not suppressed[b.ID].isin([120, 1000, -10]).any()
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'fixture.parquet'
        dep = queries.copy()
        dep[b.BLOCK] = pd.Timestamp('2200-01-01', tz='UTC')
        dep['TAXITIME_SEC_mvt'] = -1e12
        pd.concat([dep, arrivals], ignore_index=True).to_parquet(path, index=False)
        loaded_q = b.load_queries(path)
        loaded_a = b.load_arrivals([path], ['2025-07'])
        assert b.BLOCK not in loaded_q and 'TAXITIME_SEC_mvt' not in loaded_a
        assert loaded_a[b.PHASE].eq('ARR').all()
        pd.testing.assert_frame_equal(b.build(loaded_q, loaded_a), actual)
    print(f'PASS {checks} exactquantile exclusion cases;900oraclecells; negative/rawtail,month/future,missinggroup/flight,selfID,dedup,hiddenpoison,ARRArrowboundary,IDorder')


if __name__ == '__main__':
    main()
