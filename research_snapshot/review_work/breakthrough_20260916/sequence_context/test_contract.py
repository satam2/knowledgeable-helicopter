"""Independent examples for the ordered-context observation boundary."""
import unittest
import numpy as np
import pandas as pd
import cache


def raw_fixture():
    times = pd.to_datetime(['2025-07-01T00:30Z'] * 10, utc=True)
    frame = pd.DataFrame({cache.ID: np.arange(10), cache.FLIGHT: np.arange(100, 110),
        cache.PHASE: ['DEP'] * 5 + ['ARR'] * 5, cache.MOVEMENT: times,
        'ADEP_mvt': ['AAAA'] * 10, 'ADES_mvt': ['AAAA'] * 10,
        'RUNWAY_mvt': ['01'] * 10, 'STAND_mvt': ['A1'] * 10,
        'AIRCRAFT_TYPE_mvt': ['A320'] * 10, 'WK_TBL_CAT_flt': ['M'] * 10,
        'AIRCRAFT_OPERATOR_flt': ['XYZ'] * 10})
    for name in cache.CLOCKS:
        frame[name] = times - pd.Timedelta(minutes=10)
    frame.loc[0, cache.MOVEMENT] = pd.Timestamp('2025-07-01T01:00Z')
    frame.loc[1, cache.MOVEMENT] = pd.Timestamp('2025-07-01T00:59Z')
    frame.loc[2, cache.MOVEMENT] = pd.Timestamp('2025-07-01T01:00Z')
    frame.loc[3, cache.FLIGHT] = frame.loc[0, cache.FLIGHT]
    frame.loc[4, cache.MOVEMENT] = pd.Timestamp('2025-06-30T23:59Z')
    completions = pd.DataFrame({cache.ID: np.arange(5, 10), cache.BLOCK: pd.to_datetime([
        '2025-07-01T00:55Z', '2025-07-01T01:00Z', '2025-07-01T01:01Z',
        '2025-07-01T00:58Z', '2025-07-01T00:20Z'], utc=True)})
    frame.loc[8, cache.FLIGHT] = frame.loc[0, cache.FLIGHT]
    return frame, completions


class ContextBoundary(unittest.TestCase):
    def test_completion_tie_month_and_sameflight(self):
        raw, arr = raw_fixture()
        events = cache.events_from_public(raw, arr)
        query = cache.query_frame(raw.iloc[[0]])
        idx = cache.neighbor_indices(events, query)
        selected = events.iloc[idx[0][idx[0] >= 0]]
        self.assertEqual(selected[cache.ID].tolist(), [5, 1])
        self.assertTrue((selected.time_ns < query.time_ns.iloc[0]).all())
        self.assertTrue((selected.month == query.month.iloc[0]).all())

    def test_hidden_departure_columns_cannot_change_tokens(self):
        raw, arr = raw_fixture()
        original = cache.events_from_public(raw, arr)
        raw[cache.BLOCK] = pd.Timestamp('1900-01-01', tz='UTC')
        raw['TAXITIME_SEC_mvt'] = np.arange(len(raw)) * 99999
        changed = cache.events_from_public(raw, arr)
        pd.testing.assert_frame_equal(original, changed)
        self.assertNotIn(cache.BLOCK, changed)
        self.assertNotIn('TAXITIME_SEC_mvt', changed)

    def test_nearest16_each_phase_and_no_labels(self):
        raw, arr = raw_fixture()
        raw = pd.concat([raw.iloc[[1]].assign(**{cache.ID: 1000+i, cache.FLIGHT: 2000+i,
            cache.MOVEMENT: pd.Timestamp('2025-07-01T00:59Z')-pd.Timedelta(seconds=i)}) for i in range(24)])
        events = cache.events_from_public(raw, arr.iloc[:0])
        query = cache.query_frame(raw.iloc[[0]].assign(**{cache.ID: 9999, cache.FLIGHT: 9999,
            cache.MOVEMENT: pd.Timestamp('2025-07-01T01:00Z')}))
        idx = cache.neighbor_indices(events, query)[0]
        self.assertEqual(len(idx[idx >= 0]), 16)
        self.assertEqual(events.iloc[idx[idx >= 0]][cache.ID].tolist(), list(range(1015, 999, -1)))
        self.assertTrue((idx[16:] == -1).all())

    def test_duplicate_event_and_missing_flight_identity(self):
        raw, arr = raw_fixture()
        duplicate = raw.iloc[[1]].copy()
        duplicate[cache.ID] = 888
        raw = pd.concat([raw, duplicate], ignore_index=True)
        events = cache.events_from_public(raw, arr)
        self.assertNotIn(888, events[cache.ID].tolist())
        raw.loc[raw[cache.ID].isin([1, 888]), cache.FLIGHT] = np.nan
        events = cache.events_from_public(raw, arr)
        self.assertIn(888, events[cache.ID].tolist())


if __name__ == '__main__':
    unittest.main()
