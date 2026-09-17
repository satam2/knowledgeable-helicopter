import unittest
import numpy as np
import pandas as pd
from airport_state import AirportState, stage_features, COLUMNS


def fixture():
    return pd.DataFrame({'MVT_ID_mvt': range(8), 'FLIGHT_ID_mvt': [10, 11, 12, 13, 14, 15, 16, 17],
                         'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-01-01','2025-01-08','2025-02-01','2025-02-08','2025-03-01','2025-03-08','2025-04-01','2025-04-08'], utc=True),
                         'ADEP_mvt': ['AAA'] * 8, 'TAXITIME_SEC_mvt': [100., 300., 500., 700., 900., 1100., 1300., 1500.],
                         'proxy_sec': [90., np.nan, 400., 500., 800., 900., 1000., np.nan]})


IDX = {'fit': np.arange(4), 'tune': np.arange(4, 6), 'refit': np.arange(6), 'score': np.arange(6, 8)}


class Tests(unittest.TestCase):
    def test_future_labels_and_query_proxy_do_not_enter(self):
        data = fixture()
        expected, _ = stage_features(data, IDX, 'score')
        data.loc[6:, ['TAXITIME_SEC_mvt', 'proxy_sec']] = 1e12
        actual, _ = stage_features(data, IDX, 'score')
        pd.testing.assert_frame_equal(expected, actual)

    def test_whole_month_no_teacher_forcing(self):
        data = fixture()
        expected, _ = stage_features(data, IDX, 'fit')
        data.loc[2:3, 'TAXITIME_SEC_mvt'] = -1e9
        actual, _ = stage_features(data, IDX, 'fit')
        pd.testing.assert_frame_equal(expected, actual)

    def test_flight_purge_before_aggregate(self):
        data = fixture()
        data.loc[0, 'FLIGHT_ID_mvt'] = data.loc[6, 'FLIGHT_ID_mvt']
        expected, receipt = stage_features(data, IDX, 'score')
        data.loc[0, 'TAXITIME_SEC_mvt'] = 1e15
        actual, _ = stage_features(data, IDX, 'score')
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(receipt['months'][0]['taxi_history_rows'], 5)

    def test_blackout_empty_support(self):
        actual, _ = stage_features(fixture(), IDX, 'score', history_blackout_days=182)
        np.testing.assert_array_equal(actual.state_taxi_dynamic_mean_sec, 900.)
        np.testing.assert_array_equal(actual.state_source_dynamic_mean_sec, 0.)
        np.testing.assert_array_equal(actual.state_taxi_recent_count, 0.)

    def test_subset_preserves_full_history(self):
        full, _ = stage_features(fixture(), IDX, 'score')
        subset, _ = stage_features(fixture(), IDX, 'score', query_positions=[7])
        pd.testing.assert_frame_equal(full.loc[[7]], subset)
        self.assertEqual(list(full), COLUMNS)

    def test_subset_keeps_whole_query_month_purge(self):
        data = fixture()
        data.loc[0, 'FLIGHT_ID_mvt'] = data.loc[6, 'FLIGHT_ID_mvt']
        full, _ = stage_features(data, IDX, 'score')
        subset, _ = stage_features(data, IDX, 'score', query_positions=[7])
        pd.testing.assert_frame_equal(full.loc[[7]], subset)

    def test_overlap_rejected(self):
        data = fixture()
        state = AirportState('taxi').fit(data.iloc[:2], [100, 300])
        with self.assertRaises(ValueError):
            state.transform(data.iloc[:1])

    def test_raw_negative_and_long_labels_kept(self):
        data = fixture()
        state = AirportState('taxi').fit(data.iloc[:2], [-10000, 30000])
        actual = state.transform(data.iloc[2:3])
        self.assertEqual(float(actual.state_taxi_seasonal_mean_sec.iloc[0]), 10000.)

    def test_gap_decay_uses_real_dates(self):
        data = fixture()
        state = AirportState('taxi').fit(data.iloc[:4], data.TAXITIME_SEC_mvt.iloc[:4])
        near = data.iloc[[4]].copy()
        far = near.copy()
        far['MVT_TIME_UTC_mvt'] += pd.Timedelta(days=182)
        a, b = state.transform(near), state.transform(far)
        self.assertLess(abs(float(b.state_taxi_recent_shift_sec.iloc[0])), abs(float(a.state_taxi_recent_shift_sec.iloc[0])))
        self.assertAlmostEqual(float(b.state_taxi_history_age_days.iloc[0] - a.state_taxi_history_age_days.iloc[0]), 182.)


if __name__ == '__main__':
    unittest.main()
