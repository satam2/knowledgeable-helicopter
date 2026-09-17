"""Synthetic availability contracts; run with the campaign Python and -B."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
sys.path.insert(0, str(HERE))
from taxiout.availability import make_observations
from taxiout.schema import BLOCK, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET


def rows():
    result = pd.DataFrame({
        ID: [1, 2, 3, 4, 5, 6], FLIGHT_ID: [11, 22, 33, 44, 55, 66],
        PHASE: ['ARR', 'ARR', 'ARR', 'ARR', 'DEP', 'DEP'],
        MOVEMENT: ['2025-07-01 09:30', '2025-07-01 09:40', '2025-07-01 09:50',
                   '2025-07-01 09:55', '2025-07-01 10:00', '2025-07-01 10:05'],
        BLOCK: ['2025-07-01 09:40', '2025-07-01 09:55', '2025-07-01 10:00',
                '2025-07-01 10:10', '2025-07-01 09:40', '2025-07-01 09:45'],
        'ADEP_mvt': ['X', 'X', 'X', 'X', 'AAA', 'AAA'],
        'ADES_mvt': ['AAA', 'AAA', 'AAA', 'AAA', 'X', 'X'],
        'RUNWAY_mvt': ['01'] * 6, 'STAND_mvt': ['A1'] * 6,
        'WK_TBL_CAT_flt': ['H', 'M', 'M', 'H', 'M', 'H'],
        'AOBT_3_flt': ['2025-07-01 09:20'] * 6,
        'SCHED_TIME_UTC_mvt': ['2025-07-01 09:20'] * 6,
        'EOBT_1_flt': ['2025-07-01 09:20'] * 6,
        'IOBT_flt': ['2025-07-01 09:20'] * 6,
        'LOBT_flt': ['2025-07-01 09:20'] * 6,
    })
    delta = (pd.to_datetime(result[BLOCK]) - pd.to_datetime(result[MOVEMENT])).dt.total_seconds()
    result[TARGET] = np.where(result[PHASE].eq('ARR'), delta, -delta)
    return result


class ImplementationExists(unittest.TestCase):
    def test_feature_implementation_exists(self):
        self.assertTrue((HERE / 'arrival_features.py').is_file(), 'Arrival implementation is missing')


@unittest.skipUnless((HERE / 'arrival_features.py').exists(), 'Implementation not written yet')
class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        import arrival_features
        self.module = arrival_features

    def build(self, raw, selection=None):
        observations, _, _ = make_observations(raw)
        dep = observations.loc[observations[PHASE].eq('DEP')]
        if selection is not None:
            dep = dep.loc[dep[ID].isin(selection)]
        arrivals = self.module.extract_completed_arrivals(raw)
        return self.module.aviation_features(dep, observations, arrivals)

    def test_completion_ties_and_support(self):
        x = self.build(rows())
        self.assertEqual(x.loc[5, 'arr_airport_count_30m'], 2)
        self.assertEqual(x.loc[5, 'arr_airport_mean_sec_30m'], 750)
        self.assertEqual(x.loc[6, 'arr_airport_count_30m'], 3)
        self.assertEqual(x.loc[5, 'surface_arrivals_open_120m'], 2)

    def test_hidden_departure_columns_do_not_change_features(self):
        raw = rows()
        original = self.build(raw)
        raw.loc[raw[PHASE].eq('DEP'), TARGET] = -1234567
        raw.loc[raw[PHASE].eq('DEP'), BLOCK] = '2099-01-01'
        pd.testing.assert_frame_equal(original, self.build(raw))

    def test_future_completed_arrival_mutation_is_invisible(self):
        raw = rows()
        original = self.build(raw)
        raw.loc[raw[ID].eq(4), BLOCK] = '2025-07-01 11:10'
        raw.loc[raw[ID].eq(4), TARGET] = 4500
        pd.testing.assert_frame_equal(original, self.build(raw))

    def test_future_movement_mutation_is_invisible(self):
        raw = rows()
        original = self.build(raw, [5])
        raw.loc[raw[ID].eq(6), 'RUNWAY_mvt'] = '99'
        raw.loc[raw[ID].eq(6), 'WK_TBL_CAT_flt'] = 'J'
        pd.testing.assert_frame_equal(original, self.build(raw, [5]))

    def test_same_flight_counterpart_excluded(self):
        raw = rows()
        raw.loc[raw[ID].eq(5), FLIGHT_ID] = 11
        x = self.build(raw)
        self.assertEqual(x.loc[5, 'arr_airport_count_30m'], 1)
        self.assertEqual(x.loc[5, 'arr_airport_mean_sec_30m'], 900)

    def test_duplicate_arrival_event_not_double_counted(self):
        raw = rows()
        duplicate = raw.iloc[[0]].copy()
        duplicate[ID] = 101
        x = self.build(pd.concat([raw, duplicate], ignore_index=True))
        self.assertEqual(x.loc[5, 'arr_airport_count_30m'], 2)

    def test_duplicate_movement_id_rejected(self):
        raw = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            self.build(raw)

    def test_month_boundary_isolation(self):
        raw = rows()
        raw.loc[raw[ID].eq(1), MOVEMENT] = '2025-06-30 23:55'
        raw.loc[raw[ID].eq(1), TARGET] = 35100
        x = self.build(raw)
        self.assertEqual(x.loc[5, 'arr_airport_count_30m'], 1)

    def test_departure_self_and_ties_excluded(self):
        x = self.build(rows())
        self.assertEqual(x.loc[5, 'seq_dep_prior_15m'], 0)
        self.assertEqual(x.loc[6, 'seq_dep_prior_15m'], 1)

    def test_missing_and_negative_arrivals_preserve_query_cohort(self):
        raw = rows()
        raw.loc[raw[ID].eq(1), BLOCK] = None
        raw.loc[raw[ID].eq(1), TARGET] = np.nan
        raw.loc[raw[ID].eq(2), BLOCK] = '2025-07-01 09:30'
        raw.loc[raw[ID].eq(2), TARGET] = -600
        x = self.build(raw)
        self.assertEqual(x.index.tolist(), [5, 6])
        self.assertEqual(x.loc[5, 'arr_airport_count_30m'], 0)
        self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_clock_precision_and_missing_flags(self):
        raw = rows()
        raw.loc[raw[ID].eq(5), 'AOBT_3_flt'] = None
        x = self.build(raw)
        self.assertEqual(x.loc[5, 'precision_aobt_missing'], 1)
        self.assertEqual(x.loc[6, 'precision_aobt_minute_aligned'], 1)

    def test_reference_never_uses_same_or_future_month(self):
        raw = rows().loc[lambda x: x[PHASE].eq('DEP')].copy()
        raw = pd.concat([raw.assign(**{ID: [15,16], MOVEMENT: ['2025-06-01 00:00','2025-06-02 00:00'], TARGET:[600.,800.]}), raw], ignore_index=True)
        obs, labels, _ = make_observations(raw)
        first = self.module.chronological_reference(obs, labels, min_support=1)
        labels.loc[labels[ID].isin([5,6]), TARGET] = 999999
        second = self.module.chronological_reference(obs, labels, min_support=1)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.loc[5, 'physical_reference_q10_sec'], 620)
        self.assertEqual(first.loc[15, 'physical_reference_history_n'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
