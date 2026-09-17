"""Contracts for missing-source formulations; no private training occurs here."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Presence(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertTrue((HERE / 'run_missing_models.py').exists())


@unittest.skipUnless((HERE / 'run_missing_models.py').exists(), 'Not implemented')
class Contracts(unittest.TestCase):
    def setUp(self):
        import run_missing_models
        self.module = run_missing_models

    def test_day_reconstruction_keeps_negative_and_extreme_labels(self):
        y = np.array([-12., 900., 70000., 87598., 131167., 180000.])
        day, within = self.module.day_parts(y)
        np.testing.assert_equal(day, [0, 0, 0, 1, 1, 2])
        np.testing.assert_equal(day * 86400 + within, y)
        self.assertEqual(within[0], -12)

    def test_expected_value_uses_probabilities_not_class_argmax(self):
        p = np.array([[.8, .2], [.2, .8]])
        result = self.module.expected_days(p, [0, 1])
        np.testing.assert_allclose(result, [.2, .8])

    def test_day_calibration_only_uses_supplied_tuning_arrays(self):
        calibration = self.module.calibrate_days(np.array([.1, .2]), np.array([1000., 1000.]), np.array([9640., 18280.]))
        self.assertAlmostEqual(calibration['scale'], 1)
        self.assertGreaterEqual(calibration['scale'], 0)

    def examples(self):
        idx = pd.Index([1, 2, 3, 4], name='MVT_ID_mvt')
        x = pd.DataFrame({'airport': ['AAA'] * 4, 'stand': ['A'] * 4, 'flight': ['XX1'] * 4,
            'schedule_bucket': ['normal'] * 4, 'schedule_hhmm': [600] * 4}, index=idx)
        times = pd.Series(pd.to_datetime(['2025-01-02', '2025-01-03', '2025-02-01', '2025-02-02'], utc=True), index=idx)
        return x, times

    def test_templates_do_not_use_same_month_labels(self):
        x, times = self.examples()
        first = self.module.crossfit_templates(x, np.array([500., 700., 999., 1111.]), times)
        second = self.module.crossfit_templates(x, np.array([500., 700., 999999., 999999.]), times)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.iloc[:2].template_mean_sec.eq(900).all())
        self.assertTrue(first.iloc[2:].template_history_n.eq(2).all())

    def test_template_transform_rejects_training_period(self):
        x, times = self.examples()
        model = self.module.HistoricalTemplate().fit(x.iloc[:2], np.array([500., 700.]), times.iloc[:2])
        with self.assertRaises(ValueError):
            model.transform(x.iloc[:1], times.iloc[:1])

    def test_categorical_unknowns_and_missing_not_numeric_stand_distances(self):
        from taxiout.schema import ID, MOVEMENT, PHASE
        raw = pd.DataFrame({ID: [1, 2], PHASE: ['DEP', 'DEP'], MOVEMENT: pd.to_datetime(['2025-01-02', '2025-01-03'], utc=True),
            'SCHED_TIME_UTC_mvt': pd.to_datetime(['2025-01-01 23:40', '2025-01-02 23:40'], utc=True),
            'ADEP_mvt': ['AAA', 'AAA'], 'ADES_mvt': ['BBB', 'CCC'], 'STAND_mvt': ['999', None],
            'RUNWAY_mvt': ['01', None], 'AIRCRAFT_TYPE_mvt': ['A320', None],
            'FLIGHT_mvt': ['AA123', None], 'FLIGHT_RULE_mvt': ['I', None]})
        x = self.module.airport_features(raw)
        self.assertEqual(x.loc[2, 'stand'], 'm:')
        self.assertIn('999', x.loc[1, 'stand'])
        self.assertEqual(x.loc[1, 'schedule_proxy_sec'], 1200.)
        self.assertNotIn('_event_time', x)

    def test_three_real_tiny_fits_predict_and_serialize(self):
        import joblib
        import tempfile
        x, times = self.examples()
        y = np.array([900., 87598., 1000., 86460.])
        x['observed_hour'] = [1., 2., 3., 4.]
        future = x.iloc[:2].copy()
        future.index = pd.Index([10, 11], name=x.index.name)
        future_times = pd.Series(pd.to_datetime(['2025-03-01', '2025-03-02'], utc=True), index=future.index)
        for arm in self.module.ARMS:
            chosen = {'steps': 3, 'classifier_steps': 3, 'calibration': {'scale': 1.}}
            model, _ = self.module.fit_arm(arm, x, y, times, selected=chosen, threads=1)
            pred = self.module.predict_arm(model, future, future_times)
            self.assertTrue(np.isfinite(pred).all())
            self.assertEqual(pred.shape, (2,))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.joblib'
                joblib.dump(model, path)
                replay = self.module.predict_arm(joblib.load(path), future, future_times)
                np.testing.assert_equal(pred, replay)


if __name__ == '__main__':
    unittest.main(verbosity=2)
