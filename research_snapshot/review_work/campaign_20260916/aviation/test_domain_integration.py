"""Independent physical adapter contracts and tiny real-estimator integration."""

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN))
import common
import domain_adapter
import run
from aviation.arrival_features import StandRunwayReference, chronological_reference
from taxiout.schema import ID, MOVEMENT, TARGET


def frame(months=('01', '02'), per_month=120, first_id=1000):
    n = len(months) * per_month
    index = pd.Index(np.arange(first_id, first_id + n), name=ID)
    times = pd.DatetimeIndex([pd.Timestamp(f'2025-{month}-05', tz='UTC') + pd.Timedelta(minutes=i) for month in months for i in range(per_month)])
    return pd.DataFrame({'ADEP_mvt': pd.Categorical(['s3:AAA'] * n),
        'RUNWAY_mvt': pd.Categorical(['s2:01'] * n), 'STAND_mvt': pd.Categorical(['s2:A1'] * n),
        'utc_hour': np.arange(n, dtype=float) % 24,
        '__movement_ns': times.as_unit('ns').astype('int64')}, index=index)


class PhysicalIntegration(unittest.TestCase):
    def test_synthetic_observation_ids_tokens_and_timestamp_exact(self):
        x = frame()
        obs = domain_adapter.observations(x)
        np.testing.assert_array_equal(obs[ID], x.index)
        np.testing.assert_array_equal(obs.ADEP_mvt, x.ADEP_mvt)
        np.testing.assert_array_equal(obs[MOVEMENT].astype('int64'), x['__movement_ns'])
        self.assertTrue(obs.FLIGHT_ID_mvt.isna().all())

    def test_crossfit_first_month_and_same_month_isolation(self):
        x = frame()
        obs = domain_adapter.observations(x)
        labels = pd.DataFrame({ID: x.index, TARGET: np.r_[np.full(120, 500.), np.full(120, 1000.)]})
        original = chronological_reference(obs, labels)
        self.assertTrue(original.iloc[:120].physical_reference_q10_sec.eq(900).all())
        self.assertTrue(original.iloc[:120].physical_reference_history_n.eq(0).all())
        self.assertTrue(original.iloc[120:].physical_reference_q10_sec.eq(500).all())
        labels.loc[120:, TARGET] = 999999.
        pd.testing.assert_frame_equal(original, chronological_reference(obs, labels))

    def test_fitted_reference_rejects_overlap_and_future_labels(self):
        x = frame()
        obs = domain_adapter.observations(x)
        prior = StandRunwayReference().fit(obs, pd.DataFrame({ID: x.index, TARGET: 600.}))
        with self.assertRaisesRegex(ValueError, 'overlap or precede'):
            prior.transform(obs.iloc[[0]])

    def test_real_model_excludes_metadata_and_reconstructs_target(self):
        x = frame()
        y = np.r_[np.full(120, 500.), np.full(120, 900.)]
        model, evidence = domain_adapter.fit(x, y, steps=3, threads=1)
        columns = model['estimator']['encoder'].columns
        self.assertNotIn('__movement_ns', columns)
        self.assertIn('physical_reference_q10_sec', columns)
        self.assertEqual(model['prior'].n, len(x))
        later = frame(('03',), 20, first_id=9000)
        prediction = domain_adapter.predict(model, later)
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue(np.isfinite(prediction).all())
        prior = model['prior'].transform(domain_adapter.observations(later))
        residual = domain_adapter.lgbm_adapter.predict(model['estimator'], pd.concat([domain_adapter.raw_features(later), prior], axis=1))
        np.testing.assert_allclose(prediction, residual + prior.physical_reference_q10_sec)
        self.assertEqual(evidence['prior_fit_rows'], 240)

    def test_refit_builds_new_prior_on_fit_plus_tune(self):
        fit = frame(('01',), 120)
        tune = frame(('02',), 120, first_id=2000)
        first, _ = domain_adapter.fit(fit, np.full(120, 500.), steps=2, threads=1)
        combined = pd.concat([fit, tune])
        refit, _ = domain_adapter.fit(combined, np.r_[np.full(120, 500.), np.full(120, 900.)], steps=2, threads=1)
        self.assertIsNot(first['prior'], refit['prior'])
        self.assertEqual(first['prior'].n, 120)
        self.assertEqual(refit['prior'].n, 240)
        self.assertGreater(refit['prior'].last_fit, first['prior'].last_fit)

    def test_physical_cli_strips_clock_information_before_adapter(self):
        x = frame()
        meta = pd.DataFrame({ID: x.index, MOVEMENT: pd.to_datetime(x['__movement_ns'].to_numpy(), utc=True)})
        x = x.drop(columns='__movement_ns')
        clocks = ['takeoff_minus_AOBT_3_flt', 'takeoff_minus_EOBT_1_flt',
                  'takeoff_minus_IOBT_flt', 'takeoff_minus_LOBT_flt',
                  'takeoff_minus_SCHED_TIME_UTC_mvt', 'nm_actual_minus_estimated',
                  'last_minus_initial', 'proxy_missing', 'precision_aobt_second']
        for column in clocks:
            x[column] = 1000.
        captured = []
        def receive(args, model_x, model_meta, fold):
            captured.append((args, model_x, fold))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(common, 'load_data', return_value=(x, meta)), patch.object(common, 'OUT', Path(directory)), patch.object(run, 'run_one', side_effect=receive), patch.object(run, 'adapter'), patch.object(sys, 'argv', ['run_domain.py', '--mode', 'physical_base', '--folds', 'F1']):
                runpy.run_path(str(CAMPAIGN / 'run_domain.py'), run_name='__main__')
        self.assertEqual(len(captured), 1)
        args, model_x, fold = captured[0]
        self.assertEqual(args.target, 'direct')
        self.assertEqual(args.family, 'lightgbm')
        self.assertEqual(fold, 'F1')
        self.assertFalse(set(clocks) & set(model_x.columns))
        self.assertIn('__movement_ns', model_x)
        self.assertNotIn('__movement_ns', domain_adapter.raw_features(model_x))


if __name__ == '__main__':
    unittest.main(verbosity=2)
