"""Small synthetic tests for the all-airport missing-clock soft mixture."""

import sys
import unittest
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class ImplementationExists(unittest.TestCase):
    def test_missing_mixture_implementation_exists(self):
        self.assertTrue((HERE / 'run_missing_mixture.py').is_file())


@unittest.skipUnless((HERE / 'run_missing_mixture.py').exists(), 'Implementation pending')
class MixtureTests(unittest.TestCase):
    def setUp(self):
        import run_missing_mixture
        self.module = run_missing_mixture

    def test_expected_conditional_mean_uses_soft_probability(self):
        result = self.module.combine([1000, 1000, 1000], [0, .25, 1], [20, 20, 20], [3000, 3000, 3000])
        np.testing.assert_allclose(result, [3000, 2505, 1020])

    def test_invalid_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.combine([100], [1.01], [0], [100])

    def test_schedule_ineligible_reference_remains_exact(self):
        reference = np.array([100., 200., 300., 400.])
        missing_nm = [True, True, False, True]
        schedule = [120., np.nan, 110., -300.]
        predictions = self.module.replace_eligible(reference, missing_nm, schedule, [150., 450.])
        np.testing.assert_array_equal(predictions, [150., 200., 300., 450.])

    def test_regime_truth_threshold_only_training(self):
        regime = self.module.regime_labels([100, 160, 161, -50], [100, 100, 100, 100])
        np.testing.assert_array_equal(regime, [1, 1, 0, 0])

    def test_tiny_fit_and_unknown_category_predict_finite(self):
        n = 120
        x = pd.DataFrame({'airport': pd.Categorical(['A', 'B'] * (n // 2)), 'hour': np.arange(n) % 24})
        schedule = np.full(n, 1000.)
        y = np.where(np.arange(n) % 2, 1005., 3000.)
        model, evidence = self.module.fit_mixture(x, y, schedule, steps={'classifier': 3, 'good': 3, 'bad': 3}, threads=1)
        query = pd.DataFrame({'airport': pd.Categorical(['C', 'A']), 'hour': [3, 4]})
        pred = self.module.predict_mixture(model, query, [1000., 1000.])
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(pred.shape, (2,))
        self.assertEqual(evidence['fit_rows'], n)
        self.assertEqual(evidence['consistent_n'], 60)
        self.assertEqual(evidence['inconsistent_n'], 60)

    def test_single_class_and_sparse_group_use_unclipped_means(self):
        x = pd.DataFrame({'airport': pd.Categorical(['A'] * 4), 'hour': [1, 2, 3, 4]})
        y = np.array([-1000., 10000., 20000., 30000.])
        model, _ = self.module.fit_mixture(x, y, np.zeros(4), steps={'classifier': 1, 'good': 1, 'bad': 1}, threads=1)
        result = self.module.predict_mixture(model, x, np.zeros(4))
        np.testing.assert_allclose(result, np.mean(y))


if __name__ == '__main__':
    unittest.main(verbosity=2)
