import unittest
import linear_finite_tune as fit
import numpy as np


class ScaleTests(unittest.TestCase):
    def test_fit_only_scaling_and_nan(self):
        result, state = fit.scale_column([1., 3., np.nan, 999.], 3, False)
        self.assertEqual(state['mean'], 2.)
        self.assertEqual(state['scale'], 1.)
        np.testing.assert_array_equal(result, [-1., 1., np.nan, 997.])

    def test_source_specific_sentinel_and_constant(self):
        values = [-999999., 2., 2., np.inf]
        fixed, state = fit.scale_column(values, 3, True)
        np.testing.assert_array_equal(fixed, [np.nan, 0., 0., np.nan])
        self.assertEqual(state['observed_fit'], 2)
        raw, state = fit.scale_column(values, 3, False)
        self.assertTrue(np.isfinite(raw[0]))
        self.assertEqual(state['observed_fit'], 3)

    def test_arms_differ_only_in_leaf_type(self):
        a, b = [fit.parameters({'n_jobs': 2}, arm) for arm in fit.ARMS]
        self.assertEqual(a.pop('linear_tree'), False)
        self.assertEqual(b.pop('linear_tree'), True)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
