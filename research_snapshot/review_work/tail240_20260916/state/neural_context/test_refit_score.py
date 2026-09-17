import torch
import unittest
import numpy as np
import refit_score as subject


class CompositionTest(unittest.TestCase):
    def test_only_ordinary_changes_and_exact_coefficients(self):
        base = np.array([900., 1000., 1100., 1200., 1300.])
        old = np.array([890., 980., 1080., 1180., 1280.])
        new = np.array([np.nan, 960., 1060., 1160., 1260.])
        proxy = np.array([np.nan, -1., 0., 7200., 7201.])
        eligible = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
        replacement, blend = subject.compose(base, old, new, eligible, .2)
        np.testing.assert_array_equal(replacement, [900., 1000., 1096., 1196., 1300.])
        np.testing.assert_array_equal(blend, [900., 1000., 1090., 1190., 1300.])
        np.testing.assert_array_equal(base, [900., 1000., 1100., 1200., 1300.])

    def test_zero_expert_weight_preserves_baseline(self):
        base = np.array([1., 2.])
        replacement, _ = subject.compose(base, np.array([2., 3.]), np.array([4., 5.]), np.ones(2, bool), 0.)
        np.testing.assert_array_equal(replacement, base)


if __name__ == '__main__':
    unittest.main()
