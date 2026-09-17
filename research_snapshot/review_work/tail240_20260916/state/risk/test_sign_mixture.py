import unittest
import numpy as np
import run_sign_mixture as mixture


class MixtureTests(unittest.TestCase):
    def test_strict_sign_boundaries(self):
        np.testing.assert_array_equal(mixture.classes([-2000, -1801, -1800, 0, 1800, 1801]), [0, 0, 1, 1, 1, 2])

    def test_conditional_mean_uses_probabilities_not_observed_class(self):
        p = np.array([[.1, .7, .2], [.9, .05, .05]])
        means = np.array([[-3000., 0., 3000.], [-3000., 0., 3000.]])
        np.testing.assert_allclose(mixture.mixture_mean(p, means), [300., -2550.])

    def test_rejects_invalid_probability_sum(self):
        with self.assertRaises(AssertionError):
            mixture.mixture_mean(np.array([[.2, .3, .1]]), np.zeros((1, 3)))


if __name__ == '__main__':
    unittest.main()
