import unittest
import lobt_anchor_tune as fit
import numpy as np


class AnchorTests(unittest.TestCase):
    def test_observed_clock_and_missing_fallback(self):
        observed = np.array([1200., -5., -999999., 90000., np.nan, np.inf])
        proxy = np.full(6, 777.)
        np.testing.assert_array_equal(fit.anchor_of(observed, proxy), [1200., -5., -999999., 90000., 777., 777.])

    def test_same_raw_loss_with_different_origin(self):
        y = np.array([-30., 0., 1234., 86400., 172900.])
        anchor = fit.anchor_of([100., 200., np.nan, -9000., 1e6], np.full(5, 300.))
        residual = np.array([21., -13., 0., 431., -52.])
        np.testing.assert_array_equal((residual-(y-anchor))**2, (anchor+residual-y)**2)


if __name__ == '__main__':
    unittest.main()
