import unittest
import numpy as np
import pandas as pd
import normalized_missing_tune as n


class NormalizedTargetTests(unittest.TestCase):
    def test_exact_original_loss_and_inverse(self):
        x = pd.DataFrame({'schedule_proxy_sec': [-40000., 0., 1500., 60000., np.nan, -999999.]})
        y = np.array([-123., 1800., 2300., 87600., 4000., 1400.])
        scale = n.scale_of(x)
        z = n.transformed(y, scale)
        np.testing.assert_allclose(n.reconstruct(z, scale), y, atol=1e-10)
        trial = np.array([-.1, .2, .3, .5, .6, 1.])
        norm = np.mean(scale**2)
        np.testing.assert_allclose(scale**2/norm * (trial-z)**2,
                                   (n.reconstruct(trial, scale)-y)**2/norm, rtol=1e-12, atol=1e-10)
        self.assertEqual(scale[-1], 3600.)
        self.assertEqual(scale[-2], 3600.)

    def test_no_label_dependent_scale(self):
        x = pd.DataFrame({'schedule_proxy_sec': [10., 600., np.nan], 'target': [100, 200, 300]})
        expected = n.scale_of(x)
        x['target'] = -90000.
        np.testing.assert_array_equal(n.scale_of(x), expected)
        with self.assertRaises(ValueError):
            n.transformed([1., 2.], [0., 1.])


if __name__ == '__main__':
    unittest.main()
