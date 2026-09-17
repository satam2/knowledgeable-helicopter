import unittest
import numpy as np
from scaled_finite_tune import scale_of


class ScalingTests(unittest.TestCase):
    def test_missing_and_nonfinite_clocks_have_fixed_scale(self):
        np.testing.assert_array_equal(scale_of([np.nan,np.inf,-999999.,100000.], [0,0,0,1]), np.full(4,900.))

    def test_original_mse_is_preserved_for_signed_extreme_targets(self):
        scale = scale_of([-200000.,0.,900.,86400.], [0,0,0,0])
        proxy = np.array([-4000.,500.,900.,90000.])
        y = np.array([-1000.,1200.,90000.,1000000.])
        z = np.array([.2,-.3,2.,1.])
        normalizer = np.mean(scale**2)
        np.testing.assert_allclose(scale**2/normalizer*(z-(y-proxy)/scale)**2,
            (proxy+scale*z-y)**2/normalizer, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
