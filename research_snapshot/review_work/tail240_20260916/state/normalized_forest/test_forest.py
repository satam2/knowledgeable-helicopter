import unittest
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import run as subject


class WeightedForestTests(unittest.TestCase):
    def test_one_leaf_minimizes_original_raw_loss_in_scaled_basis(self):
        scale = np.array([1., 2., 3., 4.])
        y = np.array([900., 1000., 2000., -300.])
        x = np.zeros((4, 1))
        weights = scale**2 / np.mean(scale**2)
        z = subject.normalized.transformed(y, scale)
        tree = ExtraTreesRegressor(n_estimators=1, min_samples_leaf=4, n_jobs=1, random_state=20260916)
        tree.fit(x, z, sample_weight=weights)
        value = tree.predict(x)
        expected = np.sum(scale*(y-900.))/np.sum(scale**2)
        np.testing.assert_allclose(value, expected, rtol=1e-14)
        raw = subject.normalized.reconstruct(value, scale)
        np.testing.assert_allclose(weights*(value-z)**2, (raw-y)**2/np.mean(scale**2), rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
