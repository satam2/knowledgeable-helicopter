import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import gate


class ThreeExpertContracts(unittest.TestCase):
    def test_recovers_known_interior_optimum(self):
        rng = np.random.default_rng(402)
        predictions = rng.normal(1000., 250., size=(1000, 3))
        expected = np.array([.2, .3, .5])
        with threadpool_limits(2):
            weights = gate.constant_weights(predictions@expected, predictions)
        np.testing.assert_allclose(weights, expected, atol=2e-6)

    def test_simplex_boundary_and_shrinkage(self):
        rng = np.random.default_rng(403)
        predictions = rng.normal(1000., 250., size=(1000, 3))
        labels = predictions[:, 1]
        with threadpool_limits(2):
            weights, local, penalty = gate.airport_weights(labels, predictions, np.array(['A']*500+['B']*500))
        np.testing.assert_allclose(weights, [0., 1., 0.], atol=2e-6)
        self.assertGreater(penalty, 0.)
        for value in local.values():
            self.assertGreaterEqual(value.min(), 0.)
            self.assertAlmostEqual(float(value.sum()), 1.)
            np.testing.assert_allclose(value, [0., 1., 0.], atol=2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
