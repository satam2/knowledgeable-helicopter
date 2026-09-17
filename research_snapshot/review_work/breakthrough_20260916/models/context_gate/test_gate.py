import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from threadpoolctl import threadpool_limits
import gate


class GateContracts(unittest.TestCase):
    def test_squared_mixture_gradient(self):
        rng = np.random.default_rng(81)
        design = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
        experts = rng.normal(1000., 100., size=(20, 3))
        labels = rng.normal(1000., 100., size=20)
        parameters = rng.normal(0., .1, size=12)
        error = check_grad(lambda w: gate.objective(w, design, experts, labels, 10000.)[0],
                           lambda w: gate.objective(w, design, experts, labels, 10000.)[1], parameters)
        self.assertLess(error, 1e-6)

    def test_selection_signal_convexity_unknown_airport_replay(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=250)
        experts = np.column_stack([np.full(len(x), -100.), np.full(len(x), 100.), np.zeros(len(x))])
        labels = 100 * np.tanh(2*x)
        frame = pd.DataFrame({'context': x, 'airport': ['A'] * len(x)})
        with threadpool_limits(2):
            model, _ = gate.fit(frame, experts, labels)
            predicted, weights = gate.predict(model, frame, experts)
        self.assertLess(np.mean((predicted-labels)**2), 100.)
        np.testing.assert_allclose(weights.sum(1), 1.)
        self.assertTrue(np.all(weights >= 0.))
        shifted = frame.assign(airport='UNKNOWN')
        expected, _ = gate.predict(model, shifted, experts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(model, path)
            actual, _ = gate.predict(joblib.load(path), shifted, experts)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
