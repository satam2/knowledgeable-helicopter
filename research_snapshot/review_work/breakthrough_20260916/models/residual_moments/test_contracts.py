"""Exact-target identity, raw tune selection, encoder isolation and replay checks."""
import io
import unittest
import adapter
import joblib
import numpy as np
import pandas as pd


class MomentContracts(unittest.TestCase):
    def test_exact_identity_including_boundaries_and_extremes(self):
        y = np.array([-1e9, -131167., -301., -300., -299., -12., 0., 299., 300., 301., 131167., 1e9])
        parts = adapter.decompose(y)
        np.testing.assert_array_equal(sum(parts.values()), y)
        self.assertTrue(np.all(np.abs(parts["center"]) <= 300))
        self.assertTrue(np.all(parts["upper"] >= 0))
        self.assertTrue(np.all(parts["lower"] <= 0))

    def test_bad_inputs_rejected(self):
        for values in [[], [np.nan], [np.inf], [[1, 2]]]:
            with self.assertRaises(ValueError):
                adapter.decompose(values)

    def test_constant_tail_heads_have_complete_prefix_trajectory(self):
        x = pd.DataFrame({"numeric": np.arange(100, dtype=float)})
        y = np.linspace(-10, 10, 100)
        previous = adapter.MAX_TREES
        adapter.MAX_TREES = 3
        try:
            model, evidence = adapter.fit(x, y, (x, y), threads=2)
            self.assertEqual(len(evidence["selection"]["raw_mse_by_common_prefix"]), 3)
            self.assertEqual(evidence["component_diagnostics"]["upper"]["nonzero_rows"], 0)
            self.assertEqual(evidence["component_diagnostics"]["lower"]["nonzero_rows"], 0)
            self.assertTrue(np.isfinite(adapter.predict(model, x)).all())
        finally:
            adapter.MAX_TREES = previous

    def test_raw_reconstructed_prefix_selection_and_saved_replay(self):
        rng = np.random.default_rng(20260916)
        x = pd.DataFrame({"numeric": rng.normal(size=180), "airport": pd.Categorical(["A", "B"] * 90)})
        y = np.where(x.numeric.to_numpy() > 1., 3000., np.where(x.numeric.to_numpy() < -1., -2000., 20.))
        tune = pd.DataFrame({"numeric": rng.normal(size=60), "airport": pd.Categorical(["UNSEEN", "B"] * 30)})
        ty = np.where(tune.numeric.to_numpy() > 1., 3100., np.where(tune.numeric.to_numpy() < -1., -1900., 25.))
        previous = adapter.MAX_TREES
        adapter.MAX_TREES = 6
        try:
            model, evidence = adapter.fit(x, y, (tune, ty), threads=2)
            encoded = model["encoder"].transform(tune)
            curve = [float(np.mean(np.square(sum(boost.predict(encoded, num_iteration=step, num_threads=2)
                                                    for boost in model["boosters"].values()) - ty))) for step in range(1, 7)]
            self.assertEqual(model["steps"], int(np.argmin(curve)) + 1)
            np.testing.assert_allclose(curve, evidence["selection"]["raw_mse_by_common_prefix"], rtol=1e-10, atol=1e-8)
            self.assertNotIn("UNSEEN", model["encoder"].maps["airport"])
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            np.testing.assert_array_equal(adapter.predict(model, tune), adapter.predict(joblib.load(stream), tune))
            refit, info = adapter.fit(x, y, steps=model["steps"], threads=2)
            self.assertEqual(info["steps"], model["steps"])
            self.assertTrue(np.isfinite(adapter.predict(refit, tune)).all())
        finally:
            adapter.MAX_TREES = previous


if __name__ == "__main__":
    unittest.main()
