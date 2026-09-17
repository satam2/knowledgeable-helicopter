import adapter
import io
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd


def leaves(node):
    if "leaf_index" in node:
        yield node
    else:
        yield from leaves(node["left_child"])
        yield from leaves(node["right_child"])


class LinearTreeContracts(unittest.TestCase):
    def test_fit_only_scaling_native_missing_and_category_codes(self):
        x = pd.DataFrame({"number": [1., 3., -999999, np.inf], "empty": [np.nan] * 4,
                          "airport": pd.Categorical(["A", "B", "A", None])})
        encoder = adapter.ScaledNativeEncoder().fit(x)
        train = encoder.transform(x)
        self.assertEqual(encoder.means["number"], 2.)
        self.assertEqual(encoder.scales["number"], 1.)
        np.testing.assert_array_equal(train.number.iloc[:2], [-1., 1.])
        self.assertTrue(train.number.iloc[2:].isna().all())
        self.assertTrue(train["empty"].isna().all())
        query = pd.DataFrame({"number": [100.], "empty": [np.nan], "airport": pd.Categorical(["NEW"])})
        transformed = encoder.transform(query)
        self.assertEqual(transformed.number.iloc[0], 98.)
        self.assertTrue(isinstance(transformed.airport.dtype, pd.CategoricalDtype))
        self.assertEqual(transformed.airport.iloc[0], 1)
        self.assertNotIn("NEW", encoder.native.maps["airport"])

    def test_matched_parameters_only_differ_in_linear_switch(self):
        original = adapter.LINEAR_TREE
        try:
            adapter.LINEAR_TREE = False
            constant = adapter.parameters(17, 2, 200)
            adapter.LINEAR_TREE = True
            linear = adapter.parameters(17, 2, 200)
            self.assertEqual([key for key in linear if linear[key] != constant[key]], ["linear_tree"])
        finally:
            adapter.LINEAR_TREE = original

    def test_actual_linear_coefficients_exclude_categories_and_replay(self):
        rng = np.random.default_rng(4)
        x = pd.DataFrame({"number": rng.uniform(-2, 2, 400), "airport": pd.Categorical(["A", "B"] * 200)})
        y = x.number.to_numpy() * 100 + np.where(x.airport.eq("A"), 20., -20.)
        query = pd.DataFrame({"number": [-5., 5., np.nan], "airport": pd.Categorical(["A", "NEW", "B"])})
        original = adapter.LINEAR_TREE
        adapter.LINEAR_TREE = True
        try:
            captured = {}
            realfit = adapter.lgb.LGBMRegressor.fit

            def spy(estimator, frame, labels, **kwargs):
                captured["labels"] = labels.copy()
                return realfit(estimator, frame, labels, **kwargs)

            with patch.object(adapter.lgb.LGBMRegressor, "fit", spy):
                model, info = adapter.fit(x, y, steps=8, threads=2)
            np.testing.assert_array_equal(captured["labels"], y)
            dump = model["estimator"].booster_.dump_model()
            numeric_slopes = []
            for tree in dump["tree_info"]:
                for leaf in leaves(tree["tree_structure"]):
                    self.assertNotIn(1, leaf.get("leaf_features", []))
                    numeric_slopes.extend(leaf.get("leaf_features", []))
            self.assertIn(0, numeric_slopes)
            pred = adapter.predict(model, query)
            self.assertTrue(np.isfinite(pred).all())
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            np.testing.assert_array_equal(pred, adapter.predict(joblib.load(stream), query))
        finally:
            adapter.LINEAR_TREE = original

    def test_tune_stopping_and_independent_refit_counts(self):
        x = pd.DataFrame({"number": np.linspace(-2, 2, 240), "airport": pd.Categorical(["A", "B"] * 120)})
        y = x.number.to_numpy() * 50
        tune = x.iloc[-40:].copy()
        model, info = adapter.fit(x.iloc[:200], y[:200], (tune, y[-40:]), threads=2)
        self.assertTrue(1 <= info["steps"] <= 200)
        refit, refit_info = adapter.fit(x, y, steps=info["steps"], threads=2)
        self.assertEqual(refit_info["rows"], 240)
        self.assertEqual(refit_info["steps"], info["steps"])
        self.assertNotEqual(model["encoder"].means["number"], refit["encoder"].means["number"])


if __name__ == "__main__":
    unittest.main()
