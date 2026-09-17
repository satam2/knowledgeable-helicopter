"""Synthetic preprocessing, neural persistence, and unseen-category checks."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import torch
import joblib
import numpy as np
import pandas as pd

import tabm_adapter as adapter


class TabMAdapterTests(unittest.TestCase):
    def test_preprocessing_and_persistence(self):
        rng = np.random.default_rng(7)
        x = pd.DataFrame({"n": rng.normal(size=1024), "constant": np.nan,
                          "category": pd.Categorical(np.where(np.arange(1024) % 2, "A", "B"))})
        x.loc[0, "n"] = -999999
        y = np.nan_to_num(x.n.to_numpy(), nan=0.0)
        y[0] = 0.0
        y = 500 + 50 * y
        probe = pd.DataFrame({"n": [np.nan, 1.0, -999999], "constant": [np.nan]*3,
                              "category": pd.Categorical(["new", None, "A"])})
        encoder = adapter.FrameEncoder().fit(x)
        nums, cats = encoder.transform(probe)
        self.assertTrue(np.isfinite(nums.numpy()).all())
        self.assertEqual(cats[:2, 0].tolist(), [1, 0])
        self.assertNotIn("new", encoder.categories["category"])
        self.assertEqual(nums[[0, 2], 2].tolist(), [1.0, 1.0])
        model, evidence = adapter.fit(x, y, steps=2, threads=2)
        self.assertEqual(evidence["steps"], 2)
        predictions = adapter.predict(model, probe)
        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertIsInstance(model["estimator"].backbone, adapter.tabm.TabM)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(predictions, adapter.predict(joblib.load(path), probe))
        self.assertTrue(np.isfinite(evidence["history"][-1]["train_mse_sec2"]))
        print("SYNTHETIC_EVIDENCE", json.dumps(evidence), flush=True)

    def test_invalid_refit_contract(self):
        x = pd.DataFrame({"a": [1., 2.]})
        with self.assertRaises(ValueError):
            adapter.fit(x, [1., 2.], (x, [1., 2.]), steps=1)

    def test_tuning_selection_and_repeat(self):
        rng = np.random.default_rng(11)
        x = pd.DataFrame({"value": rng.normal(size=2048),
                          "airport": pd.Categorical(np.where(np.arange(2048) % 2, "A", "B"))})
        y = 120 + 25 * x.value.to_numpy()
        model, evidence = adapter.fit(x.iloc[:1536], y[:1536], (x.iloc[1536:], y[1536:]), threads=2)
        history = evidence["history"]
        best = min(history, key=lambda row: row["tune_mse_sec2"])
        self.assertEqual(evidence["steps"], best["epoch"])
        repeated = adapter.predict(model, x.iloc[1536:])
        np.testing.assert_array_equal(repeated, adapter.predict(model, x.iloc[1536:]))
        self.assertLess(best["tune_mse_sec2"], history[0]["tune_mse_sec2"])


if __name__ == "__main__":
    from common import OUT, external_path, write_json, sha256, utc_now
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TabMAdapterTests))
    dest = external_path(OUT / "neural" / "synthetic_verification.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, {"completed_utc": utc_now(), "success": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "adapter_sha256": sha256(adapter.__file__), "test_sha256": sha256(__file__),
        "command": ".review-venv/Scripts/python.exe -B -u review_work/campaign_20260916/test_tabm_adapter.py",
        "data": "synthetic only", "real_cohort_trained": False,
        "resolved_failure": "Windows c10.dll WinError1114 when importing numpy/pandas before torch; torch-first entry point passed."})
    sys.exit(0 if result.wasSuccessful() else 1)
