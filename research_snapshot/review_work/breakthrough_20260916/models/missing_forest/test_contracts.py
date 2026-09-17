"""Small CPU tests for category isolation, chronology, raw targets and replay."""
import io
import unittest
import joblib
import numpy as np
import pandas as pd
import adapter


def fixture():
    dates = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    x = pd.DataFrame({"airport": ["A", "B"] * 30, "flight": ["F1", "F2"] * 30,
                       "stand": ["S1", "S2"] * 30, "schedule_bucket": ["0_30m"] * 60,
                       "schedule_hhmm": np.arange(60) % 24 * 60, "numeric": np.arange(60, dtype=float),
                       adapter.TIME_COLUMN: dates}, index=pd.Index(np.arange(60), name="MVT_ID_mvt"))
    y = np.arange(60, dtype=float) * 2 + 400
    y[0] = -12
    y[1] = 100000
    tune = x.iloc[:4].copy()
    tune.index += 100
    tune[adapter.TIME_COLUMN] = pd.date_range("2025-04-01", periods=4, tz="UTC")
    tune["airport"] = "NEVER_SEEN"
    return x, y, tune


class ForestContracts(unittest.TestCase):
    def test_onehot_unknown_is_zero_and_numeric_median_fit_only(self):
        x, _, tune = fixture()
        raw = adapter.input_frame(x)
        raw.loc[0, "numeric"] = np.nan
        enc = adapter.encoder(raw)
        train = enc.fit_transform(raw)
        transformed = enc.transform(adapter.input_frame(tune))
        self.assertEqual(train.shape[1], transformed.shape[1])
        category = enc.named_transformers_["category"].named_steps["onehot"]
        self.assertNotIn("NEVER_SEEN", category.categories_[0])
        self.assertEqual(enc.named_transformers_["numeric"].statistics_[1], 30.)
        self.assertTrue(np.isfinite(transformed.data).all())

    def test_raw_labels_tune_selection_and_saved_replay(self):
        x, y, tune = fixture()
        model, evidence = adapter.fit(x, y, (tune, np.full(4, 700.)), threads=2)
        self.assertEqual(len(evidence["tune_candidates"]), 3)
        expected = min(evidence["tune_candidates"], key=lambda row: row["tune_mse_sec2"])["min_samples_leaf"]
        self.assertEqual(evidence["steps"], expected)
        self.assertEqual(model["prior"].global_mean, y.mean())
        prediction = adapter.predict(model, tune)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        self.assertTrue(np.allclose(prediction, adapter.predict(joblib.load(stream), tune), rtol=0, atol=1e-9))
        refit, record = adapter.fit(x, y, steps=expected, threads=2)
        self.assertEqual(len(record["tune_candidates"]), 1)
        self.assertEqual(record["steps"], expected)
        self.assertTrue(np.isfinite(adapter.predict(refit, tune)).all())


if __name__ == "__main__":
    unittest.main()
