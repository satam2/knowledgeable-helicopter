"""CPU-only context-pair and chronology tests without launching TabICL CUDA."""
import adapter
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


def frame(start, periods, offset=0):
    return pd.DataFrame({"airport": ["A"] * periods, "flight": ["F"] * periods,
                         "stand": ["S"] * periods, "schedule_bucket": ["0_30m"] * periods,
                         "schedule_hhmm": [60.] * periods, "idctx": np.arange(periods, dtype=float),
                         adapter.TIME_COLUMN: pd.date_range(start, periods=periods, freq="D", tz="UTC")},
                        index=pd.Index(np.arange(periods) + offset, name="MVT_ID_mvt"))


class ContextContracts(unittest.TestCase):
    def test_context_labels_and_chronological_features_stay_paired(self):
        x = frame("2025-01-01", 60)
        y = np.arange(60, dtype=float) * 10
        y[0], y[1] = -12, 100000
        tune = frame("2025-04-01", 4, 1000)
        captured = {}

        def fake_fit(features, labels, tuning=None, **kwargs):
            captured.update(features=features.copy(), labels=labels.copy(), tuning=tuning)
            return {"synthetic": True}, {"steps": 1}

        with patch.object(adapter.frozen, "fit", fake_fit):
            model, _ = adapter.fit(x, y, (tune, np.arange(4.)), threads=2)
        np.testing.assert_array_equal(captured["labels"], y)
        self.assertTrue(captured["features"].index.equals(x.index))
        self.assertTrue(captured["tuning"][0].index.equals(tune.index))
        self.assertNotIn(adapter.TIME_COLUMN, captured["features"])
        self.assertTrue(captured["features"].iloc[:31].template_mean_sec.eq(900).all())
        self.assertTrue(captured["tuning"][0].template_history_n.eq(len(x)).all())
        self.assertEqual(model["prior"].global_mean, y.mean())

    def test_prediction_uses_saved_prior_and_rejects_overlap(self):
        x = frame("2025-01-01", 45)
        y = np.arange(45, dtype=float)
        with patch.object(adapter.frozen, "fit", return_value=({"synthetic": True}, {"steps": 1})):
            model, _ = adapter.fit(x, y, steps=1)
        with patch.object(adapter.frozen, "predict", side_effect=lambda _, values: values.template_mean_sec.to_numpy()) as predict:
            later = frame("2025-03-01", 3, 1000)
            output = adapter.predict(model, later)
            self.assertTrue(np.isfinite(output).all())
            self.assertEqual(len(output), len(later))
            self.assertEqual(list(predict.call_args.args[1]), model["feature_columns"])
            with self.assertRaisesRegex(ValueError, "follow every fit"):
                adapter.predict(model, x)


if __name__ == "__main__":
    unittest.main()
