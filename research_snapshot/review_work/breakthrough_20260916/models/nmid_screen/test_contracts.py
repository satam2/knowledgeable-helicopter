import lightgbm
import unittest
import tempfile
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import run as runner


class CacheContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.x = pd.DataFrame({"base": [1., 2.]}, index=pd.Index([11, 12], name=runner.core.ID))
        self.extra = pd.DataFrame({runner.core.ID: [11, 12], **{name: [1., np.nan] for name in runner.FEATURES}})
        self.publish()

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, verified="passed"):
        self.extra.to_parquet(self.root / "training_features.parquet", index=False)
        runner.core.write_json(self.root / "protocol.json", {"source": "synthetic"})
        runner.core.write_json(self.root / "manifest.json", {"status": "complete", "features": runner.FEATURES,
            "training_id_order_verified": True, "protocol_sha256": runner.core.sha256(self.root / "protocol.json"),
            "outputs": {"training_features.parquet": runner.core.sha256(self.root / "training_features.parquet")}})
        runner.core.write_json(self.root / "verification.json", {"status": verified, "training_id_order_verified": True,
                                                                "manifest_sha256": runner.core.sha256(self.root / "manifest.json")})

    def test_exact_seven_columns_and_missing_encoding(self):
        result, _ = runner.append_nmid(self.x, self.root)
        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result.iloc[1, 1], -999999)

    def test_stale_receipt_and_changed_feature_rejected(self):
        runner.core.write_json(self.root / "manifest.json", {"status": "complete"})
        with self.assertRaisesRegex(ValueError, "stale"):
            runner.append_nmid(self.x, self.root)
        self.publish()
        self.extra.iloc[:1].to_parquet(self.root / "training_features.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "feature hash"):
            runner.append_nmid(self.x, self.root)

    def test_order_and_duplicate_identity_rejected(self):
        self.extra = self.extra.iloc[::-1]
        self.publish()
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            runner.append_nmid(self.x, self.root)
        self.extra[runner.core.ID] = [11, 11]
        self.publish()
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            runner.append_nmid(self.x, self.root)


class FlightIdentityReview(unittest.TestCase):
    def test_observed_identity_transform_ignores_outcomes(self):
        path = runner.core.ROOT / "review_work/breakthrough_20260916/flight_identity/build.py"
        spec = importlib.util.spec_from_file_location("review_identity_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertNotIn(runner.core.TARGET, module.COLS)
        self.assertNotIn("BLOCK_TIME_UTC_mvt", module.COLS)
        data = pd.DataFrame({runner.core.ID: [1, 2, 3], module.PHASE: ["DEP", "ARR", "DEP"],
                             module.MOVEMENT: pd.to_datetime(["2025-06-02T01:00Z"] * 3),
                             "SCHED_TIME_UTC_mvt": pd.to_datetime(["2025-06-01T23:55Z"] * 3),
                             "FLIGHT_mvt": ["AB123", "XX1", None], "CALLSIGN_flt": ["ABC123", "XX2", None],
                             "ADEP_mvt": ["LIRF", "LIRF", "EDDF"], runner.core.TARGET: [100., 200., 300.],
                             "BLOCK_TIME_UTC_mvt": pd.to_datetime(["2025-06-01T20:00Z"] * 3)})
        before = module.transform(data)
        data[runner.core.TARGET] += 100000
        data["BLOCK_TIME_UTC_mvt"] += pd.Timedelta(days=100)
        pd.testing.assert_frame_equal(before, module.transform(data))
        self.assertEqual(before[runner.core.ID].tolist(), [1, 3])
        self.assertEqual(before.identity_schedule_day_offset.tolist(), [1., 1.])
        self.assertEqual(before.identity_flight_prefix.iloc[0], module.token(pd.Series(["AB"])).iloc[0])
        self.assertEqual(before.identity_callsign_prefix.iloc[0], module.token(pd.Series(["ABC"])).iloc[0])


if __name__ == "__main__":
    unittest.main()
