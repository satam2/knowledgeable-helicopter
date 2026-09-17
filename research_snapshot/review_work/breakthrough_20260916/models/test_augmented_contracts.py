"""Cache-integrity boundary checks without CUDA fitting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import run_augmented as runner


class CacheContracts(unittest.TestCase):
    def setUp(self):
        parent = runner.core.OUT / "validation"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)
        self.x = pd.DataFrame({"base": [1., 2.]}, index=pd.Index([11, 12], name=runner.core.ID))
        self.ext = pd.DataFrame({runner.core.ID: [11, 12], "batch_surface_T_count": [1., np.inf],
                                 "batch_surface_N_count": [2., 3.], "batch_source_airport_past_count": [3., 4.],
                                 "batch_source_stand_twosided_count": [4., 5.]})
        self.publish()

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, status="complete", verified="passed"):
        path = self.root / "training_features.parquet"
        self.ext.to_parquet(path, index=False)
        runner.core.write_json(self.root / "manifest.json", {"status": status, "training_id_order_verified": True,
                                                              "outputs": {path.name: runner.core.sha256(path)}})
        runner.core.write_json(self.root / "verification.json", {"status": verified})

    def test_each_block_is_isolated(self):
        for block in runner.BATCH_PATTERNS:
            result, receipts = runner.augment(self.x, [block], self.root)
            self.assertEqual(result.shape, (2, 2))
            self.assertEqual(len(receipts[0]["columns"]), 1)
            self.assertTrue(np.isfinite(result.to_numpy()).all())
            self.assertTrue(result.index.equals(self.x.index))

    def test_incomplete_rejected(self):
        self.publish(status="running")
        with self.assertRaisesRegex(ValueError, "complete and verified"):
            runner.augment(self.x, ["surface_T"], self.root)

    def test_unverified_rejected(self):
        self.publish(verified="failed")
        with self.assertRaisesRegex(ValueError, "complete and verified"):
            runner.augment(self.x, ["surface_T"], self.root)

    def test_changed_content_rejected(self):
        self.ext.iloc[:1].to_parquet(self.root / "training_features.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.augment(self.x, ["surface_T"], self.root)

    def test_reordered_ids_rejected(self):
        self.ext = self.ext.iloc[::-1]
        self.publish()
        with self.assertRaisesRegex(ValueError, "movement IDs"):
            runner.augment(self.x, ["surface_T"], self.root)

    def test_duplicate_ids_rejected(self):
        self.ext[runner.core.ID] = [11, 11]
        self.publish()
        with self.assertRaisesRegex(ValueError, "movement IDs"):
            runner.augment(self.x, ["surface_T"], self.root)

    def test_missing_block_rejected(self):
        self.ext = self.ext.drop(columns="batch_surface_T_count")
        self.publish()
        with self.assertRaisesRegex(ValueError, "Empty feature block"):
            runner.augment(self.x, ["surface_T"], self.root)


if __name__ == "__main__":
    unittest.main()
