"""CPU-only contract checks; never invokes a GPU or reads private rows."""
import unittest
import catboost387_canary as run
import numpy as np


class CanaryContracts(unittest.TestCase):
    def test_resource_boundaries(self):
        run.check_resources(100, 200, run.RESERVE)
        for values in [(run.CAP, 200, run.RESERVE), (100, run.CAP, run.RESERVE), (100, 200, run.RESERVE-1)]:
            with self.assertRaises(MemoryError):
                run.check_resources(*values)

    def test_schema_data_and_fit_only_encoding(self):
        columns, cardinalities = run.schema()
        first, y = run.generate(128, columns, cardinalities)
        second, y2 = run.generate(128, columns, cardinalities)
        run.pd.testing.assert_frame_equal(first, second)
        np.testing.assert_array_equal(y, y2)
        self.assertEqual(first.shape, (128, 387))
        self.assertEqual(len(first.select_dtypes('category').columns), 15)
        self.assertTrue(np.isfinite(y).all())
        encoder = run.adapter.FrameEncoder().fit(first)
        self.assertEqual(len(encoder.numeric), 372)
        probe = first.iloc[:3].copy()
        name = next(iter(cardinalities))
        probe[name] = probe[name].astype('string')
        probe.loc[0, name] = 'unseen'
        probe.loc[1, name] = run.pd.NA
        encoded = encoder.transform(probe)
        self.assertEqual(encoded[name].iloc[0], 1)
        self.assertEqual(encoded[name].iloc[1], 0)
        self.assertNotIn('unseen', encoder.categories[name])
        self.assertTrue(encoded[encoder.numeric[27]].isna().all())
        with self.assertRaises(ValueError):
            encoder.transform(probe[columns[::-1]])

    def test_frozen_parameters(self):
        full = run.adapter.parameters()
        canary = run.adapter.parameters(seed=run.SEED, threads=2, device='GPU', iterations=run.ROUNDS)
        self.assertEqual(full['iterations'], 5000)
        self.assertEqual(canary['iterations'], 50)
        self.assertEqual({key:value for key,value in full.items() if key != 'iterations'},
                         {key:value for key,value in canary.items() if key != 'iterations'})
        self.assertEqual(canary['gpu_ram_part'], .65)


if __name__ == '__main__':
    unittest.main()
