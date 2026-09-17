import unittest
import numpy as np
import pandas as pd
import schema


class ExactAblation(unittest.TestCase):
    def setUp(self):
        self.features = [f'flat_{phase}{rank}_{name}' for phase in ('dep', 'arr')
                         for rank in range(1, 5) for name in schema.CLOCK_FIELDS + schema.OTHER_FIELDS]

    def test_exact_schema(self):
        dropped, kept = schema.selection(self.features)
        self.assertEqual(len(dropped), 24)
        self.assertEqual(len(kept), 88)
        self.assertEqual([c for c in kept if c.startswith('flat_arr')], self.features[56:])
        self.assertEqual(len([c for c in kept if c.startswith('flat_dep')]), 32)
        self.assertEqual(kept, [c for c in self.features if c not in dropped])

    def test_dropped_poison_and_missing_preservation(self):
        dropped, kept = schema.selection(self.features)
        frame = pd.DataFrame(np.arange(5 * 112, dtype=np.float32).reshape(5, 112), columns=self.features)
        frame.loc[0, kept] = np.nan
        expected = frame[kept].copy()
        frame[dropped] = -1e20
        pd.testing.assert_frame_equal(frame[kept], expected)
        self.assertTrue(frame.loc[0, kept].isna().all())

    def test_schema_drift_rejected(self):
        for invalid in [self.features[::-1], self.features[:-1], self.features + ['extra'],
                        [self.features[1], *self.features[1:]]]:
            with self.assertRaises(ValueError):
                schema.selection(invalid)


if __name__ == '__main__':
    unittest.main()
