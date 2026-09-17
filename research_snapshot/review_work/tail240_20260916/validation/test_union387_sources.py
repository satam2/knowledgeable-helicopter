import unittest
import numpy as np
import pandas as pd
from audit_union387_sources import mark_once, numeric_stats


class SourceContracts(unittest.TestCase):
    def test_duplicate_plus_missing_cannot_cancel(self):
        seen = np.zeros(3, dtype=bool)
        mark_once(seen, np.array([0, 2]))
        with self.assertRaises(ValueError):
            mark_once(seen, np.array([0]))
        self.assertFalse(seen.all())
        with self.assertRaises(ValueError):
            mark_once(np.zeros(3, dtype=bool), np.array([1, 1]))
        mark_once(seen, np.array([1]))
        self.assertTrue(seen.all())

    def test_preexisting_sentinel_overflow_and_new_fill_are_distinct(self):
        values = pd.Series([np.nan, np.inf, -np.inf, 1e40, -999999., -999998.99, 16777217., 3.])
        result = numeric_stats(values, True)
        self.assertEqual(result['source_nan'], 1)
        self.assertEqual(result['source_finite_sentinel'], 1)
        self.assertEqual(result['finite_to_nonfinite_float32'], 1)
        self.assertEqual(result['nonsentinel_finite_to_sentinel_float32'], 1)
        self.assertEqual(result['new_loader_sentinel'], 4)
        self.assertEqual(result['final_loader_nan'], 0)
        native = numeric_stats(values, False)
        self.assertEqual(native['new_loader_sentinel'], 0)
        self.assertEqual(native['final_loader_nan'], 4)


if __name__ == '__main__':
    unittest.main()
