import torch
import unittest
import numpy as np
import run_models


class SamplingContract(unittest.TestCase):
    def test_full_tune_and_score_eligibility_and_independent_refit(self):
        index = {'fit': np.arange(0, 100), 'tune': np.arange(100, 200),
                 'refit': np.arange(0, 200), 'score': np.arange(200, 250)}
        proxy = np.ones(250)
        proxy[[1, 110, 225]] = np.nan
        proxy[0], proxy[111], proxy[200] = -100., 99999., -1.
        selected, eligible = run_models.sampled_rows(index, proxy, 41, limit=20)
        self.assertEqual(len(selected['fit']), 20)
        self.assertEqual(len(selected['refit']), 20)
        np.testing.assert_array_equal(selected['tune'], eligible['tune'])
        np.testing.assert_array_equal(selected['score'], eligible['score'])
        self.assertEqual(len(selected['tune']), 99)
        self.assertEqual(len(index['score']), 50)
        self.assertIn(111, selected['tune'])
        self.assertIn(200, selected['score'])
        self.assertNotIn(1, selected['fit'])
        self.assertTrue(set(selected['refit'])-set(selected['fit']))


if __name__ == '__main__':
    unittest.main()
