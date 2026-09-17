import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import run_risk as risk


class RiskTests(unittest.TestCase):
    def test_streamed_loader_alignment_and_fit_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = folder / 'first.parquet'
            second = folder / 'second.parquet'
            pd.DataFrame({risk.ID:[3., 1.], 'category':['tune_only', 'b'], 'clock':[3., np.inf]}).to_parquet(first, index=False)
            pd.DataFrame({risk.ID:[2., 4.], 'category':['a', 'not_selected'], 'clock':[2., 4.]}).to_parquet(second, index=False)
            sources = [(path, 'fixture', True, ['category', 'clock']) for path in [first, second]]
            with patch.object(risk, 'feature_sources', return_value=sources), patch.object(risk, 'guard', return_value=0):
                matrix, vocab, receipts = risk.load_matrix(pd.Index([1., 2., 3.]), 2, ['category', 'clock'], folder)
            self.assertEqual(vocab, {'category':['a', 'b']})
            np.testing.assert_array_equal(matrix, [[3., -999999.], [2., 2.], [1., 3.]])
            self.assertEqual(sum(row['rows'] for row in receipts), 3)
            del matrix

    def test_fit_only_categories_and_unknown_missing(self):
        values = pd.Series(['b', 'a', 'tune_only', None])
        known = risk.categories_for(values, np.array([True, True, False, False]))
        self.assertEqual(known, ['a', 'b'])
        np.testing.assert_array_equal(risk.category_codes(values, known), [3, 2, 1, 0])

    def test_own_excludes_batch_neighbors(self):
        self.assertEqual(risk.own_columns(['ADEP_mvt', 'dep_prior_15m', 'arr_prior_60m']), ['ADEP_mvt'])

    def test_constant_baseline_and_full_capture(self):
        y = np.tile([0, 0, 0, 1], 25)
        dates = np.repeat(pd.date_range('2025-06-01', periods=10), 10)
        report = risk.metric_report(y, np.full(100, .25), .25, np.ones(100), dates)
        self.assertEqual(report['brier'], report['baseline_brier'])
        self.assertEqual(report['brier_improvement_ci95'], [0., 0.])
        self.assertEqual(report['roc_auc'], .5)
        self.assertEqual(sum(x['rows'] for x in report['calibration']), 100)
        self.assertAlmostEqual(report['top_risk']['0.1']['union_sse_capture'], .1)

    def test_probability_one_in_last_bin(self):
        y = np.array([0, 1, 0, 1])
        report = risk.metric_report(y, y.astype(float), .5, np.ones(4), ['a', 'a', 'b', 'b'])
        self.assertEqual(report['calibration'][-1]['rows'], 2)
        self.assertEqual(report['brier'], 0.)


if __name__ == '__main__':
    unittest.main()
