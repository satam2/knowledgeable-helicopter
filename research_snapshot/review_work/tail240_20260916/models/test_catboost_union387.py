import unittest
import catboost_union387_tune as trial
import numpy as np
import pandas as pd


class Contracts(unittest.TestCase):
    def test_replacement_preserves_other_components(self):
        experts = ['tree', 'catboost_combined', 'tabm_ple8']
        weights = dict(experts=experts, global_weights=[.4, .2, .4])
        aligned = pd.DataFrame([[100., 200., 300.], [500., 0., 100.]], columns=experts)
        current, candidate = trial.compose(aligned, weights, np.array([350., 50.]), np.array([120., 80.]))
        np.testing.assert_allclose(current, [220., 220.])
        np.testing.assert_allclose(candidate, [204., 236.])
        same, unchanged = trial.compose(aligned, weights, np.array([350., 50.]), aligned.catboost_combined.to_numpy())
        np.testing.assert_array_equal(same, unchanged)

    def test_paired_gain_counts_all_rows(self):
        y = np.array([10000., 0., 1., -20.])
        control = np.zeros(4)
        candidate = np.array([1000., 0., 1., -10.])
        days = np.array(['a', 'a', 'b', 'b'])
        result = trial.compare(y, candidate, control, days)
        self.assertAlmostEqual(result['mse_gain'], float(np.mean((y-control)**2-(y-candidate)**2)))
        self.assertTrue(result['all_day_removals_improve'])

    def test_fit_vocabulary_ignores_tune_only_values(self):
        matrix = np.array([[2., 1.], [0., np.nan], [1., -999999.]], dtype=np.float32)
        frame = trial.decode_frame(matrix, {'ADEP_mvt': ['AAA']}, ['ADEP_mvt', 'number'])
        enc = trial.adapter.FrameEncoder().fit(frame.iloc[:2])
        values = enc.transform(frame)
        self.assertEqual(enc.categories['ADEP_mvt'], {'AAA': 2})
        np.testing.assert_array_equal(values.ADEP_mvt.astype(int), [2, 0, 1])
        self.assertTrue(np.isnan(values.number.iloc[2]))


if __name__ == '__main__':
    unittest.main()
