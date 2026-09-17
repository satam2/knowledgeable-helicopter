import unittest
import catboost_union387_tune_v2 as trial
import numpy as np
import pandas as pd


class Contracts(unittest.TestCase):
    def test_prediction_file_does_not_need_labels(self):
        stored = pd.DataFrame({trial.ID: [2, 1], 'prediction_sec': [20., 10.]})
        np.testing.assert_array_equal(trial.align_reference(stored, [1, 2]), [10., 20.])
        for ids in [[1, 1], [1, 3], [1]]:
            with self.assertRaises(AssertionError):
                trial.align_reference(stored, ids)
        with self.assertRaises(AssertionError):
            trial.align_reference(pd.concat([stored, stored]), [1, 2, 3, 4])

    def test_component_replacement(self):
        experts = ['tree', 'catboost_combined', 'tabm_ple8']
        aligned = pd.DataFrame([[100., 200., 300.], [500., 0., 100.]], columns=experts)
        current, candidate = trial.compose(aligned, dict(experts=experts, global_weights=[.4, .2, .4]),
            np.array([350., 50.]), np.array([120., 80.]))
        np.testing.assert_allclose(current, [220., 220.])
        np.testing.assert_allclose(candidate, [204., 236.])

    def test_fit_only_vocabulary(self):
        matrix = np.array([[2., 1.], [0., np.nan], [1., -999999.]], dtype=np.float32)
        frame = trial.decode_frame(matrix, {'ADEP_mvt': ['AAA']}, ['ADEP_mvt', 'number'])
        enc = trial.adapter.FrameEncoder().fit(frame.iloc[:2])
        values = enc.transform(frame)
        self.assertEqual(enc.categories['ADEP_mvt'], {'AAA': 2})
        np.testing.assert_array_equal(values.ADEP_mvt.astype(int), [2, 0, 1])
        self.assertTrue(np.isnan(values.number.iloc[2]))


if __name__ == '__main__':
    unittest.main()
