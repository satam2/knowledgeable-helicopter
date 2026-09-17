import unittest
import preprocessed_locations_tune as trial
import numpy as np


class PreprocessingTests(unittest.TestCase):
    def test_only_unknown_physical_fields_change(self):
        columns = ['raw_clock', 'batch_source_stand_past_15m_count', 'flat_dep1_same_stand',
                   'batch_surface_T_runway_previous_wake', 'retro_past_arr_runway_count',
                   'flat_arr1_same_runway', 'batch_surface_T_observed_active_runways_15m']
        original = np.array([[100., 5., 1., 2., 8., 1., 3.], [-999999., 7., 1., 3., 9., 1., 4.]], dtype=np.float32)
        result = original.copy()
        trial.preprocess(result, columns, ['s7:UNKNOWN', 's3:007'], ['s2:NA', 's3:25L'])
        np.testing.assert_array_equal(result[1], original[1])
        np.testing.assert_array_equal(result[0], [100., -999999., 0., -999999., -999999., 0., 3.])
        np.testing.assert_array_equal(result[:, 0], original[:, 0])

    def test_noop_known_locations_and_alignment(self):
        matrix = np.ones((2, 1), dtype=np.float32)
        trial.preprocess(matrix, ['flat_dep1_same_stand'], ['s1:1', 's3:001'], ['s2:01', 's3:01L'])
        np.testing.assert_array_equal(matrix, 1.)
        with self.assertRaises(ValueError):
            trial.preprocess(matrix, ['flat_dep1_same_stand'], ['m:'], ['s2:01', 's2:02'])


if __name__ == '__main__':
    unittest.main()
