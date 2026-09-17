import unittest
import numpy as np
import pandas as pd
from semantic_locations import known_location, mask_query_features


class SemanticsTests(unittest.TestCase):
    def test_exact_unknown_tokens_and_real_category_preservation(self):
        raw = pd.Series([None, '', 'UNKNOWN', 'NA', '18L', 'N/A', 'Unknown'])
        np.testing.assert_array_equal(known_location(raw), [False]*4+[True]*3)
        encoded = pd.Series(['m:', 's0:', 's7:UNKNOWN', 's2:NA', 's3:18L'])
        np.testing.assert_array_equal(known_location(encoded, encoded=True), [False]*4+[True])

    def test_mask_preserves_empty_known_window_and_airport_features(self):
        x = pd.DataFrame({'batch_source_stand_past_15m_count': [4., 0., 3.],
            'batch_surface_T_runway_previous_wake': [3., 0., 2.],
            'retro_past_arr_runway_count': [2., 0., 1.],
            'flat_dep1_same_stand': [1., 0., 1.], 'flat_arr1_same_runway': [1., 0., 1.],
            'batch_surface_T_observed_active_runways_15m': [3., 0., 2.],
            'proxy_missing': [0., 0., 1.], 'STAND_mvt': ['UNKNOWN', 'A1', None]})
        y=mask_query_features(x, x.STAND_mvt, ['NA', '18L', '19'])
        self.assertTrue(np.isnan(y.loc[0,'batch_source_stand_past_15m_count']))
        self.assertEqual(y.loc[1,'batch_source_stand_past_15m_count'],0.)
        self.assertTrue(np.isnan(y.loc[0,'retro_past_arr_runway_count']))
        self.assertEqual(y.loc[0,'flat_dep1_same_stand'],0.)
        self.assertEqual(y.loc[0,'flat_arr1_same_runway'],0.)
        self.assertEqual(y.loc[2,'flat_arr1_same_runway'],1.)
        pd.testing.assert_series_equal(x.STAND_mvt,y.STAND_mvt)
        pd.testing.assert_series_equal(x.proxy_missing,y.proxy_missing)
        pd.testing.assert_series_equal(x.batch_surface_T_observed_active_runways_15m,y.batch_surface_T_observed_active_runways_15m)
        self.assertEqual(x.loc[0,'flat_dep1_same_stand'],1.)


if __name__ == '__main__':
    unittest.main()
