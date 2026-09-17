"""Record-context feature arithmetic and target isolation."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Presence(unittest.TestCase):
    def test_wrapper_exists(self):
        self.assertTrue((HERE / 'run_id_context.py').is_file())


@unittest.skipUnless((HERE / 'run_id_context.py').exists(), 'Not implemented')
class Features(unittest.TestCase):
    def test_peer_schedule_offset_and_calendar_alignment(self):
        import run_id_context as m
        x = pd.DataFrame({'schedule_proxy_sec': [16 * 3600.]}, index=pd.Index([5], name='MVT_ID_mvt'))
        ext = pd.DataFrame({f'id_peer_median_time_minus_query_w{w}': [-16 * 3600.] for w in [2, 8, 32]}, index=x.index)
        for w in [2, 8, 32]:
            ext[f'id_peer_time_spread_w{w}'] = 100.
        timestamps = pd.Series(pd.to_datetime(['2025-07-03 10:00'], utc=True), index=x.index)
        result = m.id_context_features(x, ext, timestamps)
        self.assertEqual(result.idctx_peer_minus_schedule_w2_sec.iloc[0], 0.)
        self.assertEqual(result.idctx_peer_query_day_delta_w2.iloc[0], -1.)
        self.assertEqual(result.idctx_peer_schedule_same_day_w2.iloc[0], 1.)
        self.assertEqual(result.idctx_peer_hour_w2.iloc[0], 18.)
        self.assertEqual(result.idctx_peer_age_w2_sec.iloc[0], 16 * 3600.)

    def test_hidden_targets_cannot_change_features(self):
        import run_id_context as m
        x = pd.DataFrame({'schedule_proxy_sec': [1200.]}, index=pd.Index([5], name='MVT_ID_mvt'))
        ext = pd.DataFrame({f'id_peer_median_time_minus_query_w{w}': [-100.] for w in [2, 8, 32]}, index=x.index)
        for w in [2, 8, 32]:
            ext[f'id_peer_time_spread_w{w}'] = 100.
        timestamps = pd.Series(pd.to_datetime(['2025-07-03 10:00'], utc=True), index=x.index)
        result = m.id_context_features(x, ext, timestamps)
        x['TAXITIME_SEC_mvt'] = 9999999.
        pd.testing.assert_frame_equal(result, m.id_context_features(x, ext, timestamps))

    def test_feature_id_order_must_match(self):
        import run_id_context as m
        x = pd.DataFrame({'schedule_proxy_sec': [1200.]}, index=[5])
        with self.assertRaises(ValueError):
            m.id_context_features(x, pd.DataFrame(index=[6]), pd.Series(pd.to_datetime(['2025-07-03'], utc=True), index=[5]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
