import unittest
import numpy as np
import pandas as pd
import features


class RankingPeerTests(unittest.TestCase):
    def test_self_exclusion_month_airport_and_input_order(self):
        meta = pd.DataFrame({features.ID:[30.,10.,20.,40.,50.], 'ADEP_mvt':['A','A','A','B','A'],
            features.TIME:pd.to_datetime(['2026-01-01 00:03Z','2026-01-01 00:01Z','2026-01-01 00:02Z',
                                         '2026-01-01 01:00Z','2026-07-01 00:00Z'])})
        actual = features.record_peers(meta)
        np.testing.assert_array_equal(actual.index,meta[features.ID])
        np.testing.assert_allclose(actual.iloc[:3]['id_peer_median_time_minus_query_w2'],[-90.,90.,0.])
        self.assertTrue(actual.iloc[3:].isna().all().all())

    def test_independent_bruteforce_values(self):
        rng = np.random.default_rng(7)
        ids = rng.permutation(np.arange(83,dtype=float))
        t = pd.Timestamp('2026-01-01',tz='UTC')+pd.to_timedelta(rng.integers(0,86400,83),unit='s')
        meta = pd.DataFrame({features.ID:ids,'ADEP_mvt':'A',features.TIME:t})
        actual = features.record_peers(meta)
        seconds = t.as_unit('ns').asi8/1e9
        order = np.argsort(ids)
        for width in [2,8,32]:
            for rank,index in enumerate(order):
                peers = seconds[np.concatenate([order[max(0,rank-width):rank],order[rank+1:rank+1+width]])]
                self.assertEqual(actual.iloc[index][f'id_peer_median_time_minus_query_w{width}'],np.median(peers)-seconds[index])
                self.assertEqual(actual.iloc[index][f'id_peer_time_spread_w{width}'],np.quantile(peers,.9)-np.quantile(peers,.1))


if __name__ == '__main__':
    unittest.main()
