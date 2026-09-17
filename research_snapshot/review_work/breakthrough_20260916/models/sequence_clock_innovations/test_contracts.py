import unittest
import numpy as np
import pandas as pd
import build


class Innovations(unittest.TestCase):
    def test_exact_36_pairs(self):
        self.assertEqual(len(build.PAIRS), 36)
        self.assertEqual(len(set((left, right) for _, left, right in build.PAIRS)), 36)
        self.assertEqual(sum(left == build.OWN[0] for _, left, right in build.PAIRS), 20)

    def test_sign_missing_and_no_clipping(self):
        own = pd.DataFrame({c: [100., np.nan, -999999., -2000000., 10000000.] for c in build.OWN})
        peer = pd.DataFrame({c: [130., 40., 40., 5., -2000000.] for c in build.PEER})
        result = build.transform(own, peer)
        for name in build.FEATURES:
            np.testing.assert_array_equal(result[name], np.array([-30., np.nan, np.nan, -2000005., 12000000.], dtype='float32'))
        peer.loc[0, build.PEER] = np.nan
        self.assertTrue(build.transform(own, peer).iloc[0].isna().all())

    def test_hidden_fields_inert_and_inputs_unchanged(self):
        own = pd.DataFrame({c: [20., 500.] for c in build.OWN})
        peers = pd.DataFrame({c: [10., 600.] for c in build.PEER})
        original = own.copy()
        expected = build.transform(own, peers)
        own['TAXITIME_SEC_mvt'] = [1e20, -1e20]
        peers['BLOCK_TIME_UTC_mvt'] = 'poison'
        pd.testing.assert_frame_equal(expected, build.transform(own, peers))
        pd.testing.assert_frame_equal(own[build.OWN], original)


if __name__ == '__main__':
    unittest.main()
