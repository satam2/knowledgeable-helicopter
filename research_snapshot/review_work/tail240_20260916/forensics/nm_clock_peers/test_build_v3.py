import unittest
import numpy as np
import pandas as pd
import build_v1 as b
from test_build_v2 import NMClockPeersTest


class ExtendedBoundaries(unittest.TestCase):
    def test_hour_edges_and_population_variance(self):
        raw = NMClockPeersTest().fixture().iloc[:1].copy()
        center = raw[b.CLOCKS[0]].iloc[0]
        rows = [raw.iloc[0].to_dict()]
        for ident, delta, value in [(20,-3600,-600),(21,3600,10000),(22,-3601,15),(23,3601,25)]:
            row = rows[0].copy()
            n = center + pd.Timedelta(seconds=delta)
            row.update({b.ID:float(ident),b.FID:float(ident),b.TIME:n+pd.Timedelta(seconds=value),
                        **{c:n for c in b.CLOCKS}})
            rows.append(row)
        frame = pd.DataFrame(rows)
        result = b.build_frame(b.prepare(frame)).set_index(b.ID).loc[1.]
        self.assertEqual(result.nmpeer_airport_60m_count, 2.)
        self.assertEqual(result.nmpeer_airport_60m_nm_mean_sec, 4700.)
        self.assertEqual(result.nmpeer_airport_60m_nm_std_sec, 5300.)
        self.assertEqual(result.nmpeer_airport_60m_exact_nm_tie_fraction, 0.)
        self.assertEqual(result.nmpeer_airport_15m_count, 0.)
        self.assertTrue(np.isnan(result.nmpeer_airport_15m_nm_mean_sec))
        frame.loc[frame[b.ID].eq(1.), 'ADEP_mvt'] = None
        result = b.build_frame(b.prepare(frame)).set_index(b.ID).loc[1.]
        self.assertEqual(result.nmpeer_airport_60m_count, 0.)


if __name__ == '__main__':
    unittest.main()
