import unittest
import numpy as np
import pandas as pd
import build_v1 as b


class NMClockPeersTest(unittest.TestCase):
    def fixture(self):
        t = pd.Timestamp('2025-01-15T12:00:00Z')
        rows = []
        for ident, flight, delta, proxy, stand in [(1,100,0,600,'S1'), (2,200,-900,-30,'S1'),
            (3,300,900,9000,'S1'), (4,400,0,300,'S2'), (5,100,60,500,'S1'),
            (6,np.nan,0,100,'S1'), (7,700,901,600,'S1'), (8,200,-900,-30,'S1')]:
            n = t + pd.Timedelta(seconds=delta)
            row = {b.ID:float(ident), b.FID:float(flight), b.PHASE:'DEP', b.TIME:n+pd.Timedelta(seconds=proxy),
                'ADEP_mvt':'AAA','STAND_mvt':stand, **{c:n for c in b.CLOCKS}}
            rows.append(row)
        rows[3]['EOBT_1_flt'] = pd.NaT
        missing = rows[0].copy()
        missing.update({b.ID:9.,b.FID:900.,b.CLOCKS[0]:pd.NaT})
        rows.append(missing)
        return pd.DataFrame(rows)

    def test_boundaries_ties_exclusions_and_raw_values(self):
        result = b.build_frame(b.prepare(self.fixture())).set_index(b.ID)
        q = result.loc[1.]
        self.assertEqual(q.nmpeer_airport_15m_count, 4.)
        self.assertEqual(q.nmpeer_airport_15m_nm_mean_sec, np.float32((-30+9000+300+100)/4))
        self.assertEqual(q.nmpeer_airport_15m_exact_nm_tie_fraction, .5)
        self.assertEqual(q.nmpeer_airport_15m_est_finite_count, 3.)
        self.assertEqual(q.nmpeer_stand_15m_count, 3.)
        self.assertEqual(q.nmpeer_airport_60m_count, 5.)
        self.assertEqual(result.loc[9., 'nmpeer_anchor_missing'], 1.)
        self.assertEqual(result.loc[9., 'nmpeer_airport_60m_count'], 0.)
        self.assertTrue(np.isnan(result.loc[9., 'nmpeer_airport_60m_nm_mean_sec']))
        self.assertEqual(result.loc[6., 'nmpeer_airport_15m_count'], 4.)

    def test_hidden_columns_and_month_boundary(self):
        raw = self.fixture()
        expected = b.build_frame(b.prepare(raw))
        raw['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('1900-01-01T00:00Z')
        raw['TAXITIME_SEC_mvt'] = -999999999.
        pd.testing.assert_frame_equal(expected, b.build_frame(b.prepare(raw)))
        x = b.prepare(raw)
        x.loc[x[b.ID].eq(3), 'month'] = '2025-02'
        self.assertEqual(b.build_frame(x).set_index(b.ID).loc[1., 'nmpeer_airport_15m_count'], 3.)
        x.loc[x[b.ID].eq(1), 'stand'] = ''
        self.assertEqual(b.build_frame(x).set_index(b.ID).loc[1., 'nmpeer_stand_60m_count'], 0.)


if __name__ == '__main__':
    unittest.main()
