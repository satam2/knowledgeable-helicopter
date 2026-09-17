import unittest
import numpy as np
import pandas as pd
import build_cache as builder
import audit_ids as base


class CacheTests(unittest.TestCase):
    def frame(self):
        data=pd.DataFrame({base.ID:[1,2,3,4],base.FLIGHT_ID:[10.,11.,12.,np.nan],base.PHASE:['DEP']*4,
            base.MOVEMENT:pd.to_datetime(['2025-01-03 10:00Z','2025-02-03 10:00Z','2025-01-03 10:02Z','2025-01-03 10:03Z'])})
        data['AOBT_3_flt']=data[base.MOVEMENT]-pd.Timedelta(minutes=10)
        data['EOBT_1_flt']=data[base.MOVEMENT]-pd.Timedelta(minutes=20)
        return data

    def test_context_month_and_query_order(self):
        data=self.frame()
        result=builder.transform(data)
        np.testing.assert_array_equal(result[base.ID],data[base.ID])
        self.assertEqual(result.nmid_peer_nm_count.iloc[0],1.)
        self.assertEqual(result.nmid_peer_nm_minus_own_nm.iloc[0],120.)
        self.assertEqual(result.nmid_peer_nm_count.iloc[1],0.)
        self.assertTrue(np.isnan(result.nmid_peer_nm_minus_own_nm.iloc[1]))
        self.assertEqual(result.nmid_source_id_present.iloc[3],0.)

    def test_hidden_inputs_rejected(self):
        data=self.frame()
        data[base.TARGET]=900.
        with self.assertRaisesRegex(ValueError,'Hidden'):
            builder.transform(data)


if __name__=='__main__':
    unittest.main()
