import unittest
import numpy as np
import pandas as pd
from audit_ids import neighbors, ID, FLIGHT_ID, PHASE, MOVEMENT


class NeighborTests(unittest.TestCase):
    def test_excludes_all_same_id_and_aggregates_other_duplicates(self):
        context=pd.DataFrame({ID:[1,2,3,4,5],FLIGHT_ID:[10.,20.,20.,30.,30.],PHASE:['DEP','DEP','ARR','DEP','ARR'],
            MOVEMENT:pd.to_datetime(['2025-01-01 01:00Z','2025-01-01 02:00Z','2025-01-01 02:00Z','2025-01-01 03:00Z','2025-01-01 03:00Z'])})
        context['AOBT_3_flt']=context[MOVEMENT]
        context['EOBT_1_flt']=context[MOVEMENT]
        queries=context.iloc[[1]].copy()
        baseline=neighbors(queries,context,width=1)
        context.loc[context[FLIGHT_ID].eq(20),'AOBT_3_flt']=pd.Timestamp('2030-01-01',tz='UTC')
        pd.testing.assert_frame_equal(baseline,neighbors(queries,context,width=1))
        self.assertEqual(baseline.nmid_peer_nm_count.iloc[0],2.)
        self.assertEqual(baseline.nmid_peer_nm_seconds.iloc[0],pd.Timestamp('2025-01-01 02:00Z').timestamp())

    def test_missing_id_has_no_neighbors(self):
        context=pd.DataFrame({ID:[1],FLIGHT_ID:[np.nan],PHASE:['DEP'],MOVEMENT:[pd.Timestamp('2025-01-01',tz='UTC')]})
        context['AOBT_3_flt']=context[MOVEMENT]
        context['EOBT_1_flt']=context[MOVEMENT]
        result=neighbors(context,context)
        self.assertTrue(np.isnan(result.nmid_peer_nm_seconds.iloc[0]))
        self.assertEqual(result.nmid_peer_nm_count.iloc[0],0.)


if __name__=='__main__':
    unittest.main()
