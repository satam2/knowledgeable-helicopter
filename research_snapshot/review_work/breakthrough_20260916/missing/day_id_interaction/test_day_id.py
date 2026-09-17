import unittest
import numpy as np
import pandas as pd
import run as wrapper


class DayIDTests(unittest.TestCase):
    def test_exact_day_target_and_context_schema(self):
        y=np.array([-12.,900.,87598.,180000.])
        day,remainder=wrapper.base.day_parts(y)
        np.testing.assert_array_equal(day*86400+remainder,y)
        ids=pd.Index([1,2,3,4],name=wrapper.ID)
        x=pd.DataFrame({'schedule_proxy_sec':[1000.,2000.,-3000.,90000.]},index=ids)
        times=pd.Series(pd.to_datetime(['2025-01-02']*4,utc=True),index=ids)
        peer=pd.DataFrame(index=ids)
        for width in [2,8,32]:
            peer[f'id_peer_median_time_minus_query_w{width}']=np.array([0.,30.,-60.,-86400.])
            peer[f'id_peer_time_spread_w{width}']=600.
        features=wrapper.context.id_context_features(x,peer,times)
        self.assertEqual(len(features.columns),38)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_probability_mean_not_argmax_day(self):
        result=wrapper.base.expected_days(np.array([[.7,.3],[.9,.1]]),[0,1])
        np.testing.assert_array_equal(result,[.3,.1])


if __name__=='__main__':
    unittest.main()
