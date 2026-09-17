import unittest
import numpy as np
import pandas as pd
from audit_dates import majority,date_neighbors,ID


class DateTests(unittest.TestCase):
    def test_ties_are_unknown(self):
        day,purity,n=majority(np.array([[1,1,2,2],[1,1,1,2],[np.nan]*4,[2,2,np.nan,np.nan]]))
        np.testing.assert_allclose(day,[np.nan,1,np.nan,2],equal_nan=True)
        np.testing.assert_allclose(purity,[.5,.75,0,1])
        np.testing.assert_array_equal(n,[4,4,0,2])

    def test_self_excluded_and_bracketing(self):
        events=pd.DataFrame({ID:[1.,2.,3.,4.,5.],'airport':['A']*5,'month':['2025-01']*5,'source_day':[10.,10.,99.,10.,10.]})
        result=date_neighbors(events.iloc[[2]],events,2)
        self.assertEqual(result.day.iloc[0],10.)
        self.assertEqual(result.purity.iloc[0],1.)
        self.assertEqual(result.bracketed.iloc[0],1.)


if __name__=='__main__':
    unittest.main()
