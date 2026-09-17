import unittest
import numpy as np
import pandas as pd
from audit import prior_matching_fraction,ID,MOVEMENT


class ContextTests(unittest.TestCase):
    def test_strictprior_ties_and_month_isolation(self):
        data=pd.DataFrame({ID:[1,2,3,4,5],MOVEMENT:pd.to_datetime(['2025-01-31 23:50Z','2025-02-01 00:00Z','2025-02-01 00:10Z','2025-02-01 00:10Z','2025-02-01 00:20Z']),
            'ADEP_mvt':['A']*5,'proxy_sec':[900.,900.,900.,1200.,900.]})
        result=prior_matching_fraction(data)
        np.testing.assert_array_equal(result.prior_count,[0,0,1,1,3])
        np.testing.assert_allclose(result.ownminute_match_fraction,[0,0,1,0,2/3])
        np.testing.assert_allclose(result.prior_modal_fraction,[0,0,1,1,2/3])


if __name__=='__main__':
    unittest.main()
