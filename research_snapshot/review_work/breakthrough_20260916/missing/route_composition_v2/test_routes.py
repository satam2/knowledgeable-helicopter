import unittest
import numpy as np
from compose import routes


class Routing(unittest.TestCase):
    def test_observed_boundaries_only(self):
        proxy=np.array([np.nan,np.inf,-np.inf,-1.,0.,7200.,7200.01])
        result=routes(proxy)
        np.testing.assert_array_equal(result['ordinary'],[0,0,0,0,1,1,0])
        np.testing.assert_array_equal(result['nonordinary'],[0,0,0,1,0,0,1])
        np.testing.assert_array_equal(result['missing'],[1,1,1,0,0,0,0])
        np.testing.assert_array_equal(sum(v.astype(int) for v in result.values()),1)


if __name__=='__main__':
    unittest.main()
