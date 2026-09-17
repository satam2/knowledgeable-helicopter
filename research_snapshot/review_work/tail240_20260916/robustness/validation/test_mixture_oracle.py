import unittest
import numpy as np
from mixture_oracle import simplex_oracle


class OracleTests(unittest.TestCase):
    def test_interior_known_weights(self):
        p=np.eye(3);w=np.array([.2,.3,.5]);result=simplex_oracle(p,p@w)
        np.testing.assert_allclose(result['weights'],w,atol=1e-12)
        self.assertLess(abs(result['mse']),1e-15)

    def test_vertex_optimum(self):
        p=np.array([[1.,2.,3.],[1.,3.,2.]])
        result=simplex_oracle(p,np.zeros(2))
        np.testing.assert_allclose(result['weights'],[1,0,0],atol=1e-12)
        self.assertAlmostEqual(result['mse'],1.)

    def test_redundant_experts(self):
        p=np.ones((10,3));result=simplex_oracle(p,np.zeros(10))
        self.assertAlmostEqual(result['mse'],1.)
        self.assertAlmostEqual(result['weights'].sum(),1.)


if __name__=='__main__':unittest.main()
