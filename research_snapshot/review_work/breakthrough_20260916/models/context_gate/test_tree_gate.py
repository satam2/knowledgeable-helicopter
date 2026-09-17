import lightgbm
import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tree_gate


class DirectMixture(unittest.TestCase):
    def test_gradient_hessian_and_zero_disagreement(self):
        d=np.array([0.,1e-12,-100.,250.])
        residual=np.array([1e8,-5.,100.,-30.])
        a=np.array([.2,.4,.3,.8])
        scale=10000.
        grad,hess=tree_gate.derivatives(a,d,residual,scale)
        epsilon=1e-5
        numerical=((d*(a+epsilon)-residual)**2-(d*(a-epsilon)-residual)**2)/(2*epsilon*scale)
        np.testing.assert_allclose(grad,numerical,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(hess,2*d*d/scale)
        self.assertEqual(grad[0],0.)
        self.assertEqual(hess[0],0.)

    def test_initial_score_prediction_offset_and_replay(self):
        n=2400
        x=pd.DataFrame({'regime':np.repeat([-1.,1.],n//2),'airport':pd.Categorical(['A']*n)})
        first=np.zeros(n)
        second=np.full(n,100.)
        y=np.where(x.regime<0,20.,80.)
        second[-1]=0.
        y[-1]=1000.
        model,evidence=tree_gate.fit(x,y,first,second,.37)
        self.assertEqual(evidence['first_objective_init_max_delta'],0.)
        self.assertEqual(evidence['zero_disagreement_rows'],1)
        predicted,alpha,unbounded=tree_gate.predict(model,x,first,second)
        raw=model['estimator'].predict(model['encoder'].transform(x),num_threads=2)
        np.testing.assert_array_equal(unbounded,raw+.37)
        self.assertEqual(len(predicted),n)
        self.assertEqual(predicted[-1],first[-1])
        self.assertEqual((predicted[-1]-y[-1])**2,1e6)
        self.assertLess(np.mean((predicted[:-1]-y[:-1])**2),1.)
        self.assertTrue(np.all((alpha>=0)&(alpha<=1)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gate.joblib'
            joblib.dump(model,path)
            replay,_,_=tree_gate.predict(joblib.load(path),x,first,second)
        np.testing.assert_array_equal(predicted,replay)


if __name__=='__main__':
    unittest.main(verbosity=2)
