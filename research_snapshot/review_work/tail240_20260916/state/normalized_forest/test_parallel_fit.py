import unittest
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from parallel_fit import ParallelFitExtraTrees


class ParallelFitTest(unittest.TestCase):
    def test_every_tree_and_prediction_bitwise_equal_weighted_and_raw(self):
        rng = np.random.default_rng(20260916)
        x = rng.normal(size=(601, 17)).astype('float32')
        y = x[:,0]*4 + x[:,1]**2 + rng.normal(size=len(x))
        weight = 1 + x[:,2]**2
        for sample_weight in [None, weight]:
            params = dict(n_estimators=300, max_features=.7, min_samples_leaf=1,
                criterion='squared_error', bootstrap=False, random_state=20260916, n_jobs=1)
            one = ExtraTreesRegressor(**params).fit(x,y,sample_weight=sample_weight)
            four = ParallelFitExtraTrees(**params).fit(x,y,sample_weight=sample_weight)
            self.assertEqual(four.n_jobs,1)
            for left, right in zip(one.estimators_,four.estimators_):
                self.assertEqual(left.random_state,right.random_state)
                for name in ['children_left','children_right','feature','threshold','impurity','n_node_samples','weighted_n_node_samples','value']:
                    np.testing.assert_array_equal(getattr(left.tree_,name),getattr(right.tree_,name))
            np.testing.assert_array_equal(one.predict(x),four.predict(x))


if __name__ == '__main__':
    unittest.main()
