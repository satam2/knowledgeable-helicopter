import unittest
import run_v1 as run


class CanaryGuards(unittest.TestCase):
    def test_resource_boundaries(self):
        run.check_memory(100,200,8*1024**3)
        with self.assertRaises(AssertionError):run.check_memory(run.CAP,200,9*1024**3)
        with self.assertRaises(AssertionError):run.check_memory(100,run.CAP,9*1024**3)
        with self.assertRaises(AssertionError):run.check_memory(100,200,8*1024**3-1)

    def test_structured_leaf_checks(self):
        node=dict(leaf_value=1.,leaf_const=1.,leaf_features=[0,1],leaf_coeff=[2.,3.])
        dump={'tree_info':[{'tree_structure':node}]}
        result=run.inspect_dump(dump,{2})
        self.assertEqual(result['linear_coefficients'],2)
        with self.assertRaises(AssertionError):run.inspect_dump(dump,{1})
        node['leaf_coeff']=[float('nan'),3.]
        with self.assertRaises(AssertionError):run.inspect_dump(dump,{2})


if __name__=='__main__':unittest.main()
