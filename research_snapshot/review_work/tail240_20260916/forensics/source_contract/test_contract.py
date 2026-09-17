import tempfile
from pathlib import Path
import unittest

import contract
import numpy as np
import pandas as pd


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, ids, values, fill=False):
        path = self.root/name
        pd.DataFrame({contract.ID:ids,'x':values}).to_parquet(path,index=False)
        return path, contract.sha256(path), fill, ['x']

    def test_duplicate_plus_missing_cancellation_rejected(self):
        sources = [self.source('a.parquet',[1,2],[10.,20.]),
                   self.source('b.parquet',[2,4],[20.,40.])]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            contract.validate(sources,pd.Index([1,2,3,4]),['x'])

    def test_reorder_and_out_of_cohort_rows_are_valid(self):
        sources = [self.source('a.parquet',[3,1,999],[30.,10.,1e100]),
                   self.source('b.parquet',[2],[20.])]
        result = contract.validate(sources,pd.Index([2,1,3]),['x'])
        self.assertEqual(result['coverage']['x']['rows'],3)
        self.assertEqual(result['sources'][0]['outside_requested_rows'],1)

    def test_missing_and_in_file_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing'):
            contract.validate([self.source('a.parquet',[1],[1.])],pd.Index([1,2]),['x'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            contract.validate([self.source('b.parquet',[1,1],[1.,1.])],pd.Index([1]),['x'])

    def test_finite_overflow_and_rounded_sentinel_rejected(self):
        for name,value,reason in [('overflow',1e100,'nonfinite'),('collision',-999999.01,'sentinel')]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError,reason):
                contract.validate([self.source(name+'.parquet',[1],[value])],pd.Index([1]),['x'])

    def test_actual_sentinel_nan_and_negative_are_preserved(self):
        source = self.source('valid.parquet',[1,2,3],[-999999.,np.nan,-17.])
        before = pd.read_parquet(source[0])
        receipt = contract.validate([source],pd.Index([1,2,3]),['x'])
        pd.testing.assert_frame_equal(before,pd.read_parquet(source[0]))
        self.assertEqual(receipt['coverage']['x']['preexisting_sentinel'],1)
        self.assertEqual(receipt['coverage']['x']['nonfinite_input'],1)

    def test_same_ids_in_disjoint_features_allowed(self):
        x = self.source('x.parquet',[1],[1.])
        path = self.root/'y.parquet'
        pd.DataFrame({contract.ID:[1],'y':['cat']}).to_parquet(path,index=False)
        result = contract.validate([x,(path,contract.sha256(path),False,['y'])],pd.Index([1]),['x','y'])
        self.assertEqual(result['coverage']['y']['rows'],1)

    def test_wrapper_never_calls_loader_on_failure(self):
        class FakeRisk:
            feature_sources = None
            def load_matrix(self,*args):
                raise AssertionError('unsafe loader called')
        bad = self.source('bad.parquet',[1],[1e100])
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            contract.load_checked(FakeRisk(),[bad],pd.Index([1]),1,['x'],self.root)


if __name__ == '__main__':
    unittest.main()
