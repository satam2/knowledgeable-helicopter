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


    def test_cross_batch_duplicate_hash_and_schema(self):
        repeated = self.source('repeat.parquet',[1,1],[1.,2.])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            contract.validate([repeated],pd.Index([1]),['x'],batch_size=1)
        good = self.source('hash.parquet',[1],[1.])
        with self.assertRaisesRegex(ValueError,'hash'):
            contract.validate([(good[0],'0'*64,False,['x'])],pd.Index([1]),['x'])
        category = self.source('category.parquet',[2],['one'])
        with self.assertRaisesRegex(ValueError,'schema'):
            contract.validate([good,category],pd.Index([1,2]),['x'])

    def test_wrapper_binds_guard_validates_path_and_restores_on_both_exits(self):
        from types import SimpleNamespace
        good = self.source('wrapper.parquet',[1],[1.])
        for fail in [False,True]:
            original = lambda columns: 'original'
            original_guard = lambda: 'original_guard'
            calls = []
            active_guard = lambda: calls.append('guard')
            def external(path):
                calls.append(('external',path))
                return path
            fake = SimpleNamespace(feature_sources=original,guard=original_guard,
                                   common=SimpleNamespace(external_path=external))
            def loader(ids,nfit,columns,folder):
                self.assertEqual(fake.feature_sources(columns),[good])
                self.assertIs(fake.guard,active_guard)
                with self.assertRaisesRegex(ValueError,'schema'):
                    fake.feature_sources(['wrong'])
                if fail:
                    raise RuntimeError('deliberate')
                return 'matrix','vocab','receipt'
            fake.load_matrix = loader
            if fail:
                with self.assertRaisesRegex(RuntimeError,'deliberate'):
                    contract.load_checked(fake,[good],pd.Index([1]),1,['x'],self.root,guard=active_guard)
            else:
                result = contract.load_checked(fake,[good],pd.Index([1]),1,['x'],self.root,guard=active_guard)
                self.assertEqual(result[:3],('matrix','vocab','receipt'))
            self.assertIs(fake.feature_sources,original)
            self.assertIs(fake.guard,original_guard)
            self.assertIn(('external',self.root),calls)


if __name__ == '__main__':
    unittest.main()
