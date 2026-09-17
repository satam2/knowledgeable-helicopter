"""Independent edge cases for packed coverage and transactional loader binding."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from audit_union387_sources import ROOT

path=ROOT/'review_work/tail240_20260916/forensics/source_contract/v2/contract.py'
spec=importlib.util.spec_from_file_location('independent_source_contract',path)
subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)


class ContractIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def source(self,filename,ids,values):
        path=self.root/filename
        pd.DataFrame({subject.ID:ids,'x':values}).to_parquet(path,index=False)
        return path,subject.sha256(path),False,['x']

    def test_packed_coverage_random_order_and_batch_boundaries(self):
        rng=np.random.default_rng(921)
        requested=pd.Index(np.arange(41))
        for attempt in range(8):
            shuffled=rng.permutation(41);cut=int(rng.integers(1,40))
            sources=[self.source(f'{attempt}a.parquet',shuffled[:cut],shuffled[:cut].astype(float)),
                     self.source(f'{attempt}b.parquet',shuffled[cut:],shuffled[cut:].astype(float))]
            receipt=subject.validate(sources,requested,['x'],batch_size=3)
            self.assertEqual(receipt['coverage']['x']['rows'],41)

    def test_cross_batch_repeat_and_omission_cannot_cancel(self):
        source=self.source('repeat.parquet',[0,1,2,0],[1.,2.,3.,4.])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            subject.validate([source],pd.Index([0,1,2,3]),['x'],batch_size=1)

    def test_binding_restored_after_success_and_loader_exception(self):
        source=self.source('valid.parquet',[1,2],[1.,2.])
        class Fake:
            def __init__(self):
                self.common=SimpleNamespace(external_path=lambda path:Path(path).resolve())
                self.feature_sources=lambda values:['original']
                self.guard=lambda:123
                self.fail=False
            def load_matrix(self,ids,nfit,columns,folder):
                self.asserted=(self.feature_sources(columns)==[source] and self.guard()==999)
                if self.fail:raise RuntimeError('deliberate')
                return np.array([[1.],[2.]],dtype='float32'),{},[]
        fake=Fake();oldsource,oldguard=fake.feature_sources,fake.guard
        _,_,_,receipt=subject.load_checked(fake,[source],pd.Index([1,2]),1,['x'],self.root,guard=lambda:999)
        self.assertTrue(fake.asserted);self.assertEqual(receipt['status'],'passed')
        self.assertIs(fake.feature_sources,oldsource);self.assertIs(fake.guard,oldguard)
        fake.fail=True
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            subject.load_checked(fake,[source],pd.Index([1,2]),1,['x'],self.root,guard=lambda:999)
        self.assertIs(fake.feature_sources,oldsource);self.assertIs(fake.guard,oldguard)

    def test_source_mutation_during_loader_is_detected_and_bindings_restored(self):
        source=self.source('mutation.parquet',[1],[1.])
        class Fake:
            def __init__(self):
                self.common=SimpleNamespace(external_path=lambda path:Path(path).resolve())
                self.feature_sources=lambda values:[]
                self.guard=lambda:None
            def load_matrix(self,*args):
                pd.DataFrame({subject.ID:[1],'x':[2.]}).to_parquet(source[0],index=False)
                return np.array([[2.]],dtype='float32'),{},[]
        fake=Fake();oldsource,oldguard=fake.feature_sources,fake.guard
        with self.assertRaisesRegex(ValueError,'changed during'):
            subject.load_checked(fake,[source],pd.Index([1]),1,['x'],self.root)
        self.assertIs(fake.feature_sources,oldsource);self.assertIs(fake.guard,oldguard)


if __name__=='__main__':unittest.main()
