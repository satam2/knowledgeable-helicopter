import tempfile
from pathlib import Path
import unittest
import run_v1 as r
import numpy as np
import pandas as pd


class BankTests(unittest.TestCase):
    def test_partitions_and_hidden_metadata(self):
        frame=pd.DataFrame({'ADEP_mvt':['B','A',None],'BLOCK_TIME_UTC_mvt':[1,2,3],r.TARGET:[-10,0,1e9]})
        values=r.airport_values(frame)
        np.testing.assert_array_equal(values,['B','A','__MISSING__'])
        frame[r.TARGET]=999
        np.testing.assert_array_equal(values,r.airport_values(frame))
        plan=r.partitions(['B','A','A'],['A','C','B'])
        self.assertEqual([x[0] for x in plan],['A','B'])
        np.testing.assert_array_equal(plan[0][1],[1,2])
        np.testing.assert_array_equal(plan[1][2],[2])

    def test_native_bank_and_global_fallback(self):
        x=np.arange(32,dtype=np.float32).reshape(16,2)
        fit=np.array(['A']*8+['B']*4)
        tune=np.array(['A','C','A','C'])
        params=dict(n_estimators=5,num_leaves=3,min_child_samples=1,learning_rate=.1,verbosity=-1,n_jobs=2,
            deterministic=True,force_col_wise=True,random_state=42)
        fallback=np.array([700.,800.,900.,1000.])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            prediction,bank,mask=r.train_bank(x,12,np.arange(12,dtype=float),np.arange(4,dtype=float),np.ones(4)*100,
                np.arange(12),np.arange(4)+12,fit,tune,fallback,['f1','f2'],{},params,3,Path(temp))
        self.assertEqual(set(bank),{'A','B'})
        self.assertEqual(bank['B']['steps'],3)
        self.assertTrue(bank['B']['fixed_global_rounds_no_tune'])
        np.testing.assert_array_equal(mask,[False,True,False,True])
        np.testing.assert_array_equal(prediction[mask],fallback[mask])
        self.assertTrue(np.isfinite(prediction).all())

    def test_original_category_mapping(self):
        series=pd.Series(['fit','tune',None])
        values=r.risk.category_codes(series,['fit'])
        np.testing.assert_array_equal(values,[2,1,0])


if __name__=='__main__':unittest.main()


