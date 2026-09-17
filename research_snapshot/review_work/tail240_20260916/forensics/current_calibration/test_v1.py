import unittest
import run_v1 as run
import numpy as np
import pandas as pd


class CalibrationContracts(unittest.TestCase):
    def test_tied_boundary_and_related_flight_purge(self):
        frame=pd.DataFrame({run.ID:np.arange(10),run.TIME:pd.to_datetime(['2025-06-01']*3+['2025-06-02']*4+['2025-06-03']*3,utc=True),
            run.FID:[10,20,None,30,40,50,60,10,None,70]})
        fit,early,late,purged,cutoff=run.partitions(frame)
        self.assertEqual(str(cutoff),'2025-06-02 00:00:00+00:00')
        self.assertEqual(np.flatnonzero(late).tolist(),list(range(3,10)))
        self.assertEqual(np.flatnonzero(purged).tolist(),[0])
        self.assertEqual(np.flatnonzero(fit).tolist(),[1,2])
        self.assertTrue(early[2] and late[8])

    def test_source_specific_missing_and_fit_vocabulary(self):
        raw=np.array([-999999,np.nan,np.inf,0.],dtype=np.float32)
        self.assertEqual(run.restore_numeric(raw,False)[0],-999999)
        self.assertTrue(np.isnan(run.restore_numeric(raw,True)[:3]).all())
        values=pd.Series(['early',None,'late_only'])
        vocabulary=run.risk.categories_for(values,np.array([True,True,False]))
        self.assertEqual(vocabulary,['early'])
        np.testing.assert_array_equal(run.risk.category_codes(values,vocabulary),[2,0,1])

    def test_centering_and_fixed_parameters(self):
        p=np.arange(45,dtype=float).reshape(5,9)
        a=run.contrasts(p)
        np.testing.assert_array_equal(a.sum(1),np.zeros(5))
        np.testing.assert_array_equal(a,run.contrasts(p+100))
        self.assertEqual(run.PARAMS,run.inherited_parameters())
        self.assertEqual(run.PARAMS['n_estimators'],200)

    def test_day_and_influence_comparison(self):
        y=np.arange(12,dtype=float)
        dates=pd.to_datetime(['2025-06-01']*4+['2025-06-02']*4+['2025-06-03']*4,utc=True)
        result=run.comparison(y,y+1,y+2,dates)
        self.assertAlmostEqual(result['gain'],1.)
        self.assertTrue(result['every_day_positive'])
        self.assertEqual(result['remove_largest_gain_rows']['10'],1.)


if __name__=='__main__':
    unittest.main()
