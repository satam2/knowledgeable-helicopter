import unittest
import numpy as np
import pandas as pd
from run_anchor_mixture import encode,expectation,replicas,physical_reference,build_anchors,ID,TARGET,TIME


class AnchorTests(unittest.TestCase):
    def test_exact_reconstruction_negative_and_outside_hull(self):
        a=np.array([[900,8100,5000,87300]]*7,dtype=float)
        y=np.array([-300,900,1000,5000,7500,87300,131167.])
        q,r,active=encode(y,a)
        np.testing.assert_allclose((q*a).sum(axis=1)+r,y,atol=1e-9,rtol=0)
        np.testing.assert_allclose(q.sum(axis=1),1)
        self.assertEqual(r[0],-1200)
        self.assertEqual(r[-1],43867)
        self.assertTrue((q>=0).all())
        self.assertTrue(((q>0).sum(axis=1)<=2).all())

    def test_missing_and_equal_anchors(self):
        a=np.array([[900,np.nan,np.nan,87300],[900,900,900,87300],[900,np.nan,-1500,87300]])
        y=np.array([8000,1000,-1700])
        q,r,mask=encode(y,a)
        self.assertFalse(mask[1,1])
        self.assertFalse(mask[1,2])
        self.assertTrue((q[~mask]==0).all())
        np.testing.assert_allclose(np.sum(np.nan_to_num(a)*q,axis=1)+r,y,atol=1e-9)
        np.testing.assert_allclose(expectation(q,np.arange(4),a)+r,y,atol=1e-9)

    def test_probability_mean_masks_unsupported(self):
        a=np.array([[900,np.nan,900,87300]])
        p=expectation(np.array([[.2,.4,.2,.2]]),[0,1,2,3],a)
        np.testing.assert_allclose(p,[44100])
        np.testing.assert_allclose(expectation(np.array([[1.]]),[1],a),[900])

    def test_replica_ids_and_weight_mass(self):
        x=pd.DataFrame({"f":[1.,2.]},index=pd.Index([101,202],name=ID))
        q=np.array([[.2,.8,0,0],[0,0,1,0]])
        rx,cls,w,ids=replicas(x,q)
        self.assertEqual(ids.tolist(),[101,101,202])
        self.assertEqual(rx.index.tolist(),ids.tolist())
        np.testing.assert_allclose(pd.Series(w).groupby(ids).sum(),[1,1])

    def test_prior_uses_earlier_labels_and_empty_fallback(self):
        history=pd.DataFrame({ID:[1,2],TIME:pd.to_datetime(["2025-01-01","2025-01-02"],utc=True),TARGET:[500.,900.],"ADEP_mvt":["LIRF"]*2,"RUNWAY_mvt":["25"]*2,"STAND_mvt":["A"]*2}).set_index(ID)
        query=pd.DataFrame({ID:[3],TIME:pd.to_datetime(["2025-02-01"],utc=True),"ADEP_mvt":["LIRF"],"RUNWAY_mvt":["25"],"STAND_mvt":["A"]}).set_index(ID)
        self.assertEqual(physical_reference(history,query).physical_prior_sec.iloc[0],700)
        self.assertEqual(physical_reference(history.iloc[:0],query).physical_prior_sec.iloc[0],900)
        with self.assertRaises(ValueError):
            physical_reference(history,history.iloc[:1])

    def test_dst_only_rome_and_missing_schedule(self):
        rows=pd.DataFrame({TIME:pd.to_datetime(["2025-01-01","2025-07-01","2025-07-01"],utc=True),"ADEP_mvt":["LIRF","LIRF","EGLL"],"schedule_sec":[np.nan,5000.,1000.]})
        refs=pd.DataFrame({"physical_prior_sec":[900.]*3})
        a,offset=build_anchors(rows,refs,True)
        np.testing.assert_array_equal(offset,[3600,7200,0])
        self.assertTrue(np.isnan(a[0,2]))
        self.assertEqual(a[2,0],a[2,1])


if __name__ == "__main__":
    unittest.main()
