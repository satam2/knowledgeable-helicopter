import unittest
import pandas as pd
import numpy as np
from run_alias_rotation_models import extensions,stability,SUFFIXES


class ExtensionTests(unittest.TestCase):
    def test_raw_unchanged_and_nonrome_duplicate_blocks(self):
        ids=pd.Index([9,2,7],name="MVT_ID_mvt")
        airports=pd.Series(["EDDF","LIRF","EGLL"],index=ids)
        raw=pd.DataFrame(np.arange(24).reshape(3,8),index=ids,columns=["opdi_"+s for s in SUFFIXES]).astype(float)
        raw.iloc[0,2]=np.nan
        pilot=pd.DataFrame(index=pd.Index([2],name=ids.name))
        for mode in ["raw","learned","lexical"]:
            for s in SUFFIXES:pilot[f"alias_{mode}_"+s]=raw.loc[2,"opdi_"+s]+(0 if mode=="raw" else 100)
        ext=extensions(ids,airports,raw,pilot)
        self.assertEqual(ext.shape,(3,24))
        self.assertTrue(ext.index.equals(ids))
        np.testing.assert_array_equal(ext.iloc[[0,2],:8],ext.iloc[[0,2],8:16])
        np.testing.assert_array_equal(ext.iloc[[0,2],:8],ext.iloc[[0,2],16:24])
        self.assertEqual(ext.iloc[0,2],-999999)
        self.assertEqual(ext.loc[2,"alias_learned_match"],108)
        with self.assertRaises(ValueError):extensions(ids,airports,raw,pilot.iloc[:0])

    def test_stability_detects_single_day_gain(self):
        t=pd.Series(pd.to_datetime(["2025-06-01","2025-06-02","2025-06-03"],utc=True))
        result=stability([0,0,0],[0,5,5],[100,0,0],t)
        self.assertFalse(result["all_day_removals_improve"])
        self.assertGreater(result["day_removals"][0]["delta_rmse"],0)


if __name__=="__main__":unittest.main()
