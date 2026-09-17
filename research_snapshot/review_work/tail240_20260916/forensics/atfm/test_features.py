import unittest
import numpy as np
import pandas as pd
from build_features import daily_features,attach,ratio,CAUSES,ID,TIME


class AtfmTests(unittest.TestCase):
    def tables(self):
        slot=pd.DataFrame({"APT_ICAO":["LIRF"]*3,"date_utc":["2026-01-01","2026-01-02","2026-01-03"],
            "FLT_DEP_1":[100,100,0],"FLT_DEP_REG_1":[20,0,0],"FLT_DEP_OUT_EARLY_1":[2,0,0],"FLT_DEP_IN_1":[17,0,0],"FLT_DEP_OUT_LATE_1":[1,0,0]})
        arrival=pd.DataFrame({"APT_ICAO":["LIRF"]*3,"date_utc":slot.date_utc,"FLT_ARR_1":[100,100,0],"DLY_APT_ARR_1":[0,np.nan,np.nan],"FLT_ARR_1_DLY":[0,np.nan,np.nan],"FLT_ARR_1_DLY_15":[0,np.nan,np.nan],**{f"DLY_APT_ARR_{c}_1":[0,np.nan,np.nan] for c in CAUSES}})
        return slot,arrival

    def test_zero_blank_and_undefined_denominator(self):
        d=daily_features(*self.tables())
        self.assertEqual(d.iloc[0].atfm_arrival_delay_minutes_per_arrival,0)
        self.assertEqual(d.iloc[0].atfm_arrival_delay_populated,1)
        self.assertTrue(np.isnan(d.iloc[1].atfm_arrival_delay_minutes_per_arrival))
        self.assertEqual(d.iloc[1].atfm_arrival_delay_populated,0)
        self.assertTrue(np.isnan(d.iloc[1].atfm_early_fraction))
        self.assertEqual(d.iloc[1].atfm_regulated_fraction,0)
        self.assertTrue(np.isnan(d.iloc[2].atfm_regulated_fraction))
        self.assertAlmostEqual(float(d.iloc[0].atfm_early_fraction),.1,places=6)

    def test_duplicate_keys_rejected(self):
        s,a=self.tables()
        with self.assertRaises(ValueError):daily_features(pd.concat([s,s.iloc[[0]]]),a)

    def test_utc_date_join_order_and_missing(self):
        d=daily_features(*self.tables())
        q=pd.DataFrame({ID:[9,2,5],"ADEP_mvt":["LIRF","LIRF","EDDF"],TIME:["2026-01-02T00:30:00+02:00","2026-01-02T02:00:00+00:00","2026-01-01T10:00:00+00:00"]})
        x=attach(q,d)
        self.assertEqual(x.index.tolist(),[9,2,5])
        self.assertEqual(x.loc[9,"atfm_regulated_fraction"],np.float32(.2))
        self.assertEqual(x.loc[2,"atfm_regulated_fraction"],0)
        self.assertEqual(x.loc[5,"atfm_slot_row_present"],0)
        self.assertTrue(np.isnan(x.loc[5,"atfm_departures"]))
        with self.assertRaises(ValueError):attach(pd.concat([q,q.iloc[[0]]]),d)


if __name__=="__main__":unittest.main()
