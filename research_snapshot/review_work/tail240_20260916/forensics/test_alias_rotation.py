import unittest
import pandas as pd
import numpy as np
from alias_rotation import build_month,ID,TIME


def query():
    return pd.DataFrame({ID:[42],TIME:pd.to_datetime(["2025-07-02T12:00:00Z"]),"FLIGHT_mvt":["ITYTY680"],"ADEP_mvt":["LIRF"],"ADES_mvt":["KJFK"]})


def public():
    return pd.DataFrame({"id":["9007199254740993","9007199254740995"],"icao24":["ABC123"]*2,"flt_id":["ITY679","ITY680"],"adep":["KJFK","LIRF"],"ades":["LIRF","KJFK"],"first_seen":pd.to_datetime(["2025-07-01T21:00Z","2025-07-02T12:01Z"]),"last_seen":pd.to_datetime(["2025-07-02T08:00Z","2025-07-02T20:00Z"])})


class RotationTests(unittest.TestCase):
    def test_alias_strict_route_exact_id_previous(self):
        x,w=build_month(query(),public(),{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_raw_match"],0)
        self.assertEqual(x.loc[42,"alias_learned_match"],1)
        self.assertEqual(x.loc[42,"alias_lexical_match"],1)
        self.assertEqual(x.loc[42,"alias_learned_previous_same_airport"],1)
        self.assertEqual(x.loc[42,"alias_learned_ground_interval_sec"],14460)
        self.assertEqual(w.iloc[0].public_id,"9007199254740995")
        self.assertEqual(w.iloc[0].previous_public_id,"9007199254740993")

    def test_blank_tail_and_duplicate_ambiguity(self):
        p=public();p["icao24"]=" "
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_match"],0)
        p=public();p=pd.concat([p,p.iloc[[1]]],ignore_index=True)
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_match_ambiguous"],1)
        self.assertEqual(x.loc[42,"alias_learned_match"],0)

    def test_overlap_and_completion_before_query(self):
        p=public();p.loc[0,"last_seen"]=pd.Timestamp("2025-07-02T12:02Z")
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_previous_leg_available"],0)
        p=public();p.loc[0,"last_seen"]=pd.Timestamp("2025-07-02T12:00:30Z")
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_previous_leg_available"],0)

    def test_month_route_and_tolerance(self):
        p=public();p.loc[0,"first_seen"]=pd.Timestamp("2025-06-30T21:00Z")
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_previous_leg_available"],0)
        p=public();p.loc[1,"ades"]="KBOS"
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_match"],0)
        p=public();p.loc[1,"first_seen"]=pd.Timestamp("2025-07-02T12:10:01Z")
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_match"],0)

    def test_previous_destination_mismatch(self):
        p=public();p.loc[0,"ades"]="LIMC"
        x,_=build_month(query(),p,{"ITYTY":"ITY"},"202507")
        self.assertEqual(x.loc[42,"alias_learned_previous_leg_available"],1)
        self.assertEqual(x.loc[42,"alias_learned_previous_same_airport"],0)
        self.assertTrue(np.isnan(x.loc[42,"alias_learned_ground_interval_sec"]))


if __name__=="__main__":unittest.main()
