import unittest
import pandas as pd
import build
from test_build import RotationTests
from build_v3 import build_month

class V3Tests(unittest.TestCase):
    def test_nonairport_latest_after_takeoff_is_ineligible(self):
        fixture = RotationTests()
        public = fixture.public()
        public.loc[1, "first_seen"] = "2025-07-01 12:05:00Z"
        bad = public.iloc[[0]].copy()
        bad["first_seen"] = "2025-07-01 10:01:00Z"
        bad["last_seen"] = "2025-07-01 12:01:00Z"
        bad["ades"] = "EDDM"
        result = build_month(fixture.query(), pd.concat([public, bad]), "202507").loc[1]
        self.assertEqual(result.opdi_previous_same_airport, 1)
        self.assertEqual(result.opdi_previous_leg_age_at_takeoff_sec, 7200)
    def test_label_columns_cannot_change_features(self):
        fixture = RotationTests()
        query = fixture.query()
        baseline = build_month(query, fixture.public(), "202507")
        query["TAXITIME_SEC_mvt"] = 999999
        query["BLOCK_TIME_UTC_mvt"] = "2020-01-01"
        pd.testing.assert_frame_equal(baseline, build_month(query, fixture.public(), "202507"))

if __name__ == "__main__":
    unittest.main()
