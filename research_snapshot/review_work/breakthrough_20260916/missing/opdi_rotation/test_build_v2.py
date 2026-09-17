import unittest
import pandas as pd
from test_build import RotationTests
from build_v2 import build_month

class MoreTests(unittest.TestCase):
    def test_prior_must_end_before_takeoff(self):
        fixture = RotationTests()
        public = fixture.public()
        public.loc[1, "first_seen"] = "2025-07-01 12:05:00Z"
        public.loc[0, "last_seen"] = "2025-07-01 12:01:00Z"
        result = build_month(fixture.query(), public, "202507").loc[1]
        self.assertEqual(result.opdi_match, 1)
        self.assertEqual(result.opdi_previous_same_airport, 0)
        self.assertTrue(pd.isna(result.opdi_ground_interval_sec))
    def test_invalid_public_leg(self):
        fixture = RotationTests()
        public = fixture.public()
        public.loc[1, "last_seen"] = "2025-07-01 11:54:00Z"
        self.assertEqual(build_month(fixture.query(), public, "202507").loc[1].opdi_match, 0)

if __name__ == "__main__":
    unittest.main()
