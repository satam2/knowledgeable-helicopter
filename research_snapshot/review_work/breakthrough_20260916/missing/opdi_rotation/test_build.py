import unittest
import pandas as pd
from build import build_month, ID, TIME

class RotationTests(unittest.TestCase):
    def query(self):
        return pd.DataFrame([{ID: 1, TIME: "2025-07-01 12:00:00Z", "FLIGHT_mvt": " a b123 ", "ADEP_mvt": "EGLL", "ADES_mvt": "LFPG"}])
    def public(self):
        return pd.DataFrame([
            {"id": 10, "icao24": "abcd", "flt_id": "PREV", "adep": "EDDF", "ades": "EGLL", "first_seen": "2025-07-01 08:00:00Z", "last_seen": "2025-07-01 10:00:00Z"},
            {"id": 11, "icao24": "abcd", "flt_id": "AB123", "adep": "EGLL", "ades": "LFPG", "first_seen": "2025-07-01 11:55:00Z", "last_seen": "2025-07-01 13:00:00Z"}])
    def test_valid_and_empty(self):
        result = build_month(self.query(), self.public(), "202507").loc[1]
        self.assertEqual(result.opdi_match, 1)
        self.assertEqual(result.opdi_ground_interval_sec, 6900)
        self.assertEqual(result.opdi_previous_leg_duration_sec, 7200)
        self.assertEqual(result.opdi_previous_leg_age_at_takeoff_sec, 7200)
        empty = build_month(self.query(), self.public().iloc[:0], "202507").loc[1]
        self.assertEqual(empty.opdi_match, 0)
        self.assertTrue(pd.isna(empty.opdi_ground_interval_sec))
    def test_ambiguity_and_destination(self):
        public = self.public()
        public = pd.concat([public, public.iloc[[1]]], ignore_index=True)
        result = build_month(self.query(), public, "202507").loc[1]
        self.assertEqual(result.opdi_match, 0)
        self.assertEqual(result.opdi_match_ambiguous, 1)
        public = self.public()
        public.loc[1, "ades"] = "LIRF"
        self.assertEqual(build_month(self.query(), public, "202507").loc[1].opdi_match, 0)
    def test_month_and_overlap(self):
        public = self.public()
        public.loc[0, "first_seen"] = "2025-06-30 23:00:00Z"
        self.assertEqual(build_month(self.query(), public, "202507").loc[1].opdi_previous_leg_available, 0)
        public = self.public()
        public.loc[0, "last_seen"] = "2025-07-01 11:55:00Z"
        self.assertEqual(build_month(self.query(), public, "202507").loc[1].opdi_previous_leg_available, 0)
    def test_boundary_and_future_mutation(self):
        public = self.public()
        public.loc[1, "first_seen"] = "2025-07-01 11:50:00Z"
        self.assertEqual(build_month(self.query(), public, "202507").loc[1].opdi_match, 1)
        public.loc[1, "first_seen"] = "2025-07-01 11:49:59Z"
        self.assertEqual(build_month(self.query(), public, "202507").loc[1].opdi_match, 0)
        baseline = build_month(self.query(), self.public(), "202507")
        future = self.public().iloc[[0]].copy()
        future["first_seen"] = "2025-07-02 08:00:00Z"
        future["last_seen"] = "2025-07-02 10:00:00Z"
        pd.testing.assert_frame_equal(baseline, build_month(self.query(), pd.concat([self.public(), future]), "202507"))

if __name__ == "__main__":
    unittest.main()
