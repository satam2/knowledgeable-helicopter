import unittest
import pandas as pd
import test_build
import test_build_v2
import test_build_v3
from build_v4 import build_month

# Rebind the fixture modules so every inherited case exercises the final wrapper.
test_build.build_month = build_month
test_build_v2.build_month = build_month
test_build_v3.build_month = build_month

class IdentityTests(unittest.TestCase):
    def test_blank_identity_cannot_displace_unique_match(self):
        fixture = test_build.RotationTests()
        public = fixture.public()
        blank = public.iloc[[1]].copy()
        blank["icao24"] = " "
        baseline = build_month(fixture.query(), public, "202507")
        observed = build_month(fixture.query(), pd.concat([blank, public], ignore_index=True), "202507")
        pd.testing.assert_frame_equal(baseline, observed)

if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in [test_build.RotationTests, test_build_v2.MoreTests, test_build_v3.V3Tests, IdentityTests]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
