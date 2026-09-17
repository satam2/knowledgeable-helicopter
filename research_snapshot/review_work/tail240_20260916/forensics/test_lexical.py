import unittest
import numpy as np
import pandas as pd
from run_lexical import lexical_features, decode_flight_tokens, NUMERIC, CATEGORICAL, prior


class LexicalTests(unittest.TestCase):
    def test_mechanisms_and_normalization(self):
        flights = pd.Series(["ITYTY680", "BAWW561D", "QFAQFA6", "WZZZZ401", "ITY680", None, "", "I22I2251", " aMx 071 ", "RYR2AA"], index=np.arange(100, 110))
        result = lexical_features(flights)
        self.assertEqual(result.index.tolist(), flights.index.tolist())
        self.assertEqual(result.lex_repeated_prefix_suffix.tolist(), [1,1,1,1,0,0,0,0,0,0])
        canonical = decode_flight_tokens(result.lex_canonical_flight)
        self.assertEqual(canonical.tolist(), ["ITY680","BAW561D","QFA6","WZZ401","ITY680","","","I22I2251","AMX071","RYR2AA"])
        self.assertEqual(result.loc[108,"lex_leading_zero"], 1)
        self.assertEqual(result.loc[109,"lex_trailing_repeat"], 1)
        self.assertEqual(result.loc[107,"lex_parse_ok"], 0)
        self.assertTrue(np.isfinite(result[NUMERIC].to_numpy()).all())
        self.assertEqual(list(result), NUMERIC + CATEGORICAL)

    def test_token_roundtrip_and_rejection(self):
        flights = pd.Series([None, "", "ITYTY680", "A:B", "  aBc  "])
        pd.testing.assert_series_equal(decode_flight_tokens(prior.token(flights)), flights.astype("string"))
        with self.assertRaises(ValueError):
            decode_flight_tokens(pd.Series(["s1:ABC"]))

    def test_batch_independent_and_order_stable(self):
        flights = pd.Series(["BAWW561D", "ITYTY680", "ITY680"], index=[8,3,7])
        full = lexical_features(flights)
        for index in flights.index:
            pd.testing.assert_frame_equal(full.loc[[index]], lexical_features(flights.loc[[index]]))
        pd.testing.assert_frame_equal(full.iloc[::-1], lexical_features(flights.iloc[::-1]))


if __name__ == "__main__":
    unittest.main()
