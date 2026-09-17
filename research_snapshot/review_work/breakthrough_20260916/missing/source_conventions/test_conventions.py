"""Synthetic source-feature availability and signed-clock contracts."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Presence(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue((HERE / 'prepare_conventions.py').exists())


@unittest.skipUnless((HERE / 'prepare_conventions.py').exists(), 'Not implemented')
class Features(unittest.TestCase):
    def frame(self):
        now = pd.Timestamp('2025-01-02 12:00', tz='UTC')
        return pd.DataFrame({'MVT_ID_mvt': [1, 2], 'PHASE_mvt': ['DEP', 'DEP'], 'MVT_TIME_UTC_mvt': [now, now],
            'AOBT_3_flt': [now-pd.Timedelta(minutes=10), pd.NaT],
            'EOBT_1_flt': [now-pd.Timedelta(minutes=40), pd.NaT],
            'IOBT_flt': [now-pd.Timedelta(minutes=40), pd.NaT],
            'LOBT_flt': [now-pd.Timedelta(minutes=40), pd.NaT],
            'SCHED_TIME_UTC_mvt': [now-pd.Timedelta(hours=1), now-pd.Timedelta(hours=1)],
            'ADEP_mvt': ['AAA', 'AAA'], 'ADEP_flt': ['AAA', None], 'ADES_mvt': ['BBB', 'BBB'],
            'ADES_flt': ['CCC', None], 'AIRCRAFT_TYPE_mvt': ['A320', 'A320'], 'AIRCRAFT_TYPE_flt': ['A320', None]})

    def test_signed_pair_gaps_and_equality(self):
        import prepare_conventions as m
        result = m.convention_features(self.frame())
        self.assertEqual(result.loc[1, 'conv_nm_minus_est_sec'], 1800)
        self.assertEqual(result.loc[1, 'conv_est_minus_init_sec'], 0)
        self.assertEqual(result.loc[1, 'conv_est_init_last_same'], 1)
        self.assertEqual(result.loc[1, 'conv_distinct_clock_count'], 3)

    def test_hidden_columns_rejected(self):
        import prepare_conventions as m
        raw = self.frame()
        raw['BLOCK_TIME_UTC_mvt'] = raw.MVT_TIME_UTC_mvt
        with self.assertRaises(ValueError):
            m.convention_features(raw)

    def test_missing_source_not_fabricated_clock_or_disagreement(self):
        import prepare_conventions as m
        result = m.convention_features(self.frame())
        self.assertEqual(result.loc[2, 'conv_available_clock_count'], 1)
        self.assertEqual(result.loc[2, 'conv_nm_minus_est_missing'], 1)
        self.assertEqual(result.loc[2, 'conv_ADES_source_disagree'], 0)
        self.assertEqual(result.loc[2, 'conv_ADES_source_missing'], 1)
        self.assertTrue(np.isfinite(result.select_dtypes('number').to_numpy()).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
