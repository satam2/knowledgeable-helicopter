import unittest
import pandas as pd
import numpy as np
import opdi_alias_dates as o


class DateHypothesisTests(unittest.TestCase):
    def test_prefix_is_explicit_and_conservative(self):
        self.assertEqual(o.lexical_prefix('ITYTY'), 'ITY')
        self.assertEqual(o.lexical_prefix('BAWW'), 'BAW')
        self.assertEqual(o.lexical_prefix('ABCD'), 'ABCD')
        self.assertIsNone(o.lexical_prefix(None))

    def test_all_day_hypotheses_and_nm_masking(self):
        query = pd.DataFrame({o.ID: [1.], o.TIME: pd.to_datetime(['2025-06-03T10:00Z']),
            'FLIGHT_mvt': ['ITYTY0680'], 'ADEP_mvt': ['LIRF'], 'ADES_mvt': ['SBGR'],
            'AOBT_3_flt': [pd.NaT], 'CALLSIGN_flt': [None]})
        public = pd.DataFrame({'id': ['a', 'b', 'c'], 'icao24': ['abc'] * 3,
            'flt_id': ['ITY680'] * 3, 'adep': ['LIRF'] * 3, 'ades': ['SBGR'] * 3,
            'first_seen': pd.to_datetime(['2025-06-02T10:02Z', '2025-06-03T10:01Z', '2025-06-04T10:03Z']),
            'last_seen': pd.to_datetime(['2025-06-02T20:02Z', '2025-06-03T20:01Z', '2025-06-04T20:03Z'])})
        x, links = o.emit_features(query, public, {'ITYTY': 'ITY'})
        self.assertEqual(x.loc[1., 'opdi_learned_day2_offset_sec'], 86280.)
        self.assertEqual(x.loc[1., 'opdi_learned_day1_offset_sec'], -60.)
        self.assertEqual(x.loc[1., 'opdi_learned_day0_offset_sec'], -86580.)
        self.assertEqual(x.loc[1., 'opdi_raw_day1_count_30min'], 0.)
        query['AOBT_3_flt'] = pd.Timestamp('2000-01-01T00:00Z')
        query['CALLSIGN_flt'] = 'WRONG'
        again, _ = o.emit_features(query, public, {'ITYTY': 'ITY'})
        pd.testing.assert_frame_equal(x, again)

    def test_ties_and_invalid_records(self):
        query = pd.DataFrame({o.ID: [1.], o.TIME: pd.to_datetime(['2025-06-03T10:00Z']),
            'FLIGHT_mvt': ['BAW12'], 'ADEP_mvt': ['LIRF'], 'ADES_mvt': ['EGLL']})
        public = pd.DataFrame({'id': ['a', 'b', 'c'], 'icao24': ['abc', 'def', ''],
            'flt_id': ['BAW12'] * 3, 'adep': ['LIRF'] * 3, 'ades': ['EGLL'] * 3,
            'first_seen': pd.to_datetime(['2025-06-03T10:01Z'] * 3),
            'last_seen': pd.to_datetime(['2025-06-03T12:01Z'] * 3)})
        x, links = o.emit_features(query, public, {})
        self.assertEqual(x.loc[1., 'opdi_raw_day1_count_30min'], 2.)
        self.assertTrue(np.isnan(x.loc[1., 'opdi_raw_day1_offset_sec']))
        self.assertTrue(links.empty)


if __name__ == '__main__':
    unittest.main()
