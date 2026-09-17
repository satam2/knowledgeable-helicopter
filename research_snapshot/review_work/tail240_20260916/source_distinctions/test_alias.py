import unittest
import pandas as pd
import numpy as np
import audit_alias as a


def fixture():
    base = {a.ID: [1., 2.], a.FLIGHT_ID: [10., 10.], a.PHASE: ['DEP', 'ARR'],
            a.MOVEMENT: pd.to_datetime(['2025-02-01T10:00Z', '2025-02-01T12:00Z']),
            'FLIGHT_mvt': ['BAWW0561D', 'BAW561D'], 'CALLSIGN_flt': ['BAW561D', 'BAW561D'],
            'ADEP_mvt': ['LIRF'] * 2, 'ADES_mvt': ['EGLL'] * 2,
            'AOBT_3_flt': pd.to_datetime(['2025-02-01T09:45Z'] * 2),
            'EOBT_1_flt': pd.to_datetime(['2025-02-01T09:40Z'] * 2),
            'SCHED_TIME_UTC_mvt': pd.to_datetime(['2025-02-01T09:30Z', '2025-02-01T12:00Z'])}
    return pd.DataFrame(base)


class AliasTests(unittest.TestCase):
    def test_normalization_and_missing(self):
        _, parsed = a.parts(pd.Series([' Baww0561d ', 'ITYTY680', None, 'INVALID']))
        self.assertEqual(parsed.iloc[0].tolist(), ['BAWW', '561', 'D'])
        self.assertEqual(parsed.iloc[1].tolist(), ['ITYTY', '680', ''])
        self.assertTrue(parsed.iloc[2].isna().all())
        self.assertTrue(parsed.iloc[3].isna().all())

    def test_unique_alias_and_masking(self):
        raw = fixture()
        frame = a.prepare(raw)
        found = a.candidates(frame.iloc[:1], frame.iloc[1:], {'BAWW': 'BAW'})
        self.assertEqual(len(found), 1)
        self.assertEqual(found.candidate_proxy_sec.iloc[0], 900)
        raw.loc[0, 'AOBT_3_flt'] = pd.NaT
        raw.loc[0, a.FLIGHT_ID] = np.nan
        raw.loc[0, 'CALLSIGN_flt'] = 'ZZZ000'
        masked = a.prepare(raw)
        repeated = a.candidates(masked.iloc[:1], masked.iloc[1:], {'BAWW': 'BAW'})
        pd.testing.assert_frame_equal(found, repeated)

    def test_ambiguity_and_negative_airtime(self):
        raw = fixture()
        second = raw.iloc[1:].copy()
        second[a.ID], second[a.FLIGHT_ID] = 3., 11.
        second['AOBT_3_flt'] += pd.Timedelta(minutes=1)
        frame = a.prepare(pd.concat([raw, second], ignore_index=True))
        found = a.candidates(frame.iloc[:1], frame.iloc[1:], {'BAWW': 'BAW'})
        report, selected = a.summarize(frame.iloc[:1], found, 'alias')
        self.assertEqual(report['known_masked']['ambiguous_rows'], 1)
        self.assertEqual(len(selected), 0)
        frame.loc[1:, a.MOVEMENT] = pd.Timestamp('2025-02-01T08:00Z')
        self.assertEqual(len(a.candidates(frame.iloc[:1], frame.iloc[1:], {'BAWW': 'BAW'})), 0)

    def test_empty_mapping_and_distinct_support(self):
        frame = a.prepare(fixture())
        self.assertEqual(len(a.candidates(frame.iloc[:1], frame.iloc[1:], {})), 0)
        repeated = pd.DataFrame({a.FLIGHT_ID: [1.] * 50, 'mp': ['BAWW'] * 50, 'cp': ['BAW'] * 50})
        self.assertEqual(a.prefix_table([repeated]), {})
        repeated[a.FLIGHT_ID] = np.arange(50)
        self.assertEqual(a.prefix_table([repeated]), {'BAWW': 'BAW'})


if __name__ == '__main__':
    unittest.main()
