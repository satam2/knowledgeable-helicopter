import unittest

import pandas as pd

import cohorts_v1 as subject


class ChronologicalCohortTests(unittest.TestCase):
    def metadata(self):
        rows = [
            (1, 'base', '2025-01-01'),
            (2, 'base-cal', '2025-04-01'),
            (3, 'base-cal', '2025-05-01'),
            (4, 'base-eval', '2025-04-02'),
            (5, 'base-eval', '2025-06-01'),
            (6, 'cal-eval', '2025-05-02'),
            (7, 'cal-eval', '2025-06-02'),
            (8, 'base-score', '2025-04-03'),
            (9, 'base-score', '2025-07-01'),
            (10, 'eval-score', '2025-06-03'),
            (11, 'eval-score', '2025-07-02'),
            (12, None, '2025-04-04'),
            (13, None, '2025-05-03'),
            (14, None, '2025-06-04'),
            (15, None, '2025-07-03'),
            (16, 'outside', '2025-08-01'),
        ]
        return pd.DataFrame(rows, columns=[subject.ID, subject.FLIGHT_ID, subject.MOVEMENT])

    def test_outer_purges_preserved_and_new_calibration_purge(self):
        frame = self.metadata()
        positions, receipt = subject.build(frame, 'F1')
        actual = {name: frame.iloc[index][subject.ID].tolist() for name, index in positions.items()}
        self.assertEqual(actual, {'base': [1, 12], 'calibration': [3, 13],
                                  'evaluation': [5, 7, 14], 'protected_score': [9, 11, 15]})
        self.assertEqual(receipt['additional_base_against_calibration'], 1)
        self.assertEqual(receipt['stages']['evaluation'], receipt['original']['stages']['tune'])
        self.assertEqual(receipt['stages']['protected_score'], receipt['original']['stages']['score'])

    def test_labels_proxy_and_hidden_block_are_rejected(self):
        for column in ['TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt', 'proxy_sec']:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'identifier/time'):
                subject.build(self.metadata().assign(**{column: 0}), 'F1')

    def test_duplicate_ids_missing_times_and_wrong_fold_fail(self):
        frame = self.metadata()
        frame.loc[1, subject.ID] = 1
        with self.assertRaises(ValueError):
            subject.build(frame, 'F1')
        frame = self.metadata()
        frame.loc[1, subject.MOVEMENT] = None
        with self.assertRaises(ValueError):
            subject.build(frame, 'F1')
        with self.assertRaises(ValueError):
            subject.build(self.metadata(), 'F2')

    def test_order_and_full_score_are_preserved(self):
        frame = self.metadata().iloc[::-1].reset_index(drop=True)
        positions, receipt = subject.build(frame, 'F1')
        self.assertEqual(frame.iloc[positions['protected_score']][subject.ID].tolist(), [15, 11, 9])
        self.assertEqual(frame.iloc[positions['base']][subject.ID].tolist(), [12, 1])
        self.assertEqual(receipt['stages']['protected_score']['n'], 3)

    def test_empty_auxiliary_stage_fails(self):
        frame = self.metadata()
        frame = frame[~frame[subject.MOVEMENT].str.startswith('2025-05')]
        with self.assertRaisesRegex(ValueError, 'empty'):
            subject.build(frame, 'F1')


if __name__ == '__main__':
    unittest.main()
