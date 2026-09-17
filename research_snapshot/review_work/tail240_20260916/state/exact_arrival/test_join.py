"""Synthetic linkage isolation contracts before any private label access."""
import unittest
import pandas as pd
import numpy as np
import join


def row(identity=1, flight=10, phase='DEP', time='2025-06-01T10:00:00Z', **extra):
    out = {join.ID:identity, join.FLIGHT:flight, join.PHASE:phase, join.TIME:pd.Timestamp(time),
           join.ORIGIN:'AAA', join.DEST:'BBB', join.ARVT:pd.Timestamp('2025-06-01T11:00:30Z'), join.OPERATOR:'OP'}
    out.update(extra)
    return out


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.dep = pd.DataFrame([row()])
        self.arr = pd.DataFrame([row(2, phase='ARR', time='2025-06-01T11:00:00Z')])

    def test_exact_route_landing_delta(self):
        result = join.link(self.dep, self.arr)
        self.assertEqual(result.delta_sec.iloc[0], 30.)
        self.assertEqual(join.oracle(self.dep.iloc[0], self.arr), ('linked', 30.))

    def test_duplicate_movement_identical_and_conflicting(self):
        same = pd.concat([self.arr, self.arr])
        self.assertEqual(join.link(self.dep, same).delta_sec.iloc[0], 30.)
        bad = self.arr.copy()
        bad[join.DEST] = 'CCC'
        with self.assertRaises(ValueError):
            join.link(self.dep, pd.concat([self.arr, bad]))

    def test_physical_duplicates_and_ambiguous_flight(self):
        duplicate = self.arr.copy()
        duplicate[join.ID] = 3
        result = join.link(self.dep, pd.concat([duplicate, self.arr]))
        self.assertEqual(result.arrival_id.iloc[0], 2)
        duplicate[join.TIME] += pd.Timedelta(minutes=1)
        result = join.link(self.dep, pd.concat([duplicate, self.arr]))
        self.assertEqual(result.status.iloc[0], 'ambiguous_arrival')
        duplicate[join.DEST] = 'CCC'
        self.assertEqual(join.link(self.dep, pd.concat([duplicate, self.arr])).status.iloc[0], 'ambiguous_arrival')

    def test_route_and_chronology(self):
        for field in [join.ORIGIN, join.DEST]:
            bad = self.arr.copy()
            bad[field] = 'XXX'
            self.assertEqual(join.link(self.dep, bad).status.iloc[0], 'route_mismatch')
        for minutes in [0, -1]:
            bad = self.arr.copy()
            bad[join.TIME] = self.dep[join.TIME] + pd.Timedelta(minutes=minutes)
            self.assertEqual(join.link(self.dep, bad).status.iloc[0], 'not_strictly_later')

    def test_missing_clock_flight_and_month(self):
        for value in [np.nan, pd.NA]:
            bad = self.dep.copy()
            bad[join.FLIGHT] = value
            self.assertEqual(join.link(bad, self.arr).status.iloc[0], 'missing_flight')
        bad = self.dep.copy()
        bad[join.ARVT] = pd.NaT
        self.assertEqual(join.link(bad, self.arr).status.iloc[0], 'missing_own_arvt')
        bad = self.arr.copy()
        bad[join.TIME] = pd.Timestamp('2025-07-01T00:00:00Z')
        self.assertEqual(join.link(self.dep, bad).status.iloc[0], 'no_arrival_in_month')

    def test_label_block_mutation_and_phase(self):
        expected = join.link(self.dep, self.arr)
        for frame in [self.dep, self.arr]:
            frame['TAXITIME_SEC_mvt'] = -1e30
            frame['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2040-01-01', tz='UTC')
        pd.testing.assert_frame_equal(join.link(self.dep, self.arr), expected)
        with self.assertRaises(ValueError):
            join.link(self.dep, self.dep)

    def test_blank_route_and_empty_arrivals(self):
        for value in ['', '   ', None]:
            self.dep[join.ORIGIN] = value
            self.arr[join.ORIGIN] = value
            self.assertEqual(join.link(self.dep, self.arr).status.iloc[0], 'route_mismatch')
            self.assertEqual(join.oracle(self.dep.iloc[0], self.arr)[0], 'route_mismatch')
        self.assertEqual(join.link(self.dep, self.arr.iloc[:0]).status.iloc[0], 'no_arrival_in_month')


if __name__ == '__main__':
    unittest.main()
