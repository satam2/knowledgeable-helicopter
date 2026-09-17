"""Raw-label, training-only nuisance means, fixed ridge and permutation contracts."""
import unittest
import numpy as np
import pandas as pd
import association_stats as subject


def frame():
    return pd.DataFrame(dict(ADEP_mvt=['A']*10+['B']*10, ADES_mvt=['C']*20,
        AIRCRAFT_OPERATOR_flt=['OP']*20, MVT_TIME_UTC_mvt=pd.date_range('2025-06-01', periods=20, freq='12h', tz='UTC'),
        linked=[True]*19+[False], delta_sec=list(np.arange(19, dtype=float))+[np.nan],
        raw_target_sec=np.arange(20)*10.-500., baseline_prediction_sec=np.zeros(20)))


class StatisticsTests(unittest.TestCase):
    def test_closed_form_and_no_label_clipping(self):
        f = frame()
        model = subject.fit_calibration(f)
        self.assertLess(model['global_error_mean'], 0.)
        self.assertEqual(model['ridge_alpha'], 100.)
        self.assertAlmostEqual(model['coefficient'], model['coefficient_closed_form'], places=12)

    def test_unlinked_unchanged_unknown_airport_and_target_mutation(self):
        f = frame()
        model = subject.fit_calibration(f)
        f.loc[0, 'ADEP_mvt'] = 'UNSEEN'
        before = subject.predictions(f, model)
        f['raw_target_sec'] = 1e12
        after = subject.predictions(f, model)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a[-1], 0.)
        self.assertEqual(before[0][0], model['global_error_mean'])

    def test_permutation_is_reproducible_and_reports_singletons(self):
        f = frame()
        model = subject.fit_calibration(f)
        a = subject.permutation_null(f, model)
        self.assertEqual(a, subject.permutation_null(f, model))
        self.assertEqual(a['repetitions'], 25)
        self.assertEqual(a['singleton_rows'], 1)


if __name__ == '__main__':
    unittest.main()
