"""CPU-only tests for centrally scheduled residual GPU model banks."""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Presence(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertTrue((HERE / 'adapter.py').exists())


@unittest.skipUnless((HERE / 'adapter.py').exists(), 'Not implemented')
class BankContracts(unittest.TestCase):
    def test_cpu_banks_fit_tune_refit_and_predict_in_input_order(self):
        import adapter
        x = pd.DataFrame({'airport': pd.Categorical(['AAA', 'BBB'] * 20), 'value': np.arange(40, dtype=float)})
        y = np.array([10., -20.] * 20) + np.arange(40) * .1
        airports = x.airport.astype(str).to_numpy()
        params = {'iterations': 4, 'depth': 2, 'learning_rate': .1, 'loss_function': 'RMSE', 'l2_leaf_reg': 8,
                  'random_seed': 20260916, 'task_type': 'CPU', 'thread_count': 1, 'allow_writing_files': False, 'verbose': False}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for per_airport in [False, True]:
                path = folder / ('airport' if per_airport else 'global')
                bank, info = adapter.fit_bank(x.iloc[:30], y[:30], airports[:30],
                    path / 'fit', params, per_airport=per_airport, tuning=(x.iloc[30:], y[30:], airports[30:]))
                pred = adapter.predict_bank(bank, x.iloc[30:], airports[30:], threads=1)
                self.assertEqual(pred.shape, (10,))
                self.assertTrue(np.isfinite(pred).all())
                refit, _ = adapter.fit_bank(x, y, airports, path / 'refit', params,
                    per_airport=per_airport, selected_steps=info['selected_steps'])
                repeated = adapter.predict_bank(refit, x.iloc[30:].iloc[::-1], airports[30:][::-1], threads=1)
                expected = adapter.predict_bank(refit, x.iloc[30:], airports[30:], threads=1)
                np.testing.assert_allclose(repeated[::-1], expected, atol=1e-12)
                self.assertEqual(len(bank['models']), 2 if per_airport else 1)

    def test_unseen_airport_is_explicit_error_not_dropped_query(self):
        import adapter
        with self.assertRaises(ValueError):
            adapter.predict_bank({'per_airport': True, 'models': {}}, pd.DataFrame({'value': [1.]}), np.array(['NEW']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
