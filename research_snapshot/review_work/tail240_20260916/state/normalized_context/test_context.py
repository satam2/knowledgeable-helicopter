import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import run as context


class ContextTests(unittest.TestCase):
    def test_selected_rows_preserve_raw_values_and_no_target_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.parquet'
            pd.DataFrame({context.ID:[3., 1., 2.], 'event_clock':[-100., np.nan, 86400.], 'category':['c', 'a', 'b']}).to_parquet(path, index=False)
            actual = context.selected_frame(path, ['event_clock', 'category'], pd.Index([1., 2.]))
            self.assertEqual(actual.index.tolist(), [1., 2.])
            self.assertTrue(np.isnan(actual.loc[1., 'event_clock']))
            self.assertEqual(actual.loc[2., 'event_clock'], 86400.)

    def test_scale_weight_identity_with_extreme_raw_targets(self):
        x = pd.DataFrame({'schedule_proxy_sec':[900., 86400., -999999.]})
        scale = context.base.scale_of(x)
        y = np.array([-100., 175000., 0.])
        z = context.base.transformed(y, scale)
        predz = np.array([1., 2., 3.])
        pred = context.base.reconstruct(predz, scale)
        np.testing.assert_allclose(scale**2*(predz-z)**2, (pred-y)**2)


if __name__ == '__main__':
    unittest.main()
