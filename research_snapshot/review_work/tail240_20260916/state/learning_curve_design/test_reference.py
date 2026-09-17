import unittest
import numpy as np
import pandas as pd
from reference import reference_predictions


class ReferenceTests(unittest.TestCase):
    def test_label_free_original_schema(self):
        frame=pd.DataFrame({'MVT_ID_mvt':[1.,2.],'prediction_sec':[10.,20.]})
        np.testing.assert_array_equal(reference_predictions(frame,[1.,2.]),[10.,20.])

    def test_order_mismatch_rejected(self):
        frame=pd.DataFrame({'MVT_ID_mvt':[1.,2.],'prediction_sec':[10.,20.]})
        with self.assertRaises(AssertionError):reference_predictions(frame,[2.,1.])


if __name__=='__main__':unittest.main()
