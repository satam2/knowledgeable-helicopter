import unittest
import numpy as np
import pandas as pd
import run as neural


class NeuralTests(unittest.TestCase):
    def test_decode_category_missing_contract_and_fit_only_unknown(self):
        matrix=np.array([[0,0,1.],[2,2,np.nan],[1,1,-999999.]],dtype=np.float32)
        x=neural.decode_frame(matrix,{'ADEP_mvt':['ABC'],'conv_clock_rank_signature':['MISSING']},
            ['ADEP_mvt','conv_clock_rank_signature','numeric'])
        self.assertTrue(pd.isna(x.iloc[0,0]))
        self.assertEqual(x.iloc[0,1],'MISSING')
        self.assertEqual(x.iloc[1,1],'MISSING')
        encoder=neural.encoders.FrameEncoder().fit(x.iloc[:2],neural=True)
        self.assertNotIn('__UNSEEN_CONTEXT_VALUE__',encoder.categories['ADEP_mvt'])
        numbers,categories=encoder.transform(x.iloc[2:])
        self.assertEqual(categories[0,0],1)
        self.assertTrue(np.isfinite(numbers).all())


if __name__=='__main__':
    unittest.main()
