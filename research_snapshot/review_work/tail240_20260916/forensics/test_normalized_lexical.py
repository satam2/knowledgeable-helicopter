import unittest
import pandas as pd
import numpy as np
from run_normalized_lexical import selected_lexical,lexical_features,FEATURES,base
from run_lexical import prior


class NormalizedLexicalTests(unittest.TestCase):
    def test_frozen_lexical_exact_and_order(self):
        ids=pd.Index([7.,2.,9.],name="MVT_ID_mvt")
        flights=pd.Series(["ITYTY078","BAWW561D",None],index=ids)
        x=pd.DataFrame({"flight":prior.token(flights),"schedule_proxy_sec":[-100.,2000.,-999999]},index=ids)
        cached=lexical_features(flights)
        result=selected_lexical(x,cached)
        self.assertEqual(list(result),FEATURES)
        pd.testing.assert_frame_equal(result,cached)
        with self.assertRaises(AssertionError):selected_lexical(x,cached.iloc[::-1])
        with self.assertRaises(ValueError):selected_lexical(x,cached.drop(columns=FEATURES[0]))

    def test_scale_unchanged_by_extra_features_and_raw_loss_identity(self):
        ids=pd.Index([1.,2.],name="MVT_ID_mvt")
        flights=pd.Series(["ITYTY680","BAWW561D"],index=ids)
        x=pd.DataFrame({"flight":prior.token(flights),"schedule_proxy_sec":[58563.,60599.]},index=ids)
        ext=selected_lexical(x,lexical_features(flights))
        scale=base.scale_of(x)
        np.testing.assert_array_equal(scale,base.scale_of(pd.concat([x,ext],axis=1)))
        y=np.array([-300.,87598.]);z=np.array([.3,1.5]);normalizer=float(np.mean(scale**2))
        np.testing.assert_allclose(scale**2/normalizer*(z-base.transformed(y,scale))**2,(base.reconstruct(z,scale)-y)**2/normalizer)


if __name__=="__main__":unittest.main()
