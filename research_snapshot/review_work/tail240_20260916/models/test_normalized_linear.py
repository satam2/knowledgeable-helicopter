import numpy as np
import pandas as pd
import normalized_linear_tune as subject


def test_fit_only_scaling_preserves_missing_categories():
    fit=pd.DataFrame({'number':[1.,3.,np.nan],'constant':[2.,2.,2.],'cat':['a','b','a']})
    tune=pd.DataFrame({'number':[101.,np.nan],'constant':[2.,2.],'cat':['unknown','a']})
    encoder=subject.prior.FrameEncoder().fit(fit)
    encoded={name:encoder.transform(frame) for name,frame in [('fit',fit),('tune',tune)]}
    result,scaler=subject.standardize(encoded,encoder)
    np.testing.assert_array_equal(scaler['means'],[2.,2.])
    np.testing.assert_array_equal(scaler['stds'],[1.,1.])
    assert result['tune'].number.iloc[0]==99.
    assert np.isnan(result['tune'].number.iloc[1])
    pd.testing.assert_series_equal(result['tune']['cat'],encoded['tune']['cat'])
