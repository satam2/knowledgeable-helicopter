import lightgbm
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import prepare
import adapter


class AdaptationTests(unittest.TestCase):
    def examples(self):
        x=pd.DataFrame({'airport':pd.Categorical(['A']*120+['B']*120),
            'schedule_proxy_sec':np.arange(240,dtype=float),'stand':pd.Categorical(['1','2']*120)})
        missing=np.arange(240)%7==0
        dates=pd.Series(pd.to_datetime(['2025-01-03']*120+['2025-02-03']*120,utc=True))
        return x,missing,dates

    def test_weights_cap_preserve_missing_and_exact_mass(self):
        missing=np.array([True]*10+[False]*100)
        p=np.r_[np.ones(10),np.zeros(99),1.]
        for ratio in adapter.RATIOS:
            weights,evidence=adapter.weights(p,missing,ratio)
            np.testing.assert_array_equal(weights[missing],np.ones(10))
            self.assertAlmostEqual(weights[~missing].sum(),10*ratio,places=7)
            self.assertLessEqual(weights.max(),20.)
            self.assertGreaterEqual(evidence['known_ess'],0.)
            if ratio:
                self.assertTrue((weights>0).all())

    def test_same_month_availability_does_not_change_own_propensity(self):
        x,missing,dates=self.examples()
        params=adapter.propensity_params(threads=1)
        params['n_estimators']=3
        fitting=lambda fx,fm:adapter.fit_propensity(fx,fm,params=params)
        first,_=adapter.crossfit(x,missing,dates,fit_function=fitting)
        changed=missing.copy()
        changed[120:]=~changed[120:]
        second,_=adapter.crossfit(x,changed,dates,fit_function=fitting)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[:120],np.full(120,.01))

    def test_raw_labels_and_weights_reach_regressor(self):
        x,missing,dates=self.examples()
        y=np.r_[[-50.,87598.],np.full(238,900.)]
        w,_=adapter.weights(np.full(240,.1),missing,1.)
        captured={}
        original=adapter.lgb.LGBMRegressor.fit
        def spy(model,features,labels,**kwargs):
            captured['y']=np.asarray(labels)
            captured['weights']=kwargs['sample_weight']
            return original(model,features,labels,**kwargs)
        with patch.object(adapter.lgb.LGBMRegressor,'fit',spy):
            model,_=adapter.fit_regression(x,y,w,steps=3,threads=1)
        np.testing.assert_array_equal(captured['y'],y)
        np.testing.assert_array_equal(captured['weights'],w)
        prediction=adapter.predict_regression(model,x)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(prediction,adapter.predict_regression(joblib.load(path),x))

    def test_zero_ratio_is_exact_missing_only(self):
        x,missing,_=self.examples()
        w,_=adapter.weights(np.full(240,.5),missing,0.)
        model,info=adapter.fit_regression(x,np.arange(240,dtype=float),w,steps=2,threads=1)
        self.assertEqual(info['rows'],int(missing.sum()))
        self.assertEqual(info['weight_sum'],float(missing.sum()))

    def test_forbidden_source_and_target_predictors(self):
        x,_,_=self.examples()
        for column in [prepare.TARGET,'proxy_missing','AOBT_3_flt','FLIGHT_ID_mvt','BLOCK_TIME_UTC_mvt']:
            bad=x.assign(**{column:0.})
            with self.assertRaisesRegex(ValueError,'Forbidden'):
                prepare.check_features(bad)


if __name__=='__main__':
    unittest.main()
