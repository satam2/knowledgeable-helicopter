"""Mock-only adapter contract tests: no estimator training or GPU initialization."""
import torch
import unittest
import inspect
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd

from preprocessing import FramePreprocessor,RetrievalPreprocessor
import realmlp_adapter
import tabdpt_adapter
from run_pilot import sample_rows


class FakeEstimator:
    def __init__(self,**params):
        if "n_epochs" in params:
            from pytabkit import RealMLP_TD_Regressor
            inspect.signature(RealMLP_TD_Regressor).bind(**params)
        self.params=params;self.max_features=128
    def fit(self,x,y,**kwargs):
        self.x=x;self.y=np.array(y);self.fit_kwargs=kwargs
        self.fit_params_={"stop_epoch":{"rmse":3}}
        return self
    def predict(self,x,**kwargs):self.predict_kwargs=kwargs;return np.full(len(x),self.y.mean())


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.x=pd.DataFrame({"a":[1.,np.nan,3.,-999999],"b":pd.Categorical(["x","x",None,"y"])})
        self.y=np.array([5.,10.,15.,20.])
        self.t=pd.DataFrame({"a":[100.,np.nan],"b":pd.Categorical(["new",None])})

    def test_preprocessor_fit_only(self):
        p=FramePreprocessor().fit(self.x)
        t=p.transform(self.t)
        self.assertEqual(p.medians.tolist(),[2.])
        self.assertEqual(t.a.iloc[1],2.)
        self.assertEqual(t.b.iloc[0],"unknown:")
        self.assertEqual(t.b.iloc[1],"missing:")
        r=RetrievalPreprocessor().fit(self.x)
        self.assertTrue(np.isfinite(r.transform(self.t)).all())

    def test_realmlp_temporal_tune_and_refit(self):
        with patch.object(realmlp_adapter,"estimator_class",return_value=FakeEstimator):
            fitted,evidence=realmlp_adapter.fit(self.x,self.y,(self.t,np.array([7.,8.])),device="cpu")
            self.assertEqual(evidence["steps"],3)
            self.assertEqual(fitted["estimator"].params["n_refit"],0)
            self.assertFalse(fitted["estimator"].params["clamp_output"])
            self.assertIn("X_val",fitted["estimator"].fit_kwargs)
            refit,_=realmlp_adapter.fit(self.x,self.y,steps=3,device="cpu")
            self.assertEqual(refit["estimator"].params["n_epochs"],64)
            self.assertEqual(refit["estimator"].params["stop_epoch"],3)
            self.assertEqual(len(refit["estimator"].fit_kwargs["val_idxs"]),0)
            self.assertEqual(len(realmlp_adapter.predict(refit,self.t)),2)

    def test_tabdpt_reference_and_prediction_limits(self):
        receipt={"sha256":"test","revision":"fixed"}
        with patch.object(tabdpt_adapter,"estimator_class",return_value=FakeEstimator),patch.object(tabdpt_adapter,"checked_checkpoint",return_value=("local.safetensors",receipt)):
            model,evidence=tabdpt_adapter.fit(self.x,self.y,(self.t,np.array([99999.,88888.])),device="cpu")
            self.assertEqual(len(model["estimator"].y),4)
            self.assertFalse(evidence["tune_labels_in_reference"])
            self.assertFalse(model["estimator"].params["compile"])
            self.assertEqual(model["estimator"].params["context_reduction"],"retrieval")
            prediction=tabdpt_adapter.predict(model,self.t)
            self.assertEqual(prediction.tolist(),[12.5,12.5])
            self.assertEqual(model["estimator"].predict_kwargs["batch_size"],16)
            self.assertEqual(model["estimator"].predict_kwargs["n_ensembles"],1)
            self.assertLessEqual(model["estimator"].predict_kwargs["context_size"],256)

    def test_original_refit_indices_and_fullscore_preserved(self):
        index={"fit":np.arange(12),"tune":np.arange(20,25),"refit":np.array([1,4,8,14,16,19,23,28]),"score":np.arange(30,45)}
        selected=sample_rows(index,5,20260916)
        self.assertEqual(len(selected["fit"]),5)
        self.assertEqual(len(selected["refit"]),5)
        self.assertTrue(set(selected["refit"]).issubset(set(index["refit"])))
        np.testing.assert_array_equal(selected["tune"],index["tune"])
        np.testing.assert_array_equal(selected["score"],index["score"])


if __name__=="__main__":unittest.main()
