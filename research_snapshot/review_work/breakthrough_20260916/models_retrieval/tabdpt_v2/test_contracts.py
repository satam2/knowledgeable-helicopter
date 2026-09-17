import torch
from torch.nn.attention import SDPBackend
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import adapter
from tabdpt.utils import flash_context


class FakeEstimator:
    def __init__(self,**kwargs):
        self.params=kwargs
        self.use_flash=kwargs['use_flash']
        self.model=SimpleNamespace(use_flash=self.use_flash)
        self.max_features=512
    def fit(self,x,y):
        self.labels=np.asarray(y).copy()
    def predict(self,x,**kwargs):
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        self.predict_params=kwargs
        return np.full(len(x),self.labels.mean())


class NonflashContract(unittest.TestCase):
    def test_official_flag_bypasses_flash_only_context(self):
        call=flash_context(lambda self:torch.nn.functional.scaled_dot_product_attention(
            torch.ones(1,1,2,4),torch.ones(1,1,3,4),torch.ones(1,1,3,6)))
        with patch('tabdpt.utils.sdpa_kernel',side_effect=RuntimeError('No available kernel. Aborting execution.')), \
             patch('tabdpt.utils.torch.cuda.is_available',return_value=True), \
             patch('tabdpt.utils.torch.cuda.get_device_capability',return_value=(8,0)), \
             patch('tabdpt.utils.torch.autocast',return_value=contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'No available kernel'):
                call(SimpleNamespace(use_flash=True))
            result=call(SimpleNamespace(use_flash=False))
        self.assertEqual(tuple(result.shape),(1,1,2,6))
        self.assertTrue(torch.isfinite(result).all())

    def test_reference_labels_budget_and_math_prediction_scope(self):
        x=pd.DataFrame({'x':[1.,2.,np.nan,4.],'airport':pd.Categorical(['A','B','A','B'])})
        y=np.array([-12.,131167.,5.,90.])
        receipt={'sha256':'synthetic','revision':'test'}
        with patch.object(adapter.frozen,'checked_checkpoint',return_value=('fake.safetensors',receipt)), \
             patch.object(adapter.frozen,'estimator_class',return_value=FakeEstimator):
            model,evidence=adapter.fit(x,y,(x,np.full(4,1e12)),device='cpu',threads=2)
        np.testing.assert_array_equal(model['estimator'].labels,y)
        self.assertFalse(model['estimator'].params['use_flash'])
        self.assertFalse(evidence['tune_labels_in_reference'])
        before=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled())
        result=adapter.predict(model,x)
        np.testing.assert_array_equal(result,np.full(4,y.mean()))
        self.assertEqual(before,(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()))
        self.assertEqual(model['estimator'].predict_params['batch_size'],16)
        self.assertEqual(adapter.REFERENCE_LIMIT,32000)
        self.assertEqual(adapter.CONTEXT,256)
        self.assertEqual(adapter.ENSEMBLES,1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
