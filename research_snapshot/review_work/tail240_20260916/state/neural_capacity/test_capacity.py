"""CPU shape/mean/gradient/reload and original official PLE bin parity checks."""
import torch
import lightgbm
import io
import unittest
import joblib
import numpy as np
import pandas as pd
import capacity_adapter as subject


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.frame = pd.DataFrame(dict(x=np.linspace(-10, 10, 101), y=np.arange(101)%11,
                                     missing=np.where(np.arange(101)%3, 3., np.nan),
                                     category=pd.Categorical(['A','B']*50+['A'])))
        cls.preparation = subject.prepare(cls.frame)

    def model(self):
        torch.manual_seed(5)
        return subject.network(self.preparation).eval()

    def test_original_network_32_mean_and_shape(self):
        model = self.model()
        self.assertIs(type(model), subject.ple.Network)
        values = subject.frozen.tensors(self.preparation['encoder'], self.frame)
        with torch.no_grad():
            expected = model(*values).mean(dim=1).numpy()
        self.assertEqual(model(*values).shape, (101,32))
        np.testing.assert_array_equal(subject.frozen.infer(model, values, 'cpu'), expected)

    def test_official_bin_parity(self):
        encoder = self.preparation['encoder']
        values, _ = subject.frozen.tensors(encoder, self.frame)
        bins, embedded, passthrough = subject.ple.fit_bins(values, len(encoder.numeric))
        torch.testing.assert_close(embedded, self.preparation['embedded'], rtol=0, atol=0)
        torch.testing.assert_close(passthrough, self.preparation['passthrough'], rtol=0, atol=0)
        for a,b in zip(bins, self.preparation['bins']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_raw_mse_gradients_and_joblib_reload(self):
        model = self.model()
        values = subject.frozen.tensors(self.preparation['encoder'], self.frame)
        target = torch.linspace(-100., 100000., len(self.frame))
        loss = (model(*values)-target[:,None]).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.output.weight.grad.abs().sum()),0)
        buffer = io.BytesIO()
        joblib.dump(model,buffer)
        buffer.seek(0)
        restored=joblib.load(buffer)
        torch.testing.assert_close(restored(*values), model(*values), rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
