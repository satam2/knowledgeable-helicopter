"""CPU contracts for official numeric embeddings and saved-model reconstruction."""
import io
import unittest
import torch
import joblib
import numpy as np
import pandas as pd
import tabm_ple_gpu as adapter


class PiecewiseContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        x = pd.DataFrame({"continuous": np.linspace(-3, 5, 96), "constant": 7.,
                          "missing": [np.nan, 1., 2.] * 32,
                          "category": pd.Categorical(["a", "b"] * 48)})
        self.encoder = adapter.FrameEncoder().fit(x, neural=True)
        self.numbers, self.categories = adapter.frozen.tensors(self.encoder, x)
        self.bins, self.embedded, self.passthrough = adapter.fit_bins(self.numbers, len(self.encoder.numeric))

    def network(self, members):
        return adapter.Network(self.numbers.shape[1], [4], self.bins, self.embedded, self.passthrough, members)

    def test_constants_and_missing_indicators_passthrough(self):
        self.assertNotIn(1, self.embedded.tolist())
        self.assertTrue(set([1, 3, 4, 5]).issubset(self.passthrough.tolist()))
        self.assertEqual(set(self.embedded.tolist()) | set(self.passthrough.tolist()), set(range(6)))

    def test_bins_only_cover_fit_features(self):
        for index, bins in zip(self.embedded, self.bins):
            self.assertEqual(float(bins[0]), float(self.numbers[:, index].min()))
            self.assertEqual(float(bins[-1]), float(self.numbers[:, index].max()))

    def test_members_independent_squared_loss_and_gradient(self):
        model = self.network(8)
        output = model(self.numbers[:5], self.categories[:5])
        self.assertEqual(tuple(output.shape), (5, 8))
        target = torch.arange(5.)[:, None]
        loss = (output - target).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_32members_and_saved_model_replay(self):
        model = self.network(32).eval()
        expected = model(self.numbers[:5], self.categories[:5]).detach()
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        loaded = joblib.load(stream)
        actual = loaded(self.numbers[:5], self.categories[:5]).detach()
        self.assertEqual(tuple(actual.shape), (5, 32))
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
