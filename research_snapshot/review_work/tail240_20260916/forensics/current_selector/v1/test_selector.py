import tempfile
from pathlib import Path
import unittest

import selector
import numpy as np
import pandas as pd
import torch


class SelectorTests(unittest.TestCase):
    def samples(self):
        rng = np.random.default_rng(9)
        x = pd.DataFrame({'number': rng.normal(size=64),
                          'category': pd.Categorical(['a', 'b', None, 'a'] * 16)})
        p = rng.normal(1000, 100, (64, 9))
        y = p[:, 0] + 0.3 * (p[:, 1]-p[:, 0])
        return x, p, y

    def test_convexity_equal_experts_and_reload(self):
        x, p, y = self.samples()
        result = selector.fit(x, p, y, epochs=2, batch_size=32)
        pred, weights = selector.predict(result, x, p)
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_allclose(weights.sum(1), 1, rtol=0, atol=1e-12)
        self.assertTrue(np.all(pred >= p.min(1)) and np.all(pred <= p.max(1)))
        equal = np.full_like(p, 1234.5)
        np.testing.assert_array_equal(selector.predict(result, x, equal)[0], equal[:, 0])
        with tempfile.TemporaryDirectory() as folder:
            selector.save(result, Path(folder))
            native = selector.load(Path(folder))
            np.testing.assert_array_equal(pred, selector.predict(native, x, p)[0])

    def test_late_values_do_not_fit_encoder_and_unknown_embedding_zero(self):
        x, p, y = self.samples()
        result = selector.fit(x, p, y, epochs=2, batch_size=32)
        before = result['encoder'].medians.copy()
        late = pd.DataFrame({'number': [1e20], 'category': pd.Categorical(['new'])})
        pred, _ = selector.predict(result, late, p[:1])
        self.assertTrue(np.isfinite(pred).all())
        np.testing.assert_array_equal(before, result['encoder'].medians)
        self.assertNotIn('new', result['encoder'].categories['category'])
        np.testing.assert_array_equal(result['network'].embeddings[0].weight[1].detach().numpy(), np.zeros(4))

    def test_hidden_gradient_after_zero_output_initialization_and_repeat(self):
        x, p, y = self.samples()
        a = selector.fit(x, p, y, epochs=2, batch_size=32)
        b = selector.fit(x, p, y, epochs=2, batch_size=32)
        self.assertGreater(a['evidence']['hidden_max_gradient'], 0)
        self.assertGreater(a['evidence']['embedding_max_gradient'], 0)
        for name, value in a['network'].state_dict().items():
            self.assertTrue(torch.equal(value, b['network'].state_dict()[name]))
        with self.assertRaises(TypeError):
            selector.fit(x, p, y, late_labels=y)


if __name__ == '__main__':
    unittest.main()
