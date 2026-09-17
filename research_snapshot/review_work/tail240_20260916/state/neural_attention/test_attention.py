"""CPU-only behavioral tests; no private labels or GPU allocation."""
import io
import sys
import unittest
from pathlib import Path
import torch
import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
from encoders import FrameEncoder
import attention_adapter as subject


def fixture():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'airport': pd.Categorical(['A', 'B'] * 20)})
    for name in subject.OWN_CLOCKS:
        frame[name] = rng.normal(900, 200, 40)
    for slot, prefix in enumerate(subject.PREFIXES):
        for field in subject.FIELDS:
            frame[prefix + field] = rng.normal(100 * slot, 20, len(frame))
        frame[prefix + 'padding_missing'] = 0.
    frame.loc[0, subject.PREFIXES[0] + subject.FIELDS[0]] = np.nan
    frame.loc[1, subject.PREFIXES[1] + subject.FIELDS[1]] = -999999.
    return frame.copy()


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.frame = fixture()
        cls.encoder = FrameEncoder().fit(cls.frame, neural=True)
        cls.stats = subject.token_statistics(cls.frame, cls.encoder)

    def network(self):
        n = 2 * len(self.encoder.numeric)
        torch.manual_seed(7)
        return subject.Network(n, [4], [], torch.empty(0, dtype=torch.long),
                               torch.arange(n), 8, self.encoder, self.stats).eval()

    def tensors(self, frame):
        return tuple(torch.from_numpy(v) for v in self.encoder.transform(frame))

    def test_zero_head_static_parity_and_base_initialization(self):
        model = self.network()
        n = 2 * len(self.encoder.numeric)
        torch.manual_seed(7)
        base = subject.ple.Network(n, [4], [], torch.empty(0, dtype=torch.long), torch.arange(n), 8).eval()
        values = self.tensors(self.frame)
        torch.testing.assert_close(model(*values), base(*values), rtol=0, atol=0)
        self.assertEqual(model(*values).shape, (40, 8))

    def test_shared_scaler_and_missing_masks(self):
        model = self.network()
        tokens, present = model.token_inputs(self.tensors(self.frame)[0])
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(present.all())
        self.assertEqual(tokens[0, 0, 24], 1)
        self.assertEqual(tokens[1, 1, 25], 1)
        raw = self.frame[[p + subject.FIELDS[2] for p in subject.PREFIXES]].to_numpy()
        expected = (raw - self.stats['means'][2]) / self.stats['scales'][2]
        np.testing.assert_allclose(tokens[:, :, 2].detach().numpy(), expected, rtol=2e-5, atol=2e-5)

    def test_timestamp_relation_sign_age_and_arrival_mask(self):
        model = self.network()
        frame = self.frame.copy()
        # QueryT=10000, ownC=9000, peerT=9700, peerC=8800.
        frame[subject.OWN_CLOCKS[0]] = 1000.
        frame[subject.PREFIXES[0] + subject.FIELDS[0]] = 900.
        frame[subject.PREFIXES[0] + 'age_sec'] = 300.
        tokens, _ = model.token_inputs(self.tensors(frame)[0])
        duration = tokens[:, 0, 14] * model.shared_scales[14] + model.shared_means[14]
        stamp = tokens[:, 0, 19] * model.shared_scales[19] + model.shared_means[19]
        torch.testing.assert_close(duration, torch.full_like(duration, 100.), rtol=0, atol=.001)
        torch.testing.assert_close(stamp, torch.full_like(stamp, -200.), rtol=0, atol=.001)
        self.assertTrue((tokens[:, 4:, 38:48] == 1).all())

    def test_all_absent_and_padded_values_ignored(self):
        model = self.network()
        torch.nn.init.normal_(model.correction_head.weight)
        torch.nn.init.normal_(model.correction_head.bias)
        frame = self.frame.copy()
        for prefix in subject.PREFIXES:
            frame[prefix + 'padding_missing'] = 1.
        numbers, categories = self.tensors(frame)
        result = model.correction(numbers, categories)
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        frame = self.frame.copy()
        frame[subject.PREFIXES[0] + 'padding_missing'] = 1.
        initial = model.correction(*self.tensors(frame))
        for field in subject.FIELDS[:-1]:
            frame[subject.PREFIXES[0] + field] = np.nan
        torch.testing.assert_close(model.correction(*self.tensors(frame)), initial, rtol=0, atol=0)

    def test_within_phase_permutation_shared_weights(self):
        model = self.network()
        torch.nn.init.normal_(model.correction_head.weight)
        frame = self.frame.copy()
        for left, right in [(0, 3), (4, 7)]:
            for field in subject.FIELDS:
                frame[subject.PREFIXES[left] + field] = self.frame[subject.PREFIXES[right] + field]
                frame[subject.PREFIXES[right] + field] = self.frame[subject.PREFIXES[left] + field]
        torch.testing.assert_close(model.correction(*self.tensors(frame)),
                                   model.correction(*self.tensors(self.frame)), rtol=3e-5, atol=3e-5)
        self.assertEqual(sum(isinstance(m, torch.nn.MultiheadAttention) for m in model.modules()), 1)

    def test_gradient_and_serialization(self):
        model = self.network()
        values = self.tensors(self.frame)
        model.correction(*values).square().sum().backward()
        model.zero_grad(set_to_none=True)
        target = torch.ones(40, 8)
        (model.correction(*values) - target).square().mean().backward()
        self.assertGreater(float(model.correction_head.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.correction_head.weight.add_(-.01 * model.correction_head.weight.grad)
        model.zero_grad(set_to_none=True)
        (model.correction(*values) - target).square().mean().backward()
        self.assertGreater(float(model.token_mlp[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = self.network()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(*values), model(*values), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
