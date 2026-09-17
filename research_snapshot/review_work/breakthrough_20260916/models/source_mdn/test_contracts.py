"""CPU-only mechanics checks; these do not establish real-data mean accuracy."""
import torch
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import adapter


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(92)

    def test_masking_and_original_clock_domain(self):
        raw = np.array([[-12., 131167., np.nan, -999999., np.inf],
                        [np.nan, np.nan, np.nan, np.nan, np.nan]])
        frame = pd.DataFrame(raw, columns=['takeoff_minus_' + name for name in adapter.CLOCKS])
        anchors, valid = adapter.raw_anchors(frame)
        np.testing.assert_array_equal(valid[0], [True, True, False, False, False])
        self.assertEqual(anchors[0, 0], -12.)
        self.assertEqual(anchors[0, 1], 131167.)
        output = torch.randn(2, 8, 18)
        a = torch.tensor(anchors)
        v = torch.tensor(valid)
        first = adapter.distribution_parameters(output, a, v, .01)
        a[~v] = torch.nan
        second = adapter.distribution_parameters(output, a, v, .01)
        weights = first[0].exp()
        self.assertTrue(torch.all(weights[:, :, :5][~v[:, None, :].expand(-1, 8, -1)] == 0))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 8))
        self.assertTrue(torch.all(weights[1, :, 5] == 1))
        torch.testing.assert_close(adapter.analytical_mean(*first[:2]), adapter.analytical_mean(*second[:2]))

    def test_analytical_mean_and_affine_inverse(self):
        output = torch.randn(3, 8, 18, dtype=torch.float64)
        anchors = torch.tensor([[10., -12., 9000., 50., 90.]], dtype=torch.float64).repeat(3, 1)
        center, scale = 100., 45.
        parameters = adapter.distribution_parameters(output, (anchors - center) / scale,
                                                    torch.ones(3, 5, dtype=torch.bool), 1. / scale)
        logw, means, scales = parameters
        actual = adapter.analytical_mean(logw, means)
        distribution = torch.distributions.StudentT(4., means, scales)
        manual = sum(logw[:, :, i].exp() * distribution.mean[:, :, i] for i in range(6))
        torch.testing.assert_close(actual, manual)
        raw_means = means * scale + center
        torch.testing.assert_close(actual * scale + center, (logw.exp() * raw_means).sum(-1))
        self.assertTrue(torch.all(scales * scale >= 1.))
        torch.testing.assert_close(raw_means[:, :, :5], anchors[:, None, :] + output[:, :, 6:11] * scale)

    def test_extreme_labels_and_separate_member_loss(self):
        output = torch.randn(4, 8, 18, requires_grad=True)
        parameters = adapter.distribution_parameters(output, torch.zeros(4, 5),
                                                    torch.ones(4, 5, dtype=torch.bool), .001)
        labels = torch.tensor([-12., 131167., -1e12, 1e12], dtype=torch.float64)
        nll = adapter.member_negative_log_likelihood(parameters, labels)
        self.assertEqual(nll.shape, (4, 8))
        self.assertTrue(torch.isfinite(nll).all())
        nll.mean().backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        logw, means, scales = parameters
        component = torch.distributions.StudentT(4., means.double(), scales).log_prob(labels[:, None, None])
        manual = -torch.logsumexp(logw.double() + component, -1)
        torch.testing.assert_close(nll, manual)
        ensemble_nll = -torch.logsumexp(-manual, -1) + np.log(8.)
        self.assertGreater(float((nll.mean(1) - ensemble_nll).max().detach()), 1e-5)

    def test_cpu_backbone_and_saved_replay(self):
        frame = pd.DataFrame({name: [-12., 10., 131167., np.nan] for name in
                              ['takeoff_minus_' + clock for clock in adapter.CLOCKS]})
        frame['airport'] = pd.Categorical(['A', 'B', 'A', None])
        encoder = adapter.FrameEncoder().fit(frame, neural=True)
        values = adapter.tensors(encoder, frame, 100., 20.)
        network = adapter.SourceMixture(values[0].shape[1], [4], 20.)
        parameters = network(*values, return_distribution=True)
        loss = adapter.member_negative_log_likelihood(parameters, torch.tensor([1., -4., 1000., 0.])).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None))
        predicted = adapter.infer(network, values, device='cpu')
        self.assertEqual(predicted.shape, (4,))
        self.assertTrue(np.isfinite(predicted).all())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.joblib'
            joblib.dump({'network': network, 'encoder': encoder}, path)
            loaded = joblib.load(path)
            replay = adapter.infer(loaded['network'], adapter.tensors(loaded['encoder'], frame, 100., 20.), device='cpu')
        np.testing.assert_array_equal(predicted, replay)


if __name__ == '__main__':
    unittest.main(verbosity=2)
