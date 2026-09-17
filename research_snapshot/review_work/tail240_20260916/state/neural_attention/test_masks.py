"""Supplementary fit-only statistic and explicit own/peer clock mask checks."""
import unittest
import torch
import lightgbm
import numpy as np
import test_attention as base


class MaskTests(base.AttentionTests):
    def test_own_peer_age_missing_and_arrival_relations(self):
        model = self.network()
        frame = self.frame.copy()
        frame.loc[2, base.subject.OWN_CLOCKS[0]] = np.nan
        frame.loc[3, base.subject.PREFIXES[0] + base.subject.FIELDS[0]] = -999999.
        frame.loc[4, base.subject.PREFIXES[0] + 'age_sec'] = np.nan
        tokens, present = model.token_inputs(self.tensors(frame)[0])
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(present.all())
        self.assertTrue((tokens[[2, 3, 4], 0, 38] == 1).all())
        self.assertTrue((tokens[[2, 3, 4], 0, 43] == 1).all())
        self.assertTrue((tokens[[2, 3, 4], 0, 14] == 0).all())
        self.assertTrue((tokens[[2, 3, 4], 0, 19] == 0).all())
        self.assertTrue((tokens[:, 4:, 38:48] == 1).all())

    def test_padded_fit_values_do_not_change_shared_statistics(self):
        frame = self.frame.copy()
        frame[base.subject.PREFIXES[0] + 'padding_missing'] = 1.
        before = base.subject.token_statistics(frame, self.encoder)
        for field in base.subject.FIELDS[:-1]:
            frame[base.subject.PREFIXES[0] + field] = 1e20
        after = base.subject.token_statistics(frame, self.encoder)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
