import torch
import unittest
import numpy as np
import pandas as pd
import adapter
import cache
from test_contract import raw_fixture


class EncoderBoundary(unittest.TestCase):
    def test_cached_missing_categories_are_missing_not_matches(self):
        fitted = adapter.CategoryEncoder().fit(pd.DataFrame({'c': ['<missing>', 'MISSING', 'm:', 'known']}))
        self.assertEqual(fitted.transform(pd.DataFrame({'c': ['<missing>', 'MISSING', 'm:']})).tolist(), [[0], [0], [0]])
        raw, arr = raw_fixture()
        events = cache.events_from_public(raw, arr)
        query = cache.query_frame(raw.iloc[[0]])
        events['operator'] = '<missing>'
        query['operator'] = '<missing>'
        store = adapter.EventStore(events, query, cache.neighbor_indices(events, query))
        encoder = store.fit_encoder(np.array([0]))
        _, categories, mask, _ = store.batch(np.array([0]), store.encoded(encoder))
        self.assertTrue((categories[mask, -1] == 0).all())

    def test_score_category_frequency_cannot_train_vocabulary(self):
        fit = pd.DataFrame({'c': ['fit', 'fit', None]})
        enc = adapter.CategoryEncoder().fit(fit)
        score = pd.DataFrame({'c': ['score'] * 100 + ['fit', None]})
        values = enc.transform(score)
        self.assertTrue((values[:100] == 1).all())
        self.assertEqual(values[100, 0], 2)
        self.assertEqual(values[101, 0], 0)
        self.assertEqual(enc.maps['c'], {'fit': 2})

    def test_only_referenced_events_fit_categories(self):
        raw, arr = raw_fixture()
        events = cache.events_from_public(raw, arr)
        queries = cache.query_frame(raw.iloc[[0, 1]])
        idx = cache.neighbor_indices(events, queries)
        store = adapter.EventStore(events, queries, idx)
        referenced = store.training_events(np.array([1]))
        events.loc[~events.index.isin(referenced), 'operator'] = 'score_only'
        store = adapter.EventStore(events, queries, idx)
        fitted = store.fit_encoder(np.array([1]))
        self.assertNotIn('score_only', fitted['cats'].maps['operator'])

    def test_control_masks_event_values_and_order(self):
        torch.manual_seed(1)
        net = adapter.Network(4, [5], [5] * 6 + [2] * 3)
        qn = torch.ones(2, 4)
        qc = torch.ones(2, 1, dtype=torch.long)
        en = torch.randn(2, 32, 22)
        ec = torch.ones(2, 32, 9, dtype=torch.long)
        mask = torch.ones(2, 32, dtype=torch.bool)
        support = torch.ones(2, 3)
        control = net(qn, qc, en, ec, mask, support, False)
        changed = net(qn, qc, en*100, ec*0, mask, support, False)
        self.assertTrue(torch.equal(control, changed))
        self.assertFalse(torch.equal(net(qn, qc, en, ec, mask, support, True),
                                     net(qn, qc, en.flip(1), ec, mask, support, True)))


if __name__ == '__main__':
    unittest.main()
