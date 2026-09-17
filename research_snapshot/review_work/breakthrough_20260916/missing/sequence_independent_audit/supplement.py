"""Independent encoding and global-neighbor-bound checks, without model fitting."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'review_work/breakthrough_20260916/sequence_context'
sys.path.insert(0, str(SOURCE))
import adapter
import cache
from test_contract import raw_fixture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    out = ROOT / 'private_runs/breakthrough_20260916/missing/sequence_independent_audit/encoding_bounds.json'
    if out.exists():
        raise ValueError('Preserve completed supplemental receipt')
    manifest = json.loads((cache.OUT/'manifest.json').read_text())
    neighbors = np.load(cache.OUT/'neighbors.npy', mmap_mode='r')
    checked = 0
    for record in manifest['records']:
        offset = record['query_offset']
        end = offset + record['query_rows']
        selected = neighbors[offset:end]
        low = record['event_offset']
        high = low + record['event_rows']
        assert np.all((selected == -1) | ((selected >= low) & (selected < high)))
        checked += len(selected)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        encoder = adapter.NumericEncoder().fit([[np.nan, -999999], [np.nan, np.inf]])
    np.testing.assert_array_equal(encoder.mean, [0, 0])
    np.testing.assert_array_equal(encoder.std, [1, 1])
    np.testing.assert_array_equal(encoder.transform([[np.nan, -999999]]), [[0, 0, 1, 1]])
    raw, arrivals = raw_fixture()
    for name in cache.CLOCKS:
        raw[name] = pd.NaT
    for name in ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt']:
        raw[name] = None
    events = cache.events_from_public(raw, arrivals)
    queries = cache.query_frame(raw.iloc[[0]])
    indices = cache.neighbor_indices(events, queries)
    store = adapter.EventStore(events, queries, indices)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        event_encoder = store.fit_encoder(np.array([0]))
    en, ec, mask, support = store.batch(np.array([0]), store.encoded(event_encoder))
    assert np.all(np.isfinite(en))
    np.testing.assert_array_equal(en[mask][:, :6], 0)
    np.testing.assert_array_equal(en[mask][:, 8:14], 1)
    np.testing.assert_array_equal(ec[mask][:, 1:], 0)
    np.testing.assert_array_equal(en[~mask], 0)
    np.testing.assert_array_equal(ec[~mask], 0)
    query_encoder = adapter.CategoryEncoder().fit(pd.DataFrame({'c': ['m:', '<missing>', 'MISSING', None, 'known']}))
    normalized = query_encoder.transform(pd.DataFrame({'c': ['m:', '<missing>', 'MISSING', None, 'unseen']}))
    query_missing_policy = normalized[:, 0].tolist()
    torch.manual_seed(20260916)
    network = adapter.Network(2, [], event_encoder['cats'].sizes+[2, 2, 2]).eval()
    qn = np.zeros((1, 2), dtype='float32')
    qc = np.zeros((1, 0), dtype='int64')
    args = adapter.tensors((qn, qc), (en, ec, mask, support), 'cpu')
    with torch.no_grad():
        baseline = network(*args, contextual=True)
        changed = [value.clone() for value in args]
        changed[2][~changed[4]] = 10000
        changed[3][~changed[4]] = 1
        assert torch.equal(baseline, network(*changed, contextual=True))
        changed[4][:] = False
        zero_a = network(*changed, contextual=True)
        changed[2][:] = -98765
        changed[3][:] = 0
        assert torch.equal(zero_a, network(*changed, contextual=True))
        assert torch.isfinite(zero_a).all()
    result = {'status': 'passed', 'global_bounds_query_rows': checked,
        'source_hashes': {p.name: sha(p) for p in [Path(__file__), SOURCE/'adapter.py', SOURCE/'cache.py']},
        'cache_manifest_sha256': sha(cache.OUT/'manifest.json'),
        'all_missing_numeric_zero_imputation_and_one_mask': True,
        'missing_event_categories_and_equality_zero': True,
        'padded_tokens_cannot_change_prediction': True,
        'zero_context_tokens_cannot_change_prediction': True,
        'query_missing_tokens_m_angle_upper_null_then_unknown': query_missing_policy,
        'no_model_fitting': True}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
