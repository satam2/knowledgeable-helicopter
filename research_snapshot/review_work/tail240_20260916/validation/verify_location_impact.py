"""Recount location-mask impact without importing the proposed mask helper."""
from verify_category_profile import ROOT, digest, read, guard
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ID = 'MVT_ID_mvt'
TIME = 'MVT_TIME_UTC_mvt'
BASE = ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
EXPECTED = ROOT / 'private_runs/tail240_20260916/state/preprocessing_semantics/v2/receipt.json'
OUT = ROOT / 'private_runs/tail240_20260916/validation/location_impact_v2'


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    expected = read(EXPECTED)
    source = ROOT / 'review_work/tail240_20260916/state/preprocessing_semantics'
    assert digest(source / 'audit_locations.py') == expected['source_sha256']
    assert digest(source / 'semantic_locations.py') == expected['helper_sha256']
    selected = []
    for batch in pq.ParquetFile(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet').iter_batches(
            batch_size=8192, columns=[ID, TIME, 'proxy_sec'], use_threads=False):
        frame = batch.to_pandas()
        selected.append(frame.loc[(frame[TIME] >= pd.Timestamp('2025-01-01', tz='UTC')) &
                                  (frame[TIME] < pd.Timestamp('2025-06-01', tz='UTC')), [ID, 'proxy_sec']])
    fit = pd.concat(selected, ignore_index=True)
    ids = pd.Index(fit[ID])
    assert ids.is_unique and len(ids) == expected['rows'] == 822377
    fit_hash = __import__('hashlib').sha256(json.dumps(fit[ID].tolist(), sort_keys=True, default=str).encode()).hexdigest()
    assert fit_hash == expected['fit_id_hash']
    del selected
    counts = Counter()
    seen = np.zeros(len(ids), dtype=bool)
    masks = {key: set() for key in ('stand', 'runway')}
    bindings = read(ROOT / 'private_runs/tail240_20260916/state/preprocessing_semantics/v1/receipt.json')['sources']
    for path in sorted(BASE.glob('training_*.parquet')):
        binding = bindings[str(path.relative_to(ROOT))]
        assert digest(path) == binding['recorded_cache_sha256']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, 'STAND_mvt', 'RUNWAY_mvt'], use_threads=False):
            frame = batch.to_pandas()
            positions = ids.get_indexer(frame[ID])
            keep = positions >= 0
            frame = frame.loc[keep]
            positions = positions[keep]
            assert len(set(positions)) == len(positions) and not seen[positions].any()
            seen[positions] = True
            finite = np.isfinite(fit.proxy_sec.iloc[positions].to_numpy())
            for field in masks:
                values = frame[field.upper() + '_mvt'].astype(str).to_numpy()
                unknown = np.zeros(len(frame), dtype=bool)
                for token in ('m:', 's0:', 's7:UNKNOWN', 's2:NA'):
                    flag = values == token
                    counts[f'{field}_{token}_all'] += int(flag.sum())
                    counts[f'{field}_{token}_finite'] += int((flag & finite).sum())
                    unknown |= flag
                masks[field].update(frame.loc[unknown, ID].tolist())
            guard()
    assert seen.all() and dict(counts) == expected['counts']
    assert {k: len(v) for k, v in masks.items()} == expected['mask_rows']
    features = read(ROOT / 'private_runs/tail240_20260916/state/neural_context/v1/F1/manifest.json')['feature_columns']
    groups = {
        'stand': [f'batch_source_stand_{direction}_{minutes}m_{metric}' for direction in ('past', 'twosided') for minutes in (15, 60)
                  for metric in ('count', 'missing_nm_share', 'negative_proxy_share', 'long_proxy_share', 'valid_proxy_count', 'valid_proxy_median_sec')],
        'query_runway': [f'batch_surface_T_runway_{metric}' for metric in ('discharge_share_15m', 'previous_wake', 'previous_wake_heavier', 'previous_wake_same')],
        'arrival_runway': [f'retro_{direction}_arr_runway_{metric}' for direction in ('past', 'future') for metric in ('count', 'mean_sec', 'std_sec', 'over_1200_share')],
        'stand_equality': [f'flat_{phase}{rank}_same_stand' for phase in ('dep', 'arr') for rank in range(1, 5)],
        'runway_equality': [f'flat_{phase}{rank}_same_runway' for phase in ('dep', 'arr') for rank in range(1, 5)],
    }
    assert all(set(cols) == set(expected['feature_groups'][key]) and set(cols) <= set(features) for key, cols in groups.items())
    affected = {}
    cache_hashes = {}
    for relative, keys in [
            ('private_runs/breakthrough_20260916/batch_context/training_features.parquet', ('stand', 'query_runway')),
            ('private_runs/breakthrough_20260916/retrospective_research/training_features.parquet', ('arrival_runway',)),
            ('private_runs/breakthrough_20260916/missing/sequence_flatten/training_features.parquet', ('stand_equality', 'runway_equality'))]:
        path = ROOT / relative
        cache_hashes[relative] = digest(path)
        columns = [col for key in keys for col in groups[key]]
        affected.update({c: dict(queries=0, finite=0, nonzero=0) for c in columns})
        coverage = {key: set() for key in keys}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[ID, *columns], use_threads=False):
            frame = batch.to_pandas()
            for key in keys:
                mask = masks['stand' if key in ('stand', 'stand_equality') else 'runway']
                part = frame.loc[frame[ID].isin(mask)]
                observed = set(part[ID])
                assert len(observed) == len(part) and not coverage[key].intersection(observed)
                coverage[key].update(observed)
                for col in groups[key]:
                    values = part[col].to_numpy(dtype=float)
                    affected[col]['queries'] += len(values)
                    affected[col]['finite'] += int(np.isfinite(values).sum())
                    affected[col]['nonzero'] += int((np.isfinite(values) & (values != 0)).sum())
            guard()
        for key in keys:
            assert coverage[key] == masks['stand' if key in ('stand', 'stand_equality') else 'runway']
    assert affected == expected['affected']
    OUT.mkdir(parents=True, exist_ok=False)
    result = dict(status='passed', verifier_sha256=digest(Path(__file__)), input_receipt_sha256=digest(EXPECTED),
                  rows=len(ids), fit_id_hash=fit_hash, counts=dict(counts), mask_rows=expected['mask_rows'],
                  feature_group_sizes={key: len(value) for key, value in groups.items()}, affected_counts_exact=True,
                  masked_ids_once_per_cache=True, base_cache_hashes_verified=True, extension_cache_hashes=cache_hashes,
                  peak_bytes=guard(), no_model_or_gpu=True,
                  limitation='Current cached impact only; UNKNOWN/NA physical-location semantics remain a hypothesis. No performance claim.')
    (OUT / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
