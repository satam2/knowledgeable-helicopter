"""Bounded exact CPU replay from reconstructed449 input rows, no fitting."""
import lightgbm
import argparse
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_fit_canary as fixture
import audit


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if peak > 2 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Bounded449 CPUreplay memory budget exceeded')
    return int(peak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=['F1', 'F3'], required=True)
    args = parser.parse_args()
    fold = args.fold
    destination = audit.OUT / 'closing_1254' / f'replay449_{fold}.json'
    if destination.exists():
        raise ValueError('Preserve previous replay receipt')
    month, nextmonth = ('07', '08') if fold == 'F1' else ('11', '12')
    folder = audit.BASE / 'deeper_sequence8' / f'lightgbm_leaf63_sequence8_aobt_allfinite_{fold}_s20260916'
    marker = audit.checked(folder)
    saved = pd.read_parquet(folder / 'candidate.parquet')
    finite = np.flatnonzero(np.isfinite(saved.proxy_sec.to_numpy()))
    chosen = finite[np.linspace(0, len(finite) - 1, 128, dtype=int)]
    selected = saved.iloc[chosen].set_index(audit.ID)
    ids = selected.index
    path = audit.ROOT / f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-{month}-01_2025-{nextmonth}-01.parquet'
    assert audit.sha256(path) == audit.read_json(path.with_suffix('.json'))['sha256']
    x = fixture.selected_frame(path, [c for c in pq.read_schema(path).names if c != audit.ID], ids)
    for receipt in marker['anchor']['feature_receipts']:
        root = Path(receipt.get('path', receipt.get('manifest'))).parent
        filename = 'features.parquet' if receipt.get('block') == 'conventions' else 'training_features.parquet'
        path = root / filename
        source = audit.read_json(root / 'manifest.json')
        expected = source['feature_sha256'] if filename == 'features.parquet' else source['outputs'][filename]
        assert audit.sha256(path) == expected
        available = pq.read_schema(path).names
        columns = [c for c in marker['feature_columns'][:225] if c in available and c not in x]
        if not columns:
            continue
        extra = fixture.selected_frame(path, columns, ids)
        for name in columns:
            if pd.api.types.is_numeric_dtype(extra[name]):
                extra[name] = extra[name].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
            else:
                extra[name] = extra[name].astype('string').fillna('MISSING').astype('category')
        x = pd.concat([x, extra], axis=1)
        del extra
        guard()
    root = audit.BASE / 'missing/sequence_flatten8'
    manifest = audit.read_json(root / 'manifest.json')
    assert manifest['outputs']['training_features.parquet'] == audit.sha256(root / 'training_features.parquet')
    extra = fixture.selected_frame(root / 'training_features.parquet', manifest['features'], ids)
    x = pd.concat([x, extra], axis=1)[marker['feature_columns']]
    assert x.shape == (128, 449)
    model = joblib.load(folder / 'model.joblib')
    guard()
    prediction = np.asarray(model['estimator'].predict(model['encoder'].transform(x), num_iteration=model['steps'], num_threads=2)) + selected.proxy_sec.to_numpy()
    difference = np.abs(prediction - selected.prediction_sec.to_numpy())
    status = 'passed' if np.array_equal(prediction, selected.prediction_sec.to_numpy()) else 'strict_replay_failed'
    report = {'status': status, 'fold': fold, 'source_sha256': audit.sha256(__file__),
        'manifest_sha256': audit.sha256(folder / 'manifest.json'), 'model_sha256': audit.sha256(folder / 'model.joblib'),
        'rows': 128, 'columns': 449, 'ID_hash': audit.object_hash(ids.tolist()), 'declared_tolerance_sec': 0.,
        'max_absolute_delta_sec': float(difference.max()), 'producer_full_saved_replay_delta': marker['reload_max_abs_delta'],
        'scope': '128evenlydistributedoriginaleligible scorerows, all449featuresindependentlyreconstructedfromverifiedcaches; sameCPUalgorithm, nofitting/GPU.',
        'peak_rss_bytes': guard()}
    audit.write_json(destination, report)
    print('REPLAY449', report, flush=True)
    assert status == 'passed', 'Strictzero tolerance failed; receipt preserved'


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
