"""Bounded CPU replay of new saved information models from verified input rows."""
import lightgbm
import argparse
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import audit
import prepare_fit_canary as fixture


def guard():
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert peak < 2 * 1024**3 and psutil.virtual_memory().available >= 8 * 1024**3
    return int(peak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['monthly', 'innovations'], required=True)
    args = parser.parse_args()
    monthly = args.arm == 'monthly'
    root = audit.BASE / ('retrospective_models/monthly_arrival_v3' if monthly else 'models/sequence_clock_innovations/models')
    family = 'lightgbm_monthly_arrival' if monthly else 'lightgbm_clock_innovations'
    caches = [audit.BASE / 'retrospective_research/monthly_arrival'] if monthly else [
        audit.BASE / 'missing/sequence_flatten', audit.BASE / 'models/sequence_clock_innovations/cache']
    output = audit.OUT / f'{args.arm}_native_replay.json'
    assert not output.exists()
    results = {}
    for fold, month, end in [('F1', '07', '08'), ('F3', '11', '12')]:
        folder = root / f'{family}_aobt_allfinite_{fold}_s20260916'
        record = audit.checked(folder)
        saved = pd.read_parquet(folder / 'candidate.parquet')
        positions = np.flatnonzero(np.isfinite(saved.proxy_sec.to_numpy()))
        selected = saved.iloc[positions[np.linspace(0, len(positions) - 1, 128, dtype=int)]].set_index(audit.ID)
        ids = selected.index
        path = audit.ROOT / f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-{month}-01_2025-{end}-01.parquet'
        assert audit.sha256(path) == audit.read_json(path.with_suffix('.json'))['sha256']
        x = fixture.selected_frame(path, [name for name in pq.read_schema(path).names if name != audit.ID], ids)
        for receipt in record['anchor']['feature_receipts']:
            cache = Path(receipt.get('path', receipt.get('manifest'))).parent
            filename = 'features.parquet' if receipt.get('block') == 'conventions' else 'training_features.parquet'
            path = cache / filename
            manifest = audit.read_json(cache / 'manifest.json')
            assert audit.sha256(path) == (manifest['feature_sha256'] if filename == 'features.parquet' else manifest['outputs'][filename])
            available = pq.read_schema(path).names
            names = [name for name in record['feature_columns'][:225] if name in available and name not in x]
            if names:
                extra = fixture.selected_frame(path, names, ids)
                for name in names:
                    if pd.api.types.is_numeric_dtype(extra[name]):
                        extra[name] = extra[name].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
                    else:
                        extra[name] = extra[name].astype('string').fillna('MISSING').astype('category')
                x = pd.concat([x, extra], axis=1)
                del extra
            guard()
        for cache in caches:
            manifest = audit.read_json(cache / 'manifest.json')
            path = cache / 'training_features.parquet'
            assert audit.sha256(path) == manifest['outputs'][path.name]
            extra = fixture.selected_frame(path, manifest['features'], ids)
            if monthly:
                extra = extra.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
            x = pd.concat([x, extra], axis=1)
            del extra
        x = x[record['feature_columns']]
        assert x.shape == (128, 240 if monthly else 373)
        model = joblib.load(folder / 'model.joblib')
        values = np.asarray(model['estimator'].predict(model['encoder'].transform(x), num_iteration=model['steps'], num_threads=2), float) + selected.proxy_sec.to_numpy(float)
        delta = float(np.max(np.abs(values - selected.prediction_sec.to_numpy(float))))
        assert delta == 0., delta
        results[fold] = {'rows': len(x), 'columns': len(x.columns), 'max_abs_delta_sec': delta,
            'manifest_sha256': audit.sha256(folder / 'manifest.json'), 'model_sha256': audit.sha256(folder / 'model.joblib'),
            'id_hash': audit.object_hash(ids.tolist()), 'peak_rss_bytes': guard()}
        print('EXACT_NATIVE_REPLAY', args.arm, fold, results[fold], flush=True)
        del x, model, saved
        gc.collect()
    audit.write_json(output, {'status': 'passed', 'source_sha256': audit.sha256(__file__), 'folds': results,
        'tolerance_sec': 0., 'scope': '128 evenly distributed eligible score rows per fold; verified cached feature reconstruction and native CPU LightGBM; no fitting or GPU.'})


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
