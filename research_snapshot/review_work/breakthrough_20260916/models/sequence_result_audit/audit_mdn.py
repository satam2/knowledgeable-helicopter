"""Frozen MDN hashes, cohort/metric checks and bounded independent CPU replay."""
import torch
import lightgbm
import sys
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import psutil
from threadpoolctl import threadpool_limits
import prepare_fit_canary as fixture
import audit

ROOT = fixture.ROOT
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models/source_mdn'))
import adapter


def main():
    torch.set_num_threads(2)
    torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(AssertionError('CPU audit must not initialize CUDA'))
    results = {}
    base = ROOT / 'private_runs/breakthrough_20260916'
    for fold, month, nextmonth in [('F1', '07', '08'), ('F3', '11', '12')]:
        folder = base / 'models/source_mdn/conventions' / f'tabm_source_mdn_finite_rawmean_{fold}_s20260916'
        record = audit.checked(folder)
        control = audit.checked(base / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916')
        assert record['fit_ids'] == control['fit_ids']
        assert record['split'] == control['split']
        assert record['seed'] == control['seed'] == 20260916
        assert len(record['feature_columns']) == 115
        history = record['fit']['history']
        best = min(history, key=lambda r: r['tune_raw_mse_sec2'])
        assert record['fit']['steps'] == best['epoch'] == record['refit']['steps']
        assert len(record['refit']['history']) == best['epoch']
        snapshot = base / 'models/source_mdn/conventions/source_snapshots'
        for name, digest in record['source_hashes'].items():
            source = snapshot / Path(name).name
            if not source.exists():
                source = ROOT / ('review_work/campaign_20260916/common.py' if name == 'prior_common.py' else f'review_work/breakthrough_20260916/models/{name}')
            assert fixture.sha256(source) == digest, name
        saved = pd.read_parquet(folder / 'candidate.parquet')
        ref, _ = audit.common.reference(fold)
        np.testing.assert_array_equal(saved[audit.ID], ref[audit.ID])
        np.testing.assert_array_equal(saved[audit.TARGET], ref[audit.TARGET])
        missing = ~np.isfinite(saved.proxy_sec.to_numpy())
        np.testing.assert_array_equal(saved.prediction_sec.to_numpy()[missing], ref.prediction_sec.to_numpy()[missing])
        for variant in ('candidate', 'blend25'):
            frame = pd.read_parquet(folder / f'{variant}.parquet')
            error = frame.prediction_sec.to_numpy(float) - frame[audit.TARGET].to_numpy(float)
            metrics = record['reports'][variant]['metrics']['overall']
            assert metrics['n'] == len(frame)
            np.testing.assert_allclose(np.square(error).sum(), metrics['sse'], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.square(error).mean()), metrics['rmse_sec'], rtol=1e-12)
        finite = np.flatnonzero(~missing)
        positions = finite[np.linspace(0, len(finite) - 1, 128, dtype=int)]
        subset = saved.iloc[positions].set_index(audit.ID)
        ids = subset.index
        path = ROOT / f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-{month}-01_2025-{nextmonth}-01.parquet'
        assert fixture.sha256(path) == fixture.read_json(path.with_suffix('.json'))['sha256']
        x = fixture.selected_frame(path, [c for c in pq.read_schema(path).names if c != audit.ID], ids)
        for receipt in record['anchor']['feature_receipts']:
            root = Path(receipt['manifest']).parent
            assert fixture.sha256(root / 'manifest.json') == receipt['manifest_sha256']
            path = root / 'features.parquet'
            assert fixture.sha256(path) == receipt['features_sha256']
            extra = fixture.selected_frame(path, receipt['columns'], ids)
            for col in extra:
                if pd.api.types.is_numeric_dtype(extra[col]):
                    extra[col] = extra[col].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
                else:
                    extra[col] = extra[col].astype('string').fillna('MISSING').astype('category')
            x = pd.concat([x, extra], axis=1)
        x = x[record['feature_columns']]
        model = joblib.load(folder / 'model.joblib')
        assert next(model['network'].parameters()).device.type == 'cpu'
        value = adapter.infer(model['network'], adapter.tensors(model['encoder'], x, model['center'], model['scale']), device='cpu', batch_size=128)
        prediction = value * model['scale'] + model['center']
        delta = np.abs(prediction - subset.prediction_sec.to_numpy())
        assert np.isfinite(prediction).all() and delta.max() < .05
        result = {'status': 'passed', 'selected_epoch': best['epoch'], 'tune_min_raw_mse_sec2': best['tune_raw_mse_sec2'],
                  'history_epochs': len(history), 'candidate_rmse': record['reports']['candidate']['metrics']['overall']['rmse_sec'],
                  'blend25_rmse': record['reports']['blend25']['metrics']['overall']['rmse_sec'],
                  'cpu_replay_rows': len(ids), 'cpu_gpu_max_abs_delta_sec': float(delta.max()),
                  'cpu_gpu_tolerance_sec': .05, 'full_saved_evaluator_replay_delta': record['reload_max_abs_delta'],
                  'missing_v2_exact_rows': int(missing.sum()), 'manifest_sha256': fixture.sha256(folder / 'manifest.json'),
                  'versus_leaf63': audit.compare(pd.read_parquet(base / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916' / 'candidate.parquet'), saved)}
        results[fold] = result
        print('MDN_AUDIT', fold, result['candidate_rmse'], result['selected_epoch'], result['cpu_gpu_max_abs_delta_sec'], flush=True)
        del model, x, extra, saved
        gc.collect()
    assert not torch.cuda.is_initialized()
    fixture.write_json(audit.OUT / 'mdn_audit.json', {'status': 'passed', 'source_sha256': fixture.sha256(__file__), 'folds': results,
        'cuda_initialized': False, 'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
        'scope': 'Full saved output/source hashes, score metrics, cohorts and missing fallback verified; independent CPU replay128 rows/fold, not full CPU replay.'})


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
