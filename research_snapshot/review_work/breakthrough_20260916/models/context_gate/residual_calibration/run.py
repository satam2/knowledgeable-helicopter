"""One fixed additive residual model on the saved early60/late40 tune split."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import lightgbm as lgb
import gc
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'sequence_result_audit'))
import run_tune
import prepare_fit_canary as fixture
from encoders import FrameEncoder

OUT = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/residual_calibration_v1'
ID, TARGET, MOVEMENT = run_tune.ID, run_tune.TARGET, run_tune.MOVEMENT
read_json, write_json, sha256, object_hash = fixture.read_json, fixture.write_json, fixture.sha256, fixture.object_hash
PARAMS = dict(n_estimators=200, max_depth=4, num_leaves=16, min_child_samples=500,
              learning_rate=.03, reg_lambda=20., objective='regression', random_state=20260916,
              n_jobs=2, verbosity=-1, subsample=1., colsample_bytree=1., deterministic=True,
              force_col_wise=True)


def budget():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if info.rss > 2 * 1024**3:
        raise MemoryError('Residual calibration exceeds2GiB processRSS')
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Host free memory below8GiB')
    return int(peak)


def features(tune, marker):
    ids = pd.Index(tune[ID])
    stamp = pd.to_datetime(tune[MOVEMENT], utc=True).iloc[0]
    start = stamp.normalize().replace(day=1)
    end = start + pd.offsets.MonthBegin(1)
    path = ROOT / f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet'
    assert sha256(path) == read_json(path.with_suffix('.json'))['sha256']
    x = fixture.selected_frame(path, [c for c in pq.read_schema(path).names if c != ID], ids)
    desired = marker['feature_columns']
    receipts = []
    for receipt in marker['anchor']['feature_receipts']:
        root = Path(receipt.get('path', receipt.get('manifest'))).parent
        filename = 'features.parquet' if receipt.get('block') == 'conventions' else 'training_features.parquet'
        path = root / filename
        source = read_json(root / 'manifest.json')
        expected = source['feature_sha256'] if filename == 'features.parquet' else source['outputs'][filename]
        assert sha256(path) == expected
        available = pq.read_schema(path).names
        columns = [c for c in desired if c in available and c not in x]
        if not columns:
            continue
        extra = fixture.selected_frame(path, columns, ids)
        for column in columns:
            if pd.api.types.is_numeric_dtype(extra[column]):
                extra[column] = extra[column].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
            else:
                extra[column] = extra[column].astype('string').fillna('MISSING').astype('category')
        x = pd.concat([x, extra], axis=1)
        receipts.append({'path': str(path), 'sha256': expected, 'columns': columns})
        del extra
        gc.collect()
        budget()
    x = x[desired]
    assert len(x.columns) == 225
    for first, second in [('lgb63', 'tabm_source'), ('catboost_conventions', 'tabm_source'), ('catboost_conventions', 'lgb63')]:
        x[f'disagreement_{first}_minus_{second}'] = (tune[first].to_numpy() - tune[second].to_numpy()).astype('float32')
    assert list(x.index) == list(ids)
    return x, receipts


def stability(tune, prediction):
    late = ~tune.early.to_numpy(bool)
    y = tune[TARGET].to_numpy(float)[late]
    base = tune.airport3.to_numpy(float)[late]
    proposed = prediction[late]
    a, b = (base - y)**2, (proposed - y)**2
    days = pd.to_datetime(tune.loc[late, MOVEMENT], utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    day_deltas = {}
    for day in np.unique(days):
        keep = days != day
        day_deltas[day] = float(np.sqrt(b[keep].mean()) - np.sqrt(a[keep].mean()))
    gain = a - b
    top = np.argsort(gain)[::-1]
    top_removal = {}
    for n in (1, 2, 5, 10):
        keep = np.ones(len(y), bool)
        keep[top[:n]] = False
        top_removal[str(n)] = float(np.sqrt(b[keep].mean()) - np.sqrt(a[keep].mean()))
    return {'leave_one_day_out_delta_rmse': day_deltas, 'remove_largest_gain_rows_delta_rmse': top_removal,
            'late_sse_gain': float(gain.sum()), 'late_baseline_sse': float(a.sum()), 'late_candidate_sse': float(b.sum())}


def main():
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    OUT.mkdir(parents=True, exist_ok=False)
    previous = read_json(run_tune.OUT / 'summary.json')
    sources = [Path(__file__), Path(fixture.__file__), Path(run_tune.__file__), HERE.parents[1] / 'encoders.py']
    protocol = {'status': 'declared', 'created_utc': run_tune.utc_now(), 'parameters': PARAMS,
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'original_tune_summary_sha256': sha256(run_tune.OUT / 'summary.json'),
        'features': 'Exactcombined225rawfeatures plus3signedexpertpairdisagreements; fit-onlyFrameEncoder.',
        'baseline': 'Savedthreeexpertairportshrinkagesimplex fittedonly onearly60; unchanged.',
        'target': 'UnclippedrawY minus savedearly-airport3prediction. Everyoriginalordinaryearlytunerow.',
        'prediction': 'Savedairport3 + unrestrictedadditiveGBDTcorrection; no clipping/projection.',
        'fit': 'Singlefixed200trees depth4 leaves16 minleaf500 lr.03 L2=20; no eval_set, earlystop or grid.',
        'split': 'Exactexistingearly60late40 originaltunepartitions; late evaluation only.',
        'scope': 'No scoreprediction/scorelabel files, GPU orfulltuneretrain. Upstreamsource manifests provide schema andfeature receipts.',
        'caveat': 'Tunealreadyselectedexpertstopping andprevious diagnostic choices; lateisnotfreshholdout.',
        'resources': '2CPUthreads,2GiBprocessRSSguard,8GiBhostreserve,rowgroupselectivefeatureloads.'}
    write_json(OUT / 'protocol.json', protocol)
    snap = OUT / 'source_snapshots'
    snap.mkdir()
    import shutil
    for path in sources:
        shutil.copyfile(path, snap / path.name)
    results = {}
    for fold in ('F1', 'F3'):
        began = time.monotonic()
        folder = OUT / fold
        folder.mkdir()
        original = previous['folds'][fold]
        path = run_tune.OUT / fold / 'tune_diagnostic.parquet'
        assert sha256(path) == original['outputs'][path.name]
        tune = pd.read_parquet(path)
        early = tune.early.to_numpy(bool)
        late = ~early
        assert len(tune) == original['original_ordinary_tune_n'] and tune[ID].is_unique
        assert object_hash(tune.loc[early, ID].tolist()) == original['early_id_hash']
        assert object_hash(tune.loc[late, ID].tolist()) == original['late_id_hash']
        marker_path = ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916/manifest.json'
        marker = read_json(marker_path)
        x, receipts = features(tune, marker)
        y = tune[TARGET].to_numpy(float)
        baseline = tune.airport3.to_numpy(float)
        enc = FrameEncoder().fit(x.iloc[np.flatnonzero(early)])
        z = enc.transform(x)
        del x
        gc.collect()
        budget()
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(z.iloc[np.flatnonzero(early)], (y - baseline)[early])
        correction = model.predict(z)
        prediction = baseline + correction
        assert model.n_estimators_ == 200 and np.isfinite(prediction).all()
        joblib.dump({'estimator': model, 'encoder': enc}, folder / 'early_model.joblib')
        replay = baseline + joblib.load(folder / 'early_model.joblib')['estimator'].predict(z)
        np.testing.assert_array_equal(prediction, replay)
        metrics = {name: {'early_rmse': run_tune.rmse(y[early], value[early]), 'late_rmse': run_tune.rmse(y[late], value[late])}
                   for name, value in {'residual_calibration': prediction, 'airport3': baseline,
                                       'global3': tune.global3.to_numpy(), 'airport2': tune.airport2.to_numpy()}.items()}
        p = tune[run_tune.EXPERTS].to_numpy()
        tune[[ID, TARGET, MOVEMENT, 'early']].assign(baseline_sec=baseline, correction_sec=correction,
                                                  prediction_sec=prediction).to_parquet(folder / 'tune_predictions.parquet', index=False)
        record = {'status': 'complete', 'metrics': metrics, 'stability': stability(tune, prediction),
            'early_rows': int(early.sum()), 'late_rows': int(late.sum()), 'early_id_hash': original['early_id_hash'],
            'late_id_hash': original['late_id_hash'], 'features': enc.columns, 'feature_receipts': receipts,
            'schema_source_manifest_sha256': sha256(marker_path), 'iterations': model.n_estimators_,
            'late_outside_expert_hull_rows': int(((prediction < p.min(1)) | (prediction > p.max(1)))[late].sum()),
            'late_correction_percentiles_sec': np.quantile(correction[late], [0, .01, .5, .99, 1]).tolist(),
            'peak_rss_bytes': budget(), 'runtime_sec': time.monotonic() - began,
            'saved_replay_max_abs_delta': 0., 'outputs': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
        write_json(folder / 'manifest.json', record)
        results[fold] = record
        print('RESIDUAL_CALIBRATION', fold, metrics, record['stability'], 'peak', record['peak_rss_bytes'], flush=True)
        del model, enc, z, tune, p, prediction, replay, y, baseline, correction
        gc.collect()
    write_json(OUT / 'summary.json', {'status': 'complete', 'folds': results, 'protocol_sha256': sha256(OUT / 'protocol.json'),
                                     'score_reads': False, 'full_tune_fit': False, 'gpu_used': False})


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
