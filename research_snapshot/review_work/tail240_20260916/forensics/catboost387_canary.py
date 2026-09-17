"""One synthetic GPU canary with a parent memory watchdog; no private rows."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import lightgbm  # Initialize the native runtime before NumPy on Windows.
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import catboost
import joblib
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / 'review_work/breakthrough_20260916/models_retrieval/combined_catboost/adapter.py'
sys.path.insert(0, str(ADAPTER.parent))
import adapter

OUT = ROOT / 'private_runs/tail240_20260916/forensics/catboost387_canary/v1'
SCHEMA = ROOT / 'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387/encoder.json'
ROWS = 100000
SEED = 20260916
ROUNDS = 50
CAP = 4 * 1024**3
RESERVE = 8 * 1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def schema():
    record = json.loads(SCHEMA.read_text())
    columns = record['columns']
    cardinalities = {key: len(value) for key, value in record['vocab'].items()}
    assert len(columns) == 387 and len(cardinalities) == 15
    return columns, cardinalities


def check_resources(current, peak, available):
    if max(current, peak) >= CAP:
        raise MemoryError('Synthetic process reached the 4 GiB ceiling')
    if available < RESERVE:
        raise MemoryError('Host available memory fell below 8 GiB')


def memory():
    info = psutil.Process().memory_info()
    return dict(rss=info.rss, peak=getattr(info, 'peak_wset', info.rss),
                available=psutil.virtual_memory().available)


def generate(rows, columns, cardinalities, seed=SEED):
    rng = np.random.default_rng(seed)
    numeric = [name for name in columns if name not in cardinalities]
    latent = rng.standard_normal((rows, 8), dtype=np.float32)
    data = {}
    for j, name in enumerate(numeric):
        values = rng.standard_normal(rows, dtype=np.float32)
        if j < 24:
            values = (j % 5 + 1) * latent[:, j % 8] + .02 * values
        if j == 24:
            values = latent[:, 0] - latent[:, 1]
        if j == 25:
            values = 2 * (latent[:, 0] - latent[:, 1])
        if j == 26:
            values.fill(1.)
        values[rng.random(rows) < .02] = np.nan
        if j == 27:
            values.fill(np.nan)
        values[rng.random(rows) < .001] = -999999
        data[name] = values
    for name, count in cardinalities.items():
        codes = rng.integers(0, count, rows)
        support = min(rows, count)
        codes[:support] = np.arange(support)
        codes[rng.random(rows) < .01] = -1
        data[name] = pd.Categorical.from_codes(codes, categories=[f'synthetic_{j:05d}' for j in range(count)])
    frame = pd.DataFrame(data)[columns]
    first_cat = frame[next(iter(cardinalities))].cat.codes.to_numpy()
    y = (50 * latent[:, 0] - 30 * latent[:, 1] + 20 * latent[:, 2]
         + 5 * (first_cat % 5) + rng.normal(0, 5, rows)).astype(np.float64)
    return frame, y


def declaration():
    columns, cardinalities = schema()
    return dict(source_sha256=sha(__file__), adapter_sha256=sha(ADAPTER),
        encoder_source_sha256=sha(sys.modules[adapter.FrameEncoder.__module__].__file__),
        schema_metadata_sha256=sha(SCHEMA), rows=ROWS, numeric=372, categorical=15,
        columns=columns, synthetic_category_cardinalities=cardinalities, seed=SEED,
        params=adapter.parameters(seed=SEED, threads=2, device='GPU', iterations=ROUNDS),
        data='Synthetic normals, correlated clock-like fields, exact collinearity, constant/all-missing columns, 2pct NaNs, .1pct sentinel; category cardinalities from metadata only, all identities synthetic.',
        fit='Fit-only frozen FrameEncoder, 100000 fit rows, fixed50 rounds, no tuning or generalization claim.',
        resources='GPU allocation required from parent; two CPU threads, GPU ram part .65, child current/OS peak below4GiB, startup12GiB, reserve8GiB; parent100ms polling, not OS hard limit.',
        scope='No private flight rows, no grid, no main code mutation. Native CBM save/load and same CPU inference.',
        projection='Observed CPU peak and global GPU usage provide heuristic row scaling only; categorical combinations, bins, GPU allocation and5000 rounds differ.')


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = declaration()
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write(path, protocol)
    return protocol


def worker():
    protocol = declare()
    assert not (OUT / 'worker_receipt.json').exists()
    initial = memory()
    check_resources(initial['rss'], initial['peak'], initial['available'])
    started = time.perf_counter()
    frame, y = generate(ROWS, protocol['columns'], protocol['synthetic_category_cardinalities'])
    before_fit = memory()
    check_resources(before_fit['rss'], before_fit['peak'], before_fit['available'])
    fit_started = time.perf_counter()
    bundle, evidence = adapter.fit(frame, y, steps=ROUNDS, seed=SEED, threads=2, device='GPU')
    fit_seconds = time.perf_counter() - fit_started
    after_fit = memory()
    check_resources(after_fit['rss'], after_fit['peak'], after_fit['available'])
    assert evidence['params'] == protocol['params'] and bundle['steps'] == ROUNDS
    assert len(bundle['encoder'].numeric) == 372 and len(bundle['encoder'].categories) == 15
    probe = frame.iloc[:1024].copy()
    for name in bundle['encoder'].categories:
        probe[name] = probe[name].astype('string')
        probe.loc[probe.index[0], name] = 'synthetic_unseen_probe'
        probe.loc[probe.index[1], name] = pd.NA
        assert 'synthetic_unseen_probe' not in bundle['encoder'].categories[name]
    encoded = bundle['encoder'].transform(probe)
    for name in bundle['encoder'].categories:
        assert encoded[name].iloc[0] == 1 and encoded[name].iloc[1] == 0
    model_path = OUT / 'model.cbm'
    bundle['estimator'].save_model(str(model_path), format='cbm')
    joblib.dump(bundle['encoder'], OUT / 'encoder.joblib')
    native = catboost.CatBoostRegressor()
    native.load_model(str(model_path), format='cbm')
    prediction = bundle['estimator'].predict(encoded, task_type='CPU', thread_count=2)
    reloaded_encoder = joblib.load(OUT / 'encoder.joblib')
    replay = native.predict(reloaded_encoder.transform(probe), task_type='CPU', thread_count=2)
    np.testing.assert_array_equal(prediction, replay)
    assert np.isfinite(prediction).all()
    full_prediction = adapter.predict(bundle, frame)
    assert np.isfinite(full_prediction).all()
    np.save(OUT / 'probe_predictions.npy', prediction)
    final = memory()
    check_resources(final['rss'], final['peak'], final['available'])
    assert declaration() == protocol
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum() + y.nbytes)
    incremental = max(0, after_fit['peak'] - before_fit['rss']) / ROWS
    projection = initial['rss'] + frame_bytes / ROWS * 996700 + incremental * 816060
    write(OUT / 'worker_receipt.json', dict(status='passed', source_sha256=sha(__file__),
        protocol_sha256=sha(OUT / 'protocol.json'), rows=ROWS, features=387,
        finite_full_fit_predictions=True, native_cpu_replay_max_abs_delta=0., replay_rows=len(probe),
        fit_only_unknown_and_missing_codes_verified=True, evidence=evidence,
        fit_seconds=fit_seconds, total_seconds=time.perf_counter()-started,
        memory=dict(initial=initial, before_fit=before_fit, after_fit=after_fit, final=final),
        projection=dict(synthetic_frame_and_target_bytes=frame_bytes,
            approximate_native_increment_bytes_per_fit_row=incremental,
            fullF1_heuristic_host_GiB=projection/1024**3,
            naive5000round_F1_seconds=fit_seconds*(816060/ROWS)*(5000/ROUNDS),
            limitation='Heuristic only: real categories, combinations, bins, retained state and5000round training can differ.'),
        no_private_rows=True, outputs={name:sha(OUT/name) for name in ['model.cbm','encoder.joblib','probe_predictions.npy']}))


def gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        used, total = [int(value.strip()) for value in result.stdout.splitlines()[0].split(',')]
        return dict(used_MiB=used, total_MiB=total)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def launch():
    protocol = declare()
    assert not (OUT / 'worker_receipt.json').exists() and not (OUT / 'watchdog_receipt.json').exists()
    assert psutil.virtual_memory().available >= 12*1024**3
    gpu_before = gpu_usage()
    assert gpu_before is not None, 'GPU monitoring must be available'
    peak = current_max = 0
    minimum_available = psutil.virtual_memory().available
    gpu_peak = gpu_before['used_MiB']
    gpu_samples = 1
    failure = None
    began = time.monotonic()
    with (OUT / 'run.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker'],
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        child = psutil.Process(proc.pid)
        last_gpu = began
        while proc.poll() is None:
            try:
                info = child.memory_info()
                peak = max(peak, getattr(info, 'peak_wset', info.rss))
                current_max = max(current_max, info.rss)
                minimum_available = min(minimum_available, psutil.virtual_memory().available)
                check_resources(info.rss, peak, minimum_available)
                if time.monotonic()-last_gpu >= 2:
                    sample = gpu_usage()
                    if sample:
                        gpu_peak = max(gpu_peak, sample['used_MiB'])
                        gpu_samples += 1
                    last_gpu = time.monotonic()
            except psutil.NoSuchProcess:
                break
            except MemoryError as error:
                failure = str(error)
                proc.terminate()
                break
            time.sleep(.1)
        exit_code = proc.wait(timeout=30)
    receipt = dict(status='passed' if exit_code == 0 and failure is None else 'failed',
        source_sha256=sha(__file__), protocol_sha256=sha(OUT/'protocol.json'), exit_code=exit_code,
        failure=failure, elapsed_seconds=time.monotonic()-began,
        observed_child_peak_bytes=peak, observed_child_max_current_bytes=current_max,
        minimum_host_available_bytes=minimum_available, gpu_before=gpu_before,
        sampled_global_gpu_peak_MiB=gpu_peak, gpu_samples=gpu_samples,
        gpu_measurement_limitation='Global device use including desktop;2s sampling may miss peaks. gpu_ram_part is allocator preference, not hard OS limit.',
        protocol=protocol)
    write(OUT/'watchdog_receipt.json', receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key != 'protocol'}), flush=True)
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--declare-only', action='store_true')
    mode.add_argument('--run-allocated-gpu', action='store_true')
    mode.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print('DECLARED_ONLY', memory(), flush=True)
    elif args.worker:
        worker()
    else:
        launch()
