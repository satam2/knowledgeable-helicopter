"""Versioned chronological expert selection/refit; evaluation labels are inaccessible."""
import os
import sys
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
if '--family' in sys.argv and sys.argv[sys.argv.index('--family')+1] in ('ple', 'catboost'):
    import torch
import lightgbm
import argparse
import gc
import importlib.util
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/risk'))
import run_risk as risk
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import schema_discovery
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/forensics/source_contract/v2'))
import contract as source_contract
ID, TARGET, MOVEMENT = common.ID, common.TARGET, common.MOVEMENT
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/robustness/chronological_v1')
COHORTS = ROOT / 'private_runs/tail240_20260916/robustness/validation/cohorts_v1'
META = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
FAMILIES = ['lgb', 'ple', 'catboost']
ADAPTERS = {
    'lgb': ROOT / 'review_work/breakthrough_20260916/deeper_lgb/adapter.py',
    'ple': ROOT / 'review_work/breakthrough_20260916/models/tabm_ple_gpu.py',
    'catboost': ROOT / 'review_work/breakthrough_20260916/models_retrieval/combined_catboost/adapter.py'}


def load_adapter(family):
    sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/models'))
    if family == 'ple':
        import tabm_ple_gpu as adapter
        import torch
        def grouped_bins(numbers, n_original_numeric):
            varying = (numbers[:, :n_original_numeric] != numbers[0, :n_original_numeric]).any(dim=0)
            embedded = torch.where(varying)[0]
            chosen = set(embedded.tolist())
            passthrough = torch.tensor([i for i in range(numbers.shape[1]) if i not in chosen], dtype=torch.long)
            bins = []
            for columns in embedded.split(16):
                bins.extend(adapter.rtdl_num_embeddings.compute_bins(numbers[:, columns],
                    n_bins=min(adapter.N_BINS, len(numbers)-1)))
            return bins, embedded, passthrough
        adapter.fit_bins = grouped_bins
        return adapter
    spec = importlib.util.spec_from_file_location('robustness_' + family, ADAPTERS[family])
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    return adapter


def declare():
    columns = common.read_json(ROOT / 'private_runs/tail240_20260916/state/neural_context/v1/protocol.json')['feature_columns']
    assert len(columns) == 387
    assert common.sha256(META) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][META.name]
    source_paths = [Path(__file__), Path(risk.__file__), Path(schema_discovery.__file__),
        Path(source_contract.__file__), Path(__file__).with_name('mixture.py'),
        ROOT / 'review_work/campaign_20260916/common.py',
        ROOT / 'review_work/campaign_20260916/lgbm_adapter.py',
        ROOT / 'review_work/breakthrough_20260916/models/encoders.py',
        ROOT / 'review_work/breakthrough_20260916/models/tabm_gpu.py', *ADAPTERS.values()]
    protocol = dict(feature_columns=columns, families=FAMILIES, folds=['F1', 'F3'],
        sources={str(p.relative_to(ROOT)): common.sha256(p) for p in source_paths},
        metadata_sha256=common.sha256(META), cohort_manifest_sha256=common.sha256(COHORTS / 'manifest.json'),
        seed=20260916, threads=2, target='raw_target-minus-proxy, all finite proxy fitting rows',
        stopping='train-only encoder; stop-only RMSE; no calibration/evaluation labels',
        refit='fresh encoder on purged train+stop; fixed stop-selected length; no post-calibration refit',
        prediction='All finite calibration and declared evaluation rows; no evaluation label reads',
        ensemble={'primary': '.5 simplex + .5 equal', 'comparator': 'equal', 'diagnostic': 'simplex',
                  'calibration': 'ordinary proxy rows only', 'score_selected_variants': False},
        evaluation={'panels': {'F1': ['2025-06', '2025-07', '2025-12'],
                              'F3': ['2025-10', '2025-11', '2025-12']},
                    'bootstrap_repetitions': 2000, 'bootstrap_seed': 20260916,
                    'tail_deletions': [1, 2, 5, 10], 'all_day_removals': True,
                    'advancement': 'Primary shrunk must improve equal in July and November, both lower95 paired-day MSEgain >0, all day removals and top10 beneficial deletion positive; both December lower95 MSEgain >=0. Historical V3 hybrid comparison is secondary and insufficient for replacement.'},
        exposure='All 2025 months previously exposed; within-run chronology only, not fresh holdout',
        release='No ranking prediction, submission, or automatic replacement of V3',
        resources={'process_gib': 20, 'startup_available_gib': 28, 'host_reserve_gib': 8,
                   'exclusive_gpu': True, 'threads': 2})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == protocol, 'Frozen source or protocol changed'
    else:
        common.write_json(path, protocol)
    return protocol


def read_stage(fold, stage):
    manifest = common.read_json(COHORTS / 'manifest.json')
    record = manifest['folds'][fold]['stages'][stage]
    path = Path(record['path'])
    if not path.is_absolute():
        path = ROOT / path
    assert common.sha256(path) == record['sha256']
    frame = pd.read_parquet(path)
    assert TARGET not in frame and frame[ID].is_unique
    assert common.object_hash(frame[ID].tolist()) == record['ordered_id_hash']
    return frame


def read_labels(frame):
    # Parquet predicate restricts label reads to the caller's supervised interval.
    low, high = frame[MOVEMENT].min(), frame[MOVEMENT].max()
    values = pd.read_parquet(META, columns=[ID, TARGET],
        filters=[(MOVEMENT, '>=', low), (MOVEMENT, '<=', high)]).set_index(ID)
    assert values.index.is_unique
    y = values.loc[frame[ID], TARGET].to_numpy(float)
    assert np.isfinite(y).all()
    return y


def decode(matrix, vocab, columns):
    values = {}
    base_categories = {'ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt',
        'airport_stand', 'airport_runway'}
    for j, name in enumerate(columns):
        if name in vocab:
            assert '__UNSEEN_CONTEXT_VALUE__' not in vocab[name]
            labels = np.asarray([np.nan if name in base_categories else 'MISSING',
                '__UNSEEN_CONTEXT_VALUE__', *vocab[name]], dtype=object)
            values[name] = pd.Categorical(labels[matrix[:, j].astype(np.int32)])
        else:
            values[name] = matrix[:, j]
    return pd.DataFrame(values, copy=False)


def guard():
    info = psutil.Process().memory_info()
    assert max(info.rss, getattr(info, 'peak_wset', info.rss)) < 20*1024**3, 'Process memory budget exceeded'
    assert psutil.virtual_memory().available >= 8*1024**3, 'Host reserve below8GiB'
    return info.rss


def start_watchdog(destination):
    finished = threading.Event()
    def monitor():
        last_report = time.monotonic()
        while not finished.wait(1):
            rss = psutil.Process().memory_info().rss
            available = psutil.virtual_memory().available
            if rss >= 20*1024**3 or available < 8*1024**3:
                common.write_json(destination/'resource_failure.json',
                    {'rss': rss, 'available': available, 'utc': common.utc_now()})
                print('RESOURCE_LIMIT_EXCEEDED', rss, available, flush=True)
                os._exit(86)
            if time.monotonic()-last_report >= 45:
                print('RESOURCE_HEARTBEAT', round(rss/1024**3, 2),
                      round(available/1024**3, 2), flush=True)
                last_report = time.monotonic()
    threading.Thread(target=monitor, name='resource-watchdog', daemon=True).start()
    return finished


def run(fold, family, phase):
    protocol = declare()
    assert psutil.virtual_memory().available >= 28*1024**3, 'Startup requires28GiB free'
    destination = common.external_path(OUT / fold / family / phase)
    destination.mkdir(parents=True, exist_ok=False)
    common.write_json(destination / 'launch.json', {'utc': common.utc_now(),
        'protocol_sha256': common.sha256(OUT/'protocol.json'), 'phase': phase})
    watchdog = start_watchdog(destination)
    started = time.monotonic()
    adapter = load_adapter(family)
    if family != 'lgb':
        import torch
        assert torch.cuda.is_available(), 'GPU families require breakthrough-venv'
    stages = ['train', 'stop'] if phase == 'select' else ['refit', 'calibration'] + sorted(
        key for key in common.read_json(COHORTS/'manifest.json')['folds'][fold]['stages']
        if key.startswith('evaluation_'))
    parts = {s: read_stage(fold, s) for s in stages}
    parts = {s: f.loc[np.isfinite(f.proxy_sec)].copy() for s, f in parts.items()}
    ids = pd.Index(pd.concat([f[ID] for f in parts.values()], ignore_index=True))
    assert ids.is_unique
    fit_stage = stages[0]
    original = risk.feature_sources
    def cached(columns):
        sources, receipt = schema_discovery.cached_discovery(original, columns)
        common.write_json(destination/'discovery.json', receipt)
        return sources
    sources = cached(protocol['feature_columns'])
    matrix, vocab, receipts, coverage = source_contract.load_checked(risk, sources,
        ids, len(parts[fit_stage]), protocol['feature_columns'], destination, guard=guard)
    common.write_json(destination/'source_coverage.json', coverage)
    x = decode(matrix, vocab, protocol['feature_columns'])
    x.index = ids
    offset, frames = 0, {}
    for name, part in parts.items():
        frames[name] = x.iloc[offset:offset+len(part)]
        offset += len(part)
    assert offset == len(x)
    events = []
    def labels(stage):
        y = read_labels(parts[stage])
        events.append({'kind': 'expert_label_read', 'stage': stage,
                       'ordered_id_hash': common.object_hash(parts[stage][ID].tolist()),
                       'target_hash': common.object_hash(y.tolist()), 'utc': common.utc_now()})
        return y - parts[stage].proxy_sec.to_numpy(float)
    y = labels(fit_stage)
    tuning = (frames['stop'], labels('stop')) if phase == 'select' else None
    steps = None
    if phase == 'refit':
        selection = common.read_json(destination.parent/'select/manifest.json')
        assert selection['protocol_sha256'] == common.sha256(OUT/'protocol.json')
        steps = selection['selected_steps']
    guard()
    print('START_FIT', fold, family, phase, len(y), steps, flush=True)
    model, evidence = adapter.fit(frames[fit_stage], y, tuning=tuning, steps=steps, seed=20260916, threads=2)
    guard()
    joblib.dump(model, destination/'model.joblib')
    common.write_json(destination/'fit_evidence.json', evidence)
    events.append({'kind': 'expert_frozen', 'stage': fit_stage, 'utc': common.utc_now(),
                   'model_sha256': common.sha256(destination/'model.joblib')})
    replay = joblib.load(destination/'model.joblib')
    predictions = {}
    for stage in stages[1:]:
        result = parts[stage].copy()
        result['prediction_sec'] = adapter.predict(model, frames[stage]) + result.proxy_sec.to_numpy(float)
        native = adapter.predict(replay, frames[stage]) + result.proxy_sec.to_numpy(float)
        difference = float(np.max(np.abs(native-result.prediction_sec.to_numpy())))
        assert difference <= 1e-6 and np.isfinite(native).all()
        result.to_parquet(destination/f'{stage}.parquet', index=False)
        predictions[stage] = {'rows': len(result), 'native_max_delta_sec': difference,
            'ordered_id_hash': common.object_hash(result[ID].tolist()),
            'sha256': common.sha256(destination/f'{stage}.parquet')}
        guard()
    events.append({'kind': 'predictions_frozen', 'utc': common.utc_now()})
    common.write_json(destination/'access.json', events)
    common.write_json(destination/'manifest.json', {'status': 'complete', 'fold': fold, 'family': family,
        'phase': phase, 'protocol_sha256': common.sha256(OUT/'protocol.json'),
        'selected_steps': evidence['steps'], 'fit_rows': len(y), 'predictions': predictions,
        'model_sha256': common.sha256(destination/'model.joblib'),
        'fit_id_hash': common.object_hash(parts[fit_stage][ID].tolist()),
        'features': protocol['feature_columns'], 'feature_sources': receipts,
        'runtime_sec': time.monotonic()-started,
        'peak_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', guard()),
        'evaluation_labels_read': False,
        'outputs': {p.name: common.sha256(p) for p in destination.iterdir()
                    if p.is_file() and p.name != 'matrix.float32'}})
    print('COMPLETE', fold, family, phase, evidence['steps'], flush=True)
    watchdog.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--phase', choices=['select', 'refit'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare:
            declare()
        else:
            assert args.fold and args.family and args.phase
            run(args.fold, args.family, args.phase)
