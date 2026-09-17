"""Prepared-only until scheduler grants GPU: batch timing then same-target replay."""
import torch
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import psutil
import tabdpt.model
import tabdpt.utils
import tabdpt.estimator
import tabdpt.regressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_pilot as core
import run_residual as residual
import adapter

OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/models_retrieval/tabdpt_batch_v3')
CANARY = OUT / 'batch_canary_cuda_s20260916'
V2 = core.ROOT / 'private_runs/breakthrough_20260916/models_retrieval/tabdpt_v2'
ATOL, RTOL = 1e-3, 1e-6


def source_paths():
    return {'v3_run.py': Path(__file__), 'v3_adapter.py': HERE / 'adapter.py',
        'v3_test_contracts.py': HERE / 'test_contracts.py',
        'frozen_v2_adapter.py': Path(adapter.v2.__file__), 'frozen_tabdpt_adapter.py': Path(adapter.v2.frozen.__file__),
        'frozen_run_pilot.py': Path(core.__file__), 'frozen_run_residual.py': Path(residual.__file__),
        'frozen_preprocessing.py': HERE.parent / 'preprocessing.py', 'prior_common.py': Path(core.common.__file__),
        'official_tabdpt_model.py': Path(tabdpt.model.__file__), 'official_tabdpt_utils.py': Path(tabdpt.utils.__file__),
        'official_tabdpt_estimator.py': Path(tabdpt.estimator.__file__), 'official_tabdpt_regressor.py': Path(tabdpt.regressor.__file__)}


def hashes():
    return {name: core.sha256(path) for name, path in source_paths().items()}


def array_hash(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256(str(x.shape).encode() + x.dtype.str.encode())
    h.update(x.tobytes())
    return h.hexdigest()


def frame_hash(frame):
    return array_hash(pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy(np.uint64))


def guard():
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('At least8GiB available hostreserve required')


def declare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint, receipt = adapter.v2.frozen.checked_checkpoint()
    payload = {'source_hashes': hashes(), 'seed': args.seed, 'threads': args.threads, 'device': 'cuda',
        'batches': [16, 64, 128], 'query_rows': 2048, 'warmup_rows': 128, 'timed_repetitions': 2,
        'reference': 'F1 originalpurged finiteNM refit32000 sample with frozeneligible_rows seed20260916, exactlymatchingv2failedattempt',
        'query': 'First2048originalF1eligible scorequeries, observationfeaturesonly forcanary; noquerylabels used for timing/equivalence/selection',
        'target': 'Original rawY-NMproxy referencevalues exactlyunchanged; no label clipping/representation change',
        'invariants': {'reference_limit': 32000, 'context': 256, 'ensembles': 1, 'checkpoint_sha256': receipt['sha256'],
            'checkpoint_revision': receipt['revision'], 'dtype': 'v2float32', 'attention': 'officialnonflash+PyTorchmathSDPA', 'preprocessor': 'frozenfitreferenceonly'},
        'selection': 'Fastest median elapsed passingbatch with retrievalcontexts exactlyequal, finitepredictions allclose tobatch16, allocatedVRAM<=12GiB; noaccuracy-basedselection',
        'prediction_tolerance': {'atol_seconds': ATOL, 'rtol': RTOL}, 'max_score_seconds': args.max_score_seconds,
        'replay': 'Exact savedrefitfeature/label/ID hashes; selectedbatch joblibreload on2048; allbatchcontexts andpredictionhashes retained',
        'scope': 'Batchthroughputcanary is not completedcohortaccuracy; fullrunner preserves originalallrows andmissingV2',
        'gpu': 'No automaticlaunch onprepare;parentexclusiveGPUslot;12GiBallocatedcap/8GiBhostreserve',
        'failed_v2_preserved': str(V2 / 'tabdpt_aobt_allfinite_F1_s20260916/manifest.json')}
    path = OUT / 'protocol_cuda_s20260916.json'
    if path.exists() and core.read_json(path) != payload:
        raise ValueError('Frozenv3protocol differs')
    core.write_json(path, payload)
    snapshot = OUT / 'source'
    snapshot.mkdir(exist_ok=True)
    for name, source in source_paths().items():
        destination = snapshot / name
        if destination.exists() and core.sha256(destination) != core.sha256(source):
            raise ValueError('Frozenv3source snapshot differs')
        if not destination.exists():
            shutil.copyfile(source, destination)
    return path, payload


def capture(model, query, batch):
    original = model['estimator']._get_context_indices
    contexts = []
    def wrapped(x, context_size, seed=None):
        result = original(x, context_size=context_size, seed=seed)
        contexts.append(np.asarray(result, dtype=np.int64).copy())
        return result
    model['estimator']._get_context_indices = wrapped
    try:
        prediction = adapter.predict_batch(model, query, batch)
    finally:
        del model['estimator'].__dict__['_get_context_indices']
    return prediction, np.concatenate(contexts)


def canary(args, protocol, declaration):
    if CANARY.exists():
        raise ValueError('Preserveexistingv3canary attempt')
    CANARY.mkdir()
    record = {'status': 'running', 'source_hashes': hashes(), 'protocol_sha256': core.sha256(protocol), 'batches': [], 'private_reference_data': True}
    core.write_json(CANARY / 'manifest.json', record)
    started = time.monotonic()
    try:
        if psutil.virtual_memory().available < 12 * 1024**3:
            raise MemoryError('Need12GiB available before basefeature loading')
        x, meta = core.common.load_data()
        index, split, _ = core.common.fold_data(meta, 'F1', full=True)
        proxy = meta.proxy_sec.to_numpy(float)
        rows, eligible, _ = residual.eligible_rows(index, proxy, 32000, args.seed)
        prior = core.read_json(V2 / 'tabdpt_aobt_allfinite_F1_s20260916/manifest.json')
        assert prior['status'] == 'failed' and 'forecast' in prior['error']
        for old_name, source in [('v2_adapter.py', Path(adapter.v2.__file__)),
            ('frozen_tabdpt_adapter.py', Path(adapter.v2.frozen.__file__)),
            ('frozen_preprocessing.py', HERE.parent / 'preprocessing.py'),
            ('official_tabdpt_model.py', Path(tabdpt.model.__file__)),
            ('official_tabdpt_estimator.py', Path(tabdpt.estimator.__file__)),
            ('official_tabdpt_utils.py', Path(tabdpt.utils.__file__))]:
            assert prior['source_hashes'][old_name] == core.sha256(source)
        actual_ids = {stage: core.object_hash(meta.iloc[pos][core.ID].tolist()) for stage, pos in rows.items()}
        assert actual_ids == prior['sampled_stage_id_hashes']
        assert core.object_hash(split) == core.object_hash(prior['split']) and list(x) == prior['columns']
        bank = x.iloc[rows['refit']].copy()
        query = x.iloc[rows['score'][:2048]].copy()
        labels = meta.iloc[rows['refit']][core.TARGET].to_numpy(float) - proxy[rows['refit']]
        score_count = len(rows['score'])
        receipt = {'feature_columns': list(x), 'reference_rows': len(bank), 'query_rows': len(query),
            'stage_id_hashes': actual_ids, 'reference_ID_sha256': array_hash(bank.index.to_numpy()),
            'query_ID_sha256': array_hash(query.index.to_numpy()), 'reference_feature_sha256': frame_hash(bank),
            'query_feature_sha256': frame_hash(query), 'raw_residual_label_sha256': array_hash(labels),
            'original_v2_manifest_sha256': core.sha256(V2 / 'tabdpt_aobt_allfinite_F1_s20260916/manifest.json'),
            'raw_hashes': prior['raw_hashes'], 'finite_score_queries': score_count}
        del x, meta, proxy
        gc.collect()
        guard()
        model, evidence = adapter.fit(bank, labels, steps=1, seed=args.seed, threads=2, device='cuda')
        receipt['encoded_reference_sha256'] = array_hash(model['processor'].transform(bank))
        receipt['estimator_reference_label_sha256'] = array_hash(np.asarray(model['estimator'].y_train))
        np.testing.assert_array_equal(np.asarray(model['estimator'].y_train), labels)
        reference_prediction, reference_contexts = None, None
        for batch in [16, 64, 128]:
            guard()
            entry = {'batch_size': batch, 'status': 'running'}
            try:
                torch.cuda.empty_cache()
                adapter.predict_batch(model, query.iloc[:128], batch)
                samples, outputs, context_samples = [], [], []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    prediction, contexts = capture(model, query, batch)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - begin)
                    outputs.append(prediction)
                    context_samples.append(contexts)
                if reference_prediction is None:
                    reference_prediction, reference_contexts = outputs[0], context_samples[0]
                context_equal = all(np.array_equal(c, reference_contexts) for c in context_samples)
                same_replay = np.allclose(outputs[0], outputs[1], atol=ATOL, rtol=RTOL)
                batch_equal = all(np.allclose(p, reference_prediction, atol=ATOL, rtol=RTOL) for p in outputs)
                peak = torch.cuda.max_memory_allocated()
                passing = context_equal and same_replay and batch_equal and peak <= 12 * 1024**3
                entry.update(status='passed' if passing else 'equivalence_or_memory_failed', seconds=samples,
                    median_seconds=float(np.median(samples)), projected_F1_finite_score_sec=float(np.median(samples) * score_count / len(query)),
                    gpu_peak_allocated_bytes=peak, gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    gpu_name=torch.cuda.get_device_name(), gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    contexts_exact_equal=context_equal, replay_allclose=same_replay, batch16_allclose=batch_equal,
                    prediction_max_abs_delta=float(max(np.max(np.abs(p-reference_prediction)) for p in outputs)),
                    context_sha256=array_hash(context_samples[0]), prediction_sha256=array_hash(outputs[0]))
                pd.DataFrame({core.ID: query.index.to_numpy(), 'residual_prediction_sec': outputs[0]}).to_parquet(CANARY / f'predictions_batch{batch}.parquet', index=False)
            except torch.cuda.OutOfMemoryError as exc:
                entry.update(status='cuda_oom', error=repr(exc))
                torch.cuda.empty_cache()
            record['batches'].append(entry)
            core.write_json(CANARY / 'progress.json', record)
            print('BATCH', entry, flush=True)
        passing = [r for r in record['batches'] if r['status'] == 'passed']
        if not passing:
            raise RuntimeError('No passingequivalent querybatch')
        chosen = min(passing, key=lambda r: r['median_seconds'])
        model['query_batch_size'] = chosen['batch_size']
        joblib.dump(model, CANARY / 'model.joblib', compress=0)
        saved_prediction = adapter.predict(joblib.load(CANARY / 'model.joblib'), query)
        assert np.allclose(saved_prediction, reference_prediction, atol=ATOL, rtol=RTOL)
        assert hashes() == declaration['source_hashes']
        record.update(status='passed', selected_batch=chosen['batch_size'], reference=receipt, fit=evidence,
            runtime_sec=time.monotonic()-started, saved_replay_max_abs_delta=float(np.max(np.abs(saved_prediction-reference_prediction))),
            source_hashes=hashes(), full_score_budget_seconds=args.max_score_seconds,
            projected_finite_score_seconds=chosen['projected_F1_finite_score_sec'], full_score_within_declared_budget=chosen['projected_F1_finite_score_sec'] <= args.max_score_seconds,
            outputs={p.name: core.sha256(p) for p in CANARY.iterdir() if p.is_file() and p.name not in ('manifest.json', 'progress.json')})
        core.write_json(CANARY / 'manifest.json', record)
        print('BATCH_CANARY_PASSED', chosen['batch_size'], flush=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), runtime_sec=time.monotonic()-started)
        core.write_json(CANARY / 'manifest.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--max-score-seconds', type=float, default=1200)
    args = parser.parse_args()
    if args.threads != 2 or args.seed != 20260916 or args.max_score_seconds != 1200:
        raise ValueError('V3fixesthreads2seed20260916maxscore1200')
    protocol, declaration = declare(args)
    if args.prepare_only:
        assert not torch.cuda.is_initialized()
        print('PREPARED', protocol, 'CUDAuninitialized/no privaterows', flush=True)
        return
    if args.canary:
        canary(args, protocol, declaration)
        return
    receipt = core.read_json(CANARY / 'manifest.json')
    assert receipt['status'] == 'passed' and receipt['source_hashes'] == hashes()
    assert receipt['full_score_within_declared_budget']
    for filename, digest in receipt['outputs'].items():
        assert core.sha256(CANARY / filename) == digest
    adapter.BATCH = receipt['selected_batch']
    core.OUT = OUT
    core.module = lambda family: adapter if family == 'tabdpt' else (_ for _ in ()).throw(ValueError('V3TabDPTonly'))
    core.hashes = hashes
    residual.hashes = hashes
    residual.source_paths = source_paths
    residual.verify_adapter_canary = lambda _: {'path': str(CANARY / 'manifest.json'), 'sha256': core.sha256(CANARY / 'manifest.json'),
        'coverage': 'Same32KF1refitcontexts,2048realquerytiming,batch16/64/128 equivalence andsavedreplay; selectedquerybatchonly change'}
    sys.argv = [str(Path(residual.__file__)), '--family', 'tabdpt', '--device', 'cuda', '--threads', '2',
        '--seed', str(args.seed), '--folds', *args.folds, '--max-score-seconds', str(args.max_score_seconds)]
    residual.main()


if __name__ == '__main__':
    main()
