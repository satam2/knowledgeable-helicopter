"""Tiny CPU synthetic retrieval and saved2048-row prediction diagnostics only."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import faiss
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
BASE = ROOT / 'private_runs/breakthrough_20260916/models_retrieval/tabdpt_batch_v3_1/batch_canary_cuda_s20260916'
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/retrospective_research/faiss_batch_diagnostic')
faiss.omp_set_num_threads(1)
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def search(index, query, batch):
    distances, indices = [], []
    for start in range(0, len(query), batch):
        d, i = index.search(query[start:start + batch], 256)
        distances.append(d)
        indices.append(i)
    return np.concatenate(distances), np.concatenate(indices)


def difference(first, second):
    d1, i1 = first
    d2, i2 = second
    ordered = np.any(i1 != i2, axis=1)
    sets = np.any(np.sort(i1, axis=1) != np.sort(i2, axis=1), axis=1)
    return {'ordered_context_rows_differ': int(ordered.sum()), 'neighbor_set_rows_differ': int(sets.sum()),
            'index_cells_differ': int((i1 != i2).sum()),
            'same_order': bool(np.array_equal(i1, i2)), 'same_sets': bool(not sets.any()),
            'rankwise_distance_max_delta': float(np.max(np.abs(d1 - d2)))}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Preserve8GiB host reserve')
    original = faiss.cvar.distance_compute_blas_threshold
    synthetic = {}
    try:
        rng = np.random.default_rng(20260916)
        for scale in [1., 0.001]:
            bank = np.asarray(1. + scale * rng.normal(size=(4096, 32)), dtype=np.float32)
            query = np.asarray(1. + scale * rng.normal(size=(128, 32)), dtype=np.float32)
            index = faiss.IndexFlatL2(32)
            index.add(bank)
            default = {batch: search(index, query, batch) for batch in [16, 64, 128]}
            faiss.cvar.distance_compute_blas_threshold = 1000000
            sequential = {batch: search(index, query, batch) for batch in [16, 64, 128]}
            faiss.cvar.distance_compute_blas_threshold = original
            synthetic[str(scale)] = {
                'default16_vs64': difference(default[16], default[64]),
                'default64_vs128': difference(default[64], default[128]),
                'forced_sequential16_vs64': difference(sequential[16], sequential[64]),
                'forced_sequential16_vs128': difference(sequential[16], sequential[128]),
                'default16_vs_sequential16': difference(default[16], sequential[16]),
                'default64_vs_sequential64': difference(default[64], sequential[64]),
                'description': 'Syntheticnormalfeatures around1; secondcase deliberatelynear-tied, not privateencodingdistribution.'}
    finally:
        faiss.cvar.distance_compute_blas_threshold = original
    marker = json.loads((BASE / 'manifest.json').read_text())
    predictions = {}
    receipts = {}
    ids = None
    for batch in [16, 64, 128]:
        path = BASE / f'predictions_batch{batch}.parquet'
        assert digest(path) == marker['outputs'][path.name]
        table = pq.read_table(path, use_threads=False).to_pandas()
        assert len(table) == 2048 and len(table.columns) == 2
        current = table.iloc[:, 0].to_numpy()
        if ids is None:
            ids = current
        np.testing.assert_array_equal(ids, current)
        predictions[batch] = table.residual_prediction_sec.to_numpy(float)
        receipts[str(batch)] = digest(path)
    comparisons = {}
    for left, right in [(16, 64), (16, 128), (64, 128)]:
        delta = np.abs(predictions[right] - predictions[left])
        comparisons[f'{left}_vs_{right}'] = {'max_abs_delta_sec': float(delta.max()),
            'nonzero_rows': int((delta > 0).sum()),
            'rows_outside_atol001_rtol1e6': int((~np.isclose(predictions[right], predictions[left], atol=.001, rtol=1e-6)).sum()),
            'median_abs_delta_sec': float(np.median(delta))}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(__file__),
              'canary_manifest_sha256': digest(BASE / 'manifest.json'), 'faiss_version': faiss.__version__,
              'faiss_compile_options': faiss.get_compile_options(), 'default_blas_threshold': original,
              'synthetic': synthetic, 'saved2048_prediction_comparisons': comparisons,
              'prediction_receipts': receipts, 'private_context_arrays_saved': False,
              'private_model_loaded': False, 'gpu_used': False, 'network_used': False,
              'limits': 'Hashonlycontexts cannot distinguish reorder, cutoffreplacement or ties. Syntheticpathreproduction shows mechanism exists, not exactprivatecause.',
              'runtime_seconds': time.monotonic() - started, 'rss_bytes': psutil.Process().memory_info().rss}
    assert result['rss_bytes'] < 2 * 1024**3
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
