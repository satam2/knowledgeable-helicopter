"""Exact F1 finite ID coverage and pre-loader numeric conversion audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'private_runs/tail240_20260916/validation/union387_source_audit_v1'
REFERENCE = ROOT / 'private_runs/tail240_20260916/models/linear_finite_tune_v1/F1'
ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
SENTINEL = -999999.


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def guard():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, 'peak_wset', memory.rss))
    assert peak < 4 * 1024**3, '4GiB peak process ceiling'
    assert psutil.virtual_memory().available >= 8 * 1024**3, '8GiB host reserve'
    return peak


def mark_once(coverage, positions):
    if len(np.unique(positions)) != len(positions) or coverage[positions].any():
        raise ValueError('Requested ID repeats within or across source files')
    coverage[positions] = True


def numeric_stats(values, fill):
    before = pd.to_numeric(values).to_numpy(dtype=np.float64, na_value=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        after = before.astype(np.float32)
    finite = np.isfinite(before)
    valid_after = np.isfinite(after)
    both = finite & valid_after
    errors = np.abs(after[both].astype(np.float64) - before[both])
    source_sentinel = finite & (before == SENTINEL)
    result = dict(rows=len(before), source_nan=int(np.isnan(before).sum()),
                  source_posinf=int(np.isposinf(before).sum()), source_neginf=int(np.isneginf(before).sum()),
                  source_finite=int(finite.sum()), source_finite_sentinel=int(source_sentinel.sum()),
                  finite_to_nonfinite_float32=int((finite & ~valid_after).sum()),
                  nonsentinel_finite_to_sentinel_float32=int((finite & ~source_sentinel & (after == SENTINEL)).sum()),
                  finite_changed_on_float32=int((errors != 0).sum()),
                  integer_changed_on_float32=int((both & (before == np.trunc(before)) & (before != after)).sum()),
                  new_loader_sentinel=int((~valid_after).sum()) if fill else 0,
                  final_loader_nan=0 if fill else int((~valid_after).sum()),
                  max_abs_float32_error=float(errors.max()) if len(errors) else 0.)
    return result


def merge_stats(total, partial):
    for key, value in partial.items():
        total[key] = max(total.get(key, 0), value) if key.startswith('max_') else total.get(key, 0) + value


def main():
    started = time.monotonic()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    guard()
    assert not OUT.exists(), 'Preserve prior runs'
    record = read(REFERENCE / 'constant/manifest.json')
    encoder = read(REFERENCE / 'encoder.json')
    protocol_path = REFERENCE.parent / 'protocol.json'
    assert record['protocol_sha256'] == sha(protocol_path)
    assert record['encoder_sha256'] == sha(REFERENCE / 'encoder.json')
    other = ROOT / 'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387/manifest.json'
    assert read(other)['feature_receipts'] == record['feature_receipts']
    columns = encoder['columns']
    categories = set(encoder['vocab'])
    assert len(columns) == len(set(columns)) == 387 and len(categories) == 15
    numeric = set(columns) - categories
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(meta_path) == read(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    # Original F1 purges are zero. Exact saved post-purge hashes enforce this selection.
    assert not any(record['split']['purged_related_departures'].values())
    chunks = {stage: [] for stage in ('fit', 'tune')}
    for batch in pq.ParquetFile(meta_path).iter_batches(batch_size=8192, columns=[ID, TIME, 'proxy_sec'], use_threads=False):
        data = batch.to_pandas()
        for stage in chunks:
            start, stop = [pd.Timestamp(value, tz='UTC') for value in record['split']['spec'][stage]]
            use = data[TIME].ge(start) & data[TIME].lt(stop) & np.isfinite(data.proxy_sec)
            chunks[stage].append(data.loc[use, ID])
    selected = {stage: pd.concat(parts, ignore_index=True) for stage, parts in chunks.items()}
    cohorts = {stage: dict(n=len(values), hash=object_hash(values.tolist())) for stage, values in selected.items()}
    assert cohorts == record['ids']
    nfit = cohorts['fit']['n']
    ids = pd.Index(pd.concat(list(selected.values()), ignore_index=True))
    assert ids.is_unique and len(ids) == 996700
    del chunks, selected
    OUT.mkdir(parents=True)
    frozen = dict(source_sha256=sha(Path(__file__)), source_test_sha256=sha(Path(__file__).with_name('test_union387_sources.py')),
                  reference_sha256=sha(REFERENCE / 'constant/manifest.json'), encoder_sha256=record['encoder_sha256'],
                  metadata_sha256=sha(meta_path), independent_control_sha256=sha(other), cohorts=cohorts,
                  columns=columns, categories=sorted(categories), sources=record['feature_receipts'],
                  scope='All original F1 finite fit/tune IDs and all union387 sources. No labels, ranking or score predictions.',
                  boundary='Stored native cache values before loader float32; does not recover precision already lost upstream.',
                  resources='1CPU,8192rowbatches,peak<4GiB,hostreserve>=8GiB,noGPU,no model.')
    write(OUT / 'protocol.json', frozen)
    coverage = np.zeros((len(columns), len(ids)), dtype=bool)
    column_index = {name: index for index, name in enumerate(columns)}
    stats = {name: {stage: {} for stage in ('fit', 'tune')} for name in numeric}
    per_source = []
    origins = {name: [] for name in numeric}
    for source in record['feature_receipts']:
        path = Path(source['path'])
        assert sha(path) == source['sha256'], path
        names = source['columns']
        is_base = 'screening_230/data/interim/features' in path.as_posix()
        is_sequence = 'sequence_flatten' in path.as_posix()
        fill = not is_base and not is_sequence
        file = pq.ParquetFile(path)
        dtypes = {name: str(file.schema_arrow.field(name).type) for name in names}
        for name in names:
            dtype = file.schema_arrow.field(name).type
            if pa.types.is_dictionary(dtype):
                dtype = dtype.value_type
            is_numeric = pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)
            assert is_numeric == (name in numeric), name
        rows = Counter(fit=0, tune=0)
        for batch in file.iter_batches(batch_size=8192, columns=[ID, *names], use_threads=False):
            data = batch.to_pandas()
            positions = ids.get_indexer(data[ID])
            keep = positions >= 0
            where = positions[keep]
            data = data.loc[keep]
            for name in names:
                mark_once(coverage[column_index[name]], where)
            for stage, use in [('fit', where < nfit), ('tune', where >= nfit)]:
                rows[stage] += int(use.sum())
                if not use.any():
                    continue
                part = data.iloc[np.flatnonzero(use)]
                for name in names:
                    if name in numeric:
                        merge_stats(stats[name][stage], numeric_stats(part[name], fill))
            guard()
        assert sum(rows.values()) == source['rows']
        origin = 'base_upstream_missing_sentinel_contract' if is_base else ('sequence_native_nan_contract' if is_sequence else 'extension_loader_fill_enabled')
        for name in names:
            if name in numeric:
                origins[name].append(dict(path=str(path.relative_to(ROOT)), dtype=dtypes[name], loader_fill=fill, origin=origin))
        per_source.append(dict(path=str(path.relative_to(ROOT)), sha256=source['sha256'], columns=names,
                               rows=dict(rows), native_dtypes=dtypes, loader_fill=fill, origin=origin))
        print('SOURCE', path.parent.name, path.name, dict(rows), 'peak', guard(), flush=True)
    assert coverage.all(), 'At least one requested ID is absent for a feature'
    for name in numeric:
        for stage in ('fit', 'tune'):
            assert stats[name][stage]['rows'] == cohorts[stage]['n']
    aggregate = {stage: {} for stage in ('fit', 'tune')}
    for values in stats.values():
        for stage in aggregate:
            merge_stats(aggregate[stage], values[stage])
    sentinels = {name: values for name, values in stats.items()
                 if any(part['source_finite_sentinel'] or part['nonsentinel_finite_to_sentinel_float32'] for part in values.values())}
    write(OUT / 'numeric_columns.json', dict(stats=stats, origins=origins))
    write(OUT / 'source_coverage.json', per_source)
    result = dict(status='passed', source_sha256=frozen['source_sha256'], protocol_sha256=sha(OUT / 'protocol.json'),
                  cohorts=cohorts, source_count=len(per_source), category_count=len(categories), numeric_count=len(numeric),
                  all387_columns_exact_once_per_requested_id=True, selected_id_feature_pairs=int(coverage.size),
                  stats=aggregate, preexisting_or_collision_sentinel_columns=sentinels,
                  outputs={name: sha(OUT / name) for name in ('protocol.json', 'numeric_columns.json', 'source_coverage.json')},
                  peak_bytes=guard(), elapsed_seconds=time.monotonic()-started, no_labels_models_or_gpu=True,
                  limitation='Finite source -999999 is reported separately from new loader fills. A finite source sentinel alone cannot prove missing versus legitimate measurement; raw provenance is required. Already-float32 upstream casts are outside this boundary.')
    write(OUT / 'receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
