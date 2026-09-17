"""Streaming deterministic own-minus-neighbor clock representation, no labels."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_clock_innovations/cache'
FLAT = ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten'
BASE = ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
ID = 'MVT_ID_mvt'
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
OWN = ['takeoff_minus_' + name for name in CLOCKS]
PEER = [f'flat_dep{rank}_offset_{name}' for rank in range(1, 5) for name in CLOCKS]
PAIRS = [(f'innovation_own_{clock}_minus_dep{rank}_{clock}_sec', f'takeoff_minus_{clock}', f'flat_dep{rank}_offset_{clock}')
         for rank in range(1, 5) for clock in CLOCKS]
PAIRS += [(f'innovation_own_AOBT_3_flt_minus_dep{rank}_{clock}_sec', OWN[0], f'flat_dep{rank}_offset_{clock}')
          for rank in range(1, 5) for clock in CLOCKS[1:]]
FEATURES = [row[0] for row in PAIRS]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def clean(values):
    result = np.asarray(values, dtype=np.float64).copy()
    result[(result == -999999) | ~np.isfinite(result)] = np.nan
    return result


def transform(own, peers):
    if len(own) != len(peers):
        raise ValueError('Own and peer rows must align')
    output = {}
    for name, left, right in PAIRS:
        a, b = clean(own[left]), clean(peers[right])
        value = (a - b).astype('float32')
        assert np.array_equal(np.isfinite(value), np.isfinite(a) & np.isfinite(b))
        output[name] = value
    return pd.DataFrame(output)


def guard():
    info = psutil.Process().memory_info()
    peak = int(getattr(info, 'peak_wset', info.rss))
    if peak > 1024**3:
        raise MemoryError('Clock innovations cache builder exceeds1GiB peakRSS')
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Host reserve below8GiB')
    return peak


def main():
    began = time.monotonic()
    assert len(PAIRS) == len(set(FEATURES)) == len(set((a, b) for _, a, b in PAIRS)) == 36
    if OUT.exists():
        raise ValueError('Preserve existing cache attempt')
    OUT.mkdir(parents=True)
    flat_manifest = read_json(FLAT / 'manifest.json')
    flat_check = read_json(FLAT / 'verification.json')
    assert flat_manifest['status'] == 'complete' and flat_check['status'] == 'passed'
    assert flat_check['manifest_sha256'] == sha(FLAT / 'manifest.json')
    assert sha(FLAT / 'training_features.parquet') == flat_manifest['outputs']['training_features.parquet']
    controls = ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten/models'
    marker = read_json(controls / 'lightgbm_aobt_allfinite_F1_s20260916/manifest.json')
    assert marker['status'] == 'complete' and len(marker['feature_columns']) == 337
    assert not set(FEATURES).intersection(marker['feature_columns'])
    assert set(OWN + PEER).issubset(marker['feature_columns'])
    pieces, receipts = [], []
    for path in sorted(BASE.glob('training_*.parquet')):
        assert sha(path) == read_json(path.with_suffix('.json'))['sha256']
        pieces.append(pq.read_table(path, columns=[ID, *OWN], use_threads=False).to_pandas())
        receipts.append({'path': str(path), 'sha256': sha(path)})
        guard()
    assert len(pieces) == 12
    own = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    assert own[ID].is_unique and len(own) == flat_manifest['rows']
    protocol = {'source_sha256': sha(__file__), 'flat_manifest_sha256': sha(FLAT / 'manifest.json'),
        'flat_verification_sha256': sha(FLAT / 'verification.json'), 'own_feature_sources': receipts,
        'features': FEATURES, 'pairs': PAIRS, 'rows': len(own), 'columns': 36,
        'read_columns': {'own': [ID, *OWN], 'peers': [ID, *PEER]}, 'label_reads': False,
        'computation': '20sameclockownminuspeer offsets +16ownNMminusotherpeerclock offsets;float64subtractionstoredfloat32; no age transforms orclipping.',
        'missing': 'Nonfinite and existingFrameEncoder sentinel-999999 inputs becomeNaN; outputfinite onlywhenbothinputsfinite.',
        'information': 'Deterministicrepresentationofexisting337inputs; no newinformation. Originalstrictpast neighborselection unchanged.',
        'availability': 'SuppliedfinalNM retrospective; no causalpublication-timeclaim.',
        'selection': 'Adaptive exposeddevelopment hypothesisafterclockablation andtreeusage; notincludedinfinal9.',
        'resources': '1CPUthread,4096row outputbatches,1GiBpeakguard; trainingonly, no rankingcache ormodel fit.'}
    write_json(OUT / 'protocol.json', protocol)
    source = pq.ParquetFile(FLAT / 'training_features.parquet')
    writer = None
    cursor = 0
    checks = []
    try:
        for batch in source.iter_batches(batch_size=4096, columns=[ID, *PEER], use_threads=False):
            peers = batch.to_pandas()
            stop = cursor + len(peers)
            local = own.iloc[cursor:stop].reset_index(drop=True)
            np.testing.assert_array_equal(local[ID], peers[ID])
            frame = transform(local, peers)
            for name, left, right in PAIRS:
                a = pd.to_numeric(local[left], errors='coerce').replace([np.inf, -np.inf, -999999], np.nan)
                b = pd.to_numeric(peers[right], errors='coerce').replace([np.inf, -np.inf, -999999], np.nan)
                expected = (a.astype('float64') - b.astype('float64')).astype('float32').to_numpy()
                np.testing.assert_array_equal(frame[name].to_numpy(), expected)
            frame.insert(0, ID, peers[ID].to_numpy())
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT / 'training_features.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            cursor = stop
            guard()
            if len(checks) < 24:
                checks.append({'row_offset': cursor - len(peers), 'rows': len(peers), 'max_abs_value': float(np.nanmax(np.abs(frame[FEATURES].to_numpy())))})
        assert cursor == len(own)
    finally:
        if writer is not None:
            writer.close()
    output = OUT / 'training_features.parquet'
    report = {'status': 'complete', 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol.json'),
        'features': FEATURES, 'rows': cursor, 'outputs': {output.name: sha(output)},
        'flat_manifest_sha256': protocol['flat_manifest_sha256'], 'runtime_sec': time.monotonic() - began,
        'peak_rss_bytes': guard(), 'all_ID_order_verified': True, 'all_cells_direct_formula_verified': cursor * len(FEATURES),
        'sampled_batch_summaries': checks, 'prediction_gain_tested': False}
    write_json(OUT / 'manifest.json', report)
    write_json(OUT / 'verification.json', {'status': 'passed', 'manifest_sha256': sha(OUT / 'manifest.json'),
        'source_sha256': sha(__file__), 'rows': cursor, 'columns': len(FEATURES), 'formula_cells_checked': cursor * len(FEATURES),
        'limitation': 'Separate directpandasformula vsnumpy kernel onallstoredinputs; not a newrawclock reconstruction orindependentneighborselectionaudit.'})
    print('COMPLETE_INNOVATIONS', cursor, len(FEATURES), 'peak', guard(), 'seconds', report['runtime_sec'], flush=True)


if __name__ == '__main__':
    main()
