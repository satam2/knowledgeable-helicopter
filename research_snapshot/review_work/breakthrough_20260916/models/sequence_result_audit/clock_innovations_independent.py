"""Raw own-clock reconstruction versus verified peer offsets and saved innovations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import gc
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import prepare_fit_canary as fixture

ROOT = fixture.ROOT
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
BASE = ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
CACHE = ROOT / 'private_runs/breakthrough_20260916/models/sequence_clock_innovations/cache'
FLAT = ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten'
OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit'
ID, MOVE, PHASE = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt'
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
OWN = ['takeoff_minus_' + clock for clock in CLOCKS]
PEER = [f'flat_dep{rank}_offset_{clock}' for rank in range(1, 5) for clock in CLOCKS]
FEATURES = [f'innovation_own_{clock}_minus_dep{rank}_{clock}_sec' for rank in range(1, 5) for clock in CLOCKS]
FEATURES += [f'innovation_own_AOBT_3_flt_minus_dep{rank}_{clock}_sec' for rank in range(1, 5) for clock in CLOCKS[1:]]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if peak > 1024**3:
        raise MemoryError('Independent clock innovations oracle exceeds1GiB')
    return int(peak)


def main():
    started = time.monotonic()
    manifest = fixture.read_json(CACHE / 'manifest.json')
    protocol = fixture.read_json(CACHE / 'protocol.json')
    verification = fixture.read_json(CACHE / 'verification.json')
    flat = fixture.read_json(FLAT / 'manifest.json')
    frozen = fixture.read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == fixture.sha256(CACHE / 'manifest.json')
    assert manifest['protocol_sha256'] == fixture.sha256(CACHE / 'protocol.json')
    assert manifest['source_sha256'] == fixture.sha256(ROOT / 'review_work/breakthrough_20260916/models/sequence_clock_innovations/build.py')
    assert manifest['features'] == FEATURES and len(FEATURES) == len(set(FEATURES)) == 36
    assert manifest['flat_manifest_sha256'] == fixture.sha256(FLAT / 'manifest.json')
    assert fixture.sha256(FLAT / 'training_features.parquet') == flat['outputs']['training_features.parquet']
    assert fixture.sha256(CACHE / 'training_features.parquet') == manifest['outputs']['training_features.parquet']
    records = []
    missing_cells = 0
    for path in sorted(RAW.glob('training_*.parquet')):
        assert fixture.sha256(path) == frozen['raw_hashes'][path.name]
        raw = pq.read_table(path, columns=[ID, MOVE, *CLOCKS], filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas().set_index(ID)
        movement = pd.to_datetime(raw[MOVE], utc=True)
        direct = pd.DataFrame({own: (movement - pd.to_datetime(raw[clock], utc=True)).dt.total_seconds().astype('float32') for own, clock in zip(OWN, CLOCKS)}, index=raw.index)
        selected = set(np.linspace(0, len(raw) - 1, 24).astype(int).tolist())
        for own in OWN:
            value = direct[own].to_numpy()
            absent = np.flatnonzero(~np.isfinite(value))
            selected.update(absent[:3].tolist())
            finite = np.flatnonzero(np.isfinite(value))
            if len(finite):
                selected.add(int(finite[np.argmax(value[finite])]))
                selected.add(int(finite[np.argmin(value[finite])]))
        positions = np.array(sorted(selected))
        ids = raw.index[positions]
        expected_own = direct.loc[ids].replace([np.inf, -np.inf, -999999], np.nan)
        base_path = BASE / path.name
        assert fixture.sha256(base_path) == fixture.read_json(base_path.with_suffix('.json'))['sha256']
        stored_own = fixture.selected_frame(base_path, OWN, ids).replace([np.inf, -np.inf, -999999], np.nan)
        np.testing.assert_array_equal(stored_own.to_numpy(), expected_own.to_numpy())
        peer = fixture.selected_frame(FLAT / 'training_features.parquet', PEER, ids).replace([np.inf, -np.inf, -999999], np.nan)
        saved = fixture.selected_frame(CACHE / 'training_features.parquet', FEATURES, ids)
        expected = {}
        for rank in range(1, 5):
            for clock in CLOCKS:
                name = f'innovation_own_{clock}_minus_dep{rank}_{clock}_sec'
                expected[name] = (expected_own['takeoff_minus_' + clock].astype('float64') - peer[f'flat_dep{rank}_offset_{clock}'].astype('float64')).astype('float32')
        for rank in range(1, 5):
            for clock in CLOCKS[1:]:
                name = f'innovation_own_AOBT_3_flt_minus_dep{rank}_{clock}_sec'
                expected[name] = (expected_own[OWN[0]].astype('float64') - peer[f'flat_dep{rank}_offset_{clock}'].astype('float64')).astype('float32')
        expected = pd.DataFrame(expected, index=ids)[FEATURES]
        assert list(expected) == list(saved)
        np.testing.assert_array_equal(saved.to_numpy(), expected.to_numpy())
        missing_cells += int(expected.isna().to_numpy().sum())
        record = {'raw_file': path.name, 'raw_sha256': frozen['raw_hashes'][path.name], 'selected_rows': len(ids),
            'feature_cells': len(ids) * 36, 'missing_cells': int(expected.isna().to_numpy().sum()),
            'max_absolute_difference_sec': 0., 'ID_hash': fixture.object_hash(ids.tolist()),
            'selection': '24evenlyspreadrows plusfirst3missing andmin/maxoffsetrows foreachownclock; no labels used'}
        records.append(record)
        print('INDEPENDENT_INNOVATIONS', record, flush=True)
        guard()
        del raw, movement, direct, expected_own, stored_own, peer, saved, expected
        gc.collect()
    result = {'status': 'passed', 'source_sha256': fixture.sha256(__file__),
        'manifest_sha256': fixture.sha256(CACHE / 'manifest.json'), 'flat_manifest_sha256': fixture.sha256(FLAT / 'manifest.json'),
        'records': records, 'total_selected_rows': sum(r['selected_rows'] for r in records),
        'total_cells': sum(r['feature_cells'] for r in records), 'missing_cells_checked': missing_cells,
        'peak_rss_bytes': guard(), 'runtime_sec': time.monotonic() - started,
        'raw_label_or_departure_block_read': False, 'gpu_used': False,
        'own_clock_reconstruction': 'Rawtimestampdifferences roundedtoexistingfloat32 representation; comparedexactlytostoredownfeatures, thenfloat64subtraction andfloat32innovation.',
        'peer_scope': 'Verifiedunchangedpeerfeaturecacheused; independentneighbor/rawpeeravailabilityaudit remainspriorverifiedreceipt.',
        'scope': 'All12monthsselectedrawqueries, notallrowrawreconstruction. Fullcache bytehashes verified.'}
    fixture.write_json(OUT / 'clock_innovations_independent.json', result)
    print('INDEPENDENT_INNOVATIONS_COMPLETE', result['total_selected_rows'], result['total_cells'], result['missing_cells_checked'], result['peak_rss_bytes'], flush=True)


if __name__ == '__main__':
    main()
