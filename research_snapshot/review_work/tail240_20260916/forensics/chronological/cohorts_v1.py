"""Label-free auxiliary cohorts nested in the unchanged original F1/F3 folds."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.config import load_config
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, utc
from taxiout.splits import make_fold

BOUNDARIES = {'F1': '2025-05-01', 'F3': '2025-09-01'}
INPUT_COLUMNS = [ID, FLIGHT_ID, MOVEMENT]
DEST = ROOT / 'private_runs/tail240_20260916/forensics/chronological_cohorts/v1'


def build(meta, fold):
    if list(meta.columns) != INPUT_COLUMNS:
        raise ValueError('Cohort builder requires exactly identifier/time columns in declared order')
    if fold not in BOUNDARIES:
        raise ValueError('Only F1/F3 auxiliary cohorts are declared')
    spec = load_config('configs/folds.yaml')[fold]
    original, original_receipt = make_fold(meta, spec)
    times = utc(meta[MOVEMENT], required=True)
    fit = original['fit']
    before = times.iloc[fit].lt(pd.Timestamp(BOUNDARIES[fold], tz='UTC')).to_numpy()
    base, calibration = fit[before], fit[~before]
    related = meta.iloc[calibration][FLIGHT_ID].dropna().unique()
    remove = meta.iloc[base][FLIGHT_ID].notna() & meta.iloc[base][FLIGHT_ID].isin(related)
    purged = int(remove.sum())
    base = base[~remove.to_numpy()]
    positions = dict(base=base, calibration=calibration, evaluation=original['tune'],
                     protected_score=original['score'])
    if any(not len(rows) for rows in positions.values()):
        raise ValueError('Auxiliary cohort contains an empty stage')
    for earlier, later in [('base', 'calibration'), ('base', 'evaluation'),
                           ('calibration', 'evaluation'), ('base', 'protected_score'),
                           ('calibration', 'protected_score'), ('evaluation', 'protected_score')]:
        a, b = positions[earlier], positions[later]
        if not times.iloc[a].max() < times.iloc[b].min():
            raise AssertionError('Non-forward auxiliary chronology')
        if set(meta.iloc[a][FLIGHT_ID].dropna()) & set(meta.iloc[b][FLIGHT_ID].dropna()):
            raise AssertionError('Related flight crosses auxiliary stages')
    receipt = dict(fold=fold, original=original_receipt,
                   calibration_start_utc=BOUNDARIES[fold],
                   additional_base_against_calibration=purged,
                   stages={name: dict(n=len(rows), id_hash=object_hash(meta.iloc[rows][ID].tolist()))
                           for name, rows in positions.items()})
    assert receipt['stages']['evaluation'] == original_receipt['stages']['tune']
    assert receipt['stages']['protected_score'] == original_receipt['stages']['score']
    receipt['cohort_hash'] = object_hash(receipt)
    return positions, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    if not args.materialize:
        parser.error('Explicit --materialize required; no model training is implemented')
    if DEST.exists():
        raise FileExistsError('Preserve existing cohort evidence; create a new source/version for changes')
    started = time.monotonic()
    if psutil.virtual_memory().available < 10 * 1024**3:
        raise MemoryError('Requires 2GiB process planning budget and 8GiB host reserve')
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit_path = ROOT / 'private_runs/screening_230/reports/data_audit.json'
    source_hash = sha256(path)
    assert source_hash == read_json(audit_path)['artifacts'][path.name]
    meta = pd.read_parquet(path, columns=INPUT_COLUMNS)
    binding_path = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
    binding = read_json(binding_path)
    built = {fold: build(meta, fold) for fold in BOUNDARIES}
    for fold, (_, receipt) in built.items():
        expected = binding['folds'][fold]['cohorts']['all']
        for stage in ['fit', 'tune', 'refit', 'score']:
            actual = receipt['original']['stages'][stage]
            assert dict(n=actual['n'], hash=actual['id_hash']) == expected[stage]
    process = psutil.Process().memory_info()
    peak = getattr(process, 'peak_wset', process.rss)
    assert peak < 2 * 1024**3 and psutil.virtual_memory().available >= 8 * 1024**3
    DEST.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for fold, (positions, receipt) in built.items():
        for stage, rows in positions.items():
            output = DEST / f'{fold}_{stage}_ids.parquet'
            meta.iloc[rows][[ID]].to_parquet(output, index=False)
            outputs[output.name] = sha256(output)
        write_json(DEST / f'{fold}_cohorts.json', receipt)
        outputs[f'{fold}_cohorts.json'] = sha256(DEST / f'{fold}_cohorts.json')
    sources = [Path(__file__), Path(__file__).with_name('test_cohorts_v1.py'),
               ROOT / 'knowledgeable-helicopter-screening/src/taxiout/splits.py',
               ROOT / 'knowledgeable-helicopter-screening/configs/folds.yaml']
    record = dict(status='cohorts_materialized_no_fits', created_utc=utc_now(),
                  source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                  metadata_sha256=source_hash, baseline_binding_sha256=sha256(binding_path),
                  input_columns=INPUT_COLUMNS, label_columns_read=[],
                  preprocessing='None; finite/ordinary proxy filtering belongs to a later declared stage',
                  folds={f: r for f, (_, r) in built.items()}, outputs=outputs,
                  peak_wset_bytes=peak, runtime_sec=time.monotonic()-started)
    write_json(DEST / 'manifest.json', record)
    print({f: r['stages'] for f, (_, r) in built.items()}, flush=True)
    print(dict(manifest_sha256=sha256(DEST / 'manifest.json'), peak_wset_bytes=peak,
               runtime_sec=record['runtime_sec']), flush=True)


if __name__ == '__main__':
    main()
