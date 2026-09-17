"""Matched combined-information LGB600 comparison with verified wind fields."""
import lightgbm
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_information as information

core = information.core
augmented = information.run_augmented
import lgbm_adapter

BLOCKS = ['conventions', 'geometry', 'source_past', 'source_twosided',
          'surface_T', 'trajectory', 'weather_T']
CACHE = core.ROOT / 'private_runs/breakthrough_20260916/weather_wind'
BASELINE = core.ROOT / 'private_runs/breakthrough_20260916/information_models' / '__'.join(BLOCKS)
OUT = core.external_path(CACHE / 'models')


def append_wind(x, ext, columns):
    if core.ID in ext:
        ext = ext.set_index(core.ID)
    if not x.index.is_unique or not ext.index.is_unique or not np.array_equal(x.index, ext.index):
        raise ValueError('Wind IDs/order differ from matched baseline')
    if list(ext) != columns or len(columns) != 23 or set(columns).intersection(x):
        raise ValueError('Wind schema differs or overlaps baseline')
    ext = ext.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
    return pd.concat([x, ext], axis=1)


def artifacts():
    manifest_path = CACHE / 'manifest.json'
    manifest = core.read_json(manifest_path)
    verification = core.read_json(CACHE / 'verification.json')
    if manifest['status'] != 'complete' or verification['status'] != 'passed':
        raise ValueError('Wind feature cache is not complete and verified')
    if verification['manifest_sha256'] != core.sha256(manifest_path):
        raise ValueError('Wind verification is stale')
    for filename in ('training_features.parquet', 'ranking_features.parquet'):
        if core.sha256(CACHE / filename) != manifest['outputs'][filename]:
            raise ValueError('Wind feature bytes changed')
    baseline = core.read_json(BASELINE / 'lightgbm_aobt_allfinite_s20260916_summary.json')
    if baseline['features']['blocks'] != BLOCKS:
        raise ValueError('Comparator blocks changed')
    return manifest, baseline


def configure():
    core.OUT = OUT
    original_hashes = core.source_hashes
    sources = [Path(__file__), Path(information.__file__), Path(augmented.__file__),
               Path(augmented.feature_screen.__file__), Path(lgbm_adapter.__file__)]
    core.source_hashes = lambda: {**original_hashes(), **{p.name: core.sha256(p) for p in sources}}
    core.importlib = SimpleNamespace(import_module=lambda name:
        lgbm_adapter if name == 'tabm_gpu' else importlib.import_module(name))
    return sources


def declare(args, sources, manifest, baseline):
    current = core.source_hashes()
    for name, digest in baseline['features']['source_hashes'].items():
        if current.get(name) != digest:
            raise ValueError(f'Matched comparator source changed: {name}')
    payload = {
        'blocks': BLOCKS + ['runway_wind_density'],
        'columns': baseline['features']['columns'] + manifest['features'],
        'baseline_receipts': baseline['features']['receipts'],
        'baseline_summary_sha256': core.sha256(BASELINE / 'lightgbm_aobt_allfinite_s20260916_summary.json'),
        'baseline_seasonal_rmse': baseline['seasonal_rmse'],
        'wind_manifest_sha256': core.sha256(CACHE / 'manifest.json'),
        'wind_verification_sha256': core.sha256(CACHE / 'verification.json'),
        'wind_features_sha256': manifest['outputs'],
        'source_hashes': current,
        'availability': 'Retrospective supplied batch plus latest prior METAR valid time at T; publication time unverified. Runway source is current2026 metadata, not historical2025.',
        'comparison': 'Exact combined-information baseline columns/order and frozen LGB600; append23 heading/bearing wind and approximate dry-density fields only. No extra METAR block.',
        'resource_policy': 'CPU only; centrally scheduled. No training during prepare-only.',
    }
    protocol = core.declare(args)
    path = protocol.with_name(protocol.stem + '_features.json')
    if path.exists():
        if core.read_json(path) != payload:
            raise ValueError('Frozen wind comparison declaration changed')
    else:
        core.write_json(path, payload)
    snapshot = OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for source in sources:
        destination = snapshot / source.name
        if destination.exists():
            if core.sha256(destination) != core.sha256(source):
                raise ValueError('Existing source snapshot differs')
        else:
            shutil.copyfile(source, destination)
    return protocol, payload


def load_features(manifest, baseline):
    x, meta = core.common.load_data()
    standard = [b for b in BLOCKS if b in ['trajectory', *augmented.BATCH_PATTERNS]]
    others = [b for b in BLOCKS if b not in standard]
    x, receipts = augmented.augment(x, standard)
    x, additional = information.additional(x, others)
    receipts += additional
    if list(x) != baseline['features']['columns'] or receipts != baseline['features']['receipts']:
        raise ValueError('Loaded combined-information baseline is not identical')
    ext = pd.read_parquet(CACHE / 'training_features.parquet')
    x = append_wind(x, ext, manifest['features'])
    return x, meta


def self_test():
    columns = [f'wx_{i}' for i in range(23)]
    x = pd.DataFrame({'base': [1., 2.]}, index=pd.Index([10, 20], name=core.ID))
    ext = pd.DataFrame(np.zeros((2, 23)), columns=columns, index=x.index)
    result = append_wind(x, ext, columns)
    assert list(result) == ['base'] + columns and result.shape == (2, 24)
    for bad in (ext.iloc[::-1], ext.set_axis([10, 10])):
        try:
            append_wind(x, bad, columns)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid movement identity accepted')
    print('PASSED wind schema/order/duplicate identity checks', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, choices=[20260916], default=20260916)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.test_only:
        self_test()
        return
    if args.threads < 1 or len(set(args.folds)) != len(args.folds):
        raise ValueError('Invalid threads or repeated folds')
    args.family, args.formulation = 'lightgbm', 'aobt_allfinite'
    sources = configure()
    manifest, baseline = artifacts()
    protocol, payload = declare(args, sources, manifest, baseline)
    if args.prepare_only:
        print('PREPARED', protocol, 'columns', len(payload['columns']), 'no model fit', flush=True)
        return
    x, meta = load_features(manifest, baseline)
    if list(x) != payload['columns']:
        raise ValueError('Prepared feature schema differs')
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info = {**info, 'wind_comparison': payload}
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {v: core.season_score(records['F1']['reports'][v]['metrics']['overall'],
                  records['F3']['reports'][v]['metrics']['overall']) for v in ('candidate', 'blend25')}
        core.write_json(OUT / 'lightgbm_aobt_allfinite_s20260916_summary.json',
                        {'seasonal_rmse': scores, 'features': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
