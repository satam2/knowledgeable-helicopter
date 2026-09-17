"""One longer ordered-history follow-up, preserving the leaf63 training contract."""
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
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'models'))
sys.path.insert(0, str(HERE.parent))
import run_full as core
import run_information
import run_augmented
import lgbm_adapter
import adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    assert args.threads == 2 and args.seed == 20260916
    args.family, args.formulation = 'lightgbm_leaf63_sequence8', 'aobt_allfinite'
    adapter.PRESET = 'leaf63'
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/deeper_sequence8')
    cache = core.ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten8'
    manifest = core.read_json(cache / 'manifest.json')
    verification = core.read_json(cache / 'verification.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == core.sha256(cache / 'manifest.json')
    assert core.sha256(cache / 'training_features.parquet') == manifest['outputs']['training_features.parquet']
    assert len(manifest['features']) == 224
    paths = [Path(__file__), Path(adapter.__file__), Path(lgbm_adapter.__file__),
             Path(run_information.__file__), Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{
        str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == 'tabm_gpu' else importlib.import_module(name))
    baseline_root = core.ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined'
    controls = {fold: core.read_json(baseline_root / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916/manifest.json')
                for fold in args.folds}
    first = controls[args.folds[0]]
    for baseline in controls.values():
        assert baseline['status'] == 'complete' and baseline['feature_columns'] == first['feature_columns']
        assert baseline['seed'] == args.seed and baseline['threads'] == args.threads
    protocol = core.declare(args)
    payload = {'added_columns': manifest['features'], 'base_columns': first['feature_columns'],
               'params': adapter.PARAMS['leaf63'], 'source_hashes': core.source_hashes(),
               'cache_manifest_sha256': core.sha256(cache / 'manifest.json'),
               'cache_verification_sha256': core.sha256(cache / 'verification.json'),
               'controls': {fold: core.sha256(baseline_root / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916/manifest.json')
                            for fold in args.folds},
               'hypothesis': 'Expand nearest4 to nearest8 per phase after robust ordered-context gain; one fixed extension, no capacity change.',
               'training': 'Same original fullfinite fit/tune/refit/score cohorts, original labels, leaf63 params and raw-MSE stopping. Missing V2 exact.',
               'availability': 'Verified prior60min sameairport/month DEP and completedARR tokens; finalNM publication time unknown.',
               'selection': 'Adaptive exposed development. No independent holdout or official score.'}
    detail = core.OUT / 'matched_protocol.json'
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen eight-event experiment changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('DECLARED', detail, flush=True)
        return
    if psutil.virtual_memory().available < 25 * 1024**3:
        raise MemoryError('449-feature deep model requires25GiB available before loading')
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    assert list(x) == first['feature_columns']
    for baseline in controls.values():
        assert receipts == baseline['anchor']['feature_receipts']
    flat = pd.read_parquet(cache / 'training_features.parquet').set_index(core.ID)
    assert flat.index.is_unique and list(flat) == manifest['features']
    np.testing.assert_array_equal(flat.index, x.index)
    assert not set(flat).intersection(x)
    x = pd.concat([x, flat], axis=1)
    del flat
    gc.collect()
    assert len(x.columns) == 449
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, ordered_history=payload)
    records = {}
    for fold in args.folds:
        idx, _, _ = core.common.fold_data(meta, fold, full=True)
        actual = {stage: {'n': int(eligible[pos].sum()), 'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())}
                  for stage, pos in idx.items()}
        assert actual == controls[fold]['fit_ids']
        record = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        assert record['fit']['params'] == controls[fold]['fit']['params']
        records[fold] = record
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
                                             records['F3']['reports'][variant]['metrics']['overall'])
                  for variant in ('candidate', 'blend25')}
        core.write_json(core.OUT / 'summary.json', {'seasonal_rmse': scores, 'declaration': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
