"""Compare verified retrospective context on the frozen combined225 LGB600 control."""
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

CACHE = core.ROOT / 'private_runs/breakthrough_20260916/retrospective_research'
BASE = core.ROOT / ('private_runs/breakthrough_20260916/information_models/'
                   'conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', nargs='+', choices=['past', 'future', 'day'], required=True)
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    assert len(args.blocks) == len(set(args.blocks))
    args.blocks = sorted(args.blocks)
    assert args.threads == 4 and args.seed == 20260916
    manifest = core.read_json(CACHE / 'manifest.json')
    verification = core.read_json(CACHE / 'verification.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == core.sha256(CACHE / 'manifest.json')
    assert manifest['outputs']['training_features.parquet'] == core.sha256(CACHE / 'training_features.parquet')
    added = [column for block in args.blocks for column in manifest['feature_groups'][block]]
    assert added and len(added) == len(set(added)) and set(added).issubset(manifest['features'])
    controls = {fold: core.read_json(BASE / f'lightgbm_aobt_allfinite_{fold}_s20260916/manifest.json')
                for fold in args.folds}
    for control in controls.values():
        assert control['status'] == 'complete' and control['threads'] == args.threads and control['seed'] == args.seed
        assert control['source_hashes']['lgbm_adapter.py'] == core.sha256(lgbm_adapter.__file__)
    first = controls[args.folds[0]]
    assert len(first['feature_columns']) == 225
    for control in controls.values():
        assert control['feature_columns'] == first['feature_columns']
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/retrospective_models' / '__'.join(args.blocks))
    args.family, args.formulation = 'lightgbm', 'aobt_allfinite'
    paths = [Path(__file__), Path(lgbm_adapter.__file__), Path(run_information.__file__),
             Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{
        str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: lgbm_adapter if name == 'tabm_gpu' else importlib.import_module(name))
    protocol = core.declare(args)
    payload = {'blocks': args.blocks, 'added_columns': added, 'base_columns': first['feature_columns'],
               'cache_manifest_sha256': core.sha256(CACHE / 'manifest.json'),
               'cache_verification_sha256': core.sha256(CACHE / 'verification.json'),
               'source_hashes': core.source_hashes(),
               'controls': {fold: core.sha256(BASE / f'lightgbm_aobt_allfinite_{fold}_s20260916/manifest.json') for fold in args.folds},
               'comparison': 'Same LGB600 adapter, params, seed, threads, original eligible rows; only specified context added.',
               'availability': 'Explicit retrospective supplied batch, month isolated and own-flight excluded. Future and whole-day fields are not real-time features.',
               'selection': 'Adaptive development hypothesis; existing score months exposed. No label-derived features or score-tuned weights.'}
    detail = core.OUT / 'matched_protocol.json'
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen retrospective experiment changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('DECLARED', detail, len(added), flush=True)
        return
    if psutil.virtual_memory().available < 20 * 1024**3:
        raise MemoryError('Combined context training requires20GiB available before load')
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    assert list(x) == first['feature_columns']
    for control in controls.values():
        assert receipts == control['anchor']['feature_receipts']
    context = pd.read_parquet(CACHE / 'training_features.parquet', columns=[core.ID, *added]).set_index(core.ID)
    assert context.index.is_unique and x.index.is_unique
    np.testing.assert_array_equal(context.index, x.index)
    assert not set(context).intersection(x.columns)
    context = context.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
    x = pd.concat([x, context], axis=1)
    del context
    gc.collect()
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, retrospective_context=payload)
    records = {}
    for fold in args.folds:
        idx, _, _ = core.common.fold_data(meta, fold, full=True)
        actual = {stage: {'n': int(eligible[pos].sum()), 'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())}
                  for stage, pos in idx.items()}
        assert actual == controls[fold]['fit_ids']
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        assert records[fold]['fit']['params'] == controls[fold]['fit']['params']
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
                                             records['F3']['reports'][variant]['metrics']['overall'])
                  for variant in ('candidate', 'blend25')}
        core.write_json(core.OUT / 'summary.json', {'seasonal_rmse': scores, 'matched_protocol_sha256': core.sha256(detail)})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
