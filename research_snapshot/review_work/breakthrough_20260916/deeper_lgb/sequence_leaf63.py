"""Follow up a matched sequence-information gain using the frozen leaf63 adapter."""
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
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    args = parser.parse_args()
    if args.threads != 2 or args.seed != 20260916:
        raise ValueError('Matched original leaf63 control uses seed20260916 and2threads')
    args.family, args.formulation = 'lightgbm_leaf63_sequence', 'aobt_allfinite'
    adapter.PRESET = 'leaf63'
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/deeper_sequence')
    cache = core.ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten'
    manifest = core.read_json(cache / 'manifest.json')
    verification = core.read_json(cache / 'verification.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == core.sha256(cache / 'manifest.json')
    assert core.sha256(cache / 'training_features.parquet') == manifest['outputs']['training_features.parquet']
    paths = [Path(__file__), Path(adapter.__file__), Path(lgbm_adapter.__file__),
             Path(run_information.__file__), Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{
        str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == 'tabm_gpu' else importlib.import_module(name))
    if psutil.virtual_memory().available < 20 * 1024**3:
        raise MemoryError('337-feature deep model load requires20GiB available')
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    baseline_root = core.ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined'
    controls = {fold: core.read_json(baseline_root / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916/manifest.json')
                for fold in args.folds}
    for baseline in controls.values():
        assert baseline['status'] == 'complete' and baseline['feature_columns'] == list(x)
        assert baseline['seed'] == args.seed and baseline['threads'] == args.threads
    flat = pd.read_parquet(cache / 'training_features.parquet').set_index(core.ID)
    np.testing.assert_array_equal(flat.index, x.index)
    assert list(flat) == manifest['features'] and len(flat.columns) == 112
    assert not set(flat).intersection(x)
    x = pd.concat([x, flat], axis=1)
    del flat
    gc.collect()
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, flattened_manifest_sha256=core.sha256(cache / 'manifest.json'),
                flattened_verification_sha256=core.sha256(cache / 'verification.json'))
    protocol = core.declare(args)
    payload = {'columns': list(x), 'info': info, 'params': adapter.PARAMS['leaf63'], 'source_hashes': core.source_hashes(),
               'control_manifests': {fold: core.sha256(baseline_root / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916/manifest.json')
                                     for fold in args.folds},
               'hypothesis': 'Fixed matched extra112ordered-event features afterLGB600 information gain287.156651->284.933155. Adaptive exposeddevelopment, not freshholdout.',
               'training': 'Identical original fullfinite fit/tune/refit/scorecohorts, originalrawlabels, matchedleaf63params, missingV2exact.'}
    detail = protocol.with_name(protocol.stem + '_sequence.json')
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen sequence-deep declaration changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    records = {}
    for fold in args.folds:
        record = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        assert record['fit_ids'] == controls[fold]['fit_ids']
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
