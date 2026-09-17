"""Matched combined-information neural follow-up using frozen training adapters."""
import torch
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'models'))
sys.path.insert(0, str(HERE.parent))
import run_full as core
import run_information
import run_augmented
import tabm_gpu
import tabm_ple_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['standard', 'ple8', 'ple32'], required=True)
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    args.family = 'tabm_combined_' + args.encoding
    args.formulation = 'aobt_allfinite'
    adapter = tabm_gpu if args.encoding == 'standard' else tabm_ple_gpu
    if args.encoding != 'standard':
        tabm_ple_gpu.MEMBERS = int(args.encoding[3:])
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/full_neural')
    paths = [Path(__file__), Path(run_information.__file__), Path(run_augmented.__file__),
             Path(run_augmented.feature_screen.__file__), Path(tabm_ple_gpu.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{
        str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == 'tabm_gpu' else importlib.import_module(name))
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    control = core.ROOT / ('private_runs/breakthrough_20260916/deeper_lgb/combined/'
                          'lightgbm_leaf63_aobt_allfinite_F1_s20260916/manifest.json')
    marker = core.read_json(control)
    if marker['status'] != 'complete' or list(x) != marker['feature_columns']:
        raise ValueError('Combined feature columns differ from frozen full-information tree control')
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, encoding=args.encoding)
    protocol = core.declare(args)
    detail = protocol.with_name(protocol.stem + '_features.json')
    payload = {'encoding': args.encoding, 'columns': list(x), 'receipts': receipts,
               'source_hashes': core.source_hashes(), 'control_manifest_sha256': core.sha256(control),
               'selection': 'Adaptive follow-up on combined-information tree result; original tune raw MSE selects epochs.',
               'comparison': 'Same225 columns as combined LightGBM; frozen TabM MSE training loop. Missing NM remains V2.',
               'availability': 'Includes declared retrospective supplied batch context; weather event time is not publication time.'}
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen combined-neural declaration changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('DECLARED', detail, 'shape', x.shape, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
                                             records['F3']['reports'][variant]['metrics']['overall'])
                  for variant in ('candidate', 'blend25')}
        core.write_json(core.OUT / f'{args.family}_s{args.seed}_summary.json',
                        {'seasonal_rmse': scores, 'features': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
