"""Isolated seed replication of frozen leaf63 combined-information training."""
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
import lgbm_adapter
import adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.seed == 20260916:
        raise ValueError('Original seed is already preserved; replication requires another seed')
    args.family, args.formulation = 'lightgbm_leaf63', 'aobt_allfinite'
    adapter.PRESET = 'leaf63'
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/deeper_replication' / f's{args.seed}')
    paths = [Path(__file__), Path(adapter.__file__), Path(lgbm_adapter.__file__),
             Path(run_information.__file__), Path(run_augmented.__file__),
             Path(run_augmented.feature_screen.__file__)]
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
    baseline = core.read_json(control)
    assert baseline['status'] == 'complete' and baseline['feature_columns'] == list(x)
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, seed_replication=True)
    protocol = core.declare(args)
    detail = protocol.with_name(protocol.stem + '_replication.json')
    payload = {'columns': list(x), 'receipts': receipts, 'params': adapter.PARAMS['leaf63'],
               'source_hashes': core.source_hashes(), 'original_manifest_sha256': core.sha256(control),
               'scope': 'Finite-NM residual expert only; fixed original missing reference and25percent blend. Not a full-pipeline seed replication.',
               'selection': 'Same parameters and full purged folds; independent tune stopping/refit. No seed selection by score.',
               'randomness': 'LightGBM deterministic CPU, no row/feature bagging; random_state may affect subsampled bin construction. Identical predictions are possible.'}
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen seed-replication declaration changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('DECLARED', detail, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
                                             records['F3']['reports'][variant]['metrics']['overall'])
                  for variant in ('candidate', 'blend25')}
        core.write_json(core.OUT / 'summary.json', {'seasonal_rmse': scores, 'replication': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
