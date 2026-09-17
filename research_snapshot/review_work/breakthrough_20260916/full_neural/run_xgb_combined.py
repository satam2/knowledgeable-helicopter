"""Matched-information XGBoost follow-up using the frozen GPU adapter."""
import lightgbm
import argparse
import gc
import shutil
import sys
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'models'))
sys.path.insert(0, str(HERE.parent))
import run_full as core
import run_information
import run_augmented
import xgb_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    args.family, args.formulation = 'xgb', 'aobt_allfinite'
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/combined_xgb')
    paths = [Path(__file__), Path(run_information.__file__), Path(run_augmented.__file__),
             Path(run_augmented.feature_screen.__file__), Path(xgb_gpu.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{
        str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    if psutil.virtual_memory().available < 18 * 1024**3:
        raise MemoryError('Combined tree load requires18GiB available host memory')
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    control = core.ROOT / ('private_runs/breakthrough_20260916/deeper_lgb/combined/'
                          'lightgbm_leaf63_aobt_allfinite_F1_s20260916/manifest.json')
    marker = core.read_json(control)
    assert marker['status'] == 'complete' and list(x) == marker['feature_columns']
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts)
    protocol = core.declare(args)
    detail = protocol.with_name(protocol.stem + '_features.json')
    payload = {'columns': list(x), 'receipts': receipts, 'source_hashes': core.source_hashes(),
               'control_manifest_sha256': core.sha256(control),
               'selection': 'Adaptive matched-information follow-up; original XGBoost adapter unchanged.',
               'comparison': 'Combined225 LightGBM/TabM information, original XGBoost capacity and raw residual objective.',
               'availability': 'Declared retrospective supplied batch; final NM publication time unknown.'}
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen combined XGBoost declaration changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('DECLARED', detail, x.shape, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {'F1', 'F3'}:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
                                             records['F3']['reports'][variant]['metrics']['overall'])
                  for variant in ('candidate', 'blend25')}
        core.write_json(core.OUT / 'summary.json', {'seasonal_rmse': scores, 'features': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
