"""Parent-scheduled all-finite CatBoost on frozen combined225 information."""
import lightgbm
import argparse
import gc
import importlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'models'))
sys.path.insert(0, str(HERE.parents[1]))
import run_full as core
import run_information
import run_augmented

spec = importlib.util.spec_from_file_location('combined_catboost_adapter', HERE / 'adapter.py')
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)

BASE = core.ROOT / ('private_runs/breakthrough_20260916/information_models/'
    'conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F2', 'F3', 'G1'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    assert args.threads == 2 and args.seed == 20260916
    assert len(args.folds) == len(set(args.folds))
    args.family, args.formulation = 'catboost_combined', 'aobt_allfinite'
    core.OUT = core.external_path(core.ROOT / 'private_runs/breakthrough_20260916/combined_catboost')
    paths = [Path(__file__), HERE / 'adapter.py', Path(run_information.__file__),
        Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{str(p.relative_to(core.ROOT)): core.sha256(p) for p in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == 'tabm_gpu' else importlib.import_module(name))
    control_path = BASE / 'lightgbm_aobt_allfinite_F1_s20260916/manifest.json'
    control = core.read_json(control_path)
    assert control['status'] == 'complete' and len(control['feature_columns']) == 225
    protocol = core.declare(args)
    detail = protocol.with_name(protocol.stem + '_features.json')
    payload = {'columns': control['feature_columns'], 'receipts': control['anchor']['feature_receipts'],
        'source_hashes': core.source_hashes(), 'params': adapter.parameters(args.seed, args.threads),
        'control_manifest_sha256': core.sha256(control_path),
        'comparison': 'Same combined225 input as full TabM and LightGBM; CatBoost5000depth8lr.05l2=8 from existingglobalconventions; allfinite instead of ordinary-only eligibility.',
        'selection': 'Adaptive matched-information development follow-up; original exposed folds; not a novel algorithm or parameter-matched capacity comparison.',
        'availability': 'Explicit retrospective supplied batch; final NM publication time unknown.',
        'reference_routes': 'Original V2 predictions retained exactly for missingNM; finite negative and long proxies included.',
        'preprocessing': 'Fit/refit-only frozen FrameEncoder; original full tune only selects stopping; no label-derived feature additions.',
        'declaration_loads_private_rows': False}
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError('Frozen combined CatBoost declaration changed')
    core.write_json(detail, payload)
    snapshot = core.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        destination = snapshot / path.name
        if destination.exists() and core.sha256(destination) != core.sha256(path):
            raise ValueError('Source snapshot changed')
        if not destination.exists():
            shutil.copyfile(path, destination)
    if args.declare_only:
        print('DECLARED', detail, len(payload['columns']), 'no row load or GPU launch', flush=True)
        return
    if psutil.virtual_memory().available < 18 * 1024**3:
        raise MemoryError('Combined CatBoost load requires18GiB available host memory')
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    assert list(x) == payload['columns'] and receipts == payload['receipts']
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, matched_information=payload)
    records = {}
    for fold in args.folds:
        marker = BASE / f'lightgbm_aobt_allfinite_{fold}_s20260916/manifest.json'
        if marker.exists():
            matched = core.read_json(marker)
            assert matched['status'] == 'complete'
            assert matched['feature_columns'] == list(x) and matched['anchor']['feature_receipts'] == receipts
            indices, _, _ = core.common.fold_data(meta, fold, full=True)
            actual = {stage: {'n': int(eligible[pos].sum()), 'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())} for stage, pos in indices.items()}
            assert actual == matched['fit_ids']
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if 'F1' in records and 'F3' in records:
        scores = {variant: core.season_score(records['F1']['reports'][variant]['metrics']['overall'],
            records['F3']['reports'][variant]['metrics']['overall']) for variant in ['candidate', 'blend25']}
        core.write_json(core.OUT / 'summary.json', {'seasonal_rmse': scores, 'features': payload})
        print('SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
