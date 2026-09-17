"""One matched 225+15 ARR-month information experiment; no capacity search."""
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
import run_augmented
import run_information
import lgbm_adapter

ROOT = core.ROOT
CACHE = ROOT / 'private_runs/breakthrough_20260916/retrospective_research/monthly_arrival'
OUT = ROOT / 'private_runs/breakthrough_20260916/retrospective_models/monthly_arrival'
CONTROL = ROOT / 'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    args.family, args.formulation = 'lightgbm_monthly_arrival', 'aobt_allfinite'
    args.folds, args.threads, args.seed = ['F1', 'F3'], 4, 20260916
    core.OUT = core.external_path(OUT)
    cache_declaration = core.read_json(CACHE / 'protocol.json')['declaration']
    added = cache_declaration['features']
    assert len(added) == 15
    controls = {f: core.read_json(CONTROL / f'lightgbm_aobt_allfinite_{f}_s20260916/manifest.json') for f in args.folds}
    first = controls['F1']
    for record in controls.values():
        assert record['status'] == 'complete' and record['feature_columns'] == first['feature_columns']
        assert len(record['feature_columns']) == 225 and record['threads'] == 4 and record['seed'] == 20260916
        assert record['source_hashes']['lgbm_adapter.py'] == core.sha256(lgbm_adapter.__file__)
    paths = [Path(__file__), Path(lgbm_adapter.__file__), Path(run_information.__file__),
        Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    old_hashes = core.source_hashes
    core.source_hashes = lambda: {**old_hashes(), **{str(p.relative_to(ROOT)): core.sha256(p) for p in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: lgbm_adapter if name == 'tabm_gpu' else importlib.import_module(name))
    protocol = core.declare(args)
    payload = {'hypothesis': 'Whole-month ARR raw-duration distributions by airport/stand/runway add information beyond combined225.',
        'source_hashes': core.source_hashes(), 'cache_protocol_sha256': core.sha256(CACHE / 'protocol.json'),
        'base_columns': first['feature_columns'], 'added_columns': added,
        'controls': {f: core.sha256(CONTROL / f'lightgbm_aobt_allfinite_{f}_s20260916/manifest.json') for f in args.folds},
        'comparison': 'Unchanged LGB600 adapter, seed, parameters, raw residual labels, full eligible cohorts and original tune stopping/refit.',
        'availability': cache_declaration['policy']['availability'],
        'scope': 'Single adaptive exposed-development information test; excluded from final9; missing V2 unchanged.'}
    detail = OUT / 'matched_protocol.json'
    if detail.exists():
        assert core.read_json(detail) == payload
    else:
        core.write_json(detail, payload)
        snapshot = OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
        for path in paths:
            shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print('MONTHLY_ARR_MATCHED_DECLARED_NO_PRIVATE_ROWS', flush=True)
        return
    assert psutil.virtual_memory().available >= 20 * 1024**3
    cache = core.read_json(CACHE / 'manifest.json')
    verification = core.read_json(CACHE / 'verification.json')
    oracle = core.read_json(CACHE / 'independent_oracle.json')
    assert cache['status'] == 'complete' and verification['status'] == oracle['status'] == 'passed'
    assert verification['manifest_sha256'] == oracle['manifest_sha256'] == core.sha256(CACHE / 'manifest.json')
    assert cache['features'] == added and cache['protocol_sha256'] == payload['cache_protocol_sha256']
    cachepath = CACHE / 'training_features.parquet'
    assert core.sha256(cachepath) == cache['outputs'][cachepath.name]
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, ['source_past', 'source_twosided', 'surface_T', 'trajectory'])
    x, extra = run_information.additional(x, ['conventions', 'geometry', 'weather_T'])
    receipts += extra
    assert list(x) == first['feature_columns']
    assert all(receipts == record['anchor']['feature_receipts'] for record in controls.values())
    context = pd.read_parquet(cachepath).set_index(core.ID)
    assert context.index.is_unique and list(context) == added
    np.testing.assert_array_equal(context.index, x.index)
    assert not set(context).intersection(x)
    x = pd.concat([x, context.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')], axis=1)
    del context
    gc.collect()
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, monthly_arrival={**payload,
        'cache_manifest_sha256': core.sha256(CACHE / 'manifest.json'),
        'independent_oracle_sha256': core.sha256(CACHE / 'independent_oracle.json')})
    records = {}
    for fold in args.folds:
        idx, split, _ = core.common.fold_data(meta, fold, full=True)
        actual = {stage: {'n': int(eligible[pos].sum()),
            'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())} for stage, pos in idx.items()}
        assert actual == controls[fold]['fit_ids'] and split == controls[fold]['split']
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        assert records[fold]['fit']['params'] == controls[fold]['fit']['params']
        gc.collect()
    scores = {v: core.season_score(records['F1']['reports'][v]['metrics']['overall'],
        records['F3']['reports'][v]['metrics']['overall']) for v in ('candidate', 'blend25')}
    core.write_json(OUT / 'summary.json', {'seasonal_rmse': scores, 'matched_protocol_sha256': core.sha256(detail)})
    print('MONTHLY_ARR_SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
