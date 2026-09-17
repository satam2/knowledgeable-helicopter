"""Matched LGB600337 versus373 representation-only clock innovations."""
import lightgbm
import argparse
import gc
import importlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import psutil
import build

HERE = Path(__file__).resolve().parent
ROOT = build.ROOT
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import run_full as core
import run_information
import run_augmented
import lgbm_adapter

OUT = build.OUT.parent / 'models'
CONTROL = build.FLAT / 'models'
BLOCKS = ['conventions', 'geometry', 'source_past', 'source_twosided', 'surface_T', 'trajectory', 'weather_T']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert args.threads == 4 and args.seed == 20260916 and len(args.folds) == len(set(args.folds))
    cache = core.read_json(build.OUT / 'manifest.json')
    verification = core.read_json(build.OUT / 'verification.json')
    flat = core.read_json(build.FLAT / 'manifest.json')
    assert cache['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == core.sha256(build.OUT / 'manifest.json')
    assert cache['source_sha256'] == core.sha256(build.__file__)
    assert cache['features'] == build.FEATURES and len(cache['features']) == 36
    assert cache['outputs']['training_features.parquet'] == core.sha256(build.OUT / 'training_features.parquet')
    assert cache['flat_manifest_sha256'] == core.sha256(build.FLAT / 'manifest.json')
    assert flat['status'] == 'complete' and len(flat['features']) == 112
    assert flat['outputs']['training_features.parquet'] == core.sha256(build.FLAT / 'training_features.parquet')
    controls = {fold: core.read_json(CONTROL / f'lightgbm_aobt_allfinite_{fold}_s20260916/manifest.json') for fold in args.folds}
    first = controls[args.folds[0]]
    for control in controls.values():
        assert control['status'] == 'complete' and control['seed'] == args.seed and control['threads'] == args.threads
        assert control['fit']['steps'] == control['refit']['steps'] == 600
        assert control['feature_columns'] == first['feature_columns'] and len(control['feature_columns']) == 337
        assert control['feature_columns'][225:] == flat['features']
        assert control['anchor']['feature_receipts'] == first['anchor']['feature_receipts']
        assert control['fit']['params'] == first['fit']['params']
        hashes = [digest for name, digest in control['source_hashes'].items() if Path(name).name == 'lgbm_adapter.py']
        assert hashes and all(digest == core.sha256(lgbm_adapter.__file__) for digest in hashes)
    paths = [Path(__file__), Path(build.__file__), HERE / 'test_contracts.py', Path(lgbm_adapter.__file__),
             Path(run_information.__file__), Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    original = core.source_hashes
    core.source_hashes = lambda: {**original(), **{str(p.relative_to(ROOT)): core.sha256(p) for p in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: lgbm_adapter if name == 'tabm_gpu' else importlib.import_module(name))
    core.OUT = core.external_path(OUT)
    args.family, args.formulation = 'lightgbm_clock_innovations', 'aobt_allfinite'
    protocol = core.declare(args)
    payload = {'source_hashes': core.source_hashes(), 'cache_manifest_sha256': core.sha256(build.OUT / 'manifest.json'),
        'cache_verification_sha256': core.sha256(build.OUT / 'verification.json'),
        'flat_manifest_sha256': core.sha256(build.FLAT / 'manifest.json'),
        'control_columns': first['feature_columns'], 'added_columns': cache['features'],
        'final_columns': first['feature_columns'] + cache['features'],
        'final_schema_sha256': core.object_hash(first['feature_columns'] + cache['features']),
        'controls': {fold: core.sha256(CONTROL / f'lightgbm_aobt_allfinite_{fold}_s20260916/manifest.json') for fold in args.folds},
        'parameters': first['fit']['params'], 'comparison': 'Same337controlinformation,36explicitclockdifferencesadded; sameLGB600parameters/fullfinitecohorts/seed16/4threads.',
        'selection': 'Afterexposedclockablation andtreeusage, fixedrepresentationhypothesis; no newinformation orcapacitygrid. Excludedfromfrozenfinal9.',
        'missing': 'StrictinputNaNpropagation inderivedfeatures; encoderusesexistingmissingsemantics. No labeltransforms/filtering/clipping; nonfiniteNM V2exact.',
        'availability': 'Originalstrictpastpeerqueryselection; suppliedfinalclocksretrospective.',
        'resources': 'NoGPU,20GiBavailablebeforefullfeatureload,4CPUthreads.'}
    detail = OUT / 'matched_protocol.json'
    if detail.exists():
        assert core.read_json(detail) == payload
    else:
        core.write_json(detail, payload)
    snapshot = OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        destination = snapshot / path.name
        if destination.exists():
            assert core.sha256(destination) == core.sha256(path)
        else:
            shutil.copyfile(path, destination)
    if args.prepare_only:
        print('PREPARED_CLOCK_INNOVATIONS', len(payload['final_columns']), detail, flush=True)
        return
    if psutil.virtual_memory().available < 20 * 1024**3:
        raise MemoryError('Clockinnovations373 requires20GiB availablebeforefeatureload')
    x, meta = core.common.load_data()
    standard = [b for b in BLOCKS if b in ['trajectory', *run_augmented.BATCH_PATTERNS]]
    x, receipts = run_augmented.augment(x, standard)
    x, extra = run_information.additional(x, [b for b in BLOCKS if b not in standard])
    receipts += extra
    assert list(x) == first['feature_columns'][:225] and receipts == first['anchor']['feature_receipts']
    for root, columns in [(build.FLAT, flat['features']), (build.OUT, cache['features'])]:
        features = pd.read_parquet(root / 'training_features.parquet', columns=[core.ID, *columns]).set_index(core.ID)
        assert features.index.is_unique and list(features) == columns
        np.testing.assert_array_equal(features.index, x.index)
        assert not set(features).intersection(x.columns)
        x = pd.concat([x, features], axis=1)
        del features
        gc.collect()
    assert list(x) == payload['final_columns'] and len(x.columns) == 373
    x, proxy, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, clock_innovations=payload)
    results = {}
    for fold in args.folds:
        index, split, _ = core.common.fold_data(meta, fold, full=True)
        actual = {stage: {'n': int(eligible[pos].sum()), 'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())} for stage, pos in index.items()}
        assert actual == controls[fold]['fit_ids'] and core.object_hash(split) == core.object_hash(controls[fold]['split'])
        result = core.run(args, fold, x, meta, proxy, eligible, info, protocol)
        assert result['fit']['params'] == controls[fold]['fit']['params']
        results[fold] = result
        gc.collect()
    if set(results) == {'F1', 'F3'}:
        seasonal = {variant: core.season_score(results['F1']['reports'][variant]['metrics']['overall'], results['F3']['reports'][variant]['metrics']['overall']) for variant in ('candidate', 'blend25')}
        core.write_json(OUT / 'summary.json', {'seasonal_rmse': seasonal, 'matched_protocol_sha256': core.sha256(detail)})
        print('SEASONAL_CLOCK_INNOVATIONS', seasonal, flush=True)


if __name__ == '__main__':
    main()
