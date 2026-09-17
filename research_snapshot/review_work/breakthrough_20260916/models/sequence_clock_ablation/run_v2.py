"""Matched LGB600 ablation: remove only 24 flattened departure clock channels."""
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
import schema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import run_full as core
import run_information
import run_augmented
import lgbm_adapter

CACHE = ROOT / 'private_runs/breakthrough_20260916/missing/sequence_flatten'
CONTROLS = CACHE / 'models'
BASE = ROOT / 'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_clock_ablation'
BUILDER = ROOT / 'review_work/breakthrough_20260916/missing/sequence_flatten/build.py'
BLOCKS = ['conventions', 'geometry', 'source_past', 'source_twosided', 'surface_T', 'trajectory', 'weather_T']


def checked(args):
    cache = core.read_json(CACHE / 'manifest.json')
    verification = core.read_json(CACHE / 'verification.json')
    assert cache['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == core.sha256(CACHE / 'manifest.json')
    assert cache['source_sha256'] == core.sha256(BUILDER)
    assert core.sha256(CACHE / 'training_features.parquet') == cache['outputs']['training_features.parquet']
    dropped, retained = schema.selection(cache['features'])
    controls, baseline = {}, {}
    for fold in args.folds:
        directory = CONTROLS / f'lightgbm_aobt_allfinite_{fold}_s20260916'
        control = core.read_json(directory / 'manifest.json')
        base = core.read_json(BASE / f'lightgbm_aobt_allfinite_{fold}_s20260916' / 'manifest.json')
        for marker in (control, base):
            assert marker['status'] == 'complete' and marker['threads'] == args.threads and marker['seed'] == args.seed
            adapter_hashes = [digest for name, digest in marker['source_hashes'].items() if Path(name).name == 'lgbm_adapter.py']
            assert adapter_hashes and all(digest == core.sha256(lgbm_adapter.__file__) for digest in adapter_hashes)
            assert marker['fit']['steps'] == marker['refit']['steps'] == 600
        assert control['fit']['params'] == base['fit']['params']
        assert control['fit_ids'] == base['fit_ids']
        assert control['anchor']['feature_receipts'] == base['anchor']['feature_receipts']
        assert len(base['feature_columns']) == 225 and control['feature_columns'] == base['feature_columns'] + cache['features']
        assert control['anchor']['flatten_manifest_sha256'] == core.sha256(CACHE / 'manifest.json')
        controls[fold], baseline[fold] = control, base
    first = baseline[args.folds[0]]
    for control in baseline.values():
        assert control['feature_columns'] == first['feature_columns']
        assert control['anchor']['feature_receipts'] == first['anchor']['feature_receipts']
        assert control['fit']['params'] == first['fit']['params']
    return cache, controls, baseline, dropped, retained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert args.threads == 4 and args.seed == 20260916 and len(args.folds) == len(set(args.folds))
    cache, controls, baseline, dropped, retained = checked(args)
    first = baseline[args.folds[0]]
    paths = [Path(__file__), HERE / 'run.py', HERE / 'schema.py', HERE / 'test_contracts.py', BUILDER,
             Path(lgbm_adapter.__file__), Path(run_information.__file__), Path(run_augmented.__file__),
             Path(run_augmented.feature_screen.__file__)]
    original = core.source_hashes
    core.source_hashes = lambda: {**original(), **{str(p.relative_to(ROOT)): core.sha256(p) for p in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: lgbm_adapter if name == 'tabm_gpu' else importlib.import_module(name))
    core.OUT = core.external_path(OUT)
    args.family, args.formulation = 'lightgbm_sequence_no_dep_clocks', 'aobt_allfinite'
    protocol = core.declare(args)
    payload = {'source_hashes': core.source_hashes(), 'cache_manifest_sha256': core.sha256(CACHE / 'manifest.json'),
        'cache_verification_sha256': core.sha256(CACHE / 'verification.json'),
        'cache_features_sha256': cache['outputs']['training_features.parquet'],
        'original_flatten_columns': cache['features'], 'dropped_columns': dropped, 'retained_added_columns': retained,
        'dropped_schema_sha256': core.object_hash(dropped), 'retained_schema_sha256': core.object_hash(retained),
        'base_columns': first['feature_columns'], 'final_columns': first['feature_columns'] + retained,
        'final_schema_sha256': core.object_hash(first['feature_columns'] + retained), 'features': 313,
        'controls': {fold: {'manifest_sha256': core.sha256(CONTROLS / f'lightgbm_aobt_allfinite_{fold}_s20260916' / 'manifest.json'),
                           'fit_ids': control['fit_ids'], 'parameters': control['fit']['params']} for fold, control in controls.items()},
        'statistical_and_compute_params': first['fit']['params'],
        'comparison': 'ExactLGB600seed16/4threads/originalallfinitecohorts againstcompleted337control;only24DEPclockchannelsremoved.',
        'scope': 'Original225untouched; retainall56ARRchannels and32otherDEPchannels. This doesnotremoveallclockinformation.',
        'target': 'RawunmodifiedY-minusNM residual; negative/longfiniteNMretained; no labelclipping/filtering/logging/resampling.',
        'missing': 'ExistingNaN/padding semantics unchanged; nonfiniteNMscore routes preserveV2exactly.',
        'availability': 'Verifiedstrictpast60minute sameairport/movementmonth peer cache; finalNMretrospective,notcausalavailability.',
        'selection': 'Designedafterexposeddevelopment scores andtreefeatureusage. EOBTusage motivatedhypothesis; importanceisnotcausalproof. Fixedablation,nogrid.',
        'resources': 'NoGPU;4CPUthreads;20GiBfreebeforefullqueryload;existing8GiBbeforefit.'}
    detail = OUT / 'matched_protocol.json'
    if detail.exists():
        assert core.read_json(detail) == payload, 'Frozen ablation protocol changed'
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
        print('PREPARED_SEQUENCE_CLOCK_ABLATION', len(dropped), len(retained), len(payload['final_columns']), detail, flush=True)
        return
    if psutil.virtual_memory().available < 20 * 1024**3:
        raise MemoryError('Sequence clock ablation requires20GiB beforefullfeatureload')
    x, meta = core.common.load_data()
    standard = [b for b in BLOCKS if b in ['trajectory', *run_augmented.BATCH_PATTERNS]]
    x, receipts = run_augmented.augment(x, standard)
    x, extra = run_information.additional(x, [b for b in BLOCKS if b not in standard])
    receipts += extra
    assert list(x) == first['feature_columns'] and receipts == first['anchor']['feature_receipts']
    flat = pd.read_parquet(CACHE / 'training_features.parquet', columns=[core.ID, *retained]).set_index(core.ID)
    assert flat.index.is_unique and list(flat) == retained
    np.testing.assert_array_equal(flat.index, x.index)
    assert not set(flat).intersection(x.columns)
    x = pd.concat([x, flat], axis=1)
    del flat
    gc.collect()
    assert list(x) == payload['final_columns'] and len(x.columns) == 313
    x, proxy, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, sequence_clock_ablation=payload, matched_protocol_sha256=core.sha256(detail))
    results = {}
    for fold in args.folds:
        index, split, _ = core.common.fold_data(meta, fold, full=True)
        actual = {stage: {'n': int(eligible[pos].sum()), 'hash': core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())} for stage, pos in index.items()}
        assert actual == controls[fold]['fit_ids']
        assert core.object_hash(split) == core.object_hash(controls[fold]['split'])
        record = core.run(args, fold, x, meta, proxy, eligible, info, protocol)
        assert record['fit']['params'] == controls[fold]['fit']['params']
        results[fold] = record
        gc.collect()
    if set(results) == {'F1', 'F3'}:
        summary = {variant: core.season_score(results['F1']['reports'][variant]['metrics']['overall'], results['F3']['reports'][variant]['metrics']['overall']) for variant in ('candidate', 'blend25')}
        core.write_json(OUT / 'summary.json', {'seasonal_rmse': summary, 'matched_protocol_sha256': core.sha256(detail)})
        print('SEASONAL_SEQUENCE_CLOCK_ABLATION', summary, flush=True)


if __name__ == '__main__':
    main()
