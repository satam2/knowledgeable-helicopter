"""Join two independently verified information gains on the frozen leaf63 model."""
import lightgbm
import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import sequence_leaf63 as frozen

CORE = frozen.core
ROOT = CORE.ROOT
ORIGINAL_OUT = ROOT / 'private_runs/breakthrough_20260916/deeper_sequence'
OUT = ROOT / 'private_runs/breakthrough_20260916/deeper_context_union'
CACHE = ROOT / 'private_runs/breakthrough_20260916/retrospective_research'
CONTROL = ROOT / 'private_runs/breakthrough_20260916/deeper_sequence_v2'


def main():
    if psutil.virtual_memory().available < 25 * 1024**3:
        raise MemoryError('387-feature union requires25GiB available before loading')
    manifest = CORE.read_json(CACHE / 'manifest.json')
    verification = CORE.read_json(CACHE / 'verification.json')
    independent = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit/retrospective_oracle.json'
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == CORE.sha256(CACHE / 'manifest.json')
    assert manifest['outputs']['training_features.parquet'] == CORE.sha256(CACHE / 'training_features.parquet')
    assert len(manifest['features']) == 50
    oracle = CORE.read_json(independent)
    assert oracle['status'] == 'passed'
    assert oracle['manifest_sha256'] == CORE.sha256(CACHE / 'manifest.json')
    assert oracle['source_sha256'] == manifest['source_sha256']
    controls = {fold: CORE.read_json(CONTROL / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916/manifest.json')
                for fold in ['F1', 'F3']}
    assert all(record['status'] == 'complete' for record in controls.values())
    payload = {'hypothesis': 'Combine ordered-event337 and retrospective50 after separately measured gains. Adaptive exposed-development follow-up, no fresh holdout.',
               'training': 'Same original leaf63 parameters, seed20260916,2CPU, full finite fit/tune/refit, every score row and raw label; missingV2 exact.',
               'information': 'Original225 + nearest4DEP/4completedARR112 + past/future/day50 =387 columns.',
               'availability': 'Explicit retrospective final batch; future completed arrivals and same-day peers are not real-time observations.',
               'wrapper_sha256': CORE.sha256(__file__), 'frozen_runner_sha256': CORE.sha256(frozen.__file__),
               'cache_manifest_sha256': CORE.sha256(CACHE / 'manifest.json'),
               'cache_verification_sha256': CORE.sha256(CACHE / 'verification.json'),
               'independent_oracle_sha256': CORE.sha256(independent),
               'added_columns': manifest['features'],
               'matched_sequence_controls': {fold: CORE.sha256(CONTROL / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916/manifest.json')
                                            for fold in controls}}
    real_external = CORE.external_path
    CORE.external_path = lambda path: real_external(OUT if Path(path) == ORIGINAL_OUT else path)
    original_hashes = CORE.source_hashes
    CORE.source_hashes = lambda: {**original_hashes(), str(Path(__file__).relative_to(ROOT)): CORE.sha256(__file__)}
    original_anchors = CORE.anchors

    def anchors(frame, meta, formulation):
        for record in controls.values():
            assert list(frame) == record['feature_columns'] and len(frame.columns) == 337
        context = pd.read_parquet(CACHE / 'training_features.parquet').set_index(CORE.ID)
        assert context.index.is_unique and list(context) == manifest['features']
        np.testing.assert_array_equal(context.index, frame.index)
        assert not set(context).intersection(frame.columns)
        context = context.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
        extended = pd.concat([frame, context], axis=1)
        del context
        gc.collect()
        assert len(extended.columns) == 387
        x, offset, eligible, info = original_anchors(extended, meta, formulation)
        for fold, control in controls.items():
            idx, _, _ = CORE.common.fold_data(meta, fold, full=True)
            actual = {stage: {'n': int(eligible[pos].sum()), 'hash': CORE.object_hash(meta.iloc[pos[eligible[pos]]][CORE.ID].tolist())}
                      for stage, pos in idx.items()}
            assert actual == control['fit_ids']
        info['context_union'] = payload
        return x, offset, eligible, info

    CORE.anchors = anchors
    original_declare = CORE.declare

    def declare(args):
        assert set(args.folds) == {'F1', 'F3'} and args.threads == 2 and args.seed == 20260916
        path = original_declare(args)
        snapshot = CORE.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
        shutil.copyfile(__file__, snapshot / Path(__file__).name)
        target = CORE.OUT / 'union_protocol.json'
        if target.exists() and CORE.read_json(target) != payload:
            raise ValueError('Frozen context union changed')
        CORE.write_json(target, payload)
        return path

    CORE.declare = declare
    original_run = CORE.run

    def run(args, fold, *arguments):
        info = arguments[-2]
        assert info['feature_receipts'] == controls[fold]['anchor']['feature_receipts']
        record = original_run(args, fold, *arguments)
        assert record['fit_ids'] == controls[fold]['fit_ids']
        assert record['fit']['params'] == controls[fold]['fit']['params']
        return record

    CORE.run = run
    frozen.main()


if __name__ == '__main__':
    main()
