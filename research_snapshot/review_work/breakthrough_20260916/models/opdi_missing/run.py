"""Matched missing-record template model with eight public OPDI observables."""
import argparse
import gc
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/missing'))
import run_missing_models as base
import run_id_context as context

CACHE = ROOT / 'private_runs/breakthrough_20260916/missing/opdi_rotation'
OUT = base.external_path(ROOT / 'private_runs/breakthrough_20260916/models/opdi_missing')
CONTROL = ROOT / 'private_runs/breakthrough_20260916/missing/id_context_v1/models'
COLUMNS = ['opdi_match', 'opdi_match_ambiguous', 'opdi_match_offset_sec',
    'opdi_previous_leg_available', 'opdi_previous_same_airport',
    'opdi_ground_interval_sec', 'opdi_previous_leg_duration_sec',
    'opdi_previous_leg_age_at_takeoff_sec']


def declare():
    sources = [Path(__file__), Path(base.__file__), Path(context.__file__), Path(base.common.__file__)]
    payload = {'source_hashes': {str(p.relative_to(ROOT)): base.sha256(p) for p in sources},
        'folds': ['F1', 'F3'], 'seed': 20260916, 'threads': 2, 'arm': 'historical_template',
        'added_features': COLUMNS, 'parameters': base.PARAMS,
        'control': 'Original complete id_context_v1 historical_template, same full missing cohorts and raw labels.',
        'fit': 'Unchanged chronological template crossfit and CatBoost raw residual; original tune stopping and fresh refit.',
        'variants': ['candidate', 'blend25'], 'blend_weight': .25,
        'availability': 'External public final OPDI flight lists, strict local callsign/origin/destination/time matching; month-isolated earlier completed aircraft leg. Retrospective, not observed off-block.',
        'privacy': 'Bulk public month downloads only; private joins local. No raw aircraft address predictor.',
        'licensing': 'Noncommercial local research with attribution. Prize/commercial eligibility unresolved; no submission.',
        'scope': 'Adaptive new-information follow-up. Excluded from fixed final9; no score-selected weights, clipping or exceptional-row removal.',
        'cache': str(CACHE), 'controls': {f: base.sha256(CONTROL / f'historical_template_{f}_s20260916/manifest.json') for f in ('F1', 'F3')}}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'protocol.json'
    if target.exists():
        assert base.read_json(target)['declaration'] == payload
    else:
        base.write_json(target, {'created_utc': base.utc_now(), 'declaration': payload})
        snapshot = OUT / 'source_snapshots'
        snapshot.mkdir()
        for source in sources:
            shutil.copyfile(source, snapshot / source.name)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    payload = declare()
    if args.declare_only:
        print('OPDI_MODEL_DECLARED_NO_PRIVATE_ROWS', flush=True)
        return
    assert psutil.virtual_memory().available >= 12 * 1024**3, 'Need12GiB admission and8GiB reserve'
    manifest = base.read_json(CACHE / 'manifest.json')
    verification = base.read_json(CACHE / 'verification.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == base.sha256(CACHE / 'manifest.json')
    assert manifest['features'] == COLUMNS
    path = CACHE / 'training_features.parquet'
    assert base.sha256(path) == manifest['outputs'][path.name]
    x, meta = base.load_data()
    audit = base.read_json(context.CACHE / 'audit.json')
    peerpath = context.CACHE / 'features.parquet'
    assert base.sha256(peerpath) == audit['feature_sha256']
    peer = pd.read_parquet(peerpath).set_index(base.ID)
    np.testing.assert_array_equal(peer.index, meta[base.ID])
    timestamps = meta.set_index(base.ID).loc[x.index, base.MOVEMENT]
    x = pd.concat([x, context.id_context_features(x, peer.loc[x.index], timestamps)], axis=1)
    del peer
    gc.collect()
    controls = {f: base.read_json(CONTROL / f'historical_template_{f}_s20260916/manifest.json') for f in ('F1', 'F3')}
    for control in controls.values():
        assert control['status'] == 'complete' and control['features_used'] == list(x)
        assert control['threads'] == 2 and control['seed'] == 20260916
    added = pd.read_parquet(path).set_index(base.ID)
    assert added.index.is_unique and list(added) == COLUMNS
    np.testing.assert_array_equal(added.index, x.index)
    assert not set(added).intersection(x)
    x = pd.concat([x, added.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')], axis=1)
    receipts = {'manifest_sha256': base.sha256(CACHE / 'manifest.json'),
        'verification_sha256': base.sha256(CACHE / 'verification.json'),
        'training_features_sha256': base.sha256(path), 'context_sha256': audit['feature_sha256']}
    execution = OUT / 'execution_inputs.json'
    assert not execution.exists(), 'Preserve prior executed attempt'
    base.write_json(execution, {'created_utc': base.utc_now(), 'receipts': receipts, 'features': list(x)})
    original_write = base.write_json

    def write_bound(path, value):
        if Path(path).name == 'manifest.json':
            value.update(features='Original airport and ID-context fields plus eight strict OPDI fields',
                source_hashes=payload['source_hashes'], anchor={'opdi': receipts},
                formulation='missing_template_idcontext_opdi', feature_columns=list(x))
        original_write(path, value)

    base.OUT = OUT
    base.write_json = write_bound
    for fold in ('F1', 'F3'):
        index, split, _ = base.common.fold_data(meta, fold, full=True)
        eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float))
        identities = {stage: {'n': int(eligible[rows].sum()),
            'hash': base.object_hash(meta.iloc[rows[eligible[rows]]][base.ID].tolist())} for stage, rows in index.items()}
        assert identities == controls[fold]['fit_ids'] and split == controls[fold]['split']
        assert psutil.virtual_memory().available >= 8 * 1024**3
        base.run_arm('historical_template', fold, x, meta, 20260916, 2)
    records = {f: base.read_json(OUT / f'models/historical_template_{f}_s20260916/manifest.json') for f in ('F1', 'F3')}
    scores = {variant: float(np.sqrt(sum(weight * records[f]['reports'][variant]['metrics']['overall']['rmse_sec']**2
        for f, weight in [('F1', 192122/344841), ('F3', 152719/344841)]))) for variant in ('candidate', 'blend25')}
    base.write_json(OUT / 'summary.json', {'seasonal_rmse': scores, 'execution_inputs_sha256': base.sha256(execution)})
    print('OPDI_MISSING_SEASONAL', scores, flush=True)


if __name__ == '__main__':
    main()
