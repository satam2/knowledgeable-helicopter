"""Completed449 artifact and matched337/225 comparisons; no GPU or fitting."""
import lightgbm
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', choices=['F1', 'F3'], default=['F1', 'F3'])
    args = parser.parse_args()
    root = audit.BASE / 'deeper_sequence8'
    cache = audit.BASE / 'missing/sequence_flatten8'
    manifest = audit.read_json(cache / 'manifest.json')
    verification = audit.read_json(cache / 'verification.json')
    payload = audit.read_json(root / 'matched_protocol.json')
    assert manifest['status'] == 'complete' and verification['status'] == 'passed'
    assert verification['manifest_sha256'] == payload['cache_manifest_sha256'] == audit.sha256(cache / 'manifest.json')
    assert manifest['outputs']['training_features.parquet'] == audit.sha256(cache / 'training_features.parquet')
    assert payload['cache_verification_sha256'] == audit.sha256(cache / 'verification.json')
    assert len(manifest['features']) == 224 and payload['added_columns'] == manifest['features']
    records = {}
    for fold in args.folds:
        folder = root / f'lightgbm_leaf63_sequence8_aobt_allfinite_{fold}_s20260916'
        rec = audit.checked(folder)
        controlroot = audit.BASE / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'
        seqroot = audit.BASE / 'deeper_sequence_v2' / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        control, sequence = audit.checked(controlroot), audit.checked(seqroot)
        audit.contract(control, rec)
        audit.contract(sequence, rec)
        assert rec['anchor']['feature_receipts'] == control['anchor']['feature_receipts'] == sequence['anchor']['feature_receipts']
        assert rec['feature_columns'] == control['feature_columns'] + manifest['features']
        assert len(rec['feature_columns']) == 449 and len(control['feature_columns']) == 225
        assert rec['anchor']['ordered_history'] == payload
        assert payload['controls'][fold] == audit.sha256(controlroot / 'manifest.json')
        snapshot = root / 'source_snapshots/lightgbm_leaf63_sequence8_aobt_allfinite_s20260916'
        for name, digest in rec['source_hashes'].items():
            assert audit.sha256(snapshot / Path(name).name) == digest
        reference, _ = audit.common.reference(fold)
        candidate = pd.read_parquet(folder / 'candidate.parquet')
        np.testing.assert_array_equal(candidate[audit.ID], reference[audit.ID])
        np.testing.assert_array_equal(candidate[audit.TARGET], reference[audit.TARGET])
        missing = ~np.isfinite(reference.proxy_sec.to_numpy())
        np.testing.assert_array_equal(candidate.prediction_sec.to_numpy()[missing], reference.prediction_sec.to_numpy()[missing])
        for variant in ('candidate', 'blend25'):
            frame = pd.read_parquet(folder / f'{variant}.parquet')
            errors = frame.prediction_sec.to_numpy() - frame[audit.TARGET].to_numpy()
            metrics = rec['reports'][variant]['metrics']['overall']
            assert metrics['n'] == len(frame)
            np.testing.assert_allclose(np.square(errors).sum(), metrics['sse'], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.square(errors).mean()), metrics['rmse_sec'], rtol=1e-12)
        result = {'status': 'passed', 'candidate_metrics': rec['reports']['candidate']['metrics']['overall'],
            'versus225': audit.compare(pd.read_parquet(controlroot / 'candidate.parquet'), candidate),
            'versus337': audit.compare(pd.read_parquet(seqroot / 'candidate.parquet'), candidate),
            'selected_iterations': rec['fit']['steps'], 'matched_parameters_seed_threads_cohorts_receipts': True,
            'full_saved_evaluator_replay_delta': rec['reload_max_abs_delta'], 'missing_V2_exact_rows': int(missing.sum()),
            'manifest_sha256': audit.sha256(folder / 'manifest.json')}
        records[fold] = result
        print('VERIFIED449', fold, result['candidate_metrics']['rmse_sec'], result['selected_iterations'], flush=True)
    result = {'status': 'passed', 'source_sha256': audit.sha256(__file__), 'folds': records,
        'cache_manifest_sha256': audit.sha256(cache / 'manifest.json'),
        'scope': 'Immutable completedfolds only; fullsavedmetrics/hash/cohort/schema checks. Producerfullsavedmodelreplayreceiptverified; no independent modelinference.'}
    if set(records) == {'F1', 'F3'}:
        result['seasonal_rmse'] = audit.season_score(records['F1']['candidate_metrics'], records['F3']['candidate_metrics'])
    output = audit.OUT / 'closing_1254' / ('audit449_' + '_'.join(args.folds) + '.json')
    if output.exists():
        raise ValueError('Preserve previous audit receipt')
    audit.write_json(output, result)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
