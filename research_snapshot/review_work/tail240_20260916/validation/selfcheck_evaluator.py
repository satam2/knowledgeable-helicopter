"""Positive unchanged-baseline fixture and negative prediction-schema checks."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import validate_candidate as validator


def main():
    root = validator.ROOT / 'private_runs/tail240_20260916/validation/evaluator_selfcheck_v1'
    assert not root.exists()
    root.mkdir(parents=True)
    binding = validator.read_json(validator.BINDING)
    protocol = {'name': 'unchanged_baseline_validation_canary',
        'baseline_binding_sha256': validator.sha256(validator.BINDING),
        'score_labels_used_for_selection': False, 'routing_uses_score_targets': False,
        'selection_periods': [], 'inference_columns': ['saved_baseline_prediction'],
        'prediction_transform': 'raw_unclipped'}
    validator.write_json(root / 'protocol.json', protocol)
    manifest = {'status': 'complete', 'protocol_sha256': validator.sha256(root / 'protocol.json'),
        'source_hashes': {str(Path(__file__).resolve()): validator.sha256(__file__)}, 'folds': {}}
    for fold, record in binding['folds'].items():
        frame = pd.read_parquet(record['prediction_path'], columns=[validator.ID, 'prediction_sec'])
        frame.to_parquet(root / f'{fold}.parquet', index=False)
        manifest['folds'][fold] = {'prediction': {'path': f'{fold}.parquet', 'sha256': validator.sha256(root / f'{fold}.parquet')},
            'split_hash': record['split_hash'], 'training_scope': 'all', 'cohorts': record['cohorts']['all'], 'prediction_scope': 'all'}
    validator.write_json(root / 'manifest.json', manifest)
    script = Path(validator.__file__).resolve()
    completed = subprocess.run([sys.executable, '-B', '-u', str(script), '--candidate', str(root), '--output', str(root / 'positive')], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = validator.read_json(root / 'positive/evaluation.json')
    assert evidence['seasonal']['baseline272']['delta_rmse'] == 0
    for fold in ('F1','F3'):
        assert evidence['folds'][fold]['changed_rows'] == 0
    # A forbidden label column must be rejected even if its file digest is valid.
    negative = root / 'forbidden_column'
    negative.mkdir()
    validator.write_json(negative / 'protocol.json', protocol)
    bad = dict(manifest)
    bad['folds'] = {key: dict(value) for key,value in manifest['folds'].items()}
    for fold in ('F1','F3'):
        frame = pd.read_parquet(root / f'{fold}.parquet')
        frame[validator.TARGET] = 0.
        frame.to_parquet(negative / f'{fold}.parquet', index=False)
        bad['folds'][fold]['prediction'] = {'path': f'{fold}.parquet', 'sha256': validator.sha256(negative / f'{fold}.parquet')}
    validator.write_json(negative / 'manifest.json', bad)
    rejected = subprocess.run([sys.executable, '-B', '-u', str(script), '--candidate', str(negative), '--output', str(negative / 'evaluation')], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'Candidate interchange must contain only IDs and predictions' in rejected.stderr
    # Separate paired-day arithmetic has an obvious known mean-loss improvement.
    y = np.zeros(20)
    before = np.ones(20)*2
    after = np.ones(20)
    check = validator.paired(y, before, after, np.repeat(['a','b','c','d'],5))
    assert check['delta_rmse'] == -1 and check['all_day_removals_improve']
    assert check['paired_day_bootstrap']['fraction_improving'] == 1
    validator.write_json(root / 'selfcheck.json', {'status': 'passed', 'source_sha256': validator.sha256(__file__),
        'evaluator_sha256': validator.sha256(script), 'full_baseline_noop': True,
        'hash_valid_forbidden_target_column_rejected': True, 'known_paired_day_arithmetic': True,
        'positive_receipt_sha256': validator.sha256(root / 'positive/evaluation.json')})
    print('EVALUATOR_SELFCHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()
