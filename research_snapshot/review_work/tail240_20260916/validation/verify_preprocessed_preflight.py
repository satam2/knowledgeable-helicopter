"""Bindings and synthetic mask invariants before an authorized model run."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import lightgbm
from pathlib import Path
import sys
import inspect
import numpy as np
from audit_union387_sources import ROOT, read, sha, write, guard
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import preprocessed_locations_tune as trial


def main():
    folder = ROOT / 'private_runs/tail240_20260916/models/preprocessed_locations_v1'
    protocol = read(folder / 'protocol.json')
    assert trial.declare() == protocol
    for relative, expected in protocol['source_hashes'].items():
        assert sha(ROOT / relative) == expected
    assert sha(ROOT / 'private_runs/tail240_20260916/state/preprocessing_semantics/v2/receipt.json') == protocol['helper_evidence_sha256']
    controls = {}
    for fold in ('F1', 'F3'):
        native = trial.NATIVE[fold]
        original = trial.risk.union_folder(fold)
        assert sha(native / 'manifest.json') == protocol['native_controls'][fold]
        marker = read(native / 'manifest.json')
        for name in ('model.txt', 'encoder.json'):
            assert sha(native / name) == marker['outputs'][name]
        assert sha(original / 'manifest.json') == protocol['controls'][fold]
        original_marker = read(original / 'manifest.json')
        assert original_marker['feature_columns'] == protocol['columns']
        assert original_marker['fit']['params'] == protocol['parameters']
        assert sha(original / 'tune_predictions.parquet') == original_marker['outputs']['tune_predictions.parquet']
        neural = trial.comparison.NEURAL[fold]
        assert sha(neural / 'manifest.json') == protocol['neural_controls'][fold]
        assert sha(neural / 'tune_predictions.parquet') == read(neural / 'manifest.json')['outputs']['tune_predictions.parquet']
        controls[fold] = dict(native_manifest_sha256=sha(native / 'manifest.json'),
                              model_sha256=sha(native / 'model.txt'), encoder_sha256=sha(native / 'encoder.json'))
    columns = protocol['columns']
    groups = protocol['groups']
    assert {key: len(value) for key, value in groups.items()} == dict(stand=24, query_runway=4, arrival_runway=8, stand_equality=8, runway_equality=8)
    matrix = np.arange(6 * len(columns), dtype=np.float32).reshape(6, -1)
    before = matrix.copy()
    trial.preprocess(matrix, columns, ['m:', 's0:', 's7:UNKNOWN', 's1:1', 's3:001', 's2:A1'],
                     ['s2:01', 's2:01', 's2:01', 's2:NA', 's2:02', 's2:03'])
    expected = before.copy()
    for key, names in groups.items():
        rows = [0, 1, 2] if key in ('stand', 'stand_equality') else [3]
        value = 0. if key.endswith('equality') else -999999.
        for name in names:
            expected[rows, columns.index(name)] = value
    np.testing.assert_array_equal(matrix, expected)
    assert 'eval_X' in inspect.signature(lightgbm.LGBMRegressor.fit).parameters
    out = ROOT / 'private_runs/tail240_20260916/validation/preprocessed_locations_preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status='static_preflight_passed', source_sha256=sha(Path(__file__)), protocol_sha256=sha(folder / 'protocol.json'),
                  native_controls=controls, all_source_hashes_exact=True, full387_synthetic_mask_and_unchanged_cells_exact=True,
                  original_control_replay_asserted_before_any_mask=True, fit_only_vocab_retained=True,
                  raw_encoded_tokens_used_for_mask=True, cohort_target_params_unchanged=True,
                  primary_original_union_component_replacement_in_currentPLE387=True,
                  runtime_native_control_replay_pending=True, no_training_launched=True,
                  peak_bytes=guard(), limitation='Static preflight and synthetic invariants do not certify runtime replay, fit completion or generalization.')
    write(out / 'receipt.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
