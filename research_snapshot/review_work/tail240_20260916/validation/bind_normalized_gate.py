"""Bind the independently verified tune gate to the frozen refit interface."""
from pathlib import Path
import validate_candidate as validation

ROOT=validation.ROOT
directory=ROOT/'private_runs/tail240_20260916/validation/normalized_missing_tune_v1'
receipt_path=directory/'receipt.json'
receipt=validation.read_json(receipt_path)
assert receipt['status']=='passed_validation' and receipt['both_point_gates'] and receipt['both_day_removal_gates']
producer=ROOT/'review_work/tail240_20260916/models/normalized_missing_tune.py'
summary=ROOT/'private_runs/tail240_20260916/models/normalized_missing_tune_v1/summary.json'
assert receipt['producer_summary_sha256']==validation.sha256(summary)
for fold in ('F1','F3'):
    record=receipt['folds'][fold]
    assert record['point_gate'] and record['every_day_removal_gate']
    assert all(item['native_replay_max_abs_delta']==0 for item in record['models'].values())
    assert all(item['all_day_removals_improve'] and item['delta_rmse']<0 for item in record['missing_comparisons'].values())
target=directory/'refit_gate.json'
assert not target.exists()
validation.write_json(target,{'status':'passed','advance_normalized_refit':True,
    'source_sha256':validation.sha256(producer),'summary_sha256':validation.sha256(summary),
    'independent_validation_receipt_sha256':validation.sha256(receipt_path),
    'gate_adapter_source_sha256':validation.sha256(__file__),
    'decision':'Declared tune-only gate passes both controls, both months, and every one-day removal. Permit separately declared refit, not promotion.',
    'remaining_risk':'Adaptive exposed development. October versus unscaled advantage reverses after top5beneficial-row removal; versus HistoricalTemplate survives top10removal by only0.704seconds. No score result yet.'})
print(target,validation.sha256(target))
