"""Final inventory and decision cross-check against immutable completed evidence."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import audit

BASE = audit.BASE
INVENTORY = BASE / 'inventory_final_1402'
DECISION = BASE / 'decision_draft_1403.json'
OUT = audit.OUT / 'final_consistency_review.json'
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve prior review'
    decision = audit.read_json(DECISION)
    decision_hash = audit.sha256(DECISION)
    inventory = audit.read_json(INVENTORY / 'inventory.json')
    outputs = audit.read_json(INVENTORY / 'outputs.json')
    for filename, digest in outputs['hashes'].items():
        assert audit.sha256(INVENTORY / filename) == digest, filename
    folds = pd.read_csv(INVENTORY / 'folds.csv')
    seasonal = pd.read_csv(INVENTORY / 'seasonal.csv')
    assert len(folds) == inventory['completed_prediction_records']
    assert len(seasonal) == inventory['seasonal_recipe_variants']
    assert seasonal.evidence_role.value_counts().to_dict() == inventory['classifications']
    assert not folds.duplicated(['recipe', 'variant', 'fold']).any()
    assert not seasonal.duplicated(['recipe', 'variant']).any()
    for path, digest in inventory['manifest_receipts'].items():
        assert audit.sha256(path) == digest, path
    for row in folds.itertuples():
        path = Path(row.manifest)
        assert audit.sha256(path.parent / (row.variant + '.parquet')) == row.prediction_sha256
        assert inventory['manifest_receipts'][row.manifest] == row.manifest_sha256
    for row in seasonal.itertuples():
        selected = folds[(folds.recipe == row.recipe) & (folds.variant == row.variant) & folds.fold.isin(WEIGHTS)]
        assert set(selected.fold) == set(WEIGHTS) and len(selected) == 2
        by_fold = selected.set_index('fold')
        close(row.rmse, np.sqrt(sum(weight * by_fold.loc[fold, 'rmse']**2 for fold, weight in WEIGHTS.items())))
        close(row.F1_rmse, by_fold.loc['F1', 'rmse'])
        close(row.F3_rmse, by_fold.loc['F3', 'rmse'])
    assert decision['evidence_cutoff_utc'] == inventory['created_utc']
    if decision['completed_window']:
        from datetime import datetime, timezone
        assert datetime.fromisoformat(decision['created_utc']) >= datetime(2026, 9, 16, 14, 13, tzinfo=timezone.utc)
    selected_rows = list(decision['comparison_rows']) + [
        {'label': 'Retained airport9 composition', 'recipe': 'missing/route_composition_v4/airport9/{fold}', 'variant': 'candidate'},
        {'label': 'Retained airport9 ordinary blend', 'recipe': 'models/context_gate/final_simplex9_v1/{fold}_score', 'variant': 'airport9'}]
    claims = []
    refs = {fold: audit.common.reference(fold)[0] for fold in WEIGHTS}
    for item in selected_rows:
        table = seasonal[(seasonal.recipe == item['recipe']) & (seasonal.variant == item['variant'])]
        assert len(table) == 1 and table.iloc[0].evidence_role == 'experimental_predictor', item
        metrics = {}
        for fold in WEIGHTS:
            subset = folds[(folds.recipe == item['recipe']) & (folds.variant == item['variant']) & folds.fold.eq(fold)]
            assert len(subset) == 1
            row = subset.iloc[0]
            folder = BASE / item['recipe'].replace('{fold}', fold)
            manifest = audit.read_json(folder / 'manifest.json')
            assert manifest['status'] == 'complete'
            path = folder / (item['variant'] + '.parquet')
            assert manifest['outputs'][path.name] == audit.sha256(path) == row.prediction_sha256
            frame = pd.read_parquet(path, columns=[audit.ID, audit.TARGET, 'prediction_sec'])
            np.testing.assert_array_equal(frame[audit.ID], refs[fold][audit.ID])
            np.testing.assert_array_equal(frame[audit.TARGET], refs[fold][audit.TARGET])
            error = frame.prediction_sec.to_numpy(float) - frame[audit.TARGET].to_numpy(float)
            assert np.isfinite(error).all()
            rmse = float(np.sqrt(np.mean(error**2)))
            close(rmse, row.rmse)
            close(np.abs(error).mean(), row.mae)
            close(error.mean(), row.bias)
            metrics[fold] = {'rows': len(frame), 'rmse': rmse, 'manifest_sha256': audit.sha256(folder / 'manifest.json')}
        value = float(np.sqrt(sum(weight * metrics[fold]['rmse']**2 for fold, weight in WEIGHTS.items())))
        close(value, table.iloc[0].rmse)
        claims.append({**item, 'folds': metrics, 'seasonal_rmse': value})
    final9 = audit.read_json(BASE / 'models/context_gate/final_simplex9_v1/score_summary.json')
    final9check = audit.read_json(BASE / 'models/context_gate/final_simplex9_v1/independent_verification.json')
    assert final9check['status'] == 'passed' and final9check['score_summary_sha256'] == audit.sha256(BASE / 'models/context_gate/final_simplex9_v1/score_summary.json')
    v4 = audit.read_json(BASE / 'missing/route_composition_v4/summary.json')
    v4check = audit.read_json(BASE / 'missing/route_composition_v4/verification.json')
    budget = audit.read_json(BASE / 'missing/route_composition_v4/error_budget/summary.json')
    assert v4['status'] == 'complete' and v4check['status'] == budget['status'] == 'passed'
    v4gains = {}
    for variant in ('global9', 'airport9'):
        values = v4['variants'][variant]['seasonal_rmse']
        matching = next(item for item in claims if item['recipe'] == f'missing/route_composition_v4/{variant}/{{fold}}')
        close(matching['seasonal_rmse'], values['composed'])
        close(values['own_ordinary'], final9['seasonal_rmse'][variant])
        close(values['composed'], budget['variants'][variant]['seasonal_rmse'])
        assert v4['variants'][variant]['every_day_removal_improves_vs_previous']
        v4gains[variant] = values['previous_composition'] - values['composed']
        assert v4gains[variant] > 2
        for fold in WEIGHTS:
            receipt = v4check['folds'][fold][variant]
            folder = BASE / f'missing/route_composition_v4/{variant}/{fold}'
            assert receipt['manifest_sha256'] == audit.sha256(folder / 'manifest.json')
            assert receipt['candidate_sha256'] == audit.sha256(folder / 'candidate.parquet')
            assert receipt['full_coefficient_and_route_replay_exact']
        b = budget['variants'][variant]
        close(sum(v['additive_rmse_gain_seconds'] for v in b['route_gains_vsV2'].values()), values['V2'] - values['composed'])
        for target in ('230', '250'):
            close(b['goals'][target]['required_mse_reduction_pct'], 100 * (1 - float(target)**2 / b['seasonal_mse']))
    assert decision['selected']['recipe'] == 'missing/route_composition_v4/global9/{fold}'
    assert decision['selected']['variant'] == 'candidate'
    assert all((audit.ROOT / path).is_file() for path in decision['supplemental_evidence'])
    research = audit.read_json(audit.ROOT / 'output/breakthrough_20260916/research_gap/sources_final_1402.json')
    research_count = 0
    for experiment in research['experiments']:
        if 'seasonal_rmse' not in experiment:
            continue
        for fold, value in experiment['folds'].items():
            assert value['status'] == 'complete'
            assert audit.sha256(audit.ROOT / value['path']) == value['sha256']
        for variant, value in experiment['seasonal_rmse'].items():
            close(value, np.sqrt(sum(weight * experiment['folds'][fold]['metrics'][variant]['rmse_sec']**2 for fold, weight in WEIGHTS.items())))
        research_count += 1
    assert audit.sha256(DECISION) == decision_hash
    report = {'status': 'passed', 'created_utc': audit.utc_now(), 'source_sha256': audit.sha256(__file__),
        'inventory_sha256': audit.sha256(INVENTORY / 'inventory.json'), 'inventory_outputs_sha256': audit.sha256(INVENTORY / 'outputs.json'),
        'decision_sha256': audit.sha256(DECISION), 'decision_path': str(DECISION),
        'completed_prediction_records': len(folds), 'seasonal_recipe_variants': len(seasonal),
        'classifications': inventory['classifications'], 'comparison_rows_verified': claims,
        'v4_gain_vs_previous_composition': v4gains, 'research_complete_pairs': research_count,
        'evidence_cutoff_utc': inventory['created_utc'], 'runtime_sec': time.monotonic() - started,
        'scope': 'All consolidated counts, every fold prediction digest and manifest receipt, every seasonal RMSE arithmetic; all decision comparison predictions independently recomputed on exact original rows; final9 and V4 receipts/hash/score/gain/error-budget consistency.',
        'limitations': ['Variants and folds are not independent discoveries.', 'Reporting eligibility does not waive XGBoost or GRU replay failures.',
            'Exposed development scores, not official results; no ranking fit or new submission verified.', 'Review binds this decision draft; later narrative edits require separate check.']}
    audit.write_json(OUT, report)
    print('FINAL_CONSISTENCY_PASSED', len(folds), len(seasonal), inventory['classifications'], 'researchpairs', research_count, flush=True)


if __name__ == '__main__':
    main()
