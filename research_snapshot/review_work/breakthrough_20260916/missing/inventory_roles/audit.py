"""Verify exact saved missing-route variants before recommending classification."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = BASE / 'missing/inventory_roles'
RECIPES = {
    'feature_screens/models/lightgbm_direct_linkage_full_{fold}_s20260916',
    'feature_screens/models/lightgbm_direct_trajectory_full_{fold}_s20260916',
    'physical_v2/models/surfacegeometry_{fold}_s20260916',
    'physical_v2/models/weather_{fold}_s20260916',
    'physical_v2/models/basephysicalfull_{fold}_s20260916',
}
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert not OUT.exists()
    inventory = BASE / 'inventory_v2_1205/seasonal.csv'
    table = pd.read_csv(inventory)
    rows = table[table.evidence_role.eq('unclassified_review_required')]
    assert len(rows) == 10 and set(rows.recipe) == RECIPES
    assert set(rows.variant) == {'missing_only', 'missing_blend25'}
    evidence = []
    for fold in ['F1', 'F3']:
        directory = ROOT / 'private_runs/next_230/models' / f'clock_and_rome_ensemble_{fold}_s20260910'
        manifest = json.loads((directory / 'manifest.json').read_text())
        path = directory / 'score_predictions.parquet'
        assert sha(path) == manifest['outputs'][path.name]
        reference = pd.read_parquet(path)
        identity = 'MVT_ID_mvt'
        missing = ~np.isfinite(reference.proxy_sec.to_numpy(float))
        ref = reference.prediction_sec.to_numpy(float)
        for recipe in sorted(RECIPES):
            folder = BASE / recipe.format(fold=fold)
            record = json.loads((folder / 'manifest.json').read_text())
            assert record['status'] == 'complete'
            frames = {}
            for variant in ['candidate', 'missing_only', 'missing_blend25']:
                path = folder / (variant + '.parquet')
                assert sha(path) == record['outputs'][path.name]
                frame = pd.read_parquet(path)
                np.testing.assert_array_equal(frame[identity], reference[identity])
                frames[variant] = frame.prediction_sec.to_numpy(float)
            expected = ref.copy()
            expected[missing] = frames['candidate'][missing]
            np.testing.assert_array_equal(frames['missing_only'], expected)
            blend = (0.75 * ref + 0.25 * expected if recipe.startswith('feature_screens/')
                     else ref + 0.25 * (expected - ref))
            np.testing.assert_array_equal(frames['missing_blend25'], blend)
            evidence.append(dict(recipe=recipe, fold=fold, manifest_sha256=sha(folder/'manifest.json'),
                complete_rows=len(ref), missing_rows=int(missing.sum()),
                missing_only_exact=True, missing_blend25_exact=True,
                finite_route_max_abs_delta=float(np.max(np.abs(blend[~missing]-ref[~missing])))))
    OUT.mkdir(parents=True)
    result = dict(status='passed', source_sha256=sha(__file__), inventory_sha256=sha(inventory),
        classified_seasonal_rows=10, recommendation='experimental_predictor', evidence=evidence,
        no_training_or_inference=True, no_collector_changes=True)
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
