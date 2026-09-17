"""Check exact cohorts and JSON representation of frozen split evidence."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import object_hash, sha256
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, TARGET


def differences(left, right, path=''):
    if isinstance(left, dict) and isinstance(right, dict):
        return sum([differences(left[k], right[k], path + '/' + k) for k in left], [])
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return sum([differences(a, b, path + '/' + str(i)) for i, (a, b) in enumerate(zip(left, right))], [])
    if left != right or type(left) is not type(right):
        return [{'path': path, 'live_type': type(left).__name__, 'saved_type': type(right).__name__,
                 'live_value': str(left), 'saved_value': str(right), 'string_values_equal': str(left) == str(right)}]
    return []


def main():
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    meta = pd.read_parquet(path, columns=[ID, FLIGHT_ID, MOVEMENT, TARGET, 'proxy_sec'])
    controls = {
        'missing': ROOT / 'private_runs/breakthrough_20260916/missing/id_context_v1/models',
        'finite': ROOT / 'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'}
    results = {}
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        for name, folder in controls.items():
            leaf = f'historical_template_{fold}_s20260916' if name == 'missing' else f'lightgbm_aobt_allfinite_{fold}_s20260916'
            manifest = json.loads((folder / leaf / 'manifest.json').read_text())
            eligible = np.isfinite(meta.proxy_sec.to_numpy(float))
            if name == 'missing':
                eligible = ~eligible
            identities = {stage: {'n': int(eligible[rows].sum()), 'hash': object_hash(meta.iloc[rows[eligible[rows]]][ID].tolist())}
                          for stage, rows in idx.items()}
            result = {'cohort_ids_equal': identities == manifest['fit_ids'],
                'direct_split_equal': split == manifest['split'],
                'canonical_split_hash_equal': object_hash(split) == object_hash(manifest['split']),
                'live_split_hash': object_hash(split), 'saved_split_hash': object_hash(manifest['split']),
                'differences': differences(split, manifest['split']), 'cohorts': identities}
            assert result['cohort_ids_equal'] and result['canonical_split_hash_equal']
            results[name + '_' + fold] = result
    output = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit/split_equality_diagnostic.json'
    assert not output.exists()
    output.write_text(json.dumps({'source_sha256': sha256(__file__), 'results': results}, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
