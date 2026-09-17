"""Independent completed leaf63 artifact/cohort and support attribution audit."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
sys.path.insert(0, str(HERE.parent / 'stacking'))
import common
from run_simplex_v2 import day_sensitivity
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET, MOVEMENT
from taxiout.metrics import season_score

OUT = ROOT / 'private_runs/breakthrough_20260916/models/deeper_audit'
BASE = ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined'


def summary(first, second, mask):
    n = int(mask.sum())
    a, b = first[mask].sum(), second[mask].sum()
    return {'n': n, 'reference_sse': float(a), 'candidate_sse': float(b), 'sse_gain': float(a-b),
            'reference_rmse': float(np.sqrt(a/n)) if n else None,
            'candidate_rmse': float(np.sqrt(b/n)) if n else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(path) == read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta = pd.read_parquet(path, columns=[ID, TARGET, MOVEMENT, 'FLIGHT_ID_mvt', 'ADEP_mvt', 'proxy_sec'])
    finite = np.isfinite(meta.proxy_sec.to_numpy())
    results = {}
    for fold in ['F1', 'F3']:
        folder = BASE / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'
        record = read_json(folder / 'manifest.json')
        assert record['status'] == 'complete'
        for name, digest in record['outputs'].items():
            assert sha256(folder / name) == digest, name
        snapshot = BASE / 'source_snapshots/lightgbm_leaf63_aobt_allfinite_s20260916'
        for name, digest in record['source_hashes'].items():
            assert sha256(snapshot / Path(name).name) == digest, name
        for receipt in record['anchor']['feature_receipts']:
            source = receipt.get('path', receipt.get('manifest'))
            digest = receipt.get('sha256', receipt.get('manifest_sha256'))
            assert source is not None and digest is not None
            assert sha256(source) == digest
        idx, split, _ = common.fold_data(meta, fold, full=True)
        assert object_hash(split) == object_hash(record['split'])
        for stage, positions in idx.items():
            selected = positions[finite[positions]]
            assert record['fit_ids'][stage] == {'n': len(selected), 'hash': object_hash(meta.iloc[selected][ID].tolist())}
        assert record['fit']['rows'] == record['fit_ids']['fit']['n']
        assert record['refit']['rows'] == record['fit_ids']['refit']['n']
        assert record['fit']['steps'] == record['refit']['steps']
        assert record['reload_max_abs_delta'] == 0.
        tune = pd.read_parquet(folder / 'tune_predictions.parquet')
        np.testing.assert_array_equal(tune[ID], meta.iloc[idx['tune'][finite[idx['tune']]]][ID])
        reference, refrecord = common.reference(fold)
        candidate = pd.read_parquet(folder / 'candidate.parquet')
        np.testing.assert_array_equal(candidate[ID], reference[ID])
        np.testing.assert_array_equal(candidate[TARGET], reference[TARGET])
        np.testing.assert_array_equal(candidate[ID], meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(candidate[TARGET], meta.iloc[idx['score']][TARGET])
        local = meta.iloc[idx['score']]
        proxy = local.proxy_sec.to_numpy(float)
        y = candidate[TARGET].to_numpy(float)
        first = np.square(reference.prediction_sec.to_numpy()-y)
        second = np.square(candidate.prediction_sec.to_numpy()-y)
        np.testing.assert_allclose(second, candidate.squared_error, rtol=0., atol=0.)
        masks = {'missing': ~np.isfinite(proxy), 'negative': np.isfinite(proxy) & (proxy < 0),
                 'ordinary': np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200),
                 'long': np.isfinite(proxy) & (proxy > 7200)}
        np.testing.assert_array_equal(candidate.prediction_sec.to_numpy()[masks['missing']], reference.prediction_sec.to_numpy()[masks['missing']])
        row = {'status': 'passed', 'manifest_sha256': sha256(folder / 'manifest.json'),
               'complete_score_n': len(candidate), 'fit_ids': record['fit_ids'], 'steps': record['fit']['steps'],
               'replay_delta': record['reload_max_abs_delta'], 'all_output_hashes_verified': True,
               'overall': summary(first, second, np.ones(len(first), bool)),
               'support': {name: summary(first, second, mask) for name, mask in masks.items()},
               'label_slices': {name: summary(first, second, mask) for name, mask in {
                   'negative': y < 0, '0_to_7200': (y>=0)&(y<=7200), 'over_7200': y>7200, 'over_86400': y>86400}.items()},
               'day_removals': day_sensitivity(reference, candidate),
               'missing_support_oracle_floor_rmse': float(np.sqrt(second[masks['missing']].sum()/len(second))),
               'missing_sse_share': float(second[masks['missing']].sum()/second.sum()),
               'top_gain_rows': []}
        order = np.argsort(first-second)[::-1]
        for i in order[:10]:
            row['top_gain_rows'].append({'id': str(candidate.iloc[i][ID]), 'label': float(y[i]), 'proxy': float(proxy[i]) if np.isfinite(proxy[i]) else None,
                                         'gain': float(first[i]-second[i])})
        row['remove_largest_gain_rows'] = {}
        for count in [1, 2, 5, 10]:
            keep = np.ones(len(first), bool)
            keep[order[:count]] = False
            row['remove_largest_gain_rows'][str(count)] = summary(first, second, keep)
        row['month_rmse_matches_manifest'] = bool(np.isclose(row['overall']['candidate_rmse'], record['reports']['candidate']['metrics']['overall']['rmse_sec']))
        old_folder = ROOT / 'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T' / f'lightgbm_aobt_allfinite_{fold}_s20260916'
        old_record = read_json(old_folder / 'manifest.json')
        assert sha256(old_folder / 'candidate.parquet') == old_record['outputs']['candidate.parquet']
        old = pd.read_parquet(old_folder / 'candidate.parquet')
        np.testing.assert_array_equal(old[ID], candidate[ID])
        row['versus_lgb600'] = {'overall': summary(old.squared_error.to_numpy(), second, np.ones(len(second), bool)),
                               'day_removals': day_sensitivity(old, candidate)}
        assert row['month_rmse_matches_manifest']
        results[fold] = row
        print('AUDITED', fold, row['overall'], 'missing_floor', row['missing_support_oracle_floor_rmse'], flush=True)
    output = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'folds': results,
              'seasonal_rmse': season_score(*[{'n': results[f]['complete_score_n'], 'sse': results[f]['overall']['candidate_sse'], 'rmse_sec': results[f]['overall']['candidate_rmse']} for f in ['F1','F3']]),
              'caveat': 'Exposed development audit; row/label removals are diagnostics only, never modified fit/score cohorts or selection.',
              'current_imported_encoder_source': {'path': str(ROOT / 'review_work/campaign_20260916/lgbm_adapter.py'),
                  'sha256': sha256(ROOT / 'review_work/campaign_20260916/lgbm_adapter.py')},
              'provenance_limit': 'Frozen runner imports NativeFrameEncoder from campaign lgbm_adapter.py, which its source_hashes omit; adapter source inspected separately, original hash cannot be retroactively proven.'}
    write_json(OUT / 'audit.json', output)
    print('SEASONAL', output['seasonal_rmse'], flush=True)


if __name__ == '__main__':
    main()
