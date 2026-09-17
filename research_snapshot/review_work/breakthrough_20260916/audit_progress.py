"""Read-only complete-cohort comparison of completed exploratory predictions."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import common

WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}


def audit(root, output):
    refs = {fold: common.reference(fold)[0] for fold in WEIGHTS}
    records = []
    details = {}
    for path in sorted(root.rglob('manifest.json')):
        manifest = common.read_json(path)
        fold = manifest.get('fold')
        if fold is None:
            match=re.search(r'(?:^|_)(F[13])(?:_|$)',path.parent.name)
            fold=match.group(1) if match else None
        if manifest.get('status') != 'complete' or fold not in WEIGHTS:
            continue
        ref = refs[fold]
        y = ref[common.TARGET].to_numpy(float)
        base = ref.prediction_sec.to_numpy(float)
        guarded = np.maximum(base, -12.)
        missing = ~np.isfinite(ref.proxy_sec.to_numpy(float))
        gap = ~missing & (np.abs(y - ref.proxy_sec.to_numpy(float)) > 1800)
        groups = {'all': np.ones(len(ref), bool), 'missing': missing,
                  'large_source_gap': gap, 'ordinary': ~(missing | gap),
                  'baseline_below_fit_support': base < -12}
        for variant in ('candidate', 'blend25'):
            prediction_path = path.parent / (variant + '.parquet')
            if not prediction_path.exists():
                continue
            if prediction_path.name in manifest.get('outputs', {}):
                assert common.sha256(prediction_path) == manifest['outputs'][prediction_path.name]
            frame = pd.read_parquet(prediction_path)
            assert np.array_equal(frame[common.ID], ref[common.ID])
            assert np.array_equal(frame[common.TARGET], ref[common.TARGET])
            pred = frame.prediction_sec.to_numpy(float)
            assert np.isfinite(pred).all()
            recipe = re.sub(r'_F[13](?=_|$)', '', str(path.parent.relative_to(root)))
            name = recipe + '/' + variant
            row = {'recipe': recipe, 'variant': variant, 'fold': fold,
                   'n': len(y), 'rmse': float(np.sqrt(np.mean((pred-y)**2))),
                   'mae': float(np.mean(np.abs(pred-y))), 'bias': float(np.mean(pred-y)),
                   'support_projected_rmse': float(np.sqrt(np.mean((np.maximum(pred,-12)-y)**2))),
                   'reference_rmse': float(np.sqrt(np.mean((base-y)**2))),
                   'guarded_reference_rmse': float(np.sqrt(np.mean((guarded-y)**2))),
                   'manifest': str(path), 'manifest_sha256': common.sha256(path)}
            for group, mask in groups.items():
                row[group+'_rows'] = int(mask.sum())
                row[group+'_mse_gain'] = float(np.sum(((base-y)**2-(pred-y)**2)[mask])/len(y))
                row[group+'_mse_gain_after_guard'] = float(np.sum(((guarded-y)**2-(np.maximum(pred,-12)-y)**2)[mask])/len(y))
            daily = pd.DataFrame({'day': ref['day'], 'reference_sse': (base-y)**2,
                                  'guard_sse': (guarded-y)**2, 'candidate_sse': (pred-y)**2,
                                  'candidate_guard_sse': (np.maximum(pred,-12)-y)**2, 'n': 1}).groupby('day').sum()
            details[name + '/' + fold] = daily.reset_index().to_dict('records')
            records.append(row)
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError('No completed predictions found')
    season = []
    for (recipe, variant), rows in table.groupby(['recipe', 'variant']):
        if set(rows.fold) != set(WEIGHTS):
            continue
        rows = rows.set_index('fold')
        row = {'recipe': recipe, 'variant': variant}
        for metric in ('rmse', 'reference_rmse', 'guarded_reference_rmse', 'support_projected_rmse'):
            row[metric] = float(np.sqrt(sum(WEIGHTS[f]*rows.loc[f, metric]**2 for f in WEIGHTS)))
        for column in table:
            if '_mse_gain' in column:
                row[column] = float(sum(WEIGHTS[f]*rows.loc[f, column] for f in WEIGHTS))
        row['gain_sec'] = row['reference_rmse']-row['rmse']
        row['gain_after_guard_sec'] = row['guarded_reference_rmse']-row['support_projected_rmse']
        season.append(row)
    output = common.external_path(output)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output/'folds.csv', index=False)
    seasonal = pd.DataFrame(season).sort_values('rmse')
    seasonal.to_csv(output/'seasonal.csv', index=False)
    common.write_json(output/'daily.json', details)
    common.write_json(output/'protocol.json', {'created_utc': common.utc_now(), 'script_sha256':common.sha256(__file__),
        'diagnostic_only': True, 'score_folds_previously_exposed': True,
        'floor': -12, 'floor_origin':'minimum label within both original fit cohorts, independently verified support_guard',
        'floor_ranking_effect':'zero rows affected in existing V2 ranking predictions',
        'sources': str(root), 'records': len(records), 'seasonal_recipes': len(season)})
    print(seasonal[['recipe','variant','rmse','gain_sec','gain_after_guard_sec']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=common.WORKSPACE/'private_runs/breakthrough_20260916')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    audit(args.root,args.output)
