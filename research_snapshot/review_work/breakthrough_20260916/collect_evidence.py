"""Immutable full-cohort inventory, including named blends and failed attempts."""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import common

FOLDS = ['F1', 'F2', 'F3', 'G1']
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}


def infer_fold(record, path):
    fold = record.get('fold')
    if fold in FOLDS:
        return fold
    match = re.search(r'(?:^|[/_])(F[123]|G1)(?=[/_]|$)', path.parent.as_posix())
    return match.group(1) if match else None


def summarize(frame, reference, fold):
    for column in (common.ID, common.TARGET):
        np.testing.assert_array_equal(frame[column], reference[column])
    y = frame[common.TARGET].to_numpy(float)
    prediction = frame.prediction_sec.to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite complete-cohort prediction')
    baseline = reference.prediction_sec.to_numpy(float)
    proxy = reference.proxy_sec.to_numpy(float)
    missing = ~np.isfinite(proxy)
    source_gap = ~missing & (np.abs(y - proxy) > 1800)
    error, base_error = prediction - y, baseline - y
    squared, base_squared = error ** 2, base_error ** 2
    day = pd.DataFrame({'day': reference.day, 'sse': squared, 'base': base_squared}).groupby('day').agg(
        sse=('sse', 'sum'), base=('base', 'sum'), n=('sse', 'size'))
    drops = np.sqrt((squared.sum() - day.sse) / (len(y) - day.n)) - np.sqrt(
        (base_squared.sum() - day.base) / (len(y) - day.n))
    result = {'fold': fold, 'rows': len(y), 'rmse': float(np.sqrt(squared.mean())),
              'mae': float(np.abs(error).mean()), 'bias': float(error.mean()),
              'p99_absolute_error': float(np.quantile(np.abs(error), .99)),
              'reference_rmse': float(np.sqrt(base_squared.mean())),
              'support_projected_rmse': float(np.sqrt(np.mean((np.maximum(prediction, -12.) - y) ** 2))),
              'guarded_reference_rmse': float(np.sqrt(np.mean((np.maximum(baseline, -12.) - y) ** 2))),
              'day_removal_delta_min': float(drops.min()), 'day_removal_delta_max': float(drops.max()),
              'days_better': int((day.sse < day.base).sum()), 'days': len(day),
              'missing_rows': int(missing.sum()), 'missing_exact_reference': bool(np.array_equal(prediction[missing], baseline[missing])),
              'missing_rmse': float(np.sqrt(squared[missing].mean())),
              'missing_sse_share': float(squared[missing].sum() / squared.sum()),
              'source_gap_rows': int(source_gap.sum()),
              'source_gap_sse_share': float(squared[source_gap].sum() / squared.sum()),
              'largest_error_one_percent_sse_share': float(np.sort(squared)[-int(np.ceil(.01 * len(y))):].sum() / squared.sum())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=common.WORKSPACE / 'private_runs/breakthrough_20260916')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = common.external_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    references = {fold: common.reference(fold)[0] for fold in FOLDS}
    rows, attempts, skipped, receipts = [], [], [], {}
    # Snapshot the manifest path list before output creation can affect discovery.
    paths = sorted(args.root.rglob('manifest.json'))
    for path in paths:
        try:
            record = common.read_json(path)
        except (json.JSONDecodeError, OSError) as error:
            skipped.append({'path': str(path), 'reason': str(error)})
            continue
        fold = infer_fold(record, path)
        if fold not in FOLDS:
            continue
        recipe = re.sub(r'(^|[/_])(F[123]|G1)(?=[/_]|$)', r'\1{fold}', path.parent.relative_to(args.root).as_posix())
        if record.get('status') != 'complete':
            attempts.append({'recipe': recipe, 'fold': fold, 'status': record.get('status'),
                             'manifest': str(path), 'error': record.get('error')})
            continue
        accepted = 0
        for filename, digest in record.get('outputs', {}).items():
            prediction_path = path.parent / filename
            if prediction_path.suffix != '.parquet':
                continue
            schema = pq.read_schema(prediction_path)
            if not {common.ID, common.TARGET, 'prediction_sec'}.issubset(schema.names):
                continue
            if common.sha256(prediction_path) != digest:
                raise ValueError(f'Prediction artifact changed: {prediction_path}')
            frame = pd.read_parquet(prediction_path, columns=[common.ID, common.TARGET, 'prediction_sec'])
            if len(frame) != len(references[fold]):
                skipped.append({'path': str(prediction_path), 'reason': 'Not original complete score cohort'})
                continue
            metrics = summarize(frame, references[fold], fold)
            metrics.update(recipe=recipe, variant=prediction_path.stem,
                family=record.get('family', 'composition_or_specialist'),
                formulation=record.get('formulation', 'see_manifest'), seed=record.get('seed'),
                runtime_sec=record.get('runtime_sec'), peak_rss_bytes=record.get('peak_rss_bytes'),
                manifest=str(path), manifest_sha256=common.sha256(path), prediction_sha256=digest)
            rows.append(metrics)
            accepted += 1
        if accepted:
            receipts[str(path)] = common.sha256(path)
    if not rows:
        raise ValueError('No completed full-cohort predictions')
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'folds.csv', index=False)
    seasonal = []
    for (recipe, variant), grouped in frame.groupby(['recipe', 'variant']):
        scores = grouped[grouped.fold.isin(WEIGHTS)].set_index('fold')
        if set(scores.index) != set(WEIGHTS) or not scores.index.is_unique:
            continue
        row = {'recipe': recipe, 'variant': variant}
        for column in ['rmse', 'reference_rmse', 'support_projected_rmse', 'guarded_reference_rmse']:
            row[column] = float(np.sqrt(sum(WEIGHTS[f] * scores.loc[f, column] ** 2 for f in WEIGHTS)))
        for column in ['mae', 'bias']:
            row[column] = float(sum(WEIGHTS[f] * scores.loc[f, column] for f in WEIGHTS))
        row.update(F1_rmse=scores.loc['F1', 'rmse'], F3_rmse=scores.loc['F3', 'rmse'],
            seasonal_gain=row['reference_rmse'] - row['rmse'],
            gain_after_common_support_floor=row['guarded_reference_rmse'] - row['support_projected_rmse'],
            both_months_improve=bool((scores.rmse < scores.reference_rmse).all()),
            every_day_removal_improves=bool((scores.day_removal_delta_max < 0).all()),
            missing_exact_reference=bool(scores.missing_exact_reference.all()),
            missing_regression_over5pct=bool(any(scores.loc[f, 'missing_rmse'] >
                np.sqrt(references[f].loc[~np.isfinite(references[f].proxy_sec), 'squared_error'].mean()) * 1.05 for f in WEIGHTS)))
        for fold in ['F2', 'G1']:
            selected = grouped[grouped.fold.eq(fold)]
            row[fold + '_rmse'] = None if selected.empty else float(selected.iloc[0].rmse)
        seasonal.append(row)
    table = pd.DataFrame(seasonal).sort_values('rmse')
    table.to_csv(output / 'seasonal.csv', index=False)
    common.write_json(output / 'inventory.json', {'created_utc': common.utc_now(), 'script_sha256': common.sha256(__file__),
        'source_root': str(args.root), 'completed_prediction_records': len(rows), 'seasonal_recipe_variants': len(table),
        'manifest_receipts': receipts, 'incomplete_attempts': attempts, 'skipped': skipped,
        'limits': 'Exposed adaptive development evidence; rows are variants/folds, not independent discoveries. Oracle files excluded unless they have complete model manifests and predictions; no oracle in current model registry. Support floor is a diagnostic and affects zero existing V2 ranking rows.',
        'weights': WEIGHTS, 'support_floor': -12,
        'metric_definition': 'RMSE from unchanged full original scoring labels; day removal is sensitivity vs frozen reference; source-gap grouping is label-aware diagnostic only.'})
    print(table[['recipe', 'variant', 'rmse', 'F2_rmse', 'G1_rmse']].head(25).to_string(index=False), flush=True)
    print('INVENTORY', len(rows), 'prediction records', len(table), 'seasonal variants', len(attempts), 'unfinished attempts', flush=True)


if __name__ == '__main__':
    main()
