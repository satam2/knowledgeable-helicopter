"""Full-cohort paired evaluation with independently loaded original labels."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET

BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
FORBIDDEN = {TARGET, 'BLOCK_TIME_UTC_mvt', 'error_sec', 'squared_error', 'score_tail', 'target_tail'}


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if peak > 4 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Validation memory reserve exceeded')
    return int(peak)


def metrics(y, p):
    if not len(y):
        return {'n': 0}
    error = np.asarray(p, float) - np.asarray(y, float)
    return {'n': len(error), 'rmse': float(np.sqrt(np.mean(error**2))), 'mse': float(np.mean(error**2)),
        'mae': float(np.mean(np.abs(error))), 'bias': float(np.mean(error)),
        'p95_abs_error': float(np.quantile(np.abs(error), .95)),
        'p99_abs_error': float(np.quantile(np.abs(error), .99)), 'sse': float(np.sum(error**2))}


def paired(y, control, candidate, days):
    base_error, error = (control-y)**2, (candidate-y)**2
    gain = base_error - error
    grouped = pd.DataFrame({'day': days, 'base': base_error, 'candidate': error}).groupby('day').agg(
        base=('base', 'sum'), candidate=('candidate', 'sum'), n=('base', 'size'))
    drops = []
    for day, row in grouped.iterrows():
        n = len(y) - row['n']
        delta = np.sqrt((error.sum()-row.candidate)/n) - np.sqrt((base_error.sum()-row['base'])/n)
        drops.append({'day': day, 'delta_rmse': float(delta)})
    order = np.argsort(gain)[::-1]
    influences = {}
    for n in (1, 2, 5, 10):
        keep = np.ones(len(y), bool)
        keep[order[:n]] = False
        influences[str(n)] = {'delta_rmse': float(np.sqrt(error[keep].mean())-np.sqrt(base_error[keep].mean())),
            'removed_sse_gain': float(gain[~keep].sum())}
    rng = np.random.default_rng(20260916)
    sample = rng.integers(0, len(grouped), size=(2000, len(grouped)))
    n = grouped.n.to_numpy()[sample].sum(axis=1)
    bootstrap = np.sqrt(grouped.candidate.to_numpy()[sample].sum(axis=1)/n) - np.sqrt(grouped['base'].to_numpy()[sample].sum(axis=1)/n)
    return {'control': metrics(y, control), 'candidate': metrics(y, candidate),
        'delta_rmse': float(np.sqrt(error.mean())-np.sqrt(base_error.mean())), 'sse_gain': float(gain.sum()),
        'days_improved': int((grouped.candidate < grouped['base']).sum()), 'days': len(grouped),
        'day_removals': drops, 'all_day_removals_improve': all(row['delta_rmse'] < 0 for row in drops),
        'remove_top_gain_rows': influences,
        'largest_single_row_share_of_net_gain': float(gain.max()/gain.sum()) if gain.sum() > 0 else None,
        'paired_day_bootstrap': {'replicates': 2000, 'seed': 20260916, 'delta_rmse_p025_p975': np.quantile(bootstrap, [.025,.975]).tolist(),
            'fraction_improving': float(np.mean(bootstrap < 0)), 'scope': 'Within-fold day resampling; does not correct adaptive search or validate temporal transfer.'}}


def file_receipt(root, entry):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = root / path
    common.external_path(path)
    assert sha256(path) == entry['sha256'], path
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = common.external_path(args.output)
    assert not output.exists(), 'Preserve prior evaluation'
    folder = common.external_path(args.candidate)
    manifest = read_json(folder / 'manifest.json')
    protocol = read_json(folder / 'protocol.json')
    assert manifest['status'] == 'complete'
    assert manifest['protocol_sha256'] == sha256(folder / 'protocol.json')
    assert protocol['baseline_binding_sha256'] == sha256(BINDING)
    assert set(manifest['folds']) == {'F1', 'F3'}
    assert protocol['score_labels_used_for_selection'] is False
    assert protocol['routing_uses_score_targets'] is False
    assert set(protocol['selection_periods']).issubset({'fit', 'tune'})
    assert not FORBIDDEN.intersection(protocol['inference_columns'])
    assert protocol['prediction_transform'] == 'raw_unclipped'
    assert manifest['source_hashes']
    for path, digest in manifest['source_hashes'].items():
        actual = Path(path) if Path(path).is_absolute() else ROOT / path
        assert sha256(actual) == digest, actual
    binding = read_json(BINDING)
    assert binding['status'] == 'passed'
    results = {}
    for fold, frozen in binding['folds'].items():
        guard()
        ref, _ = common.reference(fold)
        assert object_hash(ref[ID].tolist()) == frozen['score_id_hash']
        assert object_hash(ref[TARGET].tolist()) == frozen['score_target_hash']
        baseline_path = Path(frozen['prediction_path'])
        assert sha256(baseline_path) == frozen['prediction_sha256']
        baseline = pd.read_parquet(baseline_path)
        entry = manifest['folds'][fold]
        assert entry['split_hash'] == frozen['split_hash']
        scope = entry['training_scope']
        assert scope in frozen['cohorts']
        assert entry['cohorts'] == frozen['cohorts'][scope]
        prediction_path = file_receipt(folder, entry['prediction'])
        frame = pd.read_parquet(prediction_path)
        assert set(frame) == {ID, 'prediction_sec'}, 'Candidate interchange must contain only IDs and predictions'
        assert frame[ID].notna().all() and frame[ID].is_unique
        np.testing.assert_array_equal(frame[ID], ref[ID])
        p = frame.prediction_sec.to_numpy(float)
        y = ref[TARGET].to_numpy(float)
        base = baseline.prediction_sec.to_numpy(float)
        assert np.isfinite(p).all()
        days = pd.to_datetime(ref['day'], utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        proxy = ref.proxy_sec.to_numpy(float)
        missing = ~np.isfinite(proxy)
        ordinary = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
        observed = {'all': np.ones(len(y), bool), 'missing_nm': missing, 'ordinary_nm': ordinary,
            'finite_nonordinary_nm': np.isfinite(proxy) & ~ordinary}
        change_scope = entry['prediction_scope']
        assert change_scope in observed
        np.testing.assert_array_equal(p[~observed[change_scope]], base[~observed[change_scope]])
        # Error/target-defined masks are computed only after predictions are frozen.
        top = np.argsort((base-y)**2)[-max(1, int(np.ceil(.01*len(y)))):]
        top_mask = np.zeros(len(y), bool)
        top_mask[top] = True
        diagnostic = {'baseline_top1pct_error': top_mask, 'target_negative': y < 0,
            'target_over7200': y > 7200, 'source_gap_over1800': np.isfinite(proxy) & (np.abs(y-proxy) > 1800)}
        slices = {name: {'control': metrics(y[mask], base[mask]), 'candidate': metrics(y[mask], p[mask])}
                  for name, mask in {**observed, **diagnostic}.items()}
        airports = {str(airport): {'control': metrics(y[mask], base[mask]), 'candidate': metrics(y[mask], p[mask])}
            for airport in ref.ADEP_mvt.unique() for mask in [ref.ADEP_mvt.eq(airport).to_numpy()]}
        comparisons = {'baseline272': paired(y, base, p, days), 'originalV2': paired(y, ref.prediction_sec.to_numpy(float), p, days)}
        if 'matched_control' in entry:
            control_path = file_receipt(folder, entry['matched_control'])
            control = pd.read_parquet(control_path)
            np.testing.assert_array_equal(control[ID], ref[ID])
            comparisons['matched_control'] = paired(y, control.prediction_sec.to_numpy(float), p, days)
        results[fold] = {'comparisons': comparisons, 'slices': slices, 'airports': airports,
            'score_rows': len(y), 'score_id_hash': frozen['score_id_hash'], 'score_target_hash': frozen['score_target_hash'],
            'prediction_sha256': sha256(prediction_path), 'prediction_scope': change_scope,
            'unchanged_outside_scope': True, 'changed_rows': int(np.count_nonzero(p != base)),
            'tail_diagnostics_are_not_features': True}
    seasonal = {}
    shared = set.intersection(*(set(results[f]['comparisons']) for f in results))
    for name in sorted(shared):
        control = np.sqrt(sum(binding['weights'][f] * results[f]['comparisons'][name]['control']['mse'] for f in results))
        candidate = np.sqrt(sum(binding['weights'][f] * results[f]['comparisons'][name]['candidate']['mse'] for f in results))
        seasonal[name] = {'control_rmse': float(control), 'candidate_rmse': float(candidate), 'delta_rmse': float(candidate-control)}
    report = {'status': 'passed', 'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'baseline_binding_sha256': sha256(BINDING), 'protocol_sha256': sha256(folder / 'protocol.json'),
        'manifest_sha256': sha256(folder / 'manifest.json'), 'folds': results, 'seasonal': seasonal,
        'runtime_sec': time.monotonic()-started, 'peak_rss_bytes': guard(),
        'selection_provenance': 'Manifest declarations checked; code-level training/routing review and independent saved-model replay remain separate required evidence.',
        'promotion': 'No automatic promotion; exposed development, tail diagnostics not deployable routing; report both folds and matched controls.'}
    output.mkdir(parents=True)
    write_json(output / 'evaluation.json', report)
    print('FULL_COHORT_EVALUATION', seasonal, flush=True)


if __name__ == '__main__':
    main()
