"""Independent PLE32 endpoints, day bootstrap and influence after native replay."""
from audit_union387_sources import ROOT, read, sha, write, guard, object_hash
from pathlib import Path
import numpy as np
import pandas as pd

BASE = ROOT / 'private_runs/tail240_20260916/state/neural_capacity'
MODEL = BASE / 'v2/models/F1'
OUT = ROOT / 'private_runs/tail240_20260916/validation/neural_capacity_replay_F1_v1'
ID, TIME, TARGET = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'


def compare(y, candidate, control, dates, expected):
    a = np.square(y-candidate); b = np.square(y-control)
    unique, codes = np.unique(dates, return_inverse=True)
    count = np.bincount(codes)
    old_sse, new_sse = np.bincount(codes, weights=b), np.bincount(codes, weights=a)
    draws = np.random.default_rng(20260916).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=300)
    bootstrap = (draws @ (old_sse-new_sse)) / (draws @ count)
    day_gains = np.sqrt((b.sum()-old_sse)/(len(y)-count))-np.sqrt((a.sum()-new_sse)/(len(y)-count))
    assert len(expected['day_removals']) == len(day_gains)
    np.testing.assert_allclose(day_gains, [r['gain'] for r in expected['day_removals']], rtol=1e-9, atol=1e-9)
    result = dict(candidate_rmse=float(np.sqrt(a.mean())), control_rmse=float(np.sqrt(b.mean())),
                  gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())), mse_gain_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  all_day_removals_improve=bool((day_gains > 0).all()), day_removal_min_gain=float(day_gains.min()))
    for key in ('candidate_rmse', 'control_rmse', 'gain', 'mse_gain_ci95'):
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-10, atol=1e-8)
    assert result['all_day_removals_improve'] == expected['all_day_removals_improve']
    order = np.argsort(-(b-a), kind='stable')
    influence = {}
    for k in (1, 5, 10):
        keep = np.ones(len(y), bool); keep[order[:k]] = False
        influence[str(k)] = float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))
        np.testing.assert_allclose(influence[str(k)], expected['remove_top_beneficial_rows_gain'][str(k)], rtol=1e-10, atol=1e-9)
    result['remove_top_beneficial_rows_gain'] = influence
    return result


def main():
    assert not (OUT / 'metrics_receipt.json').exists()
    native = read(OUT / 'gpu_receipt.json')
    assert native['status'] in ('passed', 'numerical_tolerance_failed')
    assert native['producer_manifest_sha256'] == sha(MODEL / 'manifest.json')
    assert native['predictions_sha256'] == sha(OUT / 'independent_predictions.npy')
    marker, protocol = read(MODEL / 'manifest.json'), read(BASE / 'v1/protocol.json')
    assert marker['protocol_sha256'] == sha(BASE / 'v1/protocol.json')
    assert sha(BASE / 'v2/protocol.json') == read(OUT / 'cpu_receipt.json')['resource_protocol_sha256']
    for name, digest in marker['outputs'].items():
        assert sha(MODEL / name) == digest
    actual = pd.read_parquet(MODEL / 'tune_predictions.parquet')
    tune = pd.read_parquet(OUT / 'tune_metadata.parquet')
    for name in (ID, TIME, TARGET, 'proxy_sec'):
        np.testing.assert_array_equal(actual[name], tune[name])
    oldroot = ROOT / 'private_runs/tail240_20260916/state/neural_context/v1/F1'
    binding = protocol['controls']['F1']
    assert sha(oldroot / 'manifest.json') == binding['manifest_sha256']
    assert sha(oldroot / 'tune_predictions.parquet') == binding['predictions_sha256']
    old = pd.read_parquet(oldroot / 'tune_predictions.parquet').set_index(ID).loc[tune[ID]]
    np.testing.assert_array_equal(old.prediction_sec, actual.control8_prediction_sec)
    np.testing.assert_array_equal(old[TARGET], tune[TARGET])
    y = tune[TARGET].to_numpy(float); prediction = actual.prediction_sec.to_numpy(float); control = old.prediction_sec.to_numpy(float)
    days = tune[TIME].dt.floor('D').to_numpy(); ordinary = tune.proxy_sec.between(0, 7200).to_numpy()
    metrics = read(MODEL / 'metrics.json')
    reports = dict(all_finite=compare(y, prediction, control, days, metrics['all_finite']),
                   ordinary=compare(y[ordinary], prediction[ordinary], control[ordinary], days[ordinary], metrics['ordinary']))
    ensemble = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    assert sha(ensemble / 'F1_weights.json') == binding['global9_weights_sha256']
    assert sha(ensemble / 'F1_aligned_tune.parquet') == binding['global9_tune_sha256']
    weights = read(ensemble / 'F1_weights.json'); aligned = pd.read_parquet(ensemble / 'F1_aligned_tune.parquet')
    assert aligned[ID].is_unique and set(aligned[ID]) == set(tune.loc[ordinary, ID])
    own = actual.set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(own[TARGET], aligned[TARGET])
    global9 = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    w = weights['global'][weights['experts'].index('tabm_ple8')]
    current = global9+w*(own.control8_prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    candidate = global9+w*(own.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    reports['primary_current387'] = compare(aligned[TARGET].to_numpy(float), candidate, current,
                                          aligned[TIME].dt.floor('D').to_numpy(), metrics['primary_current387'])
    evidence = read(MODEL / 'fit_evidence.json')
    assert evidence['params']['members'] == 32 and evidence['only_model_change'].startswith('PLE ensemble members8to32')
    best = min(evidence['history'], key=lambda row: row['tune_mse_sec2'])
    assert best['epoch'] == marker['selected_epochs']
    np.testing.assert_allclose(best['tune_mse_sec2'], reports['all_finite']['candidate_rmse']**2, rtol=0, atol=1e-7)
    result = dict(status='metrics_passed', native_replay_status=native['status'], source_sha256=sha(Path(__file__)), manifest_sha256=sha(MODEL / 'manifest.json'),
                  native_receipt_sha256=sha(OUT / 'gpu_receipt.json'), metrics=reports, selected_epochs=marker['selected_epochs'],
                  rows=len(tune), primary_rows=int(ordinary.sum()), coefficient=w, peak_bytes=guard(),
                  no_score_or_fitting=True, scope='Incremental replay plus exact parity to earlier independently reconstructed fit encoder/bin state; full metrics recomputed from bound native predictions.',
                  limitation='Exposed tune and same-tune ensemble weights. F1 alone cannot establish seasonal gate or complete-score improvement.')
    write(OUT / 'metrics_receipt.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
