"""Matched full-cohort audits of final ARR and clock-representation follow-ups."""
import lightgbm
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['monthly', 'innovations'], required=True)
    args = parser.parse_args()
    monthly = args.arm == 'monthly'
    root = audit.BASE / ('retrospective_models/monthly_arrival_v3' if monthly else 'models/sequence_clock_innovations/models')
    control = audit.BASE / ('information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T' if monthly else 'missing/sequence_flatten/models')
    cache = audit.BASE / ('retrospective_research/monthly_arrival' if monthly else 'models/sequence_clock_innovations/cache')
    destination = audit.OUT / f'{args.arm}_matched_audit.json'
    assert not destination.exists(), 'Preserve prior audit'
    payload = audit.read_json(root / 'matched_protocol.json')
    marker = audit.read_json(cache / 'manifest.json')
    verified = audit.read_json(cache / 'verification.json')
    assert marker['status'] == 'complete' and verified['status'] == 'passed'
    cachehash = audit.sha256(cache / 'manifest.json')
    assert verified['manifest_sha256'] == cachehash
    assert audit.sha256(cache / 'training_features.parquet') == marker['outputs']['training_features.parquet']
    if monthly:
        oraclepath = cache / 'independent_oracle.json'
        oracle = audit.read_json(oraclepath)
        assert oracle['status'] == 'passed' and oracle['manifest_sha256'] == cachehash
        assert marker['protocol_sha256'] == payload['cache_protocol_sha256'] == audit.sha256(cache / 'protocol.json')
        base_columns = payload['base_columns']
    else:
        oraclepath = audit.OUT / 'clock_innovations_independent.json'
        oracle = audit.read_json(oraclepath)
        assert oracle['status'] == 'passed'
        assert payload['cache_manifest_sha256'] == cachehash
        base_columns = payload['control_columns']
    assert payload['added_columns'] == marker['features']
    family = 'lightgbm_monthly_arrival' if monthly else 'lightgbm_clock_innovations'
    snapshot = root / f'source_snapshots/{family}_aobt_allfinite_s20260916'
    for name, digest in payload['source_hashes'].items():
        basename = Path(name).name
        assert audit.sha256(snapshot / basename) == digest, name
    result = {}
    for fold in ('F1', 'F3'):
        folder = root / f'{family}_aobt_allfinite_{fold}_s20260916'
        before = control / f'lightgbm_aobt_allfinite_{fold}_s20260916'
        current, previous = audit.checked(folder), audit.checked(before)
        audit.contract(previous, current)
        assert current['source_hashes'] == payload['source_hashes']
        assert previous['feature_columns'] == base_columns
        assert current['feature_columns'] == base_columns + marker['features']
        assert current['anchor']['feature_receipts'] == previous['anchor']['feature_receipts']
        assert payload['controls'][fold] == audit.sha256(before / 'manifest.json')
        if monthly:
            binding = current['anchor']['monthly_arrival']
            assert binding == {**payload, 'cache_manifest_sha256': cachehash, 'independent_oracle_sha256': audit.sha256(oraclepath)}
        else:
            assert current['anchor']['clock_innovations'] == payload
        reference, _ = audit.common.reference(fold)
        missing = ~np.isfinite(reference.proxy_sec.to_numpy())
        variants = {}
        for variant in ('candidate', 'blend25'):
            frame = pd.read_parquet(folder / f'{variant}.parquet')
            old = pd.read_parquet(before / f'{variant}.parquet')
            np.testing.assert_array_equal(frame[audit.ID], reference[audit.ID])
            np.testing.assert_array_equal(frame[audit.TARGET], reference[audit.TARGET])
            np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing], reference.prediction_sec.to_numpy()[missing])
            error = frame.prediction_sec.to_numpy() - frame[audit.TARGET].to_numpy()
            reported = current['reports'][variant]['metrics']['overall']
            assert len(frame) == reported['n'] and np.isfinite(error).all()
            for name, value in [('sse', np.square(error).sum()), ('rmse_sec', np.sqrt(np.square(error).mean())),
                                ('mae_sec', np.abs(error).mean()), ('bias_sec', error.mean())]:
                np.testing.assert_allclose(value, reported[name], rtol=1e-12, atol=1e-10)
            variants[variant] = {'metrics': reported, 'comparison': audit.compare(old, frame)}
            if variant == 'candidate':
                candidate = frame.prediction_sec.to_numpy()
            else:
                np.testing.assert_array_equal(frame.prediction_sec.to_numpy(), reference.prediction_sec.to_numpy() + .25 * (candidate - reference.prediction_sec.to_numpy()))
        result[fold] = {'variants': variants, 'source_and_cache_bindings': True,
            'fit_ids': current['fit_ids'], 'matched_params': current['fit']['params'],
            'tune_selected_steps': current['fit']['steps'], 'refit_steps': current['refit']['steps'],
            'full_score_rows': len(reference), 'missing_V2_exact_rows': int(missing.sum()),
            'producer_full_saved_replay_delta': current['reload_max_abs_delta'],
            'manifest_sha256': audit.sha256(folder / 'manifest.json')}
        print('AUDIT', args.arm, fold, {key: value['comparison']['overall'] for key, value in variants.items()}, flush=True)
    seasonal = {variant: audit.season_score(result['F1']['variants'][variant]['metrics'], result['F3']['variants'][variant]['metrics']) for variant in ('candidate', 'blend25')}
    summary = audit.read_json(root / 'summary.json')
    assert summary['matched_protocol_sha256'] == audit.sha256(root / 'matched_protocol.json')
    for key in seasonal:
        np.testing.assert_allclose(seasonal[key], summary['seasonal_rmse'][key], rtol=1e-12)
    audit.write_json(destination, {'status': 'passed', 'source_sha256': audit.sha256(__file__),
        'arm': args.arm, 'folds': result, 'seasonal_rmse': seasonal,
        'cache_manifest_sha256': cachehash, 'cache_oracle_sha256': audit.sha256(oraclepath),
        'scope': 'Complete saved outputs, source snapshots, matched cohort/parameter/cache receipts, full raw score metrics and stability; producer full model replay verified, not independently reloaded here.'})
    print('PASSED', args.arm, seasonal, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
