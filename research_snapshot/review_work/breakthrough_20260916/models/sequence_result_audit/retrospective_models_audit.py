"""Matched retrospective-arm receipts and independent fullscore arithmetic."""
import lightgbm
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import audit


def main():
    cache = audit.BASE / 'retrospective_research'
    manifest = audit.read_json(cache / 'manifest.json')
    check = audit.read_json(cache / 'verification.json')
    oracle = audit.read_json(audit.OUT / 'retrospective_oracle.json')
    assert manifest['status'] == 'complete' and check['status'] == oracle['status'] == 'passed'
    assert check['manifest_sha256'] == oracle['manifest_sha256'] == audit.sha256(cache / 'manifest.json')
    assert manifest['outputs']['training_features.parquet'] == audit.sha256(cache / 'training_features.parquet')
    assert manifest['source_sha256'] == audit.sha256(audit.ROOT / 'review_work/breakthrough_20260916/models_retrieval/retrospective_research/build.py')
    base = audit.BASE / 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
    arms = {'base225': base, 'past243': audit.BASE / 'retrospective_models/past',
            'all275': audit.BASE / 'retrospective_models/day__future__past'}
    detail = {name: audit.read_json(folder / 'matched_protocol.json') for name, folder in arms.items() if name != 'base225'}
    results = {}
    for fold in ('F1', 'F3'):
        records, frames = {}, {}
        reference, _ = audit.common.reference(fold)
        for name, root in arms.items():
            folder = root / f'lightgbm_aobt_allfinite_{fold}_s20260916'
            rec = audit.checked(folder)
            records[name] = rec
            frame = pd.read_parquet(folder / 'candidate.parquet')
            frames[name] = frame
            np.testing.assert_array_equal(frame[audit.ID], reference[audit.ID])
            np.testing.assert_array_equal(frame[audit.TARGET], reference[audit.TARGET])
            missing = ~np.isfinite(reference.proxy_sec.to_numpy())
            np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing], reference.prediction_sec.to_numpy()[missing])
            for variant in ('candidate', 'blend25'):
                saved = pd.read_parquet(folder / f'{variant}.parquet')
                errors = saved.prediction_sec.to_numpy() - saved[audit.TARGET].to_numpy()
                reported = rec['reports'][variant]['metrics']['overall']
                assert reported['n'] == len(saved)
                np.testing.assert_allclose(np.square(errors).sum(), reported['sse'], rtol=1e-12)
                np.testing.assert_allclose(np.sqrt(np.square(errors).mean()), reported['rmse_sec'], rtol=1e-12)
            if name != 'base225':
                payload = detail[name]
                assert payload['cache_manifest_sha256'] == audit.sha256(cache / 'manifest.json')
                assert payload['cache_verification_sha256'] == audit.sha256(cache / 'verification.json')
                assert payload['controls'][fold] == audit.sha256(base / f'lightgbm_aobt_allfinite_{fold}_s20260916' / 'manifest.json')
                assert payload == rec['anchor']['retrospective_context']
                expected = [c for block in payload['blocks'] for c in manifest['feature_groups'][block]]
                assert payload['added_columns'] == expected
                assert rec['feature_columns'] == payload['base_columns'] + expected
                assert rec['source_hashes'] == payload['source_hashes']
                snapshot = root / 'source_snapshots/lightgbm_aobt_allfinite_s20260916'
                for path, digest in rec['source_hashes'].items():
                    assert audit.sha256(snapshot / Path(path).name) == digest
            assert rec['seed'] == 20260916 and rec['threads'] == 4
            assert rec['fit']['steps'] == rec['refit']['steps'] == 600
        for name in ('past243', 'all275'):
            audit.contract(records['base225'], records[name])
            assert records[name]['anchor']['feature_receipts'] == records['base225']['anchor']['feature_receipts']
            assert detail[name]['base_columns'] == records['base225']['feature_columns']
        assert len(records['past243']['feature_columns']) == 243
        assert len(records['all275']['feature_columns']) == 275
        assert detail['past243']['blocks'] == ['past']
        assert detail['all275']['blocks'] == ['day', 'future', 'past']
        comparisons = {'past_vs_base': audit.compare(frames['base225'], frames['past243']),
                       'all_vs_base': audit.compare(frames['base225'], frames['all275']),
                       'all_vs_past': audit.compare(frames['past243'], frames['all275'])}
        results[fold] = {'comparisons': comparisons, 'metrics': {name: rec['reports']['candidate']['metrics']['overall'] for name, rec in records.items()},
            'manifest_sha256': {name: audit.sha256(root / f'lightgbm_aobt_allfinite_{fold}_s20260916' / 'manifest.json') for name, root in arms.items()},
            'full_saved_evaluator_replay_delta': {name: rec['reload_max_abs_delta'] for name, rec in records.items()},
            'fullscore_rows': len(reference), 'missing_V2_exact_rows': int(missing.sum()),
            'matched_cohorts_schema_params_receipts_verified': True}
        print('RETROSPECTIVE_MATCHED', fold, {name: value['overall'] for name, value in comparisons.items()}, flush=True)
    seasonal = {name: audit.season_score(results['F1']['metrics'][name], results['F3']['metrics'][name]) for name in arms}
    for name in ('past243', 'all275'):
        assert np.isclose(seasonal[name], audit.read_json(arms[name] / 'summary.json')['seasonal_rmse']['candidate'], rtol=1e-12)
    audit.write_json(audit.OUT / 'retrospective_models_audit.json', {'status': 'passed', 'created_utc': audit.utc_now(),
        'source_sha256': audit.sha256(__file__), 'folds': results, 'seasonal_rmse': seasonal,
        'scope': 'All savedoutputs/source snapshots/cohorts/schema/rawscoremetrics verified. Producer fullsavedmodelreplay0 checked; models not independentlyreloaded.',
        'interpretation': 'Past243 isstrictwindowdirectiononly, notcausalavailability: base225 alreadyincludesretrospectivefeatures andfinalNMclocks. All275 adds32future/dayfields jointly; notseparatefuture/day attribution.',
        'cache_manifest_sha256': audit.sha256(cache / 'manifest.json'), 'independent_cache_oracle_sha256': audit.sha256(audit.OUT / 'retrospective_oracle.json')})
    print('RETROSPECTIVE_SEASONAL', seasonal, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
