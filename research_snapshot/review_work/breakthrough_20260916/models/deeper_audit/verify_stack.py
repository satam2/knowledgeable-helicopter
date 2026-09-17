"""Independent tune-weight reconstruction and full score replay for simplex_v3."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import audit

ROOT = audit.ROOT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stacking'))
import simplex
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET
from taxiout.metrics import season_score

OUT = audit.OUT
V3 = OUT / 'simplex_v3'


def main():
    declaration = read_json(V3 / 'protocol.json')['declaration']
    meta = pd.read_parquet(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet',
                           columns=[ID, 'proxy_sec', 'ADEP_mvt']).set_index(ID)
    results = {}
    for fold in ['F1', 'F3']:
        directory = V3 / fold
        record = read_json(directory / 'manifest.json')
        assert record['status'] == 'complete'
        assert record['protocol_sha256'] == sha256(V3 / 'protocol.json')
        for filename, digest in record['outputs'].items():
            assert sha256(directory / filename) == digest
        tune = pd.read_parquet(V3 / 'tune_only_audit' / f'{fold}_aligned_tune.parquet')
        weights = read_json(directory / 'weights.json')
        recalculated = simplex.solve(tune[TARGET], tune.tabm_source, tune.lgb_leaf63, tune.ADEP_mvt)
        assert weights == recalculated
        reference, _ = audit.common.reference(fold)
        local = meta.loc[reference[ID]]
        eligible = np.isfinite(local.proxy_sec) & local.proxy_sec.between(0, 7200)
        eligible = eligible.to_numpy()
        controls = {'reference': reference}
        source = {}
        for expert in declaration['experts']:
            receipt = declaration['source_receipts'][fold][expert]
            path = Path(receipt['directory']) / 'candidate.parquet'
            assert sha256(path) == receipt['score_sha256']
            source[expert] = pd.read_parquet(path)
            controls[expert + '_ordinary'] = pd.read_parquet(directory / f'{expert}.parquet')
        controls['leaf63_full'] = source['lgb_leaf63']
        rows = {}
        for name, conditioned in [('global_simplex', False), ('airport_shrunk_simplex', True)]:
            frame = pd.read_parquet(directory / f'{name}.parquet')
            np.testing.assert_array_equal(frame[ID], reference[ID])
            np.testing.assert_array_equal(frame[TARGET], reference[TARGET])
            expected = reference.prediction_sec.to_numpy().copy()
            expected[eligible] = simplex.predict(weights, source['tabm_source'].prediction_sec.to_numpy()[eligible],
                                                 source['lgb_leaf63'].prediction_sec.to_numpy()[eligible],
                                                 local.ADEP_mvt.to_numpy()[eligible], conditioned)
            np.testing.assert_array_equal(frame.prediction_sec, expected)
            comparisons = {}
            for control, control_frame in controls.items():
                comparisons[control] = {'overall': audit.summary(control_frame.squared_error.to_numpy(), frame.squared_error.to_numpy(), np.ones(len(frame), bool)),
                                        'day_removals': audit.day_sensitivity(control_frame, frame)}
            error = frame.squared_error.to_numpy()
            rows[name] = {'complete_score_n': len(frame), 'replay_max_delta': 0., 'comparisons': comparisons,
                          'protected_n': int((~eligible).sum()), 'protected_sse': float(error[~eligible].sum()),
                          'protected_sse_share': float(error[~eligible].sum()/error.sum()),
                          'protected_support_floor_rmse': float(np.sqrt(error[~eligible].sum()/len(error))),
                          'rmse_sec': float(np.sqrt(error.mean())), 'sse': float(error.sum()), 'n': len(error)}
            print('VERIFIED', fold, name, rows[name]['rmse_sec'], 'floor', rows[name]['protected_support_floor_rmse'], flush=True)
        results[fold] = {'weights_recomputed_exactly': True, 'global_leaf63_weight': weights['global_alpha'], 'variants': rows}
    seasonal = {name: season_score(*[results[f]['variants'][name] for f in ['F1','F3']]) for name in rows}
    output = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'folds': results, 'seasonal_rmse': seasonal,
              'declaration_sha256': sha256(V3 / 'protocol.json'),
              'interpretation': 'Exposed development results; original tune also selected expert epochs; no fresh heldout claim.'}
    write_json(OUT / 'simplex_v3_verification.json', output)
    print('SEASONAL', seasonal, flush=True)


if __name__ == '__main__':
    main()
