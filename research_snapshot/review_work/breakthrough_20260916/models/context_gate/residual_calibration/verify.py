"""Independent saved-artifact arithmetic and protocol checks, no model fitting."""
import lightgbm
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run


def main():
    summary = run.read_json(run.OUT / 'summary.json')
    assert summary['status'] == 'complete' and summary['score_reads'] is False and summary['full_tune_fit'] is False
    protocol = run.read_json(run.OUT / 'protocol.json')
    assert run.sha256(run.OUT / 'protocol.json') == summary['protocol_sha256']
    assert protocol['parameters'] == run.PARAMS
    for name, digest in protocol['source_hashes'].items():
        assert run.sha256(run.ROOT / name) == digest
        assert run.sha256(run.OUT / 'source_snapshots' / Path(name).name) == digest
    for fold in ('F1', 'F3'):
        folder = run.OUT / fold
        marker = run.read_json(folder / 'manifest.json')
        for name, digest in marker['outputs'].items():
            assert run.sha256(folder / name) == digest
        original = pd.read_parquet(run.run_tune.OUT / fold / 'tune_diagnostic.parquet')
        saved = pd.read_parquet(folder / 'tune_predictions.parquet')
        for name in (run.ID, run.TARGET, run.MOVEMENT, 'early'):
            np.testing.assert_array_equal(original[name], saved[name])
        np.testing.assert_array_equal(original.airport3, saved.baseline_sec)
        np.testing.assert_array_equal(saved.prediction_sec, saved.baseline_sec + saved.correction_sec)
        for stage in ('early', 'late'):
            keep = saved.early.to_numpy(bool) if stage == 'early' else ~saved.early.to_numpy(bool)
            assert int(keep.sum()) == marker[f'{stage}_rows']
            assert run.object_hash(saved.loc[keep, run.ID].tolist()) == marker[f'{stage}_id_hash']
            for variant, column in [('residual_calibration', 'prediction_sec'), ('airport3', 'baseline_sec')]:
                rmse = float(np.sqrt(np.square(saved.loc[keep, column] - saved.loc[keep, run.TARGET]).mean()))
                assert np.isclose(rmse, marker['metrics'][variant][f'{stage}_rmse'], rtol=1e-12)
        model = joblib.load(folder / 'early_model.joblib')
        assert model['estimator'].n_estimators_ == marker['iterations'] == 200
        assert model['estimator'].get_params() == {**model['estimator'].get_params(), **run.PARAMS}
        assert model['encoder'].columns == marker['features'] and len(marker['features']) == 228
        print('VERIFIED', fold, marker['metrics']['residual_calibration']['late_rmse'], flush=True)
    run.write_json(run.OUT / 'independent_verification.json', {'status': 'passed', 'source_sha256': run.sha256(__file__),
        'summary_sha256': run.sha256(run.OUT / 'summary.json'), 'full_saved_metrics_cohorts_hashes_verified': True,
        'no_fit_no_GPU': True})


if __name__ == '__main__':
    main()
