"""Frozen two-season preprocessing endpoint; no fitting or promotion side effects."""
import preprocessed_locations_tune as trial
import numpy as np


def main():
    protocol = trial.declare()
    records, hashes = {}, {}
    for fold in ['F1', 'F3']:
        folder = trial.OUT / fold
        record = trial.common.read_json(folder / 'manifest.json')
        assert record['status'] == 'complete' and record['no_score_prediction']
        assert record['protocol_sha256'] == trial.common.sha256(trial.OUT / 'protocol.json')
        for name, digest in record['outputs'].items():
            assert trial.common.sha256(folder / name) == digest
        records[fold] = record['results']
        hashes[fold] = trial.common.sha256(folder / 'manifest.json')
    seasonal = {}
    for endpoint in ['matched', 'primary']:
        pairs = [records[fold][endpoint] for fold in ['F1', 'F3']]
        result = dict(rmse=float(np.sqrt(sum(w*p['rmse']**2 for w, p in zip(protocol['weights'], pairs)))),
            reference_rmse=float(np.sqrt(sum(w*p['reference_rmse']**2 for w, p in zip(protocol['weights'], pairs)))),
            both_months_positive=all(p['gain'] > 0 for p in pairs),
            all_day_removals_positive=all(p['all_day_removals_improve'] for p in pairs))
        result['gain'] = result['reference_rmse']-result['rmse']
        seasonal[endpoint] = result
    primary = seasonal['primary']
    output = trial.common.external_path(trial.OUT / 'assessment.json')
    assert not output.exists()
    trial.common.write_json(output, dict(status='complete', source_sha256=trial.common.sha256(__file__),
        protocol_sha256=trial.common.sha256(trial.OUT / 'protocol.json'), manifests=hashes,
        folds=records, seasonal=seasonal,
        gate_passed=primary['gain'] >= 2 and primary['both_months_positive'] and primary['all_day_removals_positive'],
        no_score_prediction=True, limitation='Exposed June/October tune comparison only. Original complete-cohort local score unchanged. No automatic refit or promotion.'))
    print(seasonal, flush=True)


if __name__ == '__main__':
    main()
