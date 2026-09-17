"""Assess the two frozen CatBoost information endpoints without refitting."""
import catboost_union387_tune_v2 as fit
import numpy as np


def main():
    protocol = fit.declare()
    output = fit.common.external_path(fit.OUT.parent / 'catboost_union387_assessment_v1')
    output.mkdir(parents=True, exist_ok=False)
    folds, hashes = {}, {}
    for fold in ['F1', 'F3']:
        folder = fit.OUT / fold
        marker = fit.common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete' and marker['no_score_prediction']
        assert marker['protocol_sha256'] == fit.common.sha256(fit.OUT / 'protocol.json')
        for name, digest in marker['outputs'].items():
            assert fit.common.sha256(folder / name) == digest
        hashes[fold] = fit.common.sha256(folder / 'manifest.json')
        folds[fold] = marker['metrics']
    seasonal = {}
    for endpoint in ['matched_all_finite', 'primary_current387']:
        pairs = [folds[f][endpoint] for f in ['F1', 'F3']]
        value = dict(rmse=float(np.sqrt(sum(w*p['rmse']**2 for w, p in zip(protocol['weights'], pairs)))),
            reference_rmse=float(np.sqrt(sum(w*p['reference_rmse']**2 for w, p in zip(protocol['weights'], pairs)))),
            both_months_and_days_positive=all(p['gain'] > 0 and p['all_day_removals_improve'] for p in pairs))
        value['gain'] = value['reference_rmse']-value['rmse']
        seasonal[endpoint] = value
    primary = seasonal['primary_current387']
    fit.common.write_json(output / 'receipt.json', dict(status='complete', source_sha256=fit.common.sha256(__file__),
        protocol_sha256=fit.common.sha256(fit.OUT / 'protocol.json'), manifests=hashes, folds=folds,
        seasonal=seasonal, gate_passed=primary['gain'] >= 2 and primary['both_months_and_days_positive'],
        no_score_prediction=True, limitation='Exposed tune comparison. No automatic score refit, promotion or official score claim.'))
    print(seasonal, flush=True)


if __name__ == '__main__':
    main()
