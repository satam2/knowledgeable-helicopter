"""Pre-fit incremental ensemble endpoint against the already completed PLE387."""
import torch
import lightgbm
import argparse
import numpy as np
import pandas as pd
import run_attention as subject

common = subject.common
OUT = common.external_path(subject.OUT.parent / 'incremental_assessment_v1')


def declare():
    producer = subject.declare()
    record = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=common.sha256(subject.OUT / 'protocol.json'),
        endpoint='Every ordinary tune row; candidate originaloldPLE component replacement compared with global9+wPLE*(saved387-old225)',
        formula='candidate-current387 = wPLE*(candidate387attention-saved387)',
        controls=producer['controls'], season_weights=producer['season_weights'],
        statistics='RMSE/paired day bootstrap MSE interval/every day removal/topbeneficial-row removal; bothmonths/daypositive and>=2seasonalgain materiality reported',
        role='Additional incremental evidence beyond already completed PLE387; original old225 gate unchanged. Old225 improvement alone cannot justify newpackage advancement. No automatic score authorization.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def assess():
    protocol = declare()
    folds = {}
    for fold in ['F1', 'F3']:
        folder = subject.OUT / fold
        manifest = common.read_json(folder / 'manifest.json')
        assert manifest['status'] == 'complete'
        assert manifest['protocol_sha256'] == protocol['producer_protocol_sha256']
        path = folder / 'tune_predictions.parquet'
        assert common.sha256(path) == manifest['outputs'][path.name]
        own = pd.read_parquet(path).set_index(subject.ID)
        binding = protocol['controls'][fold]
        for name, field in [(f'{fold}_weights.json', 'global9_weights_sha256'), (f'{fold}_aligned_tune.parquet', 'global9_tune_sha256')]:
            assert common.sha256(subject.ENSEMBLE / name) == binding[field]
        weights = common.read_json(subject.ENSEMBLE / f'{fold}_weights.json')
        aligned = pd.read_parquet(subject.ENSEMBLE / f'{fold}_aligned_tune.parquet').set_index(subject.ID)
        own = own.loc[aligned.index]
        oldpath = subject.CONTROLS[fold] / 'tune_predictions.parquet'
        assert common.sha256(oldpath) == binding['tune_predictions_sha256']
        old = pd.read_parquet(oldpath).set_index(subject.ID).loc[aligned.index]
        np.testing.assert_array_equal(old.prediction_sec, own.control387_prediction_sec)
        np.testing.assert_array_equal(old[subject.TARGET], aligned[subject.TARGET])
        np.testing.assert_array_equal(own[subject.TARGET], aligned[subject.TARGET])
        base = aligned[weights['experts']].to_numpy() @ np.asarray(weights['global'])
        weight = weights['global'][weights['experts'].index('tabm_ple8')]
        current387 = base + weight * (old.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
        candidate = base + weight * (own.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
        folds[fold] = subject.context.paired(aligned[subject.TARGET].to_numpy(), candidate, current387,
            aligned[subject.TIME].dt.floor('D').to_numpy())
    candidate = float(np.sqrt(sum(w * folds[f]['candidate_rmse']**2 for f, w in zip(['F1', 'F3'], protocol['season_weights']))))
    control = float(np.sqrt(sum(w * folds[f]['control_rmse']**2 for f, w in zip(['F1', 'F3'], protocol['season_weights']))))
    positive = all(r['gain'] > 0 and r['all_day_removals_improve'] for r in folds.values())
    output = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        seasonal_candidate_rmse=candidate, seasonal_current387_rmse=control, incremental_gain=control-candidate,
        both_months_and_all_days_positive=positive, incremental_two_second_materiality=positive and control-candidate >= 2,
        original_gate_unchanged=True, no_score_authorization=True, folds=folds)
    path = OUT / 'assessment.json'
    assert not path.exists()
    common.write_json(path, output)
    print({k:v for k,v in output.items() if k!='folds'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    else:
        assess()
