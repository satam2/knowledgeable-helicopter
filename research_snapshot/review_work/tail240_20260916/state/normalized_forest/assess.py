"""Predeclared seasonal complete-tune hard gate for normalized ExtraTrees."""
import argparse
from pathlib import Path
import run as subject

common, np, pd = subject.common, subject.np, subject.pd
OUT = common.external_path(subject.OUT.parent / 'assessment_v1')


def declare():
    prior = subject.declare()
    value = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=common.sha256(subject.OUT / 'protocol.json'),
        reference=prior['reference'], candidate=prior['candidate'], gate=prior['gate'],
        comparison='Both fixed original folds required. Original complete tune labels/cohorts, unchanged finite predictions. Seasonal sqrt(192122/344841 * JuneMSE +152719/344841 * OctMSE).',
        effect='Only missing branch fixed25 complement; no bound272/269 score claim, no scoring authorization, no new weights or thresholds.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return value


def assess():
    declaration = declare()
    records = {}
    for fold in ['F1', 'F3']:
        folder = subject.OUT / fold
        manifest = common.read_json(folder / 'manifest.json')
        assert manifest['status'] == 'complete'
        assert manifest['protocol_sha256'] == declaration['producer_protocol_sha256']
        for name, digest in manifest['outputs'].items():
            assert common.sha256(folder / name) == digest
        control = common.read_json(folder / 'control_replay.json')
        assert control['max_abs_delta'] <= 1e-7 and control['native_original_max_abs_delta'] <= 1e-7
        assert common.read_json(folder / 'control_pass_before_scaled_fit.json')['passed']
        frame = pd.read_parquet(folder / 'full_tune.parquet')
        assert len(frame) == manifest['full_tune_rows']
        assert common.object_hash(frame[subject.ID].tolist()) == manifest['full_tune_id_hash']
        finite = np.isfinite(frame.proxy_sec.to_numpy())
        np.testing.assert_array_equal(frame.loc[finite,'prediction_sec'], frame.loc[finite,'reference_prediction_sec'])
        report = common.read_json(folder / 'metrics.json')
        for field, column in [('candidate', 'prediction_sec'), ('control', 'reference_prediction_sec')]:
            check = subject.scores(frame[subject.TARGET], frame[column])
            np.testing.assert_allclose(check['sse'], report['full_tune_fixed25'][field]['sse'], rtol=1e-12)
            assert check['n'] == report['full_tune_fixed25'][field]['n']
        records[fold] = dict(manifest_sha256=common.sha256(folder / 'manifest.json'),
            missing_beats_raw=report['vs_raw']['gain']>0,
            missing_beats_normalized_lgb=report['vs_normalized_lgb']['gain']>0,
            full_tune_gain=report['full_tune_fixed25']['gain'],
            full_tune_all_day_removals_improve=report['full_tune_fixed25']['all_day_removals_improve'],
            metrics=report)
    candidate = subject.season_score(*(records[f]['metrics']['full_tune_fixed25']['candidate'] for f in ['F1','F3']))
    control = subject.season_score(*(records[f]['metrics']['full_tune_fixed25']['control'] for f in ['F1','F3']))
    gate = control-candidate >= 2 and all(r['missing_beats_raw'] and r['missing_beats_normalized_lgb']
        and r['full_tune_gain']>0 and r['full_tune_all_day_removals_improve'] for r in records.values())
    output = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        seasonal_candidate_rmse=candidate, seasonal_reference_rmse=control, gain=control-candidate,
        hard_gate_passed=gate, no_score_refit_authorized=True, folds=records)
    path = OUT / 'assessment.json'
    assert not path.exists(), 'Preserve completed assessment'
    common.write_json(path, output)
    print({k:v for k,v in output.items() if k!='folds'}, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT/'protocol.json'),flush=True)
    else:
        assess()
