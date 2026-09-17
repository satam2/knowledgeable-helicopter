"""Fixed primary endpoint; no posthoc choice between ensemble candidates."""
import argparse
import numpy as np
import run_attention as subject

common = subject.common
OUT = common.external_path(subject.OUT.parent / 'assessment_v1')


def declare():
    producer = subject.declare()
    record = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=common.sha256(subject.OUT / 'protocol.json'),
        primary=producer['primary'], secondary=producer['secondary'], gate=producer['gate'],
        season_weights=producer['season_weights'], no_score_authorization=True)
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
        for name, digest in manifest['outputs'].items():
            assert common.sha256(folder / name) == digest
        assert common.read_json(folder / 'control_replay.json')['max_abs_delta_sec'] <= 1e-6
        assert manifest['native_reload_max_delta_sec'] == 0
        folds[fold] = common.read_json(folder / 'metrics.json')
    records = {}
    for name in ['all_finite', 'ordinary', 'primary_replacement', 'secondary_blend25']:
        candidate = float(np.sqrt(sum(w * folds[f][name]['candidate_rmse']**2 for f, w in zip(['F1', 'F3'], protocol['season_weights']))))
        control = float(np.sqrt(sum(w * folds[f][name]['control_rmse']**2 for f, w in zip(['F1', 'F3'], protocol['season_weights']))))
        records[name] = dict(candidate_rmse=candidate, control_rmse=control, gain=control-candidate,
            both_months_positive=all(folds[f][name]['gain'] > 0 for f in folds),
            all_day_removals_positive=all(folds[f][name]['all_day_removals_improve'] for f in folds))
    primary = records['primary_replacement']
    gate = primary['gain'] >= 2 and primary['both_months_positive'] and primary['all_day_removals_positive']
    record = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        hard_gate_passed=gate, no_score_authorization=True, seasonal=records, folds=folds,
        interpretation='Only primary replacement can qualify; fixed25 and matched387 are secondary exposed-tune diagnostics')
    path = OUT / 'assessment.json'
    assert not path.exists(), 'Preserve completed result'
    common.write_json(path, record)
    print(dict(hard_gate_passed=gate, seasonal=records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    else:
        assess()
