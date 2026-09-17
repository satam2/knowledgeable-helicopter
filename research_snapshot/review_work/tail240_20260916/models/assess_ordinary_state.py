"""Apply the declared two-month state399 gate without score data."""
import ordinary_state_tune as fit
import numpy as np

common = fit.common
OUT = common.external_path(fit.ROOT / 'private_runs/tail240_20260916/models/ordinary_state_assessment_v1')


def run():
    protocol = fit.declaration()
    markers = {}
    for fold in ('F1', 'F3'):
        folder = fit.OUT / fold
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete' and marker['no_score_prediction']
        assert marker['source_sha256'] == common.sha256(fit.__file__) == protocol['source_sha256']
        assert marker['protocol_sha256'] == common.sha256(fit.OUT / 'protocol.json')
        assert marker['native_replay_max_abs_delta'] == 0
        for name, digest in marker['outputs'].items():
            assert common.sha256(folder / name) == digest
        markers[fold] = marker
    weight = 192122/344841
    pair = [markers[f]['ordinary_fixed25'] for f in ('F1','F3')]
    candidate = float(np.sqrt(weight*pair[0]['rmse']**2 + (1-weight)*pair[1]['rmse']**2))
    reference = float(np.sqrt(weight*pair[0]['reference_rmse']**2 + (1-weight)*pair[1]['reference_rmse']**2))
    gain = reference - candidate
    gate = gain >= 2 and all(markers[f][n]['mse_gain'] > 0 and markers[f][n]['all_day_removals_improve'] for f in markers for n in ('matched','ordinary_fixed25'))
    OUT.mkdir(exist_ok=False)
    report = dict(status='complete', source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(fit.OUT / 'protocol.json'),
        manifests={f:common.sha256(fit.OUT/f/'manifest.json') for f in markers},
        seasonal_ordinary_fixed25=dict(candidate=candidate, reference=reference, gain=gain),
        declared_gate_passed=gate, score_predictions_read=False,
        limitation='Exposed tune evidence; independent replay required. July2026 sixmonth label blackout remains untested even if this adjacent-month gate passes.')
    common.write_json(OUT / 'summary.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    run()
