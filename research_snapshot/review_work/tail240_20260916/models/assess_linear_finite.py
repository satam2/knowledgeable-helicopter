"""Evaluate the frozen linear-leaf primary without selecting the control arm."""
import linear_finite_tune as fit
import argparse
import numpy as np

common = fit.common
OUT = common.external_path(fit.OUT.parent / 'linear_finite_assessment_v1')


def declaration():
    producer = fit.declaration()
    record = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=common.sha256(fit.OUT / 'protocol.json'),
        primary=producer['primary'], gate=producer['gate'], weights=[192122/344841, 152719/344841],
        role='Constantcontrol and matchedleaf comparison reported; primarylinear only canqualify. No automatic score/refit authority.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def run():
    protocol = declaration()
    markers = {}
    for fold in ['F1', 'F3']:
        folder = fit.OUT / fold
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete' and marker['no_score_prediction']
        assert marker['source_sha256'] == common.sha256(fit.__file__)
        assert marker['protocol_sha256'] == protocol['producer_protocol_sha256']
        for name, digest in marker['outputs'].items():
            assert common.sha256(folder / name) == digest
        for arm in fit.ARMS:
            child = common.read_json(folder / arm / 'manifest.json')
            assert child['status'] == 'complete' and child['native_replay_max_abs_delta'] == 0
            assert child['results'] == marker['arms'][arm]
            for name, digest in child['outputs'].items():
                assert common.sha256(folder / arm / name) == digest
        markers[fold] = marker
    results = {}
    for arm in fit.ARMS:
        pairs = [markers[f]['arms'][arm]['ordinary_fixed25'] for f in ['F1', 'F3']]
        candidate = float(np.sqrt(sum(w*p['rmse']**2 for w, p in zip(protocol['weights'], pairs))))
        reference = float(np.sqrt(sum(w*p['reference_rmse']**2 for w, p in zip(protocol['weights'], pairs))))
        results[arm] = dict(rmse=candidate, reference_rmse=reference, gain=reference-candidate,
            both_months_all_days_positive=all(p['gain'] > 0 and p['all_day_removals_improve'] for p in pairs))
    primary = results['linear']
    gate = primary['gain'] >= 2 and primary['both_months_all_days_positive']
    output = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        manifests={f: common.sha256(fit.OUT / f / 'manifest.json') for f in markers},
        seasonal_fixed25=results, matched={f: markers[f]['matched_linear_vs_constant'] for f in markers},
        primary_linear_gate_passed=gate, no_score_prediction=True,
        limitation='Exposed development comparisons. Complementarity primary neednotwinstandalone; constantarm notalternatepromotion.')
    path = OUT / 'summary.json'
    assert not path.exists()
    common.write_json(path, output)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        run()
