"""Predeclared disjoint-route composition of independently tested components."""
import normalized_missing_tune as prior
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

common, ROOT, ID = prior.common, prior.ROOT, prior.ID
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/neural_missing_integration_v1')
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
NEURAL = ROOT / 'private_runs/tail240_20260916/state/neural_context/refit_score_v3'
MISSING = ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v2/observed_schedule_scale_blend25'
NEURAL_PROTOCOL = '12089583f1f87a6b5030718b882ceee057bb9982b1fbac0e418e18c4710ace88'
VARIANTS = ['replacement', 'blend25']


def declare():
    assert common.sha256(NEURAL / 'protocol.json') == NEURAL_PROTOCOL
    missing = common.read_json(MISSING / 'manifest.json')
    missing_protocol = common.read_json(MISSING / 'protocol.json')
    neural_protocol = common.read_json(NEURAL / 'protocol.json')
    value = dict(name='neural387_and_normalized_missing25_disjoint_routes',
        source_sha256=common.sha256(__file__), baseline_binding_sha256=common.sha256(BINDING),
        neural_protocol_sha256=NEURAL_PROTOCOL,
        missing_manifest_sha256=common.sha256(MISSING / 'manifest.json'),
        missing_protocol_sha256=common.sha256(MISSING / 'protocol.json'),
        missing_predictions={f: missing['folds'][f]['prediction']['sha256'] for f in ('F1', 'F3')},
        variants=VARIANTS, score_labels_used_for_selection=False, routing_uses_score_targets=False,
        selection_periods=['fit', 'tune'], prediction_transform='raw_unclipped',
        inference_columns=sorted(set(neural_protocol['feature_columns'] + missing_protocol['inference_columns'] + ['proxy_sec'])),
        tail_definition='No target-defined inference groups.',
        routing_definition='Finite 0<=proxy_sec<=7200 receives declared neural variant; nonfinite proxy receives frozen normalizedmissing25; all finite nonordinary remains exact V4 baseline.',
        selection_rule='Both variants reported; fixed old PLE component weight replacement and fixed25 neural blend. Missing fixed25 unchanged. No new fit, tuning, clipping, calibration, or score-based variant selection.',
        justification='Neural matched-family tune gate passed, ensemble 2-second threshold failed and is not represented as passed. Missing component independently verified 269.863486 complete-cohort local score.',
        exposure='July and November are exposed development folds. No ranking predictions or submission.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return value


def run():
    protocol = declare()
    binding = common.read_json(BINDING)
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'proxy_sec']).set_index(ID)
    sources = {str(Path(__file__).relative_to(ROOT)): common.sha256(__file__)}
    exchanges = {}
    for variant in VARIANTS:
        dest = OUT / variant
        dest.mkdir(exist_ok=False)
        exchange_protocol = dict(protocol, name=protocol['name'] + '_' + variant,
            integration_protocol_sha256=common.sha256(OUT / 'protocol.json'), selected_variant=variant)
        common.write_json(dest / 'protocol.json', exchange_protocol)
        exchanges[variant] = dict(dest=dest, folds={}, component_manifests={})
    for fold in ('F1', 'F3'):
        bound = binding['folds'][fold]
        assert common.sha256(bound['prediction_path']) == bound['prediction_sha256']
        baseline = pd.read_parquet(bound['prediction_path'], columns=[ID, 'prediction_sec'])
        proxy = meta.loc[baseline[ID], 'proxy_sec'].to_numpy(float)
        finite = np.isfinite(proxy)
        ordinary = finite & (proxy >= 0) & (proxy <= 7200)
        missing_path = MISSING / f'{fold}.parquet'
        assert common.sha256(missing_path) == protocol['missing_predictions'][fold]
        missing = pd.read_parquet(missing_path)
        np.testing.assert_array_equal(missing[ID], baseline[ID])
        np.testing.assert_array_equal(missing.prediction_sec.to_numpy()[finite], baseline.prediction_sec.to_numpy()[finite])
        marker = common.read_json(NEURAL / fold / 'manifest.json')
        assert marker['status'] == 'complete' and marker['protocol_sha256'] == NEURAL_PROTOCOL
        assert marker['native_replay_max_abs_delta_sec'] == 0
        for variant, exchange in exchanges.items():
            neural_path = NEURAL / fold / f'{variant}.parquet'
            assert common.sha256(neural_path) == marker['outputs'][neural_path.name]
            neural = pd.read_parquet(neural_path, columns=[ID, 'prediction_sec'])
            np.testing.assert_array_equal(neural[ID], baseline[ID])
            np.testing.assert_array_equal(neural.prediction_sec.to_numpy()[~ordinary], baseline.prediction_sec.to_numpy()[~ordinary])
            result = baseline.copy()
            values = result.prediction_sec.to_numpy(copy=True)
            values[ordinary] = neural.prediction_sec.to_numpy()[ordinary]
            values[~finite] = missing.prediction_sec.to_numpy()[~finite]
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values[finite & ~ordinary], baseline.prediction_sec.to_numpy()[finite & ~ordinary])
            result['prediction_sec'] = values
            path = exchange['dest'] / f'{fold}.parquet'
            result.to_parquet(path, index=False)
            exchange['folds'][fold] = dict(prediction=dict(path=path.name, sha256=common.sha256(path)),
                split_hash=bound['split_hash'], training_scope='all', cohorts=bound['cohorts']['all'], prediction_scope='all',
                matched_control=dict(path=str(missing_path), sha256=common.sha256(missing_path)))
            exchange['component_manifests'][fold] = common.sha256(NEURAL / fold / 'manifest.json')
    for exchange in exchanges.values():
        dest = exchange['dest']
        common.write_json(dest / 'manifest.json', dict(status='complete', protocol_sha256=common.sha256(dest / 'protocol.json'),
            folds=exchange['folds'], source_hashes=sources, neural_component_manifests=exchange['component_manifests'],
            missing_component_manifest_sha256=protocol['missing_manifest_sha256']))
    print('COMPLETE both fixed compositions; metrics delegated to independent evaluator', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    else:
        run()
