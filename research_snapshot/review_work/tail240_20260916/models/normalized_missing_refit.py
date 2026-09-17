"""Gated full refit of the declared observed-scale missing-clock experiment."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import normalized_missing_tune as prior

common, shared = prior.common, prior.shared
ROOT, ID, TARGET = prior.ROOT, prior.ID, prior.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v1')
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
ARMS = ['unscaled', 'observed_schedule_scale']


def declaration():
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(source_sha256=common.sha256(__file__),
        prior_source_sha256=common.sha256(prior.__file__),
        prior_protocol_sha256=common.sha256(prior.OUT / 'protocol.json'),
        prior_summary_sha256=common.sha256(prior.OUT / 'summary.json'),
        baseline_binding_sha256=common.sha256(BINDING), arms=ARMS,
        params=prior.PARAMS, variants={'candidate': 1., 'blend25': .25},
        fitting='Original full missing-NM refit. Refit-only category vocabulary and scale weight normalizer; each arm uses its own frozen tune best iteration.',
        prediction='Raw 900+observable_scale*prediction, no clipping. Only missing NM changes; finite routes bit-identical to global9.',
        gate='Independent normalized tune replay and both-month leave-one-day gain against both controls before executing refit.',
        evaluation='All original score rows and raw labels. Exposed development, no ranking fit or submission.',
        source_hashes={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in
            [__file__, prior.__file__, shared.__file__, shared.base.__file__, shared.identity.__file__,
             ROOT / 'review_work/breakthrough_20260916/models/encoders.py']})
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return value


def run(receipt_path):
    protocol = declaration()
    receipt = common.read_json(receipt_path)
    assert receipt['status'] == 'passed'
    # The independently produced gate must explicitly authorize advancement.
    assert receipt['advance_normalized_refit'] is True
    assert receipt['source_sha256'] == common.sha256(prior.__file__)
    assert receipt['summary_sha256'] == common.sha256(prior.OUT / 'summary.json')
    assert common.read_json(prior.OUT / 'summary.json')['results']['both_month_gate']
    binding = common.read_json(BINDING)
    x, meta = shared.load_missing()
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    y = meta[TARGET].to_numpy(float)
    exchanges = {}
    for arm in ARMS:
        for variant, weight in protocol['variants'].items():
            dest = OUT / f'{arm}_{variant}'
            dest.mkdir(exist_ok=False)
            exchange_protocol = dict(name=f'normalized_missing_{arm}_{variant}',
                baseline_binding_sha256=common.sha256(BINDING), score_labels_used_for_selection=False,
                routing_uses_score_targets=False, selection_periods=['fit', 'tune'],
                inference_columns=list(x), prediction_transform='raw_unclipped',
                tail_definition='AllmissingNM, no target-defined route',
                routing_definition='MissingNM only; finite global9 frozen',
                selection_rule=f'Frozen tune tree count, {arm}, fixed weight {weight}',
                source_protocol_sha256=common.sha256(OUT / 'protocol.json'))
            common.write_json(dest / 'protocol.json', exchange_protocol)
            exchanges[(arm, variant)] = dict(dest=dest, folds={})
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        assert common.object_hash(split) == binding['folds'][fold]['split_hash']
        rows = {s: p[missing[p]] for s, p in idx.items()}
        cohorts = {s: dict(n=len(p), hash=common.object_hash(meta.iloc[p][ID].tolist())) for s,p in rows.items()}
        assert cohorts == binding['folds'][fold]['cohorts']['missing_nm']
        frames = {s: x.loc[meta.iloc[rows[s]][ID]] for s in ('refit', 'score')}
        baseline_path = shared.GLOBAL / fold / 'candidate.parquet'
        marker = common.read_json(shared.GLOBAL / fold / 'manifest.json')
        assert common.sha256(baseline_path) == marker['outputs']['candidate.parquet']
        baseline = pd.read_parquet(baseline_path, columns=[ID, 'prediction_sec'])
        np.testing.assert_array_equal(baseline[ID], meta.iloc[idx['score']][ID])
        encoder = prior.FrameEncoder().fit(frames['refit'])
        matrices = {s: encoder.transform(f) for s,f in frames.items()}
        for arm in ARMS:
            dest = OUT / 'models' / fold / arm
            dest.mkdir(parents=True, exist_ok=False)
            previous = common.read_json(prior.OUT / fold / arm / 'manifest.json')
            assert previous['source_sha256'] == protocol['prior_source_sha256']
            assert previous['protocol_sha256'] == protocol['prior_protocol_sha256']
            scales = {s: np.ones(len(f)) if arm == 'unscaled' else prior.scale_of(f) for s,f in frames.items()}
            normalizer = float(np.mean(scales['refit']**2))
            weights = scales['refit']**2 / normalizer
            target = prior.transformed(y[rows['refit']], scales['refit'])
            dataset = prior.lgb.Dataset(matrices['refit'], label=target, weight=weights,
                categorical_feature=list(encoder.categories))
            model = prior.lgb.train(prior.PARAMS, dataset, num_boost_round=previous['steps'])
            prediction = prior.reconstruct(model.predict(matrices['score']), scales['score'])
            assert np.isfinite(prediction).all()
            joblib.dump(dict(model=model, encoder=encoder, arm=arm), dest / 'model.joblib')
            saved = joblib.load(dest / 'model.joblib')
            replay = prior.reconstruct(saved['model'].predict(saved['encoder'].transform(frames['score'])), scales['score'])
            np.testing.assert_array_equal(prediction, replay)
            pd.DataFrame({ID: frames['score'].index, 'prediction_sec': prediction,
                'scale': scales['score']}).to_parquet(dest / 'missing_predictions.parquet', index=False)
            mask = missing[idx['score']]
            for variant, weight in protocol['variants'].items():
                exchange = exchanges[(arm,variant)]
                result = baseline.copy()
                values = result.prediction_sec.to_numpy(copy=True)
                values[mask] += weight * (prediction-values[mask])
                result['prediction_sec'] = values
                np.testing.assert_array_equal(values[~mask], baseline.prediction_sec.to_numpy()[~mask])
                path = exchange['dest'] / f'{fold}.parquet'
                result.to_parquet(path, index=False)
                exchange['folds'][fold] = dict(prediction=dict(path=path.name, sha256=common.sha256(path)),
                    split_hash=binding['folds'][fold]['split_hash'], training_scope='missing_nm',
                    cohorts=cohorts, prediction_scope='missing_nm')
            common.write_json(dest / 'manifest.json', dict(status='complete', fold=fold, arm=arm,
                steps=previous['steps'], split=split, fit_ids=cohorts, normalizer=normalizer,
                source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
                tune_manifest_sha256=common.sha256(prior.OUT / fold / arm / 'manifest.json'),
                gate_receipt_path=str(receipt_path), gate_receipt_sha256=common.sha256(receipt_path),
                reload_max_abs_delta=0., outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file()}))
            print('REFIT', fold, arm, previous['steps'], flush=True)
    for (arm,variant), exchange in exchanges.items():
        dest = exchange['dest']
        if arm == 'observed_schedule_scale':
            for fold in ('F1','F3'):
                control = exchanges[('unscaled',variant)]['dest'] / f'{fold}.parquet'
                exchange['folds'][fold]['matched_control'] = dict(path=str(control), sha256=common.sha256(control))
        common.write_json(dest / 'manifest.json', dict(status='complete',
            protocol_sha256=common.sha256(dest / 'protocol.json'), folds=exchange['folds'],
            source_hashes=protocol['source_hashes']))
    print('COMPLETE refit, four declared exchange variants; scores left to independent evaluator', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        assert args.gate_receipt is not None
        run(args.gate_receipt)
