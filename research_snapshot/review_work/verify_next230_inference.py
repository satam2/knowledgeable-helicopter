"""Independent saved-model replay, starting from the frozen baseline bundle."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from next230_common import OUT, OLD, load_data, load_reference, load_extension, same_split
from next230_features import known_designator, blend_weights, apply_blend
from taxiout.artifacts import read_json, write_json, sha256
from taxiout.predict import load_bundle, predict_features


def checked_record(path):
    record = read_json(path / 'manifest.json')
    if record['status'] != 'complete':
        raise ValueError('Cannot replay incomplete model')
    for name, digest in record['outputs'].items():
        if sha256(path / name) != digest:
            raise ValueError(f'Corrupt replay input: {path}/{name}')
    return record


def model_predict(path, x):
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model.predict(x, thread_count=4)


def replay(path, baseline, x, extensions, meta):
    record = checked_record(path)
    name = record['candidate']
    values = baseline.prediction_sec.to_numpy().copy()
    if 'seed_sensitivity' in record:
        from next230_clock_gate import gate_features
        dependencies={k:Path(v) for k,v in record['seed_sensitivity'].items()}
        for key,dependency in dependencies.items():
            filename='expert.json' if key=='direct' else 'manifest.json'
            if sha256(dependency/filename)!=record['dependency_hashes'][key]:
                raise ValueError('Seed sensitivity dependency changed')
        residual=replay(dependencies['residual'],baseline,x,extensions,meta)
        values=replay(dependencies['selected'],baseline,x,extensions,meta)
        direct=model_predict(dependencies['direct']/'component.cbm',x)
        weights=np.clip(model_predict(dependencies['gate']/'gate.cbm',gate_features(x,residual,direct)),0,1)
        use=baseline.route.eq('residual').to_numpy()
        values[use]=residual[use]
        return apply_blend(values,direct,baseline.route,weights)
    if 'calibrated_mixture_path' in record:
        from next230_calibrate_mixture import calibrate_probability
        from next230_schedule_mixture import predict_mixture
        mixture_path = Path(record['calibrated_mixture_path'])
        source = checked_record(mixture_path)
        if sha256(mixture_path/'manifest.json') != record['mixture_manifest_sha256'] or not same_split(source['split'],record['split']):
            raise ValueError('Calibrated mixture source mismatch')
        schema = read_json(mixture_path/'mixture.json')
        inputs = extensions['schedule']
        classifier = CatBoostClassifier()
        classifier.load_model(str(mixture_path/'reliability.cbm'))
        probability = calibrate_probability(classifier.predict_proba(inputs,thread_count=4)[:,1],read_json(path/'calibration.json'))
        candidate = predict_mixture(meta.schedule_sec.to_numpy(),probability,schema['consistent_mean'],
                                    model_predict(mixture_path/'inconsistent.cbm',inputs))
        use = baseline.route.eq('rome_schedule_residual').to_numpy()
        values[use] = candidate[use]
        return values
    if record.get('mixture'):
        from next230_schedule_mixture import predict_mixture
        schema = read_json(path / 'mixture.json')
        inputs = extensions['schedule']
        if list(inputs) != schema['columns'] or {c: str(inputs[c].dtype) for c in inputs} != schema['dtypes']:
            raise ValueError('Mixture schema mismatch')
        classifier = CatBoostClassifier()
        classifier.load_model(str(path / 'reliability.cbm'))
        correction = model_predict(path / 'inconsistent.cbm', inputs)
        candidate = predict_mixture(meta.schedule_sec.to_numpy(), classifier.predict_proba(inputs, thread_count=4)[:, 1],
                                    schema['consistent_mean'], correction)
        use = baseline.route.eq('rome_schedule_residual').to_numpy()
        values[use] = candidate[use]
        return values
    if 'composition' in record:
        for component in record['composition']:
            for component_path in component['paths']:
                if sha256(Path(component_path)/'manifest.json') != component['manifest_hashes'][component_path]:
                    raise ValueError('Composition dependency manifest changed')
            use = baseline.route.isin(component['routes']).to_numpy()
            sources = [replay(Path(p), baseline, x, extensions, meta) for p in component['paths']]
            averaged = np.mean(sources, axis=0)
            values[use] = averaged[use]
        return values
    if 'direct_expert_path' in record:
        direct_path = Path(record['direct_expert_path'])
        expert = read_json(direct_path / 'expert.json')
        if sha256(direct_path / 'expert.json') != record['expert_manifest_sha256'] or not same_split(expert['split'], record['split']):
            raise ValueError('Direct expert provenance mismatch')
        for file, digest in expert['outputs'].items():
            if sha256(direct_path / file) != digest:
                raise ValueError('Direct expert hash mismatch')
        residual_path = Path(record['residual_expert_path'])
        residual_record = checked_record(residual_path)
        if sha256(residual_path / 'manifest.json') != record['residual_manifest_sha256'] or not same_split(residual_record['split'], record['split']):
            raise ValueError('Residual expert provenance mismatch')
        residual_values = replay(residual_path, baseline, x, extensions, meta)
        use = baseline.route.eq('residual').to_numpy()
        values[use] = residual_values[use]
        direct = model_predict(direct_path / 'component.cbm', x)
        if record.get('learned_gate'):
            from next230_clock_gate import gate_features
            inputs=gate_features(x,residual_values,direct)
            schema=read_json(path/'gate_schema.json')
            if list(inputs)!=schema['columns'] or {c:str(inputs[c].dtype) for c in inputs}!=schema['dtypes']:
                raise ValueError('Learned gate schema mismatch')
            weights=np.clip(model_predict(path/'gate.cbm',inputs),0,1)
            return apply_blend(values,direct,baseline.route,weights)
        gate = read_json(path / 'gate.json')
        return apply_blend(values, direct, baseline.route, blend_weights(gate, meta.ADEP_mvt, record['conditioned']))
    if name.startswith('clock_'):
        experts = Path(record['expert_path']) if 'expert_path' in record else path
        expert_record = checked_record(experts)
        if not same_split(record['split'], expert_record['split']):
            raise ValueError('Gate/expert split mismatch')
        direct = model_predict(experts / 'component.cbm', x)
        gate = read_json(experts / 'gate.json')
        conditioned = record.get('conditioned', True)
        return apply_blend(values, direct, baseline.route, blend_weights(gate, meta.ADEP_mvt, conditioned))
    schema = read_json(path / 'component_schema.json')
    inputs = x if schema['feature_block'] == 'base' else extensions[schema['feature_block']]
    if list(inputs) != schema['columns'] or {c: str(inputs[c].dtype) for c in inputs} != schema['dtypes']:
        raise ValueError('Saved component feature schema mismatch')
    correction = model_predict(path / 'component.cbm', inputs)
    if schema['component_route'] == 'rome':
        use = baseline.route.eq('rome_schedule_residual').to_numpy()
        if 'fallback' in schema:
            fallback_path = Path(schema['fallback'])
            fallback_record = checked_record(fallback_path)
            if not same_split(record['split'], fallback_record['split']):
                raise ValueError('Fallback split mismatch')
            fallback = model_predict(fallback_path / 'component.cbm', x)
            known = known_designator(inputs, schema['designator_support'])
            correction[~known] = fallback[~known]
        values[use] = meta.schedule_sec.to_numpy()[use] + correction[use]
    else:
        use = baseline.route.isin(['residual', 'residual_long_proxy']).to_numpy()
        values[use] = meta.proxy_sec.to_numpy()[use] + correction[use]
    return values


def verify(candidates=None):
    x, meta, _ = load_data()
    all_extensions = {kind: load_extension(x, kind) for kind in ['runway', 'schedule']}
    receipts = []
    for fold in ['F1', 'F2', 'F3', 'G1']:
        paths = [p.parent for p in (OUT / 'models').glob('*/manifest.json')
                 if (r := read_json(p))['status'] == 'complete' and r['fold'] == fold
                 and (not candidates or r['candidate'] in candidates)]
        if not paths:
            continue
        record, reference, idx, split = load_reference(fold, meta)
        sx, sm = x.iloc[idx['score']], meta.iloc[idx['score']]
        ext = {k: v.iloc[idx['score']] for k, v in all_extensions.items()}
        pipeline, models, config = load_bundle(OLD / 'models' / record['run_id'])
        baseline = predict_features(pipeline, models, config, sx, sm)
        if not np.allclose(baseline.prediction_sec, reference.prediction_sec, rtol=0, atol=1e-9):
            raise ValueError('Frozen baseline full-model replay differs')
        for path in sorted(paths):
            values = replay(path, baseline, sx, ext, sm)
            expected = pd.read_parquet(path / 'score_predictions.parquet')
            delta = float(np.max(np.abs(values - expected.prediction_sec)))
            if delta > 1e-9:
                raise ValueError(f'Full component replay mismatch: {path.name}, {delta}')
            receipts.append({'run': path.name, 'max_abs_delta_sec': delta,
                             'manifest_sha256': sha256(path / 'manifest.json')})
            print(f'REPLAY {path.name}: {delta}', flush=True)
    write_json(OUT / 'inference_verification.json', {'verified': receipts, 'count': len(receipts),
        'verifier_sha256': sha256(__file__), 'status': 'all selected saved pipelines reproduce score predictions'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+')
    verify(parser.parse_args().candidates)
