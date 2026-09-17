"""Fixed final nine-expert simplex, separate declare/prepare/score stages."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / 'sequence_result_audit'))
import gate
import run_tune
import audit
from taxiout.metrics import evaluate, paired_stability, season_score

BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = BASE / 'models/context_gate/final_simplex9_v1'
PRIOR4 = BASE / 'models/context_gate/simplex4_context_v1'
EXPERTS = ['tabm_source', 'tabm_combined', 'lgb63', 'lgb63_sequence', 'tabm_ple8', 'lgb63_sequence8', 'lgb63_union', 'catboost_combined', 'xgb_combined']
ID, TARGET, MOVEMENT = run_tune.ID, run_tune.TARGET, run_tune.MOVEMENT
read_json, write_json, sha256, object_hash, utc_now = run_tune.read_json, run_tune.write_json, run_tune.sha256, run_tune.object_hash, run_tune.utc_now


def registry(fold):
    return {'tabm_source': BASE / 'models/augmented/source_past__source_twosided' / f'tabm_aobt_allfinite_{fold}_s20260916',
            'tabm_combined': BASE / 'full_neural' / f'tabm_combined_standard_aobt_allfinite_{fold}_s20260916',
            'lgb63': BASE / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
            'lgb63_sequence': BASE / 'deeper_sequence_v2' / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916',
            'tabm_ple8': BASE / 'full_neural' / f'tabm_combined_ple8_aobt_allfinite_{fold}_s20260916',
            'lgb63_sequence8': BASE / 'deeper_sequence8' / f'lightgbm_leaf63_sequence8_aobt_allfinite_{fold}_s20260916',
            'lgb63_union': BASE / 'deeper_context_union' / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916',
            'catboost_combined': BASE / 'combined_catboost' / f'catboost_combined_aobt_allfinite_{fold}_s20260916',
            'xgb_combined': BASE / 'combined_xgb' / f'xgb_aobt_allfinite_{fold}_s20260916'}


def declare():
    protocol = {'experts': EXPERTS, 'folds': ['F1', 'F3'], 'threads': 2,
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), Path(gate.__file__), Path(run_tune.__file__), Path(audit.__file__)]},
        'source_directories': {fold: {name: str(path) for name, path in registry(fold).items()} for fold in ('F1', 'F3')},
        'prior4_control_sha256': sha256(PRIOR4 / 'score_summary.json'),
        'deadline_policy': 'All9expertsrequired. Atcampaign deadline ifanymissing/failed/incomplete, reportfinal9unrun; neverdropanexpert orsubstitute variant basedonscores.',
        'finality': 'Onefinalcomplementaritytestoverfixedpending/completedbranches; no new modelgeneration,scoregridoradaptiveexpertremoval.',
        'global_objective': 'Originalordinarytune rawSSE, convexweights>=0 andsum1, nointercept.',
        'airport_objective': 'Originalairporttune rawSSE +1000*globalmean(sum((experts-meanexperts)^2))*sum((weights-globalweights)^2).',
        'unknown_airport': 'Globalweights.', 'prediction_scope': 'OnlyfiniteNMproxy0..7200 inclusive. Everyotherroute preserves V2exactly.',
        'cohort': 'AlloriginalpurgedordinarytunefromJune/October; all9expertmodels originallyfitallfiniteNM.',
        'selection': 'These9expertsfixedbeforeanynewscoreaccess; no coefficientgrids, extrapolation orscoreselectedweights.',
        'stages': 'Declarefirst; prepareonlywhenall9expertscompletebothfolds; savweightsbeforeseparate scorestage.',
        'seed_scope': 'Expertseed20260916only. Deterministicmixture coefficients are notmultiseedpipeline evidence.',
        'caveat': 'Adaptive exposeddevelopment; originaltunealreadyselectedexpertstopping. Notfreshholdoutorleaderboard.'}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert read_json(path) == protocol, 'Preserve frozen protocol'
    else:
        write_json(path, protocol)
        snap = OUT / 'source_snapshots'
        snap.mkdir()
        for name in protocol['source_hashes']:
            shutil.copyfile(ROOT / name, snap / Path(name).name)
    return protocol


def prepare(protocol):
    if (OUT / 'preparation.json').exists():
        raise ValueError('Preserve completed preparation')
    receipts = {}
    for fold in ('F1', 'F3'):
        receipts[fold] = {}
        for name, folder in registry(fold).items():
            marker = read_json(folder / 'manifest.json')
            assert marker['status'] == 'complete', f'Await complete {fold}/{name}'
            assert marker['fit']['steps'] == marker['refit']['steps']
            assert marker['fit']['rows'] == marker['fit_ids']['fit']['n']
            assert marker['refit']['rows'] == marker['fit_ids']['refit']['n']
            assert marker['reload_max_abs_delta'] == 0.
            for filename in ('fit_model.joblib', 'model.joblib', 'tune_predictions.parquet'):
                assert sha256(folder / filename) == marker['outputs'][filename]
            receipts[fold][name] = {'directory': str(folder), 'manifest_sha256': sha256(folder / 'manifest.json'),
                'outputs': marker['outputs'], 'fit_ids': marker['fit_ids'], 'split': marker['split']}
        first = receipts[fold][EXPERTS[0]]
        assert all(r['fit_ids'] == first['fit_ids'] and r['split'] == first['split'] for r in receipts[fold].values())
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(meta_path) == read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    results = {}
    for fold in ('F1', 'F3'):
        split = receipts[fold][EXPERTS[0]]['split']
        start, end = split['spec']['tune']
        frame = pd.read_parquet(meta_path, columns=[ID, TARGET, MOVEMENT, 'ADEP_mvt', 'proxy_sec'],
            filters=[(MOVEMENT, '>=', pd.Timestamp(start, tz='UTC')), (MOVEMENT, '<', pd.Timestamp(end, tz='UTC'))]).set_index(ID)
        assert len(frame) == split['stages']['tune']['n']
        assert object_hash(frame.index.tolist()) == split['stages']['tune']['id_hash']
        finite = np.isfinite(frame.proxy_sec)
        ordinary = finite & frame.proxy_sec.between(0, 7200)
        selected = frame.loc[ordinary]
        p = []
        for name in EXPERTS:
            source = pd.read_parquet(Path(receipts[fold][name]['directory']) / 'tune_predictions.parquet').set_index(ID)
            np.testing.assert_array_equal(source.index, frame.index[finite])
            assert object_hash(source.index.tolist()) == receipts[fold][name]['fit_ids']['tune']['hash']
            p.append(source.loc[selected.index, 'prediction_sec'].to_numpy())
        p = np.column_stack(p)
        y = selected[TARGET].to_numpy(float)
        airports = selected.ADEP_mvt.astype(str).to_numpy()
        assert np.isfinite(p).all() and np.isfinite(y).all()
        global_w, local, penalty = gate.airport_weights(y, p, airports)
        assert np.all(global_w >= 0) and np.isclose(global_w.sum(), 1.)
        assert all(np.all(w >= 0) and np.isclose(w.sum(), 1.) for w in local.values())
        weights = {'experts': EXPERTS, 'global': global_w.tolist(), 'airports': {a: w.tolist() for a, w in local.items()},
            'penalty': penalty, 'shrinkage_pseudorows': 1000, 'tune_rows': len(y), 'tune_id_hash': object_hash(selected.index.tolist())}
        write_json(OUT / f'{fold}_weights.json', weights)
        selected.assign(**{name: p[:, i] for i, name in enumerate(EXPERTS)}).reset_index().to_parquet(OUT / f'{fold}_aligned_tune.parquet', index=False)
        results[fold] = {'weights_sha256': sha256(OUT / f'{fold}_weights.json'),
            'aligned_tune_sha256': sha256(OUT / f'{fold}_aligned_tune.parquet'), 'weights': weights,
            'global_tune_rmse': run_tune.rmse(y, p @ global_w),
            'airport_tune_rmse': run_tune.rmse(y, np.sum(p * np.array([local[a] for a in airports]), axis=1))}
        print('PREPARED_SIMPLEX9', fold, weights, flush=True)
        del frame, selected, p, source
        gc.collect()
    write_json(OUT / 'preparation.json', {'status': 'prepared', 'created_utc': utc_now(), 'sources': receipts, 'folds': results,
        'protocol_sha256': sha256(OUT / 'protocol.json'), 'score_rows_read': False, 'peak_rss_bytes': peak()})


def peak():
    info = psutil.Process().memory_info()
    return int(getattr(info, 'peak_wset', info.rss))


def verify_weights(preparation):
    for fold in ('F1', 'F3'):
        rec = preparation['folds'][fold]
        path = OUT / f'{fold}_aligned_tune.parquet'
        assert sha256(path) == rec['aligned_tune_sha256']
        assert sha256(OUT / f'{fold}_weights.json') == rec['weights_sha256']
        saved = read_json(OUT / f'{fold}_weights.json')
        data = pd.read_parquet(path)
        assert object_hash(data[ID].tolist()) == saved['tune_id_hash']
        p, y, airports = data[EXPERTS].to_numpy(), data[TARGET].to_numpy(), data.ADEP_mvt.astype(str).to_numpy()
        w, local, penalty = gate.airport_weights(y, p, airports)
        np.testing.assert_allclose(w, saved['global'], rtol=0, atol=1e-12)
        assert penalty == saved['penalty']
        for a, value in local.items():
            np.testing.assert_allclose(value, saved['airports'][a], rtol=0, atol=1e-12)


def score(protocol):
    preparation = read_json(OUT / 'preparation.json')
    assert preparation['status'] == 'prepared' and preparation['protocol_sha256'] == sha256(OUT / 'protocol.json')
    if (OUT / 'scoring_protocol.json').exists():
        raise ValueError('Preserve completed or attempted scoring')
    verify_weights(preparation)
    write_json(OUT / 'scoring_protocol.json', {'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'preparation_sha256': sha256(OUT / 'preparation.json'), 'variants': ['global9', 'airport9'],
        'coefficient_replay_passed_before_score': True, 'training_on_score': False})
    results = {}
    for fold in ('F1', 'F3'):
        folder = OUT / f'{fold}_score'
        folder.mkdir()
        weights = read_json(OUT / f'{fold}_weights.json')
        reference, refmarker = run_tune.common.reference(fold)
        airports = reference.ADEP_mvt.astype(str).to_numpy()
        proxy = reference.proxy_sec.to_numpy()
        eligible = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
        experts, controls = [], {'reference': reference}
        for name in EXPERTS:
            receipt = preparation['sources'][fold][name]
            directory = Path(receipt['directory'])
            assert sha256(directory / 'manifest.json') == receipt['manifest_sha256']
            assert receipt['split'] == refmarker['split']
            path = directory / 'candidate.parquet'
            assert sha256(path) == receipt['outputs'][path.name]
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame[ID], reference[ID])
            np.testing.assert_array_equal(frame[TARGET], reference[TARGET])
            experts.append(frame.prediction_sec.to_numpy())
            if name in ('lgb63', 'lgb63_sequence'):
                controls[name] = frame
        p = np.column_stack(experts)
        for label, directory, filename in [('v3_global', BASE / 'models/deeper_audit/simplex_v3' / fold, 'global_simplex.parquet'),
            ('v3_airport', BASE / 'models/deeper_audit/simplex_v3' / fold, 'airport_shrunk_simplex.parquet'),
            ('previous_airport3', BASE / 'models/context_gate/simplex3_v1' / f'{fold}_score', 'airport3.parquet')]:
            marker = read_json(directory / 'manifest.json')
            assert sha256(directory / filename) == marker['outputs'][filename]
            controls[label] = pd.read_parquet(directory / filename)
        assert sha256(PRIOR4 / 'score_summary.json') == protocol['prior4_control_sha256']
        prior = read_json(PRIOR4 / f'{fold}_score' / 'manifest.json')
        for label in ('global4', 'airport4'):
            path = PRIOR4 / f'{fold}_score' / f'{label}.parquet'
            assert sha256(path) == prior['outputs'][path.name]
            controls['prior_' + label] = pd.read_parquet(path)
        variants = {}
        for variant in ('global9', 'airport9'):
            coefficients = np.tile(weights['global'], (eligible.sum(), 1)) if variant == 'global9' else np.array([weights['airports'].get(a, weights['global']) for a in airports[eligible]])
            assert np.all(coefficients >= 0) and np.allclose(coefficients.sum(1), 1.)
            values = reference.prediction_sec.to_numpy().copy()
            values[eligible] = np.sum(p[eligible] * coefficients, axis=1)
            np.testing.assert_array_equal(values[~eligible], reference.prediction_sec.to_numpy()[~eligible])
            assert np.all(values[eligible] >= p[eligible].min(1) - 1e-8) and np.all(values[eligible] <= p[eligible].max(1) + 1e-8)
            frame = reference.drop(columns=[TARGET, 'error_sec', 'squared_error', 'label_bin', 'month', 'day'], errors='ignore').copy()
            frame['prediction_sec'] = values
            metrics, errors = evaluate(frame, reference[[ID, TARGET]])
            errors.to_parquet(folder / f'{variant}.parquet', index=False)
            np.testing.assert_array_equal(pd.read_parquet(folder / f'{variant}.parquet').prediction_sec, values)
            comparisons = {name: audit.compare(control, errors) for name, control in controls.items()}
            variants[variant] = {'metrics': metrics, 'comparisons': comparisons,
                'v3_airport_block_stability': paired_stability(controls['v3_airport'], errors, repetitions=1000),
                'protected_rows': int((~eligible).sum()), 'protected_v2_exact': True,
                'support_floor_rmse': float(np.sqrt(errors.squared_error.to_numpy()[~eligible].sum() / len(errors))),
                'saved_prediction_replay_delta': 0.}
        rec = {'status': 'complete', 'variants': variants, 'weights_sha256': sha256(OUT / f'{fold}_weights.json'),
            'outputs': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
        write_json(folder / 'manifest.json', rec)
        results[fold] = rec
        print('SCORED_SIMPLEX9', fold, {name: row['metrics']['overall']['rmse_sec'] for name, row in variants.items()}, flush=True)
    seasonal = {name: season_score(results['F1']['variants'][name]['metrics']['overall'], results['F3']['variants'][name]['metrics']['overall']) for name in ('global9', 'airport9')}
    write_json(OUT / 'score_summary.json', {'status': 'complete', 'created_utc': utc_now(), 'seasonal_rmse': seasonal,
        'folds': results, 'peak_rss_bytes': peak(), 'source_sha256': sha256(__file__), 'seed_scope': protocol['seed_scope']})
    print('SEASONAL_SIMPLEX9', seasonal, flush=True)


def readiness():
    states = {}
    for fold in ('F1', 'F3'):
        states[fold] = {}
        for expert, directory in registry(fold).items():
            path = directory / 'manifest.json'
            states[fold][expert] = read_json(path).get('status', 'unknown') if path.exists() else 'missing'
    ready = all(state == 'complete' for fold in states.values() for state in fold.values())
    report = {'status': 'ready' if ready else 'unrun_pending_required_experts', 'created_utc': utc_now(), 'states': states,
              'all9_required': True, 'no_selective_exclusion': True, 'protocol_sha256': sha256(OUT / 'protocol.json')}
    destination = OUT / ('readiness_' + utc_now().replace(':', '-').replace('+', '_') + '.json')
    write_json(destination, report)
    print('FINAL9_READINESS', report, flush=True)
    return ready


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['declare', 'readiness', 'prepare', 'score'], required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        protocol = declare()
        if args.stage == 'readiness':
            readiness()
        elif args.stage == 'prepare':
            if not readiness():
                raise RuntimeError('Final9 unrun: all nine experts must complete both folds')
            prepare(protocol)
        elif args.stage == 'score':
            score(protocol)
