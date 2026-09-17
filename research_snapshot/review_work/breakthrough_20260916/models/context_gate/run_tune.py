"""One declared contextual gate, early60/late40 original tune only."""
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
sys.path.insert(0, str(HERE.parent / 'stacking'))
import common
import simplex
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET, MOVEMENT

OUT = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/early60_late40_v1'
BASE = ROOT / 'private_runs/breakthrough_20260916'
EXPERTS = ['tabm_source', 'lgb63', 'catboost_conventions']


def registry(fold):
    return {'tabm_source': BASE / 'models/augmented/source_past__source_twosided' / f'tabm_aobt_allfinite_{fold}_s20260916',
            'lgb63': BASE / 'deeper_lgb/combined' / f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
            'catboost_conventions': BASE / 'missing/conventions_models/models' / f'global_conventions_{fold}_s20260916'}


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(prediction))**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {'created_utc': utc_now(), 'source_hashes': {path.name: sha256(path) for path in [Path(__file__), Path(gate.__file__), Path(simplex.__file__)]},
                'experts': EXPERTS, 'folds': ['F1', 'F3'], 'threads': 2,
                'cohort': 'All original purged tune rows with finite observable proxy in[0,7200]; no label filters',
                'split': 'Sort original tune by movementtime then ID; early rows strictly before timestamp at floor(0.6*N), late at/after boundary',
                'features': '85 verified source conventions plus airport and two signed expert disagreements; early-fit median, missingindicator, standardscale, onehot',
                'model': 'Single linear softmax gate over three experts; convex weights, raw mixture squared error, no bestexpert classifier',
                'penalty': gate.PENALTY, 'max_iterations': gate.MAX_ITERATIONS,
                'penalty_scaling': 'Loss divided by early global mixture MSE; .001 sum squared nonintercept coefficients',
                'controls': ['three individual experts', 'two-expert global/airport simplex', 'three-expert global/airport simplex'],
                'airport3_penalty': '1000 times early global mean sum centered expertprediction squares',
                'selection': 'One model only; report late40 comparison before any fullTune training or score evaluation',
                'restriction': 'No scoreprediction or scorelabel access; no fullTune retrain; no GPU',
                'caveat': 'Tune labels already select expert stopping; chronological split is a diagnostic, not fresh holdout'}
    write_json(OUT / 'protocol.json', protocol)
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(meta_path) == read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, MOVEMENT, 'FLIGHT_ID_mvt', 'proxy_sec', 'ADEP_mvt'])
    convention_root = BASE / 'missing/source_conventions'
    convention_manifest = read_json(convention_root / 'manifest.json')
    assert sha256(convention_root / 'features.parquet') == convention_manifest['feature_sha256']
    conventions = pd.read_parquet(convention_root / 'features.parquet').set_index(ID)
    assert conventions.index.is_unique and list(conventions) == convention_manifest['columns']
    assert len(conventions.columns) == 85
    assert TARGET not in conventions and 'BLOCK_TIME_UTC_mvt' not in conventions
    results = {}
    for fold in ['F1', 'F3']:
        started = time.monotonic()
        directory = OUT / fold
        directory.mkdir()
        idx, split = common.make_fold(meta, common.load_config('configs/folds.yaml')[fold])
        original = meta.iloc[idx['tune']].set_index(ID)
        eligible = np.isfinite(original.proxy_sec) & original.proxy_sec.between(0, 7200)
        selected = original.loc[eligible].reset_index().sort_values([MOVEMENT, ID], kind='stable').set_index(ID)
        start, end = split['spec']['tune']
        labels = pd.read_parquet(meta_path, columns=[ID, TARGET], filters=[(MOVEMENT, '>=', pd.Timestamp(start, tz='UTC')),
                                                                           (MOVEMENT, '<', pd.Timestamp(end, tz='UTC'))]).set_index(ID)
        y = labels.loc[selected.index, TARGET].to_numpy(float)
        predictions, receipts = [], {}
        for expert, folder in registry(fold).items():
            record = read_json(folder / 'manifest.json')
            assert record['status'] == 'complete' and object_hash(record['split']) == object_hash(split)
            path = folder / 'tune_predictions.parquet'
            assert sha256(path) == record['outputs'][path.name]
            frame = pd.read_parquet(path).set_index(ID)
            expected = original.index[eligible] if expert == 'catboost_conventions' else original.index[np.isfinite(original.proxy_sec)]
            np.testing.assert_array_equal(frame.index, expected)
            assert record['fit_ids']['tune']['hash'] == object_hash(frame.index.tolist())
            predictions.append(frame.loc[selected.index, 'prediction_sec'].to_numpy())
            receipts[expert] = {'tune_sha256': sha256(path), 'manifest_sha256': sha256(folder/'manifest.json'), 'fit_ids': record['fit_ids']}
        p = np.column_stack(predictions)
        assert np.isfinite(p).all() and np.isfinite(y).all()
        cutoff = selected.iloc[int(.6*len(selected))][MOVEMENT]
        early = (selected[MOVEMENT] < cutoff).to_numpy()
        late = ~early
        features = conventions.loc[selected.index].copy()
        features['airport'] = selected.ADEP_mvt.astype(str)
        airports = features.airport.to_numpy()
        with threadpool_limits(2):
            w3, airport3, penalty = gate.airport_weights(y[early], p[early], airports[early])
            w2 = simplex.solve(y[early], p[early, 0], p[early, 1], airports[early])
            model, evidence = gate.fit(features.iloc[np.flatnonzero(early)], p[early], y[early])
            proposed, weights = gate.predict(model, features, p)
        predictions_out = {name: p[:, i] for i, name in enumerate(EXPERTS)}
        predictions_out.update(global3=p@w3, airport3=np.sum(p*np.array([airport3.get(a, w3) for a in airports]), axis=1),
                               global2=simplex.predict(w2, p[:,0], p[:,1], airports, False),
                               airport2=simplex.predict(w2, p[:,0], p[:,1], airports, True), context_gate=proposed)
        metrics = {name: {'early_rmse': rmse(y[early], values[early]), 'late_rmse': rmse(y[late], values[late])}
                   for name, values in predictions_out.items()}
        joblib.dump(model, directory/'early_model.joblib')
        with threadpool_limits(2):
            replay, _ = gate.predict(joblib.load(directory/'early_model.joblib'), features, p)
        np.testing.assert_array_equal(proposed, replay)
        pd.DataFrame({ID: selected.index, TARGET: y, MOVEMENT: selected[MOVEMENT], 'early': early,
                      **predictions_out}).reset_index(drop=True).to_parquet(directory/'tune_diagnostic.parquet', index=False)
        record = {'status': 'complete', 'cutoff': str(cutoff), 'early_n': int(early.sum()), 'late_n': int(late.sum()),
                  'original_ordinary_tune_n': len(y), 'early_id_hash': object_hash(selected.index[early].tolist()),
                  'late_id_hash': object_hash(selected.index[late].tolist()), 'metrics': metrics, 'gate_fit': evidence,
                  'global3_weights': w3.tolist(), 'airport3_weights': {a:w.tolist() for a,w in airport3.items()},
                  'airport3_penalty': penalty, 'global2': w2, 'receipts': receipts,
                  'conventions_manifest_sha256': sha256(convention_root/'manifest.json'),
                  'late_mean_gate_weights': weights[late].mean(0).tolist(), 'runtime_sec': time.monotonic()-started,
                  'outputs': {path.name: sha256(path) for path in directory.iterdir() if path.is_file()}}
        write_json(directory/'manifest.json', record)
        results[fold] = record
        print('TUNE_CONTEXT', fold, metrics, 'optimizer', evidence, flush=True)
    write_json(OUT/'summary.json', {'created_utc': utc_now(), 'folds': results,
                                   'protocol_sha256': sha256(OUT/'protocol.json'), 'score_read': False, 'full_tune_retrain': False})


if __name__ == '__main__':
    main()
