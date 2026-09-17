"""Freeze three-expert weights from complete original tune only; no score reads."""
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import gate
import run_tune

ROOT = run_tune.ROOT
OUT = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/simplex3_v1'
EXPERTS = run_tune.EXPERTS
common = run_tune.common
read_json, write_json, sha256, object_hash, utc_now = run_tune.read_json, run_tune.write_json, run_tune.sha256, run_tune.object_hash, run_tune.utc_now
ID, TARGET, MOVEMENT = run_tune.ID, run_tune.TARGET, run_tune.MOVEMENT


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {}
    for fold in ['F1', 'F3']:
        sources[fold] = {}
        for expert, folder in run_tune.registry(fold).items():
            record = read_json(folder/'manifest.json')
            assert record['status'] == 'complete'
            if expert == 'catboost_conventions':
                assert record['tune']['selected_steps'] == record['refit']['selected_steps']
                assert record['tune']['groups']['global']['fit_rows'] == record['fit_ids']['fit']['n']
                assert record['refit']['groups']['global']['fit_rows'] == record['fit_ids']['refit']['n']
                assert record['tune']['groups']['global']['tune_rows'] == record['fit_ids']['tune']['n']
                model_files = ['fit_bank.json', 'model_bank.json', 'fit_models\\global.cbm', 'refit_models\\global.cbm']
            else:
                assert record['fit']['steps'] == record['refit']['steps']
                assert record['fit']['rows'] == record['fit_ids']['fit']['n']
                assert record['refit']['rows'] == record['fit_ids']['refit']['n']
                model_files = ['fit_model.joblib', 'model.joblib']
            for name in [*model_files, 'tune_predictions.parquet']:
                assert sha256(folder/name) == record['outputs'][name]
            sources[fold][expert] = {'directory': str(folder), 'manifest_sha256': sha256(folder/'manifest.json'),
                                    'outputs': record['outputs'], 'fit_ids': record['fit_ids'], 'split': record['split'],
                                    'model_parity_verified': True, 'source_hashes': record.get('source_hashes', {})}
    protocol = {'created_utc': utc_now(), 'experts': EXPERTS, 'sources': sources, 'threads': 2,
                'source_hashes': {path.name:sha256(path) for path in [Path(__file__), Path(gate.__file__), Path(run_tune.__file__)]},
                'global_objective': 'Minimize all-original-ordinary-tune mean(rawY-sum(weights*expert))^2, weights>=0,sum(weights)=1',
                'airport_objective': 'Local raw SSE +1000*globalmean(sum((expert-meanexpert)^2))*sum((weights-globalweights)^2)',
                'no_intercept': True, 'shrinkage_pseudorows': 1000, 'unknown_airport': 'globalweights',
                'prediction_scope': 'Only finite proxy in[0,7200]; missing,negative,long preserve V2 exactly',
                'expert_cohort_difference': 'CatBoost fit/tune/refit only ordinaryproxy; TabM/LGB63 allfiniteNM. Mixture uses exact common originalordinarytune cohort.',
                'restriction': 'Preparation only, no scoreprediction/scorelabel reads; no grid or contextual gate refit',
                'selection_rationale': 'Threeexpertglobal beats twoexpertglobal in both chronological early60/late40 tune checks',
                'development_caveat': 'Original tune already selected expert epochs; candidates selected in exposeddevelopment search',
                'historical_limitation': 'Leaf63 original source receipt omits imported campaign/lgbm_adapter.py encoder; no retroactive source claim.'}
    write_json(OUT/'protocol.json', protocol)
    meta_path = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(meta_path) == read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    chronological = read_json(run_tune.OUT/'summary.json')
    results = {}
    for fold in ['F1', 'F3']:
        records = sources[fold]
        split = records[EXPERTS[0]]['split']
        assert all(object_hash(r['split']) == object_hash(split) for r in records.values())
        start, end = split['spec']['tune']
        frame = pd.read_parquet(meta_path, columns=[ID, TARGET, MOVEMENT, 'ADEP_mvt', 'proxy_sec'],
                                filters=[(MOVEMENT, '>=', pd.Timestamp(start, tz='UTC')), (MOVEMENT, '<', pd.Timestamp(end, tz='UTC'))])
        assert len(frame) == split['stages']['tune']['n']
        assert object_hash(frame[ID].tolist()) == split['stages']['tune']['id_hash']
        frame = frame.set_index(ID)
        ordinary = np.isfinite(frame.proxy_sec) & frame.proxy_sec.between(0, 7200)
        selected = frame.loc[ordinary]
        predictions = []
        for expert in EXPERTS:
            receipt = records[expert]
            path = Path(receipt['directory'])/'tune_predictions.parquet'
            assert sha256(path) == receipt['outputs'][path.name]
            source = pd.read_parquet(path).set_index(ID)
            expected = frame.index[ordinary] if expert == 'catboost_conventions' else frame.index[np.isfinite(frame.proxy_sec)]
            np.testing.assert_array_equal(source.index, expected)
            assert object_hash(source.index.tolist()) == receipt['fit_ids']['tune']['hash']
            predictions.append(source.loc[selected.index, 'prediction_sec'].to_numpy())
        p = np.column_stack(predictions)
        y = selected[TARGET].to_numpy(float)
        airports = selected.ADEP_mvt.astype(str).to_numpy()
        with threadpool_limits(2):
            global_w, local, penalty = gate.airport_weights(y, p, airports)
        assert np.all(global_w >= 0) and np.isclose(global_w.sum(), 1.)
        assert all(np.all(w>=0) and np.isclose(w.sum(),1.) for w in local.values())
        weights = {'experts': EXPERTS, 'global': global_w.tolist(), 'airports': {a:w.tolist() for a,w in local.items()},
                   'penalty': penalty, 'shrinkage_pseudorows': 1000, 'tune_rows': len(y),
                   'tune_ids_hash': object_hash(selected.index.tolist())}
        write_json(OUT/f'{fold}_weights.json', weights)
        earlylate = chronological['folds'][fold]['metrics']
        results[fold] = {'weights': weights, 'full_tune_global_rmse': run_tune.rmse(y,p@global_w),
                         'full_tune_airport_rmse': run_tune.rmse(y,np.sum(p*np.array([local[a] for a in airports]),axis=1)),
                         'early_late': earlylate,
                         'late_delta_global3_vs_v3global2': earlylate['global3']['late_rmse']-earlylate['global2']['late_rmse'],
                         'late_delta_airport3_vs_v3airport2': earlylate['airport3']['late_rmse']-earlylate['airport2']['late_rmse'],
                         'weights_sha256': sha256(OUT/f'{fold}_weights.json')}
        print('PREPARED_SIMPLEX3', fold, results[fold], flush=True)
        del frame, selected, p, predictions, source
        gc.collect()
    results_payload = {'created_utc': utc_now(), 'folds': results, 'score_read': False, 'status': 'prepared',
                       'protocol_sha256': sha256(OUT/'protocol.json'),
                       'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}
    write_json(OUT/'preparation.json', results_payload)


if __name__ == '__main__':
    main()
