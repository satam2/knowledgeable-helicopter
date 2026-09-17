"""Recalculate all scores and paired diagnostics from hashed local predictions."""

import json
import numpy as np
import pandas as pd

from next230_common import OUT, OLD, verify_protocol
from taxiout.artifacts import read_json, sha256, write_json
from taxiout.metrics import paired_stability, scores, season_score
from taxiout.schema import ID, TARGET


def summarize():
    verify_protocol()
    references = {}
    for fold in ['F1', 'F2', 'F3', 'G1']:
        record = read_json(OLD / 'refinements' / f'H_gpu_full_{fold}.json')
        path = OLD / 'models' / record['run_id'] / 'score_predictions.parquet'
        assert sha256(path) == record['outputs'][path.name]
        references[fold] = pd.read_parquet(path)
    candidates, incomplete, receipts = {}, [], []
    for manifest in sorted((OUT / 'models').glob('*/manifest.json')):
        run = read_json(manifest)
        if run['status'] != 'complete':
            incomplete.append(manifest.parent.name)
            continue
        for name, digest in run['outputs'].items():
            if sha256(manifest.parent / name) != digest:
                raise ValueError(f'Artifact hash failure: {manifest.parent.name}/{name}')
        record = pd.read_parquet(manifest.parent / 'score_predictions.parquet')
        ref = references[run['fold']]
        if not np.array_equal(ref[ID], record[ID]) or not np.array_equal(ref[TARGET], record[TARGET]):
            raise ValueError('Scoring IDs or labels changed')
        overall = scores(record[TARGET], record.prediction_sec)
        if abs(overall['rmse_sec'] - run['metrics']['overall']['rmse_sec']) > 1e-9:
            raise ValueError('Saved metrics mismatch')
        if not np.allclose(record.squared_error, (record.prediction_sec - record[TARGET]) ** 2, atol=0, rtol=1e-14):
            raise ValueError('Squared errors mismatch')
        receipts.append({'run': manifest.parent.name, 'manifest_sha256': sha256(manifest)})
        if run['candidate'] == 'clock_experts':
            continue
        name = run['candidate'] + (f"_seed{run['seed']}" if run['seed'] != 20260910 else '')
        candidate = candidates.setdefault(name, {'folds': {}})
        rome = record.ADEP_mvt.eq('LIRF')
        missing = record.proxy_status.eq('missing')
        masks = {'rome_missing': rome & missing, 'rome_present': rome & ~missing,
                 'all_missing': missing, 'rome_missing_ordinary_0_1800': rome & missing & record[TARGET].between(0, 1800),
                 'rome_missing_extreme_above_7200': rome & missing & record[TARGET].gt(7200)}
        if 'component_known_flight' in record:
            masks['rome_missing_unseen_flight'] = rome & missing & ~record.component_known_flight
            masks['rome_missing_known_flight'] = rome & missing & record.component_known_flight
        slices = {key: {'reference': scores(ref.loc[mask, TARGET], ref.loc[mask, 'prediction_sec']),
                        'candidate': scores(record.loc[mask, TARGET], record.loc[mask, 'prediction_sec'])}
                  for key, mask in masks.items() if mask.any()}
        candidate['folds'][run['fold']] = {'overall': overall, 'slices': slices,
            'paired': paired_stability(ref, record), 'trees': run.get('trees'),
            'model_path': str(manifest.parent), 'protected_routes_equal': run['protected_routes_equal'],
            'reload_max_abs_delta': run.get('reload_max_abs_delta'), 'training': run.get('training')}
    base_scores = {f: scores(r[TARGET], r.prediction_sec) for f, r in references.items()}
    base_seasonal = season_score(base_scores['F1'], base_scores['F3'])
    for name, candidate in candidates.items():
        folds = candidate['folds']
        if {'F1', 'F3'}.issubset(folds):
            value = season_score(folds['F1']['overall'], folds['F3']['overall'])
            candidate['seasonal_rmse_sec'] = value
            candidate['seasonal_gain_sec'] = base_seasonal - value
            candidate['screen_gates'] = {
                'minimum_gain': base_seasonal - value >= max(2, .005 * base_seasonal),
                'both_screen_folds_improve': all(folds[f]['paired']['delta_rmse_second_minus_first'] < 0 for f in ['F1', 'F3']),
                'screen_gain_survives_day_removal': all(folds[f]['paired']['leave_one_day_out_delta_range'][1] < 0 for f in ['F1', 'F3']),
                'missing_regression_below_5pct': all(folds[f]['slices']['all_missing']['candidate']['rmse_sec'] <=
                    1.05 * folds[f]['slices']['all_missing']['reference']['rmse_sec'] for f in ['F1', 'F3'])}
            candidate['eligible_for_broader_checks'] = all(candidate['screen_gates'].values())
            if {'F2', 'G1'}.issubset(folds):
                candidate['all_fold_improvement'] = all(v['paired']['delta_rmse_second_minus_first'] < 0 for v in folds.values())
    summary = {'reference': {'name': 'H_gpu_full', 'seasonal_rmse_sec': base_seasonal, 'folds': base_scores},
               'candidates': candidates, 'complete_containers': len(receipts), 'incomplete': incomplete,
               'verified_manifests': receipts, 'target_sec': 230,
               'caveat': 'Adaptive reused development folds. F2/G1 share October. Not a leaderboard result.'}
    write_json(OUT / 'summary.json', summary)
    rows = [{'candidate': name, 'seasonal': c.get('seasonal_rmse_sec'),
             **{f: c['folds'].get(f, {}).get('overall', {}).get('rmse_sec') for f in ['F1', 'F2', 'F3', 'G1']},
             'advance': c.get('eligible_for_broader_checks')} for name, c in candidates.items()]
    pd.DataFrame(rows).to_csv(OUT / 'scores.csv', index=False)
    print(json.dumps({'scores': rows, 'incomplete': incomplete}, indent=2))
    return summary


if __name__ == '__main__':
    summarize()
