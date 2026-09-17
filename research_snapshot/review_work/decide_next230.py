"""Conservative development decision, with complete audit of rejected directions."""

import json
import numpy as np
import pandas as pd

from next230_common import OUT, OLD
from taxiout.artifacts import read_json, write_json
from taxiout.schema import TARGET


def decide():
    summary = read_json(OUT / 'summary.json')
    if summary['incomplete']:
        raise ValueError('Finish or diagnose incomplete runs before deciding')
    eligible = {name: c for name, c in summary['candidates'].items()
                if c.get('eligible_for_broader_checks') and c.get('all_fold_improvement')}
    best_name = min(eligible, key=lambda name: eligible[name]['seasonal_rmse_sec']) if eligible else 'H_gpu_full'
    best = eligible.get(best_name, summary['reference'])
    value = best['seasonal_rmse_sec']
    reference_value = summary['reference']['seasonal_rmse_sec']
    error_shares, floor_mse = {}, 0.
    for fold, weight in [('F1', 192122/344841), ('F3', 152719/344841)]:
        if best_name == 'H_gpu_full':
            ref = read_json(OLD / 'refinements' / f'H_gpu_full_{fold}.json')
            path = OLD / 'models' / ref['run_id']
        else:
            path = best['folds'][fold]['model_path']
        from pathlib import Path
        p = pd.read_parquet(Path(path) / 'score_predictions.parquet')
        missing = p.proxy_status.eq('missing')
        floor_mse += weight * p.loc[~missing, 'squared_error'].sum() / len(p)
        for airport in ['LIRF', 'other']:
            location = p.ADEP_mvt.eq('LIRF') if airport == 'LIRF' else p.ADEP_mvt.ne('LIRF')
            for status, mask in [('missing', missing), ('present_or_invalid', ~missing)]:
                key = f'{airport}_{status}'
                error_shares[key] = error_shares.get(key, 0) + weight * p.loc[location & mask, 'squared_error'].sum() / len(p)
    error_shares = {k: float(100*v/value**2) for k, v in error_shares.items()}
    ranked = sorted([{'candidate': n, 'seasonal_rmse_sec': c['seasonal_rmse_sec'],
                      'gain_sec': c['seasonal_gain_sec'], 'screen_gates': c['screen_gates'],
                      'broader_fold_checks': c.get('all_fold_improvement', 'not completed')}
                     for n, c in summary['candidates'].items() if 'seasonal_rmse_sec' in c],
                    key=lambda c: c['seasonal_rmse_sec'])
    result = {'recommended_development_reference': best_name, 'seasonal_rmse_sec': value,
              'reference_seasonal_rmse_sec': reference_value, 'gain_sec': reference_value-value,
              'mse_reduction_pct': 100*(1-(value/reference_value)**2),
              'target_sec': 230, 'target_reached': value <= 230, 'intermediate_280_reached': value <= 280,
              'additional_mse_reduction_to_230_pct': max(0,100*(1-(230/value)**2)),
              'perfect_missing_floor_sec': float(np.sqrt(floor_mse)), 'seasonal_error_share_pct': error_shares,
              'ranked_candidates': ranked, 'complete_containers': summary['complete_containers'],
              'status': 'local development evidence only; no final all-year fit, submission or external score',
              'caveats': ['All score folds were used in previous development; selection optimism remains.',
                          'F2 and G1 reuse October outcomes and are not independent.',
                          'GPU fits are nondeterministic; seed checks are evidence, not a reproducibility guarantee.',
                          'No claim that 230 is attainable from the available covariates alone.']}
    write_json(OUT / 'decision.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    decide()
