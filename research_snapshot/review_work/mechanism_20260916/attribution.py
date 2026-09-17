"""Exact paired-error decomposition explains ensemble benefit without causal claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'campaign_20260916'))
from common import WORKSPACE, ID, TARGET, reference, read_json, write_json, sha256, external_path, utc_now
import numpy as np
import pandas as pd

OUT=external_path(WORKSPACE/'private_runs/mechanism_20260916')
W={'F1':192122/344841,'F3':152719/344841}
RECIPES=['lightgbm_correction_base_s200k','lightgbm_correction_base_full',
         'lightgbm_correction_arrival_full','catboost_correction_base_s200k','tabm_correction_base_s200k']


def main():
    result={}
    for recipe in RECIPES:
        stats={'reference_mse':0.,'expert_mse':0.,'blend_mse':0.,'cross_error':0.,'prediction_disagreement_mse':0.,
               'reference_error_dot_change':0.,'groups':{},'airports':{},'folds':{}}
        provenance={}
        for fold,w in W.items():
            a,ar=reference(fold)
            root=WORKSPACE/'private_runs/campaign_20260916/models'/f'{recipe}_{fold}_s20260916'
            rec=read_json(root/'manifest.json')
            for name in ['candidate.parquet','blend25.parquet']:
                assert sha256(root/name)==rec['outputs'][name]
            b=pd.read_parquet(root/'candidate.parquet')
            c=pd.read_parquet(root/'blend25.parquet')
            assert np.array_equal(a[ID],b[ID]) and np.array_equal(a[TARGET],b[TARGET])
            y=a[TARGET].to_numpy(float)
            er=a.prediction_sec.to_numpy()-y
            ee=b.prediction_sec.to_numpy()-y
            eb=c.prediction_sec.to_numpy()-y
            d=ee-er
            assert np.allclose(eb,er+.25*d,atol=1e-9,rtol=0)
            for key,v in [('reference_mse',er**2),('expert_mse',ee**2),('blend_mse',eb**2),
                          ('cross_error',er*ee),('prediction_disagreement_mse',d**2),('reference_error_dot_change',er*d)]:
                stats[key]+=float(w*np.mean(v))
            missing=~np.isfinite(a.proxy_sec.to_numpy())
            disagreement=~missing&(np.abs(y-a.proxy_sec.to_numpy())>1800)
            groups={'missing_clock':missing,'finite_source_gap_over_30min':disagreement,
                    'finite_source_gap_at_most_30min':~(missing|disagreement)}
            for group,m in groups.items():
                v=stats['groups'].setdefault(group,{'row_weight':0.,'reference_mse_contribution':0.,'blend_mse_contribution':0.,'mse_gain_contribution':0.})
                v['row_weight']+=float(w*m.mean())
                v['reference_mse_contribution']+=float(w*np.sum(er[m]**2)/len(a))
                v['blend_mse_contribution']+=float(w*np.sum(eb[m]**2)/len(a))
                v['mse_gain_contribution']+=float(w*np.sum(er[m]**2-eb[m]**2)/len(a))
            for airport in a.ADEP_mvt.unique():
                m=a.ADEP_mvt.eq(airport).to_numpy()
                v=stats['airports'].setdefault(str(airport),{'mse_gain_contribution':0.})
                v['mse_gain_contribution']+=float(w*np.sum(er[m]**2-eb[m]**2)/len(a))
            stats['folds'][fold]={'error_correlation':float(np.corrcoef(er,ee)[0,1]),
                'reference_rmse':float(np.sqrt(np.mean(er**2))),'expert_rmse':float(np.sqrt(np.mean(ee**2))),
                'blend_rmse':float(np.sqrt(np.mean(eb**2))),'expert_better_row_pct':float(100*np.mean(ee**2<er**2)),
                'selected_iterations':rec['tune']['steps'],'fit_rows':rec['fit_ids']['fit']['n'],
                'refit_rows':rec['fit_ids']['refit']['n']}
            provenance[fold]=sha256(root/'manifest.json')
        for name in ['reference','expert','blend']:
            stats[name+'_rmse']=float(np.sqrt(stats[name+'_mse']))
        stats['blend_gain_sec']=stats['reference_rmse']-stats['blend_rmse']
        stats['mse_gain']=stats['reference_mse']-stats['blend_mse']
        stats['linear_gain_term']=-.5*stats['reference_error_dot_change']
        stats['quadratic_penalty_term']=.25**2*stats['prediction_disagreement_mse']
        assert np.isclose(stats['mse_gain'],stats['linear_gain_term']-stats['quadratic_penalty_term'])
        stats['expert_error_cosine']=stats['cross_error']/np.sqrt(stats['expert_mse']*stats['reference_mse'])
        for v in stats['groups'].values():
            v['share_of_total_gain_pct']=100*v['mse_gain_contribution']/stats['mse_gain']
        stats['manifest_hashes']=provenance
        result[recipe]=stats
    base=result['lightgbm_correction_base_full']
    arrival=result['lightgbm_correction_arrival_full']
    report={'created_utc':utc_now(),'recipes':result,
        'increments':{'base_full_blend_gain_sec':base['blend_gain_sec'],
            'added_arrival_block_gain_sec':base['blend_rmse']-arrival['blend_rmse'],
            'total_arrival_blend_gain_sec':arrival['blend_gain_sec'],
            'sampling_to_full_blend_gain_sec':result['lightgbm_correction_base_s200k']['blend_rmse']-base['blend_rmse']},
        'explanation':'Squared loss blend gain equals -2w*E[e_ref*(p_expert-p_ref)]-w^2*E[(p_expert-p_ref)^2]. A weaker standalone expert can help if errors differ usefully.',
        'limits':'Paired feature ablation is predictive association, not proof of physical causality. All comparisons exposed development cohorts; no new weights fitted.',
        'script_sha256':sha256(__file__)}
    write_json(OUT/'attribution.json',report)
    print(report['increments'],flush=True)
    print('Best blend gain by disjoint group',arrival['groups'],flush=True)


if __name__=='__main__':
    main()
