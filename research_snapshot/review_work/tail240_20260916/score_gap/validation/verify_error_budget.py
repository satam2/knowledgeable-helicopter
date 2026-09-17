"""Independently check parent error-budget arithmetic from bound replay rows."""
from replay_weight_policy import ROOT,OUT,ID,TARGET,TIME,read,sha,SEASONS,guard
import json
import numpy as np
import pandas as pd


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-9)


def main():
    path=ROOT/'private_runs/tail240_20260916/score_gap/error_budget/analysis.json'
    report=read(path);assert report['status']=='complete'
    assert report['source_sha256']==sha(ROOT/'review_work/tail240_20260916/score_gap/error_budget.py')
    verified=read(OUT/'receipt.json');assert verified['status']=='passed'
    for relative,digest in verified['source_bindings'].items():assert sha(ROOT/relative)==digest
    rng=np.random.default_rng(20260917);candidate_boot=np.zeros(4000);baseline_boot=np.zeros(4000)
    contributions={};month={};removals=[]
    for fold,weight in SEASONS.items():
        source=OUT/f'{fold}_replay.parquet';assert sha(source)==verified['outputs'][source.name]
        frame=pd.read_parquet(source)
        frame['c']=(frame.local_fold_weights_raw-frame[TARGET])**2
        frame['b']=(frame.v2_raw-frame[TARGET])**2
        assert len(frame)==report['folds'][fold]['rows']
        close(np.sqrt(frame.c.mean()),report['folds'][fold]['candidate_rmse'])
        close(np.sqrt(frame.b.mean()),report['folds'][fold]['baseline_rmse'])
        routes={'route:'+value:frame.route.eq(value) for value in frame.route.unique()}
        airports={'airport:'+value:frame.ADEP_mvt.astype(str).eq(value) for value in frame.ADEP_mvt.astype(str).unique()}
        groups={**routes,**airports,'rome_missing':frame.ADEP_mvt.eq('LIRF')&frame.route.eq('missing')}
        for name,mask in groups.items():
            bucket=contributions.setdefault(name,np.zeros(3))
            bucket+=weight*np.array([int(mask.sum()),float(frame.loc[mask,'b'].sum()),float(frame.loc[mask,'c'].sum())])/len(frame)
        days=frame.groupby(frame[TIME].dt.floor('D'),sort=True).agg(n=(ID,'size'),b=('b','sum'),c=('c','sum'))
        draws=rng.multinomial(len(days),np.full(len(days),1/len(days)),size=4000)
        baseline_boot+=weight*(draws@days.b.to_numpy())/(draws@days.n.to_numpy())
        candidate_boot+=weight*(draws@days.c.to_numpy())/(draws@days.n.to_numpy())
        month[fold]=float(frame.c.mean())
        for day,rec in days.iterrows():
            remain=(float(frame.c.sum())-rec.c)/(len(frame)-rec.n)
            removals.append((fold,str(day),remain))
        for fraction,detail in report['folds'][fold]['tails'].items():
            count=int(np.ceil(float(fraction)*len(frame)));assert count==detail['rows']
            top=frame.c.nlargest(count)
            close(top.sum()/frame.c.sum(),detail['sse_share'])
            close(np.sqrt(top.iloc[-1]),detail['minimum_abs_error_sec'])
        gains=frame.b-frame.c
        for count,value in report['folds'][fold]['remove_most_beneficial_rows_gain'].items():
            remaining=frame.drop(index=gains.nlargest(int(count)).index)
            close(np.sqrt(remaining.b.mean())-np.sqrt(remaining.c.mean()),value)
    total_c=sum(SEASONS[f]*month[f] for f in SEASONS)
    total_b=verified['summary']['v2_raw']['seasonal_rmse']**2
    for name,values in contributions.items():
        record=report['slices'][name];mass,b,c=values
        for key,value in [('row_mass',mass),('baseline_mse_contribution',b),('candidate_mse_contribution',c),
                          ('candidate_sse_share',c/total_c),('local_mse_gain_contribution',b-c),
                          ('local_mse_gain_share',(b-c)/(total_b-total_c)),('candidate_slice_rmse',np.sqrt(c/mass))]:close(value,record[key])
    bootstrap=report['local_day_bootstrap']
    close(np.quantile(np.sqrt(candidate_boot),[.025,.975]),bootstrap['candidate_rmse_ci95'])
    close(np.quantile(np.sqrt(baseline_boot),[.025,.975]),bootstrap['baseline_rmse_ci95'])
    close(np.quantile(np.sqrt(baseline_boot)-np.sqrt(candidate_boot),[.025,.975]),bootstrap['paired_gain_ci95'])
    lookup={(v['fold'],v['day']):v['candidate_rmse'] for v in report['day_removals']}
    for fold,day,value in removals:close(np.sqrt(total_c+SEASONS[fold]*(value-month[fold])),lookup[(fold,day)])
    result=dict(status='passed',parent_source_sha256=report['source_sha256'],parent_analysis_sha256=sha(path),
        verifier_sha256=sha(__file__),independent_replay_receipt_sha256=sha(OUT/'receipt.json'),
        exact_original_labels_predictions_routes=True,all_slices_tails_influences_and_dayremovals_verified=True,
        bootstrap_4000_stratified_day_replicates_reproduced=True,bootstrap=bootstrap,
        interpretation='Conditional sensitivity of exposed development months, not selection-adjusted and not a future2026 prediction interval. Local error concentration and hypothetical single-route arithmetic cannot locate or attribute the actual ranking error.',
        peak_bytes=guard(),gpu_used=False,fit_used=False)
    dest=OUT/'parent_error_budget_review.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
