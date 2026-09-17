"""Post-submission local error concentration; no ranking truth or model fitting."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
import lightgbm
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET=common.ID,common.TARGET
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/score_gap/error_budget')
CAND=ROOT/'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
WEIGHTS={'F1':192122/344841,'F3':152719/344841}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    marker=common.read_json(CAND/'manifest.json')
    assert marker['status']=='complete'
    slices={};folds={};boot=[];frames=[];bindings={}
    rng=np.random.default_rng(20260917)
    for fold,w in WEIGHTS.items():
        ref,refmark=common.reference(fold)
        path=CAND/f'{fold}.parquet'
        assert common.sha256(path)==marker['folds'][fold]['prediction']['sha256']
        cur=pd.read_parquet(path)
        np.testing.assert_array_equal(cur[ID],ref[ID])
        y=ref[TARGET].to_numpy(float)
        p=cur.prediction_sec.to_numpy(float)
        old=ref.prediction_sec.to_numpy(float)
        e2=(y-p)**2;b2=(y-old)**2
        proxy=ref.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        ordinary=~missing&(proxy>=0)&(proxy<=7200)
        routes=np.select([missing,ordinary],['missing','ordinary'],default='finite_nonordinary')
        groups={**{f'route:{r}':routes==r for r in np.unique(routes)},
                **{f'airport:{a}':ref.ADEP_mvt.eq(a).to_numpy() for a in ref.ADEP_mvt.unique()},
                'rome_missing':ref.ADEP_mvt.eq('LIRF').to_numpy()&missing}
        for key,mask in groups.items():
            value=slices.setdefault(key,dict(row_mass=0.,baseline_mse_contribution=0.,candidate_mse_contribution=0.))
            value['row_mass']+=w*mask.mean()
            value['baseline_mse_contribution']+=w*b2[mask].sum()/len(y)
            value['candidate_mse_contribution']+=w*e2[mask].sum()/len(y)
        days=ref[common.MOVEMENT].dt.floor('D')
        codes,unique=pd.factorize(days,sort=True)
        counts=np.bincount(codes)
        total_b=np.bincount(codes,weights=b2)
        total_c=np.bincount(codes,weights=e2)
        draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=4000)
        sampled_n=draws@counts
        boot.append((w,(draws@total_b)/sampled_n,(draws@total_c)/sampled_n))
        tails={}
        order=np.argsort(-e2,kind='stable')
        for fraction in [.0001,.001,.01]:
            n=int(np.ceil(len(y)*fraction))
            tails[str(fraction)]=dict(rows=n,sse_share=float(e2[order[:n]].sum()/e2.sum()),
                minimum_abs_error_sec=float(np.sqrt(e2[order[n-1]])))
        gains=b2-e2;gainorder=np.argsort(-gains,kind='stable')
        influences={}
        for count in [1,5,10,100]:
            keep=np.ones(len(y),bool);keep[gainorder[:count]]=False
            influences[str(count)]=float(np.sqrt(b2[keep].mean())-np.sqrt(e2[keep].mean()))
        folds[fold]=dict(rows=len(y),days=len(unique),baseline_rmse=float(np.sqrt(b2.mean())),
            candidate_rmse=float(np.sqrt(e2.mean())),candidate_mae=float(np.abs(y-p).mean()),
            candidate_bias=float((p-y).mean()),tails=tails,remove_most_beneficial_rows_gain=influences)
        frames.append(pd.DataFrame(dict(fold=fold,day=days.to_numpy(),baseline_sse=b2,candidate_sse=e2,
                                       weighted_sse=w*e2/len(y),weight=w/len(y))))
        bindings[fold]=dict(candidate_sha256=common.sha256(path),reference_outputs=refmark['outputs'])
    baseline_mse=sum(WEIGHTS[f]*v['baseline_rmse']**2 for f,v in folds.items())
    candidate_mse=sum(WEIGHTS[f]*v['candidate_rmse']**2 for f,v in folds.items())
    for value in slices.values():
        value['candidate_sse_share']=value['candidate_mse_contribution']/candidate_mse
        value['local_mse_gain_contribution']=value['baseline_mse_contribution']-value['candidate_mse_contribution']
        value['local_mse_gain_share']=value['local_mse_gain_contribution']/(baseline_mse-candidate_mse)
        value['candidate_slice_rmse']=math.sqrt(value['candidate_mse_contribution']/value['row_mass'])
    assert abs(candidate_mse**.5-268.662991126314)<1e-8
    official=common.read_json(ROOT/'private_runs/tail240_20260916/final_submission_v3_v2/organizer_result.json')['score']
    oldofficial=common.read_json(ROOT/'private_runs/submission_v2/organizer_v2_result.json')['score']
    bboot=sum(w*b for w,b,c in boot);cboot=sum(w*c for w,b,c in boot)
    frame=pd.concat(frames,ignore_index=True)
    sensitivity=[]
    for (fold,day),group in frame.groupby(['fold','day']):
        n=folds[fold]['rows']
        replacement=(folds[fold]['candidate_rmse']**2*n-group.candidate_sse.sum())/(n-len(group))
        adjusted=candidate_mse+WEIGHTS[fold]*(replacement-folds[fold]['candidate_rmse']**2)
        sensitivity.append(dict(fold=fold,day=str(day),candidate_rmse=float(np.sqrt(adjusted))))
    excess=official**2-candidate_mse
    scenarios={key:dict(required_route_rmse_increase_fraction=math.sqrt(1+excess/slices['route:'+key]['candidate_mse_contribution'])-1)
               for key in ['missing','ordinary']}
    report=dict(status='complete',source_sha256=common.sha256(__file__),bindings=bindings,folds=folds,slices=slices,
        baseline_local_rmse=baseline_mse**.5,candidate_local_rmse=candidate_mse**.5,
        baseline_official_rmse=oldofficial,candidate_official_rmse=official,
        baseline_gap=oldofficial-baseline_mse**.5,candidate_gap=official-candidate_mse**.5,
        local_gain=baseline_mse**.5-candidate_mse**.5,official_gain=oldofficial-official,
        candidate_excess_mse=excess,candidate_relative_mse_gap=excess/candidate_mse,
        local_day_bootstrap=dict(repetitions=4000,seed=20260917,
            candidate_rmse_ci95=np.quantile(np.sqrt(cboot),[.025,.975]).tolist(),
            baseline_rmse_ci95=np.quantile(np.sqrt(bboot),[.025,.975]).tolist(),
            paired_gain_ci95=np.quantile(np.sqrt(bboot)-np.sqrt(cboot),[.025,.975]).tolist()),
        day_removal_range=[min(v['candidate_rmse'] for v in sensitivity),max(v['candidate_rmse'] for v in sensitivity)],
        day_removals=sensitivity,hypothetical_single_route_explanation=scenarios,
        limitations=['Day bootstrap conditions on exposed development data and chosen models; not selection-adjusted or a2026predictiveinterval.',
          'No ranking labels or official subgroup errors available. Local slices and prediction changes cannot identify actual official error locations.',
          'Hypothetical route increases hold every other local squared error fixed and are arithmetic illustrations, not causal attribution.',
          'Error-ranked groups are retrospective diagnostics only, not usable inference routing.'])
    common.write_json(OUT/'analysis.json',report)
    print({k:report[k] for k in ['baseline_gap','candidate_gap','local_gain','official_gain','candidate_excess_mse','local_day_bootstrap','day_removal_range']})


if __name__=='__main__':main()
