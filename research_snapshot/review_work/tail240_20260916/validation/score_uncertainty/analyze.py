"""Frozen-score seasonal day bootstrap and additive error-budget uncertainty."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_candidate as v
OUT=ROOT/'private_runs/tail240_20260916/validation/score_uncertainty/v1'
CANDIDATE=ROOT/'private_runs/tail240_20260916/models/normalized_missing_refit_v2/observed_schedule_scale_blend25'
EVALUATION=ROOT/'private_runs/tail240_20260916/validation/normalized_refit_v2/observed_schedule_scale_blend25_evaluation/evaluation.json'
REPLICATES=20000
SEED=20260916


def guard():
    info=psutil.Process().memory_info();peak=getattr(info,'peak_wset',info.rss)
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return int(peak)


def declare():
    paths=[v.BINDING,CANDIDATE/'protocol.json',CANDIDATE/'manifest.json',EVALUATION]
    binding=v.read_json(v.BINDING)
    for fold,entry in binding['folds'].items():
        paths.extend([Path(entry['manifest_path']),Path(entry['prediction_path'])])
        reference=ROOT/'private_runs/next_230/models'/f'clock_and_rome_ensemble_{fold}_s20260910'
        paths.extend([reference/'manifest.json',reference/'score_predictions.parquet'])
    marker=v.read_json(CANDIDATE/'manifest.json')
    for entry in marker['folds'].values():
        value=entry['prediction']
        paths.append(CANDIDATE/value['path'])
    record={'source_sha256':v.sha256(__file__),'inputs':{str(path):v.sha256(path) for path in paths},
        'scope':'Saved complete July/November score predictions only. Original labels and all rows retained. No model fitting, predictor selection, score-based routing, label deletion or changed score metric.',
        'bootstrap':{'unit':'UTC calendar day within each fold, sampled with replacement independently across folds; original day count per fold',
            'replicates':REPLICATES,'seed':SEED,'weights':binding['weights'],
            'score':'sqrt(sum(fixed seasonal fold weight * resampled fold SSE/resampled fold row count))',
            'paired':'Same day draw applied to current269 and frozen272 baseline. Percentile95 intervals conditional on existing predictions and observed days.'},
        'diagnostics':'Frozen posthoc missing-NM, proxyordinary[0,7200], Y>7200 and Y>=86400 masks; disjoint MSE components use full row denominators. Additive bootstrap variance contribution Cov(component_MSE,total_MSE)/Var(total_MSE); covariance shares may be negative and are not causal.',
        'comparators':{'local_target':240.,'official_leader_reported':243.5805,'prior_local_v2':292.8133464057956,'prior_official_v2':294.626,
            'provenance':'Comparator numbers supplied by parent task context; no live leaderboard retrieval or fresh official result. One historic local/official pair cannot identify transfer/calibration.'},
        'limits':'Adaptive exposed local folds; day bootstrap neither corrects model search nor estimates crossyear/January transfer or unseen extreme-event frequency. Diagnostic attainable-error-floor claims prohibited.',
        'resources':'1 CPU, historical peak RSS<2GiB, host reserve>=8GiB'}
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert v.read_json(path)==record
    else:v.write_json(path,record)
    return record


def interval(values):
    return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),
        'p025_p50_p975':np.quantile(values,[.025,.5,.975]).tolist()}


def run():
    protocol=declare()
    assert not (OUT/'receipt.json').exists()
    binding=v.read_json(v.BINDING);marker=v.read_json(CANDIDATE/'manifest.json')
    eval_report=v.read_json(EVALUATION)
    names=['missing_y_ge_day','finite_y_ge_day','missing_y_lt_day','finite_nonordinary_y_lt_day','ordinary_y_lt_day']
    rng=np.random.default_rng(SEED)
    draws=np.zeros((REPLICATES,len(names)),dtype=float)
    baseline_draws=np.zeros(REPLICATES,dtype=float)
    reports={};point_components=np.zeros(len(names));baseline_point=0.;pooled_rows=[]
    for fold in ['F1','F3']:
        frozen=binding['folds'][fold];weight=binding['weights'][fold]
        ref,_=v.common.reference(fold)
        assert v.object_hash(ref[v.ID].tolist())==frozen['score_id_hash']
        assert v.object_hash(ref[v.TARGET].tolist())==frozen['score_target_hash']
        prediction_path=v.file_receipt(CANDIDATE,marker['folds'][fold]['prediction'])
        candidate=pd.read_parquet(prediction_path);baseline=pd.read_parquet(frozen['prediction_path'])
        np.testing.assert_array_equal(candidate[v.ID],ref[v.ID]);np.testing.assert_array_equal(baseline[v.ID],ref[v.ID])
        p=candidate.prediction_sec.to_numpy(float);base=baseline.prediction_sec.to_numpy(float)
        y=ref[v.TARGET].to_numpy(float);proxy=ref.proxy_sec.to_numpy(float)
        assert np.isfinite(p).all() and np.isfinite(y).all()
        missing=~np.isfinite(proxy);ordinary=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200);extreme=y>=86400
        masks=[missing&extreme,~missing&extreme,missing&~extreme,~missing&~ordinary&~extreme,ordinary&~extreme]
        assert np.array_equal(np.sum(masks,axis=0),np.ones(len(y),dtype=int))
        np.testing.assert_array_equal(p[~missing],base[~missing])
        error=(y-p)**2;base_error=(y-base)**2
        actual_rmse=float(np.sqrt(error.mean()))
        expected=eval_report['folds'][fold]['comparisons']['baseline272']['candidate']['rmse']
        assert abs(actual_rmse-expected)<1e-10
        dates=pd.to_datetime(ref['day'],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        days,codes=np.unique(dates,return_inverse=True);counts=np.bincount(codes)
        daily=np.column_stack([np.bincount(codes,weights=np.where(mask,error,0),minlength=len(days)) for mask in masks])
        daily_baseline=np.bincount(codes,weights=base_error,minlength=len(days))
        boot_weights=rng.multinomial(len(days),np.full(len(days),1/len(days)),size=REPLICATES)
        denominators=boot_weights@counts
        components=(boot_weights@daily)/denominators[:,None]
        fold_draws=np.sqrt(components.sum(axis=1))
        draws+=weight*components
        baseline_draws+=weight*(boot_weights@daily_baseline)/denominators
        point_components+=weight*daily.sum(axis=0)/len(y)
        baseline_point+=weight*base_error.mean()
        slice_masks={**dict(zip(names,masks)),'missing_nm':missing,'y_ge_day':extreme,'y_over7200':y>7200,'y_negative':y<0}
        slices={name:{'rows':int(mask.sum()),'row_fraction':float(mask.mean()),'days':int(len(np.unique(codes[mask]))),
            'sse':float(error[mask].sum()),'sse_fraction':float(error[mask].sum()/error.sum()),
            'weighted_full_denominator_mse':float(weight*error[mask].sum()/len(y))} for name,mask in slice_masks.items()}
        extreme_records=[]
        for i in np.flatnonzero(extreme):
            extreme_records.append({'id':float(ref[v.ID].iloc[i]),'day':dates[i],'airport':str(ref.ADEP_mvt.iloc[i]),
                'target_sec':float(y[i]),'prediction_sec':float(p[i]),'proxy_missing':bool(missing[i]),
                'squared_error':float(error[i]),'seasonal_mse_contribution':float(weight*error[i]/len(y))})
        daily_table=pd.DataFrame({'day':days,'rows':counts,'candidate_sse':daily.sum(axis=1),'baseline_sse':daily_baseline,
            'y_ge_day_rows':np.bincount(codes,weights=extreme.astype(int),minlength=len(days)).astype(int)})
        for i,name in enumerate(names):daily_table[name+'_sse']=daily[:,i]
        daily_table.to_parquet(OUT/(fold+'_daily_error_budget.parquet'),index=False)
        pooled_rows.extend({'fold':fold,'weighted_sse':float(weight*value/len(y))} for value in error)
        reports[fold]={'rows':len(y),'days':len(days),'rmse':actual_rmse,'baseline_rmse':float(np.sqrt(base_error.mean())),
            'bootstrap_rmse':interval(fold_draws),'slices':slices,'y_ge_day_records':extreme_records,
            'largest_day_sse_fraction':float(daily.sum(axis=1).max()/error.sum()),
            'largest_row_sse_fraction':float(error.max()/error.sum())}
        print('FOLD',fold,actual_rmse,reports[fold]['bootstrap_rmse'],flush=True);guard()
    total_mse=draws.sum(axis=1);rmse_draws=np.sqrt(total_mse);base_rmse_draws=np.sqrt(baseline_draws)
    point=float(np.sqrt(point_components.sum()));baseline=float(np.sqrt(baseline_point))
    assert abs(point-269.863486485)<1e-8 and abs(baseline-binding['seasonal_rmse'])<1e-10
    variance=float(np.var(total_mse,ddof=1));shares={}
    for i,name in enumerate(names):
        covariance=float(np.cov(draws[:,i],total_mse,ddof=1)[0,1])
        shares[name]={'point_mse_contribution':float(point_components[i]),'point_mse_fraction':float(point_components[i]/point_components.sum()),
            'component_bootstrap_mse_std':float(np.std(draws[:,i],ddof=1)),'covariance_with_total_mse':covariance,
            'share_total_mse_variance':covariance/variance}
    assert abs(sum(row['share_total_mse_variance'] for row in shares.values())-1)<1e-10
    # Preserve the full-score denominator while reporting additive row concentration.
    contributions=np.sort(np.array([row['weighted_sse'] for row in pooled_rows]))[::-1]
    concentration={str(n):float(contributions[:n].sum()/point_components.sum()) for n in [1,2,5,10,20,100]}
    target=protocol['comparators']['local_target']
    report={'status':'complete','source_sha256':v.sha256(__file__),'protocol_sha256':v.sha256(OUT/'protocol.json'),
        'folds':reports,'seasonal':{'current_rmse':point,'baseline_rmse':baseline,'gain':baseline-point,
            'bootstrap_current_rmse':interval(rmse_draws),'bootstrap_baseline_rmse':interval(base_rmse_draws),
            'paired_bootstrap_gain':interval(base_rmse_draws-rmse_draws),
            'replicate_fraction_current_below240':float(np.mean(rmse_draws<target)),
            'replicate_fraction_current_below_reported_official_leader':float(np.mean(rmse_draws<protocol['comparators']['official_leader_reported'])),
            'threshold_fraction_warning':'Empirical within-local-fold bootstrap fractions are not probabilities of official target achievement.',
            'local240_mse_reduction_required':point**2-target**2,'local240_mse_fraction_reduction_required':1-target**2/point**2,
            'components':shares,'largest_row_mse_shares':concentration},
        'interpretation_limits':protocol['limits'],'comparators':protocol['comparators'],'peak_rss_bytes':guard(),
        'outputs':{p.name:v.sha256(p) for p in OUT.glob('*_daily_error_budget.parquet')}}
    v.write_json(OUT/'receipt.json',report)
    print('COMPLETE',point,report['seasonal']['bootstrap_current_rmse'],shares,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(1):
        if args.declare_only:print(declare())
        else:run()
