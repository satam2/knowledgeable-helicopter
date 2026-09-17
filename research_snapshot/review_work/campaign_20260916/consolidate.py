"""Complete-cohort comparison and error budget; no score-fit selection weights."""
import numpy as np
import pandas as pd
from common import OUT, WORKSPACE, ID, TARGET, reference, read_json, write_json, sha256, utc_now

W={'F1':192122/344841,'F3':152719/344841}


def seasonal(records,variant):
    return float(np.sqrt(sum(W[f]*records[f]['reports'][variant]['metrics']['overall']['rmse_sec']**2 for f in W)))


def budget(frames):
    rows=[]
    for fold,frame in frames.items():
        f=frame.copy()
        f['weight']=W[fold]/len(f)
        f['weighted_sse']=f.squared_error*f.weight
        rows.append(f)
    x=pd.concat(rows,ignore_index=True)
    total=float(x.weighted_sse.sum())
    mask_missing=x.proxy_status.eq('missing') if 'proxy_status' in x else x.route.astype(str).str.contains('missing|rome')
    disagreement=np.isfinite(x.proxy_sec)&((x[TARGET]-x.proxy_sec).abs()>1800)
    union=mask_missing|disagreement
    result={'weighted_mse':total,'seasonal_rmse_sec':float(np.sqrt(total)),
        'missing_clock_error_share_pct':float(100*x.loc[mask_missing,'weighted_sse'].sum()/total),
        'missing_clock_row_share_pct':float(100*x.loc[mask_missing,'weight'].sum()),
        'disagreement_above_30min_error_share_pct':float(100*x.loc[disagreement,'weighted_sse'].sum()/total),
        'disagreement_above_30min_row_share_pct':float(100*x.loc[disagreement,'weight'].sum()),
        'missing_or_disagreement_error_share_pct':float(100*x.loc[union,'weighted_sse'].sum()/total),
        'missing_or_disagreement_row_share_pct':float(100*x.loc[union,'weight'].sum()),
        'airport_error_share_pct':{str(k):float(v*100/total) for k,v in x.groupby('ADEP_mvt',observed=True).weighted_sse.sum().items()},
        'route_error_share_pct':{str(k):float(v*100/total) for k,v in x.groupby('route',observed=True).weighted_sse.sum().items()},
        'largest_1pct_error_share_pct':sum(W[f]*float(frame.nlargest(max(1,int(np.ceil(len(frame)*.01))),'squared_error').squared_error.sum())/len(frame) for f,frame in frames.items())*100/total,
        'note':'Airport, route, missing and tail shares overlap. Largest 1pct selected per month, diagnostic only.'}
    return result


def main():
    refs={f:reference(f)[0] for f in W}
    base_rmse={f:float(np.sqrt(v.squared_error.mean())) for f,v in refs.items()}
    base=float(np.sqrt(sum(W[f]*base_rmse[f]**2 for f in W)))
    rows=[{'candidate':'submitted_reference','variant':'reference','seasonal_rmse_sec':base,'gain_sec':0.,
           'F1_rmse_sec':base_rmse['F1'],'F3_rmse_sec':base_rmse['F3'],'status':'frozen_local_reference',
           'promote':False,'training':'full prior recipe','seed_scope':'prior pipeline mixed seeds'}]
    grouped={}
    failed=[]
    for p in sorted([*(OUT/'models').glob('*/manifest.json'),*(OUT/'missing_mixture/models').glob('*/manifest.json')]):
        rec=read_json(p)
        if rec['status']!='complete':
            failed.append({'path':str(p),'status':rec['status'],'error':rec.get('error')})
            continue
        fold=rec['name'].split('_')[-2]
        if fold not in W:
            continue
        key=rec['name'].replace('_'+fold+'_','_SCREEN_')
        grouped.setdefault(key,{})[fold]=(rec,p.parent)
    for key,records in grouped.items():
        if set(records)!=set(W):
            continue
        recs={f:r[0] for f,r in records.items()}
        for variant in recs['F1']['reports']:
            score=seasonal(recs,variant)
            each={f:recs[f]['reports'][variant] for f in W}
            missing_ratios=[]
            for f in W:
                a=refs[f]
                pred=pd.read_parquet(records[f][1]/f'{variant}.parquet')
                assert np.array_equal(a[ID],pred[ID]) and np.array_equal(a[TARGET],pred[TARGET])
                missing=a.proxy_status.eq('missing')
                missing_ratios.append(float(np.sqrt(pred.loc[missing,'squared_error'].mean()/a.loc[missing,'squared_error'].mean())))
            both=all(each[f]['metrics']['overall']['rmse_sec']<base_rmse[f] for f in W)
            stable=all(each[f]['stability']['leave_one_day_out_delta_range'][1]<0 for f in W)
            missing_ok=max(missing_ratios)<=1.05
            passes=base-score>=2 and both and stable and missing_ok
            r0=recs['F1']
            rows.append(dict(candidate=key,variant=variant,seasonal_rmse_sec=score,gain_sec=base-score,
                F1_rmse_sec=each['F1']['metrics']['overall']['rmse_sec'],F3_rmse_sec=each['F3']['metrics']['overall']['rmse_sec'],
                seasonal_mae_sec=sum(W[f]*each[f]['metrics']['overall']['mae_sec'] for f in W),
                seasonal_bias_sec=sum(W[f]*each[f]['metrics']['overall']['bias_sec'] for f in W),
                max_missing_rmse_ratio=max(missing_ratios),both_improve=both,day_removal_stable=stable,
                meets_screen_gates=passes,promote=passes and r0['full'],training='full' if r0['full'] else 'sample200k',
                family=r0['family'],target=r0['target'],features=r0['features'],seed=r0['seed'],seed_scope='new expert seed; submitted-reference components fixed',
                fit_refit_seconds=sum((recs[f].get('tuning_runtime_sec') or 0)+(recs[f].get('refit_runtime_sec') or 0) for f in W) if all(recs[f].get('tuning_runtime_sec') is not None for f in W) else None,
                inference_seconds=sum(recs[f]['inference_runtime_sec'] for f in W) if all(recs[f].get('inference_runtime_sec') is not None for f in W) else None,
                peak_rss_gb=max(recs[f].get('peak_rss_bytes') or 0 for f in W)/2**30,
                peak_rss_scope='both folds' if all(recs[f].get('peak_rss_bytes') is not None for f in W) else 'partial known-fold lower bound',
                F1_trees_epochs=recs['F1']['tune']['steps'],F3_trees_epochs=recs['F3']['tune']['steps'],
                F1_leave_day_delta_max=each['F1']['stability']['leave_one_day_out_delta_range'][1],
                F3_leave_day_delta_max=each['F3']['stability']['leave_one_day_out_delta_range'][1],
                status='complete development screen'))
    table=pd.DataFrame(rows).sort_values('seasonal_rmse_sec')
    table.to_csv(OUT/'experiment_table.csv',index=False)
    summary=dict(created_utc=utc_now(),reference_seasonal_rmse_sec=base,rows=table.to_dict('records'),
        completed_fit_records=sum(len(v) for v in grouped.values()),incomplete_or_failed=failed,
        sampled_gate_passers=[r for r in rows if r.get('meets_screen_gates') and r.get('training')=='sample200k'],
        caveat='All score months previously exposed; variants correlated; gates do not remove selection bias.',
        error_budget_reference=budget(refs))
    # pandas optional columns are null in JSON, never fabricated values.
    summary['rows']=__import__('json').loads(table.to_json(orient='records'))
    best=table.loc[table.candidate.ne('submitted_reference')].iloc[0]
    paths=grouped[best.candidate]
    best_frames={f:pd.read_parquet(paths[f][1]/f'{best.variant}.parquet') for f in W}
    summary['error_budget_best_new']=dict(candidate=best.candidate,variant=best.variant,**budget(best_frames))
    pattern=[]
    for f in W:
        a,b=refs[f],best_frames[f]
        ae=a.prediction_sec.to_numpy()-a[TARGET].to_numpy()
        be=b.prediction_sec.to_numpy()-b[TARGET].to_numpy()
        pattern.append(dict(fold=f,error_correlation=float(np.corrcoef(ae,be)[0,1]),
            weighted_mse_change=float(W[f]*(np.mean(be**2)-np.mean(ae**2))),
            mean_abs_prediction_change_sec=float(np.mean(np.abs(b.prediction_sec-a.prediction_sec)))))
    summary['best_new_error_pattern']=pattern
    write_json(OUT/'summary.json',summary)
    print(table[['candidate','variant','seasonal_rmse_sec','gain_sec','promote']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
