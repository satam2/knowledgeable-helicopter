"""Weighted full-cohort error budget for frozen unfloored composition; diagnostic only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET=common.ID,common.TARGET
MODEL=ROOT/'private_runs/breakthrough_20260916/missing/route_composition_v2'
OUT=MODEL/'error_budget'
COUNTS={'F1':192122,'F3':152719}
TOTAL=sum(COUNTS.values())
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    OUT.mkdir(exist_ok=False)
    verified=common.read_json(MODEL/'verification.json')
    assert verified['status']=='passed'
    rows=[]
    seasonal={name:0. for name in ['V2','V3','composed']}
    sources={}
    per_fold={}
    top=[]
    for fold,count in COUNTS.items():
        directory=MODEL/fold
        manifest=common.read_json(directory/'manifest.json')
        assert manifest['status']=='complete'
        assert common.sha256(directory/'candidate.parquet')==manifest['outputs']['candidate.parquet']==verified['folds'][fold]['composition_artifact_sha256']
        current=pd.read_parquet(directory/'candidate.parquet')
        reference,_=common.reference(fold)
        v3dir=ROOT/'private_runs/breakthrough_20260916/models/deeper_audit/simplex_v3'/fold
        v3record=common.read_json(v3dir/'manifest.json')
        assert common.sha256(v3dir/'airport_shrunk_simplex.parquet')==v3record['outputs']['airport_shrunk_simplex.parquet']
        v3=pd.read_parquet(v3dir/'airport_shrunk_simplex.parquet')
        for frame in [reference,v3]:
            np.testing.assert_array_equal(current[ID],frame[ID])
            np.testing.assert_array_equal(current[TARGET],frame[TARGET])
        y=current[TARGET].to_numpy(float)
        proxy=current.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        gap=~missing&(np.abs(y-proxy)>1800)
        group_masks={'missing':missing,'large_source_gap':gap,'remaining':~(missing|gap)}
        assert np.all(sum(m.astype(int) for m in group_masks.values())==1)
        for airport in sorted(current.ADEP_mvt.unique()):
            group_masks['airport_'+str(airport)]=current.ADEP_mvt.eq(airport).to_numpy()
        for route in ['ordinary','nonordinary','missing']:
            group_masks['route_'+route]=current.composition_route.eq(route).to_numpy()
        squared=current.squared_error.to_numpy(float)
        k=int(np.ceil(.01*len(current)))
        worst=np.argsort(squared)[-k:]
        mask=np.zeros(len(current),bool)
        mask[worst]=True
        group_masks['composition_top1pct']=mask
        group_masks['composition_remaining99pct']=~mask
        preds={'V2':reference.prediction_sec.to_numpy(float),'V3':v3.prediction_sec.to_numpy(float),'composed':current.prediction_sec.to_numpy(float)}
        weight=count/TOTAL
        for name,pred in preds.items():
            error=(pred-y)**2
            seasonal[name]+=weight*error.mean()
            for group,select in group_masks.items():
                sse=float(error[select].sum())
                contribution=weight*sse/len(y)
                rows.append(dict(fold=fold,predictor=name,group=group,rows=int(select.sum()),
                    raw_sse=sse,ranking_equivalent_sse=count*sse/len(y),weighted_mse=contribution,
                    row_mass=weight*float(select.mean()),within_fold_sse_share=float(sse/error.sum())))
        labels=current[[ID,TARGET,'prediction_sec','ADEP_mvt','day','composition_route','proxy_sec','squared_error']].iloc[worst].copy()
        labels['fold']=fold
        labels['weighted_mse_per_row']=labels.squared_error*(weight/len(y))
        labels['diagnostic_error_group']=np.where(missing[worst],'missing',np.where(gap[worst],'large_source_gap','remaining'))
        top.append(labels)
        per_fold[fold]=dict(rows=len(y),top1pct_rows=k,top1pct_sse_share=float(squared[worst].sum()/squared.sum()),
            composition_rmse=float(np.sqrt(squared.mean())))
        sources[fold]=dict(manifest_sha256=common.sha256(directory/'manifest.json'),candidate_sha256=manifest['outputs']['candidate.parquet'])
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'fold_groups.csv',index=False)
    grouped=table.groupby(['predictor','group'])[['rows','ranking_equivalent_sse','weighted_mse','row_mass']].sum().reset_index()
    grouped['total_mse']=grouped.predictor.map(seasonal)
    grouped['sse_share']=grouped.weighted_mse/grouped.total_mse
    grouped.to_csv(OUT/'seasonal_groups.csv',index=False)
    pd.concat(top,ignore_index=True).sort_values('weighted_mse_per_row',ascending=False).to_parquet(OUT/'top1pct_diagnostic_rows.parquet',index=False)
    current=grouped.loc[grouped.predictor.eq('composed')].set_index('group')
    before=grouped.loc[grouped.predictor.eq('V2')].set_index('group')
    rmse={k:float(np.sqrt(v)) for k,v in seasonal.items()}
    manifest_summary=common.read_json(MODEL/'summary.json')
    for name in rmse:
        assert abs(rmse[name]-manifest_summary['seasonal_rmse'][name])<1e-10
    parts={}
    for name in ['ordinary','nonordinary','missing']:
        key='route_'+name
        gain=float(before.loc[key,'weighted_mse']-current.loc[key,'weighted_mse'])
        parts[name]=dict(weighted_mse_gain=gain,ranking_equivalent_sse_gain=gain*TOTAL,
            share_of_total_mse_gain=gain/(seasonal['V2']-seasonal['composed']),
            additive_rmse_gain_seconds=gain/(rmse['V2']+rmse['composed']))
    assert abs(sum(p['additive_rmse_gain_seconds'] for p in parts.values())-(rmse['V2']-rmse['composed']))<1e-10
    goals={str(goal):dict(target_mse=goal**2,necessary_mse_reduction=seasonal['composed']-goal**2,
        necessary_mse_reduction_pct=100*(1-goal**2/seasonal['composed']),
        ranking_equivalent_sse_reduction=(seasonal['composed']-goal**2)*TOTAL,
        necessary_rmse_reduction=rmse['composed']-goal) for goal in [250,230]}
    groups={name:dict(weighted_mse=float(current.loc[name,'weighted_mse']),ranking_equivalent_sse=float(current.loc[name,'ranking_equivalent_sse']),
        sse_share=float(current.loc[name,'sse_share']),weighted_row_share=float(current.loc[name,'row_mass']),rows=int(current.loc[name,'rows']),
        hypothetical_rmse_if_group_errors_zero=float(np.sqrt(max(0,seasonal['composed']-current.loc[name,'weighted_mse']))))
        for name in ['missing','large_source_gap','remaining','composition_top1pct']}
    airports=current.loc[current.index.str.startswith('airport_')].sort_values('weighted_mse',ascending=False)
    airport_report=[dict(airport=k.replace('airport_',''),weighted_mse=float(v.weighted_mse),sse_share=float(v.sse_share),weighted_row_share=float(v.row_mass),rows=int(v.rows)) for k,v in airports.iterrows()]
    summary=dict(status='passed',source_sha256=common.sha256(__file__),composition_verification_sha256=common.sha256(MODEL/'verification.json'),
        component_sources=sources,seasonal_rmse=rmse,seasonal_mse=seasonal,groups=groups,airports=airport_report,
        route_gain_vsV2=parts,goals=goals,folds=per_fold,
        weighting='F1weight192122/344841,F3weight152719/344841; weightedMSE=weight*groupSSE/foldN; ranking_equivalentSSE=weightedMSE*344841.',
        limits='Sourcegapabs(Y-proxy)>1800 andtop1percent error groups use targets and are diagnostic only, unavailableforinference. Airports overlap groups. Zero-errorremoval is hypothetical errorbudget arithmetic, not achievablemodelperformance/noisefloor. Mainmodelunfloored.')
    common.write_json(OUT/'summary.json',summary)
    print('ERROR_BUDGET',groups,flush=True)
    print('ROUTE_GAINS',parts,flush=True)
    print('GOALS',goals,flush=True)
    print('AIRPORTS',airport_report[:5],flush=True)


if __name__=='__main__':
    main()
