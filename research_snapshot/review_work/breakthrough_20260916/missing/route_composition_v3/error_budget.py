"""Both fixed compositions: weighted source-gap/route/airport error budget."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET=common.ID,common.TARGET
BASE=ROOT/'private_runs/breakthrough_20260916'
MODEL=BASE/'missing/route_composition_v3'
OUT=MODEL/'error_budget'
COUNTS={'F1':192122,'F3':152719}
TOTAL=sum(COUNTS.values())
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    assert not OUT.exists()
    verified=common.read_json(MODEL/'verification.json')
    assert verified['status']=='passed'
    OUT.mkdir()
    rows=[]
    totals={}
    receipts={}
    top=[]
    for fold,count in COUNTS.items():
        reference,_=common.reference(fold)
        labels=reference[TARGET].to_numpy(float)
        proxy=reference.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        large=~missing&(np.abs(labels-proxy)>1800)
        weight=count/TOTAL
        previous=pd.read_parquet(BASE/'missing/route_composition_v2'/fold/'candidate.parquet')
        previous_manifest=common.read_json(BASE/'missing/route_composition_v2'/fold/'manifest.json')
        assert common.sha256(BASE/'missing/route_composition_v2'/fold/'candidate.parquet')==previous_manifest['outputs']['candidate.parquet']
        np.testing.assert_array_equal(previous[ID],reference[ID])
        np.testing.assert_array_equal(previous[TARGET],reference[TARGET])
        for variant in ['global4','airport4']:
            path=MODEL/variant/fold/'candidate.parquet'
            assert common.sha256(path)==verified['folds'][fold][variant]['candidate_sha256']
            current=pd.read_parquet(path)
            np.testing.assert_array_equal(current[ID],reference[ID])
            np.testing.assert_array_equal(current[TARGET],reference[TARGET])
            groups={'missing':missing,'large_source_gap':large,'remaining':~(missing|large)}
            for a in sorted(reference.ADEP_mvt.unique()):
                groups['airport_'+str(a)]=reference.ADEP_mvt.eq(a).to_numpy()
            for route in ['ordinary','nonordinary','missing']:
                groups['route_'+route]=current.composition_route.eq(route).to_numpy()
            error=current.squared_error.to_numpy(float)
            worst=np.argsort(error)[-int(np.ceil(len(current)*.01)):]
            mask=np.zeros(len(error),bool)
            mask[worst]=True
            groups['top1pct']=mask
            for name,frame in [('V2',reference),('previous_composition',previous),('composed',current)]:
                squared=(frame.prediction_sec.to_numpy(float)-labels)**2
                totals[(variant,name)]=totals.get((variant,name),0)+weight*squared.mean()
                for group,select in groups.items():
                    rows.append(dict(variant=variant,fold=fold,predictor=name,group=group,
                        rows=int(select.sum()),weighted_mse=weight*float(squared[select].sum())/len(labels),
                        row_mass=weight*float(select.mean())))
            topframe=current.iloc[worst][[ID,TARGET,'prediction_sec','proxy_sec','ADEP_mvt','day','squared_error','composition_route']].copy()
            topframe['variant']=variant
            topframe['fold']=fold
            topframe['weighted_mse_per_row']=topframe.squared_error*(weight/len(labels))
            top.append(topframe)
            receipts[variant+'/'+fold]=verified['folds'][fold][variant]
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'fold_groups.csv',index=False)
    aggregate=table.groupby(['variant','predictor','group'],as_index=False)[['rows','weighted_mse','row_mass']].sum()
    aggregate['total_mse']=[totals[(r.variant,r.predictor)] for r in aggregate.itertuples()]
    aggregate['sse_share']=aggregate.weighted_mse/aggregate.total_mse
    aggregate.to_csv(OUT/'seasonal_groups.csv',index=False)
    pd.concat(top,ignore_index=True).to_parquet(OUT/'top1pct_diagnostic_rows.parquet',index=False)
    compose_summary=common.read_json(MODEL/'summary.json')
    summaries={}
    for variant in ['global4','airport4']:
        current=aggregate[(aggregate.variant==variant)&(aggregate.predictor=='composed')].set_index('group')
        base=aggregate[(aggregate.variant==variant)&(aggregate.predictor=='V2')].set_index('group')
        mse=totals[(variant,'composed')]
        rmse=float(np.sqrt(mse))
        assert abs(rmse-compose_summary['variants'][variant]['seasonal_rmse']['composed'])<1e-10
        groups={g:dict(weighted_mse=float(current.loc[g,'weighted_mse']),sse_share=float(current.loc[g,'sse_share']),
            weighted_row_share=float(current.loc[g,'row_mass']),rows=int(current.loc[g,'rows']),
            hypothetical_rmse_if_zero=float(np.sqrt(max(0,mse-current.loc[g,'weighted_mse'])))) for g in ['missing','large_source_gap','remaining','top1pct']}
        routes={}
        for route in ['ordinary','nonordinary','missing']:
            gain=float(base.loc['route_'+route,'weighted_mse']-current.loc['route_'+route,'weighted_mse'])
            routes[route]=dict(weighted_mse_gain=gain,additive_rmse_gain_seconds=gain/(float(np.sqrt(totals[(variant,'V2')]))+rmse))
        airports=current[current.index.str.startswith('airport_')].sort_values('weighted_mse',ascending=False)
        summaries[variant]=dict(seasonal_rmse=rmse,seasonal_mse=mse,groups=groups,route_gains_vsV2=routes,
            airports=[dict(airport=g.removeprefix('airport_'),sse_share=float(r.sse_share),weighted_row_share=float(r.row_mass)) for g,r in airports.iterrows()],
            goals={str(goal):dict(required_rmse_reduction=rmse-goal,required_mse_reduction_pct=100*(1-goal**2/mse)) for goal in [250,230]})
    common.write_json(OUT/'summary.json',dict(status='passed',source_sha256=common.sha256(__file__),
        verification_sha256=common.sha256(MODEL/'verification.json'),sources=receipts,variants=summaries,
        limits='Hidden-label source gap and worst1percent groups are diagnostics only. Zero-error counterfactuals are arithmetic bounds, not achievable forecasts. Both variants retained; no score-selected winner.',
        outputs={p.name:common.sha256(p) for p in OUT.iterdir() if p.is_file()}))
    print('ERROR_BUDGET',summaries,flush=True)


if __name__=='__main__':
    main()
