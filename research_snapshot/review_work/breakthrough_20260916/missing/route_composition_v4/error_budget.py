"""Final9 weighted error budgets for both predeclared route compositions."""
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
MODEL=ROOT/'private_runs/breakthrough_20260916/missing/route_composition_v4'
OUT=MODEL/'error_budget'
COUNTS={'F1':192122,'F3':152719}
TOTAL=sum(COUNTS.values())
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    assert not OUT.exists()
    verified=common.read_json(MODEL/'verification.json')
    assert verified['status']=='passed'
    declared=common.read_json(MODEL/'protocol.json')['declaration']
    OUT.mkdir()
    rows=[]
    totals={}
    topframes=[]
    for fold,count in COUNTS.items():
        reference,_=common.reference(fold)
        y=reference[TARGET].to_numpy(float)
        proxy=reference.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        large=~missing&(np.abs(y-proxy)>1800)
        weight=count/TOTAL
        for variant in declared['variants']:
            path=MODEL/variant/fold/'candidate.parquet'
            assert common.sha256(path)==verified['folds'][fold][variant]['candidate_sha256']
            current=pd.read_parquet(path)
            receipt=declared['fixed_sources'][fold][variant]
            previous_path=Path(receipt['directory'])/receipt['filename']
            assert common.sha256(previous_path)==receipt['prediction_sha256']
            previous=pd.read_parquet(previous_path)
            for f in [current,previous]:
                np.testing.assert_array_equal(f[ID],reference[ID])
                np.testing.assert_array_equal(f[TARGET],reference[TARGET])
            masks={'missing':missing,'large_source_gap':large,'remaining':~(missing|large)}
            for airport in sorted(reference.ADEP_mvt.unique()):
                masks['airport_'+str(airport)]=reference.ADEP_mvt.eq(airport).to_numpy()
            for route in ['ordinary','nonordinary','missing']:
                masks['route_'+route]=current.composition_route.eq(route).to_numpy()
            worst=np.argsort(current.squared_error.to_numpy())[-int(np.ceil(.01*len(current))):]
            mask=np.zeros(len(current),bool)
            mask[worst]=True
            masks['top1pct']=mask
            for name,frame in [('V2',reference),('previous_composition',previous),('composed',current)]:
                squared=(frame.prediction_sec.to_numpy(float)-y)**2
                totals[(variant,name)]=totals.get((variant,name),0.)+weight*squared.mean()
                for group,selected in masks.items():
                    rows.append(dict(variant=variant,fold=fold,predictor=name,group=group,rows=int(selected.sum()),
                        weighted_mse=weight*float(squared[selected].sum())/len(y),row_mass=weight*float(selected.mean())))
            top=current.iloc[worst][[ID,TARGET,'prediction_sec','proxy_sec','ADEP_mvt','day','squared_error','composition_route']].copy()
            top['variant']=variant
            top['fold']=fold
            top['weighted_mse_per_row']=top.squared_error*weight/len(y)
            topframes.append(top)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'fold_groups.csv',index=False)
    aggregate=table.groupby(['variant','predictor','group'],as_index=False)[['rows','weighted_mse','row_mass']].sum()
    aggregate['total_mse']=[totals[(r.variant,r.predictor)] for r in aggregate.itertuples()]
    aggregate['sse_share']=aggregate.weighted_mse/aggregate.total_mse
    aggregate.to_csv(OUT/'seasonal_groups.csv',index=False)
    pd.concat(topframes,ignore_index=True).to_parquet(OUT/'top1pct_diagnostic_rows.parquet',index=False)
    scored=common.read_json(MODEL/'summary.json')
    summaries={}
    for variant in declared['variants']:
        current=aggregate[(aggregate.variant==variant)&(aggregate.predictor=='composed')].set_index('group')
        base=aggregate[(aggregate.variant==variant)&(aggregate.predictor=='V2')].set_index('group')
        mse=totals[(variant,'composed')]
        rmse=float(np.sqrt(mse))
        assert abs(rmse-scored['variants'][variant]['seasonal_rmse']['composed'])<1e-10
        groups={g:dict(weighted_mse=float(current.loc[g,'weighted_mse']),sse_share=float(current.loc[g,'sse_share']),
            weighted_row_share=float(current.loc[g,'row_mass']),rows=int(current.loc[g,'rows']),
            hypothetical_rmse_if_zero=float(np.sqrt(max(0,mse-current.loc[g,'weighted_mse'])))) for g in ['missing','large_source_gap','remaining','top1pct']}
        routes={}
        for route in ['ordinary','nonordinary','missing']:
            gain=float(base.loc['route_'+route,'weighted_mse']-current.loc['route_'+route,'weighted_mse'])
            routes[route]=dict(weighted_mse_gain=gain,additive_rmse_gain_seconds=gain/(float(np.sqrt(totals[(variant,'V2')]))+rmse))
        assert abs(sum(r['additive_rmse_gain_seconds'] for r in routes.values())-(np.sqrt(totals[(variant,'V2')])-rmse))<1e-10
        airports=current[current.index.str.startswith('airport_')].sort_values('weighted_mse',ascending=False)
        summaries[variant]=dict(seasonal_rmse=rmse,seasonal_mse=mse,groups=groups,route_gains_vsV2=routes,
            airports=[dict(airport=g.removeprefix('airport_'),sse_share=float(r.sse_share),weighted_row_share=float(r.row_mass)) for g,r in airports.iterrows()],
            goals={str(goal):dict(required_rmse_reduction=rmse-goal,required_mse_reduction_pct=100*(1-goal**2/mse)) for goal in [250,230]})
    common.write_json(OUT/'summary.json',dict(status='passed',source_sha256=common.sha256(__file__),
        verification_sha256=common.sha256(MODEL/'verification.json'),variants=summaries,
        limits='Hidden-labelsourcegap andworst1percent are diagnostic only. Zero-errorcounterfactualsarithmetic,not forecasts. Bothpredeclaredvariantsretained.',
        outputs={p.name:common.sha256(p) for p in OUT.iterdir() if p.is_file()}))
    print('FINAL9_ERROR_BUDGET',summaries,flush=True)


if __name__=='__main__':
    main()
