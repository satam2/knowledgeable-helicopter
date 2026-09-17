"""Diagnostic error allocation of the tune-fitted ordinary-route stacker."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'campaign_20260916'))
import common

ROOT=common.WORKSPACE/'private_runs/breakthrough_20260916/models/stacking/simplex_v2'
OUT=common.external_path(common.WORKSPACE/'private_runs/breakthrough_20260916/stacker_error_budget')
WEIGHTS={'F1':192122/344841,'F3':152719/344841}


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[];season={};sources={};top=[]
    for fold,weight in WEIGHTS.items():
        ref,_=common.reference(fold)
        path=ROOT/fold
        record=common.read_json(path/'manifest.json')
        name='airport_shrunk_simplex.parquet'
        assert common.sha256(path/name)==record['outputs'][name]
        current=pd.read_parquet(path/name)
        assert np.array_equal(current[common.ID],ref[common.ID])
        assert np.array_equal(current[common.TARGET],ref[common.TARGET])
        y=current[common.TARGET].to_numpy(float)
        proxy=current.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        gap=~missing&(np.abs(y-proxy)>1800)
        groups={'missing':missing,'large_source_gap':gap,'ordinary_gap':~(missing|gap)}
        for airport in current.ADEP_mvt.unique():
            groups['airport_'+str(airport)]=current.ADEP_mvt.eq(airport).to_numpy()
        for route,pred in [('reference',ref.prediction_sec.to_numpy(float)),
                           ('stacker',current.prediction_sec.to_numpy(float)),
                           ('stacker_guarded',np.maximum(current.prediction_sec.to_numpy(float),-12.))]:
            error=pred-y
            total=float(np.mean(error**2))
            season[route]=season.get(route,0.)+weight*total
            for key,mask in groups.items():
                rows.append({'fold':fold,'predictor':route,'group':key,'rows':int(mask.sum()),
                             'mse_contribution':float(np.sum(error[mask]**2)/len(y)),
                             'weighted_mse_contribution':float(weight*np.sum(error[mask]**2)/len(y)),
                             'row_weight':float(weight*mask.mean()),'sse_share':float(np.sum(error[mask]**2)/np.sum(error**2))})
        top.append(current.nlargest(100,'squared_error').assign(fold=fold))
        sources[fold]=common.sha256(path/'manifest.json')
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'fold_groups.csv',index=False)
    grouped=table.groupby(['predictor','group'])[['weighted_mse_contribution','row_weight']].sum().reset_index()
    grouped['total_mse']=grouped.predictor.map(season)
    grouped['sse_share']=grouped.weighted_mse_contribution/grouped.total_mse
    grouped.to_csv(OUT/'seasonal_groups.csv',index=False)
    pd.concat(top,ignore_index=True).to_parquet(OUT/'largest_errors_diagnostic.parquet',index=False)
    common.write_json(OUT/'summary.json',{'created_utc':common.utc_now(),'source_sha256':common.sha256(__file__),
        'model_manifest_hashes':sources,'seasonal_rmse':{k:float(np.sqrt(v)) for k,v in season.items()},
        'limits':'Target-defined source-gap groups and largest errors are diagnostics only; unavailable for inference routing. Airport shares overlap source groups.',
        'necessary_mse_reduction_from_stacker_pct':{str(goal):float(100*(1-goal**2/season['stacker'])) for goal in [250,243.5805,230]}})
    print(grouped.loc[grouped.predictor.eq('stacker'),['group','row_weight','sse_share']].to_string(index=False),flush=True)
    print({k:float(np.sqrt(v)) for k,v in season.items()},flush=True)
