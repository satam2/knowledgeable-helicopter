"""Saved model split-gain diagnostics, not causal importance or selection weights."""
import sys
from pathlib import Path
import joblib
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'campaign_20260916'))
import common


def block(column):
    for prefix,group in [('conv_','clock_conventions'),('histgeom_','geometry'),('weather_','weather'),
                         ('batch_source_','peer_source'),('batch_surface_','surface_inventory'),
                         ('retro_own_','trajectory'),('source_','trajectory_source_flags')]:
        if column.startswith(prefix):
            return group
    return 'base'


if __name__=='__main__':
    root=common.WORKSPACE/'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
    out=common.external_path(common.WORKSPACE/'private_runs/breakthrough_20260916/information_inspection')
    out.mkdir(parents=True,exist_ok=False)
    rows=[];sources={}
    for fold in ['F1','F3']:
        directory=root/f'lightgbm_aobt_allfinite_{fold}_s20260916'
        manifest=common.read_json(directory/'manifest.json')
        assert manifest['status']=='complete'
        path=directory/'model.joblib'
        assert common.sha256(path)==manifest['outputs'][path.name]
        model=joblib.load(path)
        estimator=model['estimator'].booster_
        table=pd.DataFrame({'feature':model['encoder'].columns,'gain':estimator.feature_importance(importance_type='gain'),
                            'splits':estimator.feature_importance(importance_type='split')})
        table['fold']=fold
        table['block']=table.feature.map(block)
        table['gain_fraction']=table.gain/table.gain.sum()
        rows.append(table)
        sources[fold]=common.sha256(directory/'manifest.json')
        print(fold,table.sort_values('gain',ascending=False)[['feature','gain_fraction']].head(15).to_dict('records'),flush=True)
    features=pd.concat(rows,ignore_index=True)
    features.to_csv(out/'features.csv',index=False)
    blocks=features.groupby(['fold','block'])[['gain','gain_fraction','splits']].sum().reset_index()
    blocks.to_csv(out/'blocks.csv',index=False)
    common.write_json(out/'protocol.json',{'script_sha256':common.sha256(__file__),'created_utc':common.utc_now(),
        'manifest_hashes':sources,'interpretation':'Training split-gain usage only. Correlated features and cardinality bias invalidate causal interpretation; separate feature ablations are needed.'})
    print(blocks[['fold','block','gain_fraction']].to_string(index=False),flush=True)
