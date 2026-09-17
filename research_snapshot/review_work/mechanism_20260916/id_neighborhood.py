"""Retrospective label-free record-neighbor timing, outcome used only to audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'campaign_20260916'))
from common import WORKSPACE, ID, TARGET, MOVEMENT, load_data, write_json, external_path, utc_now, sha256
import numpy as np
import pandas as pd

OUT=external_path(WORKSPACE/'private_runs/mechanism_20260916/id_neighbors')


if __name__=='__main__':
    x,m=load_data()
    del x
    t=pd.to_datetime(m[MOVEMENT],utc=True)
    m['month_key']=t.dt.strftime('%Y-%m')
    seconds=t.astype('int64').to_numpy()/1e6 if t.dtype.unit=='us' else t.astype('int64').to_numpy()/1e9
    m['t_sec']=seconds
    features=pd.DataFrame({ID:m[ID]})
    for width in [2,8,32]:
        estimate=np.full(len(m),np.nan)
        dispersion=np.full(len(m),np.nan)
        for _,pos in m.groupby(['ADEP_mvt','month_key'],observed=True).indices.items():
            pos=pos[np.argsort(m.iloc[pos][ID].to_numpy())]
            times=seconds[pos]
            pad=np.pad(times,(width,width),constant_values=np.nan)
            windows=np.lib.stride_tricks.sliding_window_view(pad,2*width+1)
            peers=np.concatenate([windows[:,:width],windows[:,width+1:]],axis=1)
            estimate[pos]=np.nanmedian(peers,axis=1)
            dispersion[pos]=np.nanquantile(peers,.9,axis=1)-np.nanquantile(peers,.1,axis=1)
        features[f'id_peer_median_time_minus_query_w{width}']=estimate-seconds
        features[f'id_peer_time_spread_w{width}']=dispersion
    OUT.mkdir(parents=True,exist_ok=True)
    features.to_parquet(OUT/'features.parquet',index=False)
    reports={}
    missing=~np.isfinite(m.proxy_sec.to_numpy())
    y=m[TARGET].to_numpy(float)
    for period in ['2025-07','2025-11','all']:
        base=np.ones(len(m),bool) if period=='all' else m.month_key.eq(period).to_numpy()
        result={}
        for label,mask in [('missing_all',missing),('missing_over12h',missing&(y>43200)),('missing_day_plus_2h',missing&(y>=86400)&(y<=93600)),('ordinary',~missing)]:
            use=base&mask
            result[label]={'n':int(use.sum()),'windows':{}}
            for width in [2,8,32]:
                delta=features[f'id_peer_median_time_minus_query_w{width}'].to_numpy()
                result[label]['windows'][str(width)]={'median_delta_sec':float(np.nanmedian(delta[use])) if use.any() else None,
                    'peer_median_12h_earlier_n':int(np.sum(delta[use]<-43200)),
                    'peer_median_within_2h_of_hidden_block_n':int(np.sum(np.abs(delta[use]+y[use])<7200))}
        reports[period]=result
    write_json(OUT/'audit.json',dict(created_utc=utc_now(),policy='retrospective supplied DEP record order, sameairport+month, self excluded, no target in features',
        reports=reports,feature_sha256=sha256(OUT/'features.parquet'),script_sha256=sha256(__file__)))
    print(reports['all']['missing_day_plus_2h'],flush=True)
    print(reports['2025-11']['missing_day_plus_2h'],flush=True)
