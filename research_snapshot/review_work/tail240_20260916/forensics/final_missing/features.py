"""Pure observed DEP record-neighbor calculation transferred from frozen builder."""
import warnings
import numpy as np
import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'


def record_peers(metadata):
    required = [ID,'ADEP_mvt',TIME]
    if list(metadata.columns) != required or not metadata[ID].is_unique or metadata[ID].isna().any():
        raise ValueError('Exact unique observed ID/airport/time fields required')
    meta = metadata.copy()
    times = pd.to_datetime(meta[TIME],utc=True,errors='raise')
    if times.isna().any():
        raise ValueError('Movement timestamp missing')
    meta['month_key'] = times.dt.strftime('%Y-%m')
    seconds = times.astype('int64').to_numpy()/(1e6 if times.dtype.unit=='us' else 1e9)
    result = pd.DataFrame(index=pd.Index(meta[ID],name=ID))
    for width in [2,8,32]:
        median = np.full(len(meta),np.nan)
        spread = np.full(len(meta),np.nan)
        for _,positions in meta.groupby(['ADEP_mvt','month_key'],observed=True).indices.items():
            positions = positions[np.argsort(meta.iloc[positions][ID].to_numpy())]
            local = seconds[positions]
            padded = np.pad(local,(width,width),constant_values=np.nan)
            windows = np.lib.stride_tricks.sliding_window_view(padded,2*width+1)
            peers = np.concatenate([windows[:,:width],windows[:,width+1:]],axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                median[positions] = np.nanmedian(peers,axis=1)
                spread[positions] = np.nanquantile(peers,.9,axis=1)-np.nanquantile(peers,.1,axis=1)
        result[f'id_peer_median_time_minus_query_w{width}'] = median-seconds
        result[f'id_peer_time_spread_w{width}'] = spread
    return result
