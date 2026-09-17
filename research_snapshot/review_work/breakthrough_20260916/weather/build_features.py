"""Offline prior-issued METAR features, independent of private departure labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'campaign_20260916'))
import common
from taxiout.schema import MOVEMENT, PHASE

PUBLIC = common.WORKSPACE/'output/breakthrough_20260916/public_weather'
OUT = common.external_path(common.WORKSPACE/'private_runs/breakthrough_20260916/weather')
NUMERIC = ['tmpf','dwpf','relh','drct','sknt','alti','vsby','gust','skyl1','skyl2','skyl3']
STALE_SECONDS = 10800.


def prepare_weather(raw):
    weather = raw.copy()
    weather['valid'] = pd.to_datetime(weather['valid'],utc=True,errors='raise')
    for column in NUMERIC:
        weather[column] = pd.to_numeric(weather[column].replace({'M':np.nan,'T':np.nan}),errors='coerce')
    weather = weather.sort_values(['station','valid'],kind='stable').drop_duplicates(['station','valid'],keep='last')
    weather['temperature_c'] = (weather.tmpf-32)*5/9
    weather['dewpoint_depression_c'] = (weather.tmpf-weather.dwpf)*5/9
    weather['wind_east_kt'] = -weather.sknt*np.sin(np.deg2rad(weather.drct))
    weather['wind_north_kt'] = -weather.sknt*np.cos(np.deg2rad(weather.drct))
    weather['wind_direction_missing'] = weather.drct.isna().astype(float)
    weather['visibility_reported_miles'] = weather.vsby
    weather['pressure_inhg'] = weather.alti
    weather['gust_reported_kt'] = weather.gust
    weather['wind_speed_kt'] = weather.sknt
    weather['humidity_pct'] = weather.relh
    weather['lowest_cloud_reported_ft'] = weather[['skyl1','skyl2','skyl3']].min(axis=1)
    codes = weather.wxcodes.fillna('').replace('M','').astype(str)
    for key, pattern in [('rain','RA|DZ'),('snow','SN|SG|PL'),('fog','FG|BR'),('thunder','TS'),('freezing','FZ')]:
        weather[key+'_reported'] = codes.str.contains(pattern,regex=True).astype(float)
    keep = ['temperature_c','dewpoint_depression_c','wind_east_kt','wind_north_kt','wind_direction_missing',
            'visibility_reported_miles','pressure_inhg','gust_reported_kt','wind_speed_kt','humidity_pct',
            'lowest_cloud_reported_ft','rain_reported','snow_reported','fog_reported','thunder_reported','freezing_reported']
    return weather[['station','valid',*keep]], keep


def prior_features(queries, weather, fields, query_column, prefix):
    result = pd.DataFrame(np.nan,index=queries.index,columns=[prefix+c for c in fields],dtype='float32')
    age = np.full(len(queries),np.nan)
    available = np.zeros(len(queries),dtype=bool)
    for station, positions in queries.groupby('ADEP_mvt',observed=True).indices.items():
        observations = weather.loc[weather.station.eq(str(station))].sort_values('valid')
        if observations.empty:
            continue
        times = observations.valid.to_numpy(dtype='datetime64[ns]').astype('int64')
        query = pd.to_datetime(queries.iloc[positions][query_column],utc=True)
        qns = query.to_numpy(dtype='datetime64[ns]').astype('int64')
        pick = np.searchsorted(times,qns,side='left')-1
        valid = query.notna().to_numpy() & (pick>=0)
        clipped = np.maximum(pick,0)
        ages = (qns-times[clipped])/1e9
        accepted = valid & (ages>0) & (ages<=STALE_SECONDS)
        age[positions] = np.where(valid,ages,np.nan)
        available[positions] = accepted
        result.iloc[positions[accepted],:] = observations.iloc[clipped[accepted]][fields].to_numpy(dtype='float32')
    result[prefix+'age_sec'] = age.astype('float32')
    result[prefix+'available'] = available.astype('float32')
    return result


def transform(queries,weather,fields):
    q = queries.reset_index(drop=True).copy()
    q['nm_query'] = pd.to_datetime(q[MOVEMENT],utc=True)-pd.to_timedelta(q.proxy_sec,unit='s')
    t = prior_features(q,weather,fields,MOVEMENT,'weather_T_')
    n = prior_features(q,weather,fields,'nm_query','weather_N_')
    return pd.concat([q[[common.ID]],t,n],axis=1)


def main():
    if (OUT/'manifest.json').exists():
        saved=common.read_json(OUT/'manifest.json')
        assert saved['status']=='complete' and saved['script_sha256']==common.sha256(__file__)
        for filename,digest in saved['outputs'].items():
            assert common.sha256(OUT/filename)==digest
        print('REUSED verified completed weather cache',flush=True)
        return
    manifest = common.read_json(PUBLIC/'manifest.json')
    assert manifest['status']=='complete'
    frames=[]
    for receipt in manifest['receipts']:
        path = PUBLIC/f"{receipt['station']}_{receipt['period']}.csv"
        assert common.sha256(path)==receipt['sha256']
        frames.append(pd.read_csv(path,keep_default_na=False))
    raw = pd.concat(frames,ignore_index=True)
    weather,fields = prepare_weather(raw)
    meta_path=common.WORKSPACE/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=common.read_json(common.WORKSPACE/'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path)==audit['artifacts'][meta_path.name]
    query=pd.read_parquet(meta_path,columns=[common.ID,'ADEP_mvt',MOVEMENT,'proxy_sec'])
    ranking_paths=list(common.RAW.glob('ranking*.parquet'))
    assert len(ranking_paths)==1
    frozen=common.read_json(common.WORKSPACE/'private_runs/submission_v2/protocol.json')
    assert common.sha256(ranking_paths[0])==frozen['raw_hashes'][ranking_paths[0].name]
    rank=pd.read_parquet(ranking_paths[0],columns=[common.ID,'ADEP_mvt',MOVEMENT,'AOBT_3_flt',PHASE])
    rank=rank.loc[rank[PHASE].eq('DEP')].copy()
    rank['proxy_sec']=(pd.to_datetime(rank[MOVEMENT],utc=True)-pd.to_datetime(rank.AOBT_3_flt,utc=True)).dt.total_seconds()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'created_utc':common.utc_now(),'script_sha256':common.sha256(__file__),
        'public_manifest_sha256':common.sha256(PUBLIC/'manifest.json'),'hidden_columns_loaded':[],
        'features':'Most recent strictly prior observation valid timestamp at supplied takeoff T and finalNMoffblock N.',
        'stale_seconds':STALE_SECONDS,'duplicates':'stable input order, last record for station+issue; archive corrections may be retrospective',
        'availability':'Weather archive uses past observation valid timestamps; issue/publication/correction times unavailable. N is finalNM record and can exceed T for negative proxies, so N query is not guaranteed prospective to takeoff.',
        'no_clipping':'Original labels never loaded; visibility/ceiling are reported values with source-specific censoring; absent wx code is no code reported, not sensor proof.',
        'runway_relative_wind':False,'weather_counts':weather.groupby('station').size().to_dict()}
    summaries={}
    for name,queries in [('training',query),('ranking',rank)]:
        features=transform(queries,weather,fields)
        assert np.array_equal(features[common.ID],queries[common.ID])
        features.to_parquet(OUT/f'{name}_features.parquet',index=False)
        summaries[name]={'rows':len(features),'takeoff_coverage':float(features.weather_T_available.mean()),
            'nm_coverage':float(features.weather_N_available.mean()),
            'airport_coverage':features.assign(airport=queries.ADEP_mvt.to_numpy()).groupby('airport').weather_T_available.mean().to_dict()}
    common.write_json(OUT/'manifest.json',dict(protocol,status='complete',summary=summaries,
        outputs={p.name:common.sha256(p) for p in OUT.glob('*_features.parquet')}))
    print(summaries,flush=True)


if __name__=='__main__':
    main()
