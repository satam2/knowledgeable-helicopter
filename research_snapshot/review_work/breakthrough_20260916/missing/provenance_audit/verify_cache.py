"""Independent explicit-window validation of three source-frequency features."""
import build_cache as builder
import audit as core
import numpy as np
import pandas as pd


def verify(meta,cache,seed):
    time=pd.to_datetime(meta[core.MOVEMENT],utc=True)
    month=time.dt.strftime('%Y-%m').to_numpy()
    airport=meta.ADEP_mvt.astype(str).to_numpy()
    proxy=meta.proxy_sec.to_numpy(float)
    finite=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    cached=cache.set_index(core.ID)
    selected=np.random.default_rng(seed).choice(len(meta),80,replace=False)
    for pos in selected:
        selected_time=time.iloc[pos]
        mask=(airport==airport[pos])&(month==month[pos])&finite&(time>=selected_time-pd.Timedelta(hours=1))&(time<selected_time)
        count=int(mask.sum())
        if finite[pos] and count:
            rounded=np.floor((proxy[mask]+30)/60)
            own=np.floor((proxy[pos]+30)/60)
            share=np.mean(rounded==own)
        else:
            share=0.
        expected=np.array([count,share,float(finite[pos])],dtype=np.float32)
        actual=cached.loc[meta.iloc[pos][core.ID],builder.COLS].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)
    return dict(queries=len(selected),columns=len(builder.COLS),max_abs_delta=0.)


def main():
    out=builder.OUT/'verification.json'
    if out.exists():
        raise ValueError('Completedverification retained')
    marker=core.read_json(builder.OUT/'manifest.json')
    for name,digest in marker['outputs'].items():
        assert core.sha256(builder.OUT/name)==digest
    training=pd.read_parquet(builder.OUT/'training_features.parquet')
    ranking=pd.read_parquet(builder.OUT/'ranking_features.parquet')
    path=core.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    meta=pd.read_parquet(path,columns=[core.ID,core.MOVEMENT,'ADEP_mvt','proxy_sec'])
    np.testing.assert_array_equal(training[core.ID],meta[core.ID])
    a=verify(meta,training,20260916)
    raw=core.pq.read_table(core.common.RAW/'ranking.parquet',columns=[core.ID,'PHASE_mvt',core.MOVEMENT,'ADEP_mvt','AOBT_3_flt'],use_threads=False).to_pandas()
    rank=raw.loc[raw.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    rank['proxy_sec']=(pd.to_datetime(rank[core.MOVEMENT],utc=True)-pd.to_datetime(rank.AOBT_3_flt,utc=True)).dt.total_seconds()
    np.testing.assert_array_equal(ranking[core.ID],rank[core.ID])
    b=verify(rank,ranking,20260917)
    core.write_json(out,dict(status='passed',created_utc=core.utc_now(),manifest_sha256=core.sha256(builder.OUT/'manifest.json'),
        verifier_sha256=core.sha256(__file__),checks=dict(training=a,ranking=b),training_id_order_verified=True,ranking_id_order_verified=True,
        labels_loaded=False,method='Explicitairport/month/time Booleanmasks androundedproxy equality; no producer function called.'))
    print('VERIFIED',a,b,flush=True)


if __name__=='__main__':
    main()
