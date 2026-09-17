"""Own-clock controlled NMID audit; no model/cache or score-label selection."""
import audit_ids as base
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    output=base.OUT/'incremental_audit.json'
    if output.exists():
        raise ValueError('Completed incremental audit preserved')
    marker=base.read_json(base.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions/manifest.json')
    path=base.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions/features.parquet'
    assert base.sha256(path)==marker['feature_sha256']
    columns=[base.ID,'conv_clock_rank_signature','conv_nm_est_gap_bucket']
    conventions=pd.read_parquet(path,columns=columns).set_index(base.ID)
    protocol=dict(created_utc=base.utc_now(),script_sha256=base.sha256(__file__),base_script_sha256=base.sha256(base.__file__),
        purpose='Determine whether NMID context transfers beyond own clock conventions and ownproxy five-minute bins.',
        groups='airport+sourceorder+sourcegap, plus ownproxy floor(proxy/300), plus peerNM/initial fixed prior bins; all groups mean raw residual, shrinkage50 to airportmean.',
        labels='Original fit/tune only; no score evaluation. No label clipping.',
        convention_cache_sha256=marker['feature_sha256'])
    base.write_json(base.OUT/'incremental_protocol.json',protocol)
    frozen=base.read_json(base.ROOT/'private_runs/submission_v2/protocol.json')
    frames=[]
    monthly=[]
    for path in sorted(base.common.RAW.glob('training*.parquet'))[:10]:
        assert base.sha256(path)==frozen['raw_hashes'][path.name]
        raw=base.pq.read_table(path,columns=base.COLS,use_threads=False).to_pandas()
        dep=raw.loc[raw[base.PHASE].eq('DEP')].copy()
        peers=base.neighbors(dep,raw)
        nm=base.seconds(dep.AOBT_3_flt)
        takeoff=base.seconds(dep[base.MOVEMENT])
        flight=dep[base.FLIGHT_ID].to_numpy(float)
        frame=pd.DataFrame({base.ID:dep[base.ID].to_numpy(),
            'peer_bin':pd.cut(peers.nmid_peer_nm_seconds.to_numpy()-nm,base.BINS,labels=False),
            'initial_peer_bin':pd.cut(peers.nmid_peer_initial_seconds.to_numpy()-nm,base.BINS,labels=False),
            'peer_nm_minus_own_nm':peers.nmid_peer_nm_seconds.to_numpy()-nm})
        frames.append(frame)
        within=[]
        day=pd.to_datetime(dep[base.MOVEMENT],utc=True).dt.strftime('%Y-%m-%d')
        groups=dep.assign(day=day).groupby(['ADEP_mvt','day'],observed=True).indices
        for (airport,date),positions in groups.items():
            if len(positions)<50:
                continue
            within.append(dict(airport=airport,day=date,n=len(positions),
                nm=base.rho(flight[positions],nm[positions]),takeoff=base.rho(flight[positions],takeoff[positions])))
        monthly.append(dict(file=path.name,within_airport_day_groups=len(within),
            median_nm_rho=float(np.nanmedian([r['nm'] for r in within])),
            median_takeoff_rho=float(np.nanmedian([r['takeoff'] for r in within])),
            peerclock_delta_vs_ownproxy_rho=base.rho(frame.peer_nm_minus_own_nm,takeoff-nm)))
    meta_path=base.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=base.read_json(base.ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert base.sha256(meta_path)==audit['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path)
    context=pd.concat(frames).set_index(base.ID).reindex(meta[base.ID])
    context['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    context['residual']=meta[base.TARGET].to_numpy(float)-meta.proxy_sec.to_numpy(float)
    proxy=meta.proxy_sec.to_numpy(float)
    context['proxy_bin']=np.floor(np.where(np.isfinite(proxy),proxy,0)/300)
    for column in columns[1:]:
        context[column]=conventions.loc[meta[base.ID],column].to_numpy()
    for column in ['peer_bin','initial_peer_bin']:
        context[column]=context[column].fillna(-1).astype(int)
    keys=['airport','conv_clock_rank_signature','conv_nm_est_gap_bucket']
    models={'airport':['airport'],'conventions':keys,'own_proxy':keys+['proxy_bin'],
        'own_proxy_and_nmid':keys+['proxy_bin','peer_bin'],
        'own_proxy_and_nmid_initial':keys+['proxy_bin','peer_bin','initial_peer_bin']}
    reports={}
    for fold in ['F1','F3']:
        idx,_,_=base.common.fold_data(meta,fold,full=True)
        ordinary=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        fitrows=idx['fit'][ordinary[idx['fit']]]
        tunerows=idx['tune'][ordinary[idx['tune']]]
        fit,tune=context.iloc[fitrows],context.iloc[tunerows]
        reports[fold]={}
        for method,group in models.items():
            prediction=proxy[tunerows]+base.grouped_predict(fit,tune,group)
            reports[fold][method]=base.stats(prediction-meta.iloc[tunerows][base.TARGET].to_numpy(float))
    base.write_json(output,dict(created_utc=base.utc_now(),protocol_sha256=base.sha256(base.OUT/'incremental_protocol.json'),
        months=monthly,tune_predictions=reports,score_labels_used=False))
    print({fold:{method:values['rmse'] for method,values in methods.items()} for fold,methods in reports.items()},flush=True)


if __name__=='__main__':
    main()
