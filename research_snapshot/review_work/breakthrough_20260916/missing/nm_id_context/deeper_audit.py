"""Training-only ordering attribution and predeclared thirteen-bin tune controls."""
import audit_ids as base
import numpy as np
import pandas as pd

EDGES=[-np.inf,-86400,-21600,-7200,-3600,-1800,-300,300,1800,3600,7200,21600,86400,np.inf]


def distribution(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    return dict(n=len(values),minimum=float(values.min()),median=float(np.median(values)),
        q90=float(np.quantile(values,.9)),q99=float(np.quantile(values,.99)),maximum=float(values.max())) if len(values) else dict(n=0)


def ordering(dep,labels):
    dep=dep.merge(labels[[base.ID,base.TARGET]],on=base.ID,validate='one_to_one')
    takeoff=base.seconds(dep[base.MOVEMENT])
    dates=pd.to_datetime(dep[base.MOVEMENT],utc=True).dt.strftime('%Y-%m-%d')
    values={'takeoff':takeoff,'hiddenB':takeoff-dep[base.TARGET].to_numpy(float),
        'N':base.seconds(dep.AOBT_3_flt),'EOBT':base.seconds(dep.EOBT_1_flt),
        'IOBT':base.seconds(dep.IOBT_flt),'schedule':base.seconds(dep.SCHED_TIME_UTC_mvt)}
    numeric=dep[base.FLIGHT_ID].to_numpy(float)
    rows=[]
    for (airport,date),positions in dep.assign(day=dates).groupby(['ADEP_mvt','day'],observed=True).indices.items():
        valid=positions[np.isfinite(numeric[positions])]
        if len(valid)<50:
            continue
        sortedpos=valid[np.argsort(numeric[valid],kind='stable')]
        for name,value in values.items():
            observed=sortedpos[np.isfinite(value[sortedpos])]
            differences=np.diff(value[observed])
            rows.append(dict(airport=airport,day=date,clock=name,n=len(observed),
                rho=base.rho(numeric[observed],value[observed]),
                adjacent_reverse_share=float(np.mean(differences<0)),
                adjacent_equal_share=float(np.mean(differences==0))))
    table=pd.DataFrame(rows)
    report=[]
    for (airport,clock),group in table.groupby(['airport','clock']):
        report.append(dict(airport=airport,clock=clock,days=len(group),
            median_day_rho=float(group.rho.median()),mean_adjacent_reverse_share=float(group.adjacent_reverse_share.mean()),
            mean_adjacent_equal_share=float(group.adjacent_equal_share.mean())))
    unique=np.unique(numeric[np.isfinite(numeric)])
    return dict(rows=len(dep),nm_known=int(np.isfinite(numeric).sum()),nm_unique=len(unique),
        duplicate_nm_departure_rows=int(np.isfinite(numeric).sum()-len(unique)),
        nm_id_gaps=distribution(np.diff(unique)),id_mod10_counts={str(int(i)):int(n) for i,n in zip(*np.unique(unique%10,return_counts=True))},
        airport_day_clocks=report)


def additive(fit,tune,parentkeys,childkeys):
    fitted=base.grouped_predict(fit,fit,parentkeys)
    baseline=base.grouped_predict(fit,tune,parentkeys)
    leftover=fit.copy()
    leftover['residual']=fit.residual.to_numpy()-fitted
    groups=leftover.groupby(childkeys,observed=True,dropna=False).residual.agg(['mean','size']).reset_index()
    matched=tune[childkeys].merge(groups,on=childkeys,how='left',validate='many_to_one')
    count=matched['size'].fillna(0).to_numpy()
    return baseline+matched['mean'].fillna(0).to_numpy()*count/(count+50.)


def main():
    out=base.OUT/'deeper_audit.json'
    if out.exists():
        raise ValueError('Completed deeper audit retained')
    protocol=dict(created_utc=base.utc_now(),script_sha256=base.sha256(__file__),base_sha256=base.sha256(base.__file__),
        ordering='Only Jan-May2025 originalF1fit labels; perairportday NMID order compared to T,hiddenB,N,EOBT,IOBT,schedule.',
        models='Raw residual NM-airportoffblock conditional means; 13fixedbins peerNM-ownN and peerInitial-ownN; fixed50-row shrinkage.',
        additive='Base ownconventions+ownproxy5min table, then fit-only residual conditionalmeans byairport+13peerbin; avoids productgroup sparsity.',
        availability='RETROSPECTIVE suppliedmonth observations; sameNMflight strictly excluded; no score-label routing or prediction.',
        edges=[str(edge) for edge in EDGES])
    base.write_json(base.OUT/'deeper_protocol.json',protocol)
    audited=base.read_json(base.ROOT/'private_runs/screening_230/reports/data_audit.json')
    metadata=base.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert base.sha256(metadata)==audited['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata)
    f1,_,_=base.common.fold_data(meta,'F1',full=True)
    fitlabels=meta.iloc[f1['fit']][[base.ID,base.TARGET]]
    frozen=base.read_json(base.ROOT/'private_runs/submission_v2/protocol.json')
    frames=[]
    order=[]
    for index,path in enumerate(sorted(base.common.RAW.glob('training*.parquet'))[:10]):
        assert base.sha256(path)==frozen['raw_hashes'][path.name]
        raw=base.pq.read_table(path,columns=base.COLS,use_threads=False).to_pandas()
        dep=raw.loc[raw[base.PHASE].eq('DEP')].copy()
        if index<5:
            order.append(dict(file=path.name,**ordering(dep,fitlabels)))
        peers=base.neighbors(dep,raw)
        nm=base.seconds(dep.AOBT_3_flt)
        frames.append(pd.DataFrame({base.ID:dep[base.ID].to_numpy(),
            'nm_bin':pd.cut(peers.nmid_peer_nm_seconds.to_numpy()-nm,EDGES,labels=False),
            'initial_bin':pd.cut(peers.nmid_peer_initial_seconds.to_numpy()-nm,EDGES,labels=False)}))
    directory=base.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    marker=base.read_json(directory/'manifest.json')
    assert base.sha256(directory/'features.parquet')==marker['feature_sha256']
    names=['conv_clock_rank_signature','conv_nm_est_gap_bucket']
    conv=pd.read_parquet(directory/'features.parquet',columns=[base.ID,*names]).set_index(base.ID)
    context=pd.concat(frames,ignore_index=True).set_index(base.ID).reindex(meta[base.ID])
    for col in ['nm_bin','initial_bin']:
        context[col]=context[col].fillna(-1).astype(int)
    for col in names:
        context[col]=conv.loc[meta[base.ID],col].to_numpy()
    proxy=meta.proxy_sec.to_numpy(float)
    context['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    context['proxy_bin']=np.floor(np.where(np.isfinite(proxy),proxy,0)/300)
    context['residual']=meta[base.TARGET].to_numpy(float)-proxy
    parent=['airport',*names,'proxy_bin']
    reports={}
    for fold in ['F1','F3']:
        idx,_,_=base.common.fold_data(meta,fold,full=True)
        eligible=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        a=idx['fit'][eligible[idx['fit']]]
        b=idx['tune'][eligible[idx['tune']]]
        fit,tune=context.iloc[a],context.iloc[b]
        predictions=dict(airport=base.grouped_predict(fit,tune,['airport']),
            nm13=base.grouped_predict(fit,tune,['airport','nm_bin']),
            initial13=base.grouped_predict(fit,tune,['airport','initial_bin']),
            own_control=base.grouped_predict(fit,tune,parent),
            own_plus_nm13=additive(fit,tune,parent,['airport','nm_bin']),
            own_plus_initial13=additive(fit,tune,parent,['airport','initial_bin']),
            own_plus_joint13=additive(fit,tune,parent,['airport','nm_bin','initial_bin']))
        reports[fold]={name:base.stats(proxy[b]+pred-meta.iloc[b][base.TARGET].to_numpy(float)) for name,pred in predictions.items()}
    result=dict(created_utc=base.utc_now(),protocol_sha256=base.sha256(base.OUT/'deeper_protocol.json'),
        ordering=order,tune_predictions=reports,score_labels_used=False,feature_cache_written=False)
    base.write_json(out,result)
    print({fold:{name:values['rmse'] for name,values in models.items()} for fold,models in reports.items()},flush=True)


if __name__=='__main__':
    main()
