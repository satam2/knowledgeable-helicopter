"""Explicit timestamp units and matched airport-only residual control."""
import diagnostic as old
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
common=old.common
ROOT,CACHE,RAW=old.ROOT,old.CACHE,old.RAW
ID,TARGET,TIME=old.ID,old.TARGET,old.TIME
OUT=old.OUT/'attempt_v2'


def seconds(values):
    value=pd.to_datetime(values,utc=True)
    out=value.dt.as_unit('ns').astype('int64').to_numpy()/1e9
    out[value.isna().to_numpy()]=np.nan
    return out


def main():
    OUT.mkdir(exist_ok=False)
    test=pd.Series(pd.to_datetime(['2025-01-01T00:00:00Z',None],utc=True)).dt.as_unit('us')
    assert seconds(test)[0]==1735689600 and np.isnan(seconds(test)[1])
    common.write_json(OUT/'protocol.json',dict(source_sha256=common.sha256(__file__),helper_sha256=common.sha256(old.__file__),
        corrections='Explicitns timestampnormalization; compareairport-only secondstagevsairport+arrivalgapbin withsame50shrink.',
        prior_attempt_invalid=True,scope='Same fixedseed sample20kfitrowspermonth+fullJune/Octtune; strictcompletedARRcontextandNMtime<T; noscorelabels.',
        earlier_evidence='Existingarrival_clock_source showedweakwithinairportcorrelation; thischecksconditionaltransfer understrictercompletedarrivalavailability.'))
    marker=common.read_json(CACHE/'manifest.json')
    audit=common.read_json(ROOT/'private_runs/breakthrough_20260916/missing/sequence_independent_audit/verification.json')
    assert audit['status']=='passed' and audit['manifest_sha256']==common.sha256(CACHE/'manifest.json')
    neighbors=np.load(CACHE/'neighbors.npy',mmap_mode='r')
    assert common.sha256(CACHE/'neighbors.npy')==marker['neighbors_sha256']
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    meta=pd.read_parquet(metadata,columns=[ID,TIME,'FLIGHT_ID_mvt','ADEP_mvt','proxy_sec'])
    meta[TARGET]=0.
    eligible=np.isfinite(meta.proxy_sec)&meta.proxy_sec.between(0,7200)
    convroot=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    assert common.sha256(convroot/'features.parquet')==common.read_json(convroot/'manifest.json')['feature_sha256']
    conv=pd.read_parquet(convroot/'features.parquet',columns=[ID,'conv_clock_rank_signature']).set_index(ID)
    selected=set()
    folds={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        fit=idx['fit'][eligible.iloc[idx['fit']].to_numpy()]
        tune=idx['tune'][eligible.iloc[idx['tune']].to_numpy()]
        months=pd.to_datetime(meta.iloc[fit][TIME],utc=True).dt.strftime('%Y-%m').to_numpy()
        sample=[]
        for i,month in enumerate(sorted(np.unique(months))):
            choices=fit[months==month]
            sample.extend(np.random.default_rng(20260916+i).choice(choices,min(20000,len(choices)),replace=False))
        fit=np.sort(sample)
        selected.update(fit.tolist()+tune.tolist())
        folds[fold]=(fit,tune,split)
    selected=np.array(sorted(selected))
    rawfiles=sorted(RAW.glob('training_*.parquet'))
    features=[]
    profiles=[]
    verified=0
    for i,part in enumerate(marker['records'][:10]):
        offset=part['query_offset']
        local=selected[(selected>=offset)&(selected<offset+part['query_rows'])]-offset
        for kind in ['events','queries']:
            assert common.sha256(CACHE/part[kind+'_file'])==part[kind+'_sha256']
        events=pd.read_parquet(CACHE/part['events_file'],dtype_backend='pyarrow')
        queries=pd.read_parquet(CACHE/part['queries_file'],dtype_backend='pyarrow')
        raw=[]
        for path in rawfiles[max(0,i-1):min(len(rawfiles),i+2)]:
            assert common.sha256(path)==marker['raw_hashes'][path.name]
            raw.append(pq.read_table(path,columns=[ID,'ARVT_3_flt'],filters=[('PHASE_mvt','=','ARR')],use_threads=False).to_pandas())
        mapping=pd.concat(raw).drop_duplicates(ID).set_index(ID).ARVT_3_flt
        clock=pd.to_datetime(events[ID].map(mapping),utc=True)
        nm=seconds(clock)
        movement=events.movement_ns.to_numpy('int64')/1e9
        arr=events.phase.eq('ARR').to_numpy()
        gap=np.where(arr,nm-movement,np.nan)
        known=gap[np.isfinite(gap)]
        profiles.append(dict(file=part['file'],known=len(known),median=float(np.median(known)),median_absolute=float(np.median(np.abs(known))),abs60_share=float(np.mean(np.abs(known)>60)),abs1800_share=float(np.mean(np.abs(known)>1800)),abs12h_share=float(np.mean(np.abs(known)>43200))))
        for start in range(0,len(local),4096):
            positions=local[start:start+4096]
            index=np.asarray(neighbors[offset+positions])
            index=np.where(index>=0,index-part['event_offset'],-1)
            safe=np.maximum(index,0)
            qt=queries.iloc[positions].time_ns.to_numpy('int64')[:,None]/1e9
            valid=(index>=0)&arr[safe]&np.isfinite(gap[safe])&(nm[safe]<qt)
            matrix=np.where(valid,gap[safe],np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                means=np.nanmean(matrix,axis=1)
                median=np.nanmedian(matrix,axis=1)
            count=valid.sum(axis=1)
            frame=pd.DataFrame({ID:queries.iloc[positions][ID].to_numpy(),'arrival_gap_mean':means,'arrival_gap_median':median,'arrival_gap_count':count})
            features.append(frame)
            if start==0:
                for q in range(min(10,len(positions))):
                    observed=[]
                    for eventidx in index[q]:
                        if eventidx<0 or not arr[eventidx] or pd.isna(clock.iloc[eventidx]):
                            continue
                        if clock.iloc[eventidx].value>=queries.iloc[positions[q]].time_ns:
                            continue
                        arrival=events.iloc[eventidx]
                        delta=(clock.iloc[eventidx]-pd.Timestamp(arrival.movement_ns,tz='UTC')).total_seconds()
                        observed.append(delta)
                    assert len(observed)==count[q]
                    np.testing.assert_allclose(means[q],np.mean(observed) if observed else np.nan,rtol=0,atol=1e-8,equal_nan=True)
                    verified+=1
        print('ARRIVAL_V2',profiles[-1],flush=True)
        del events,queries,raw,mapping,clock
        gc.collect()
    data=pd.concat(features).set_index(ID)
    reports={}
    for fold,(fitrows,tunerows,split) in folds.items():
        chosen=meta.iloc[np.r_[fitrows,tunerows]]
        values=data.loc[chosen[ID]].copy()
        values['airport']=chosen.ADEP_mvt.astype('string').fillna('<missing>').to_numpy()
        values['order']=conv.loc[chosen[ID],'conv_clock_rank_signature'].astype('string').fillna('<missing>').to_numpy()
        values['proxy_bin']=np.floor(chosen.proxy_sec.to_numpy()/300).astype(int)
        values['arrival_gap_bin']=pd.cut(values.arrival_gap_mean,old.EDGES).astype('string').fillna('missing')
        labels=pq.read_table(metadata,columns=[ID,TARGET],filters=[(TIME,'>=',pd.Timestamp(split['spec']['fit'][0],tz='UTC').to_pydatetime()),(TIME,'<',pd.Timestamp(split['spec']['tune'][1],tz='UTC').to_pydatetime())],use_threads=False).to_pandas().set_index(ID)
        values['residual']=labels.loc[chosen[ID],TARGET].to_numpy()-chosen.proxy_sec.to_numpy()
        fit,tune=values.iloc[:len(fitrows)],values.iloc[len(fitrows):]
        keys=['airport','order','proxy_bin']
        base=old.grouped(fit,tune,keys)
        leftovers=fit.copy()
        leftovers['residual']=fit.residual.to_numpy()-old.grouped(fit,fit,keys)
        control=base+old.grouped(leftovers,tune,['airport'],parent=0.)
        proposed=base+old.grouped(leftovers,tune,['airport','arrival_gap_bin'],parent=0.)
        y=tune.residual.to_numpy()
        a,b=(control-y)**2,(proposed-y)**2
        days=pd.to_datetime(chosen.iloc[len(fitrows):][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        daily=pd.DataFrame({'day':days,'gain':a-b}).groupby('day').gain.sum()
        saved=tune.copy()
        saved['airport_control_prediction']=control
        saved['arrival_context_prediction']=proposed
        saved.to_parquet(OUT/(fold+'_tune.parquet'))
        reports[fold]=dict(fit_n=len(fit),tune_n=len(tune),airport_control_rmse=float(np.sqrt(a.mean())),arrival_context_rmse=float(np.sqrt(b.mean())),
            improvement=float(np.sqrt(a.mean())-np.sqrt(b.mean())),days_improved=int((daily>0).sum()),days=len(daily),
            coverage=float(tune.arrival_gap_count.gt(0).mean()),arrival_gap_bin_counts=tune.arrival_gap_bin.value_counts().to_dict())
        print('VALID_TUNE',fold,reports[fold],flush=True)
    common.write_json(OUT/'diagnostic.json',dict(status='complete',source_sha256=common.sha256(__file__),folds=reports,arrival_profiles=profiles,
        independent_timestamp_oracle_queries=verified,unit_regression_passed=True,
        outputs={p.name:common.sha256(p) for p in OUT.glob('*.parquet')},
        caveat='Train/tuneconditionalmean screen only; no matchedLGBfit,newcache,scorelabels orGPU. Existinglanding-prior sourceaudit weak; thisstrictcompleted-arrivalvariantdoesnotprovepublicationtimes.'))


if __name__=='__main__':
    main()
