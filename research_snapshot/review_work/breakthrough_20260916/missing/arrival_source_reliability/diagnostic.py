"""Bounded fit/tune diagnostic of prior completed-arrival clock disagreement."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/missing/arrival_source_reliability'
CACHE=ROOT/'private_runs/breakthrough_20260916/sequence_context'
RAW=ROOT/'data/09-15-2026-18-55-03_files_list'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
EDGES=[-np.inf,-300,-60,-15,15,60,300,np.inf]


def grouped(fit,query,keys,shrink=50.,parent=None):
    base=float(fit.residual.mean()) if parent is None else parent
    groups=fit.groupby(keys,observed=True).residual.agg(['sum','size']).reset_index()
    matched=query[keys].merge(groups,on=keys,how='left',validate='many_to_one')
    count=matched['size'].fillna(0).to_numpy()
    return (matched['sum'].fillna(0).to_numpy()+shrink*base)/(count+shrink)


def main():
    OUT.mkdir(exist_ok=False)
    protocol=dict(source_sha256=common.sha256(__file__),folds=['F1','F3'],seed=20260916,
        fit_sample='Up to20000 originalordinaryfitrowspermonth, seeded, alloriginalordinarytunerows June/October.',
        information='ARVT3 minusairportlanding for alreadycompletedotherARRfromverifiedlast16cache; requireARVT3strictbeforequeryT.',
        fixed_features='Mean/median signedsourcegap,abs60tailshare,NMminuteroundedshare,support; fixedmean-gapbins[-inf,-300,-60,-15,15,60,300,inf].',
        test='Ownairport/clockorder/5minproxy-bin mean50shrink then fit-only additiveairport/arrivalgapbin correction50shrink; nohypergrid.',
        hidden_boundary='NoDEPBLOCK/TAXITIME rawreads; labelsonlyauditedoriginalfit/tuneintervals; no scorepredictions.',
        availability='Verifiedstrictcompletedarrivalcontext; NMfinalmetadata publicationunproven; sourceagreementnotphysicaltaxiin.')
    common.write_json(OUT/'protocol.json',protocol)
    marker=common.read_json(CACHE/'manifest.json')
    auditpath=ROOT/'private_runs/breakthrough_20260916/missing/sequence_independent_audit/verification.json'
    availability=common.read_json(auditpath)
    assert availability['status']=='passed' and availability['manifest_sha256']==common.sha256(CACHE/'manifest.json')
    neighbors=np.load(CACHE/'neighbors.npy',mmap_mode='r')
    assert common.sha256(CACHE/'neighbors.npy')==marker['neighbors_sha256']
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    meta=pd.read_parquet(meta_path,columns=[ID,TIME,'FLIGHT_ID_mvt','ADEP_mvt','proxy_sec'])
    meta[TARGET]=0.
    ordinary=np.isfinite(meta.proxy_sec)&meta.proxy_sec.between(0,7200)
    convroot=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    cm=common.read_json(convroot/'manifest.json')
    assert common.sha256(convroot/'features.parquet')==cm['feature_sha256']
    conv=pd.read_parquet(convroot/'features.parquet',columns=[ID,'conv_clock_rank_signature']).set_index(ID)
    requested=set()
    foldids={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        fit=idx['fit'][ordinary.iloc[idx['fit']].to_numpy()]
        tune=idx['tune'][ordinary.iloc[idx['tune']].to_numpy()]
        months=pd.to_datetime(meta.iloc[fit][TIME],utc=True).dt.strftime('%Y-%m').to_numpy()
        picks=[]
        for i,month in enumerate(sorted(np.unique(months))):
            options=fit[months==month]
            picks.extend(np.random.default_rng(20260916+i).choice(options,min(len(options),20000),replace=False).tolist())
        fit=np.sort(picks)
        requested.update(fit.tolist())
        requested.update(tune.tolist())
        foldids[fold]=(fit,tune,split)
    requested=np.array(sorted(requested))
    pieces=[]
    arrival_stats=[]
    rawfiles=sorted(RAW.glob('training_*.parquet'))
    for i,part in enumerate(marker['records'][:10]):
        offset=part['query_offset']
        positions=requested[(requested>=offset)&(requested<offset+part['query_rows'])]-offset
        if not len(positions):
            continue
        for kind in ['events','queries']:
            assert common.sha256(CACHE/part[kind+'_file'])==part[kind+'_sha256']
        events=pd.read_parquet(CACHE/part['events_file'],dtype_backend='pyarrow')
        queries=pd.read_parquet(CACHE/part['queries_file'],dtype_backend='pyarrow')
        arrivals=[]
        for path in rawfiles[max(0,i-1):min(len(rawfiles),i+2)]:
            assert common.sha256(path)==marker['raw_hashes'][path.name]
            arrivals.append(pq.read_table(path,columns=[ID,'ARVT_3_flt'],filters=[('PHASE_mvt','=','ARR')],use_threads=False).to_pandas())
        nm=pd.concat(arrivals).drop_duplicates(ID).set_index(ID).ARVT_3_flt
        clock=pd.to_datetime(events[ID].map(nm),utc=True)
        seconds=clock.astype('int64').to_numpy()/1e9
        seconds[clock.isna().to_numpy()]=np.nan
        landing=events.movement_ns.to_numpy('int64')/1e9
        gap=seconds-landing
        is_arr=events.phase.eq('ARR').to_numpy()
        gap[~is_arr]=np.nan
        known=gap[np.isfinite(gap)]
        arrival_stats.append(dict(file=part['file'],arrivals=int(is_arr.sum()),known_gap=len(known),median_gap=float(np.median(known)),
            median_abs_gap=float(np.median(np.abs(known))),abs60_share=float(np.mean(np.abs(known)>60)),
            abs1800_share=float(np.mean(np.abs(known)>1800)),abs12hour_share=float(np.mean(np.abs(known)>43200))))
        for start in range(0,len(positions),4096):
            local=positions[start:start+4096]
            index=np.asarray(neighbors[offset+local])
            index=np.where(index>=0,index-part['event_offset'],-1)
            safe=np.maximum(index,0)
            qt=queries.iloc[local].time_ns.to_numpy('int64')[:,None]/1e9
            valid=(index>=0)&is_arr[safe]&np.isfinite(gap[safe])&(seconds[safe]<qt)
            values=np.where(valid,gap[safe],np.nan)
            count=valid.sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                means=np.nanmean(values,axis=1)
                medians=np.nanmedian(values,axis=1)
            data=pd.DataFrame({ID:queries.iloc[local][ID].to_numpy(),'arrival_gap_mean':means,'arrival_gap_median':medians,
                'arrival_gap_count':count,'arrival_gap_abs60_share':np.divide((valid&(np.abs(gap[safe])>60)).sum(axis=1),count,out=np.full(len(local),np.nan),where=count>0),
                'arrival_nm_minute_share':np.divide((valid&(np.mod(seconds[safe],60)==0)).sum(axis=1),count,out=np.full(len(local),np.nan),where=count>0)})
            pieces.append(data)
        print('ARRIVAL_DIAGNOSTIC',part['file'],len(positions),arrival_stats[-1],flush=True)
        del events,queries,arrivals,nm,clock
        gc.collect()
    features=pd.concat(pieces,ignore_index=True).set_index(ID)
    assert features.index.is_unique
    reports={}
    for fold,(fitrows,tunerows,split) in foldids.items():
        rows=np.r_[fitrows,tunerows]
        chosen=meta.iloc[rows]
        data=features.loc[chosen[ID]].copy()
        data['airport']=chosen.ADEP_mvt.astype('string').fillna('<missing>').to_numpy()
        data['order']=conv.loc[chosen[ID],'conv_clock_rank_signature'].astype('string').fillna('<missing>').to_numpy()
        data['proxy_bin']=np.floor(chosen.proxy_sec.to_numpy()/300).astype(int)
        data['arrival_gap_bin']=pd.cut(data.arrival_gap_mean,EDGES).astype('string').fillna('missing')
        begin=pd.Timestamp(split['spec']['fit'][0],tz='UTC').to_pydatetime()
        end=pd.Timestamp(split['spec']['tune'][1],tz='UTC').to_pydatetime()
        labels=pq.read_table(meta_path,columns=[ID,TARGET],filters=[(TIME,'>=',begin),(TIME,'<',end)],use_threads=False).to_pandas().set_index(ID)
        data['residual']=labels.loc[chosen[ID],TARGET].to_numpy()-chosen.proxy_sec.to_numpy()
        fit=data.iloc[:len(fitrows)]
        tune=data.iloc[len(fitrows):]
        keys=['airport','order','proxy_bin']
        base=grouped(fit,tune,keys)
        fitbaseline=grouped(fit,fit,keys)
        leftovers=fit.copy()
        leftovers['residual']=fit.residual.to_numpy()-fitbaseline
        correction=grouped(leftovers,tune,['airport','arrival_gap_bin'],parent=0.)
        full=base+correction
        y=tune.residual.to_numpy()
        baseline_rmse=float(np.sqrt(np.mean((base-y)**2)))
        full_rmse=float(np.sqrt(np.mean((full-y)**2)))
        output=tune.copy()
        output['baseline_residual_prediction']=base
        output['arrival_context_residual_prediction']=full
        output.to_parquet(OUT/(fold+'_tune_diagnostic.parquet'))
        fit.groupby(['airport','arrival_gap_bin'],observed=True).residual.agg(['size','mean','std']).reset_index().to_parquet(OUT/(fold+'_fit_gap_groups.parquet'),index=False)
        tune.groupby(['airport','arrival_gap_bin'],observed=True).residual.agg(['size','mean','std']).reset_index().to_parquet(OUT/(fold+'_tune_gap_groups.parquet'),index=False)
        day=pd.to_datetime(chosen.iloc[len(fitrows):][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        a,b=(base-y)**2,(full-y)**2
        daily=pd.DataFrame({'day':day,'gain':a-b}).groupby('day').gain.sum()
        reports[fold]=dict(fit_n=len(fit),tune_n=len(tune),fit_ids_hash=common.object_hash(chosen.iloc[:len(fitrows)][ID].tolist()),
            tune_ids_hash=common.object_hash(chosen.iloc[len(fitrows):][ID].tolist()),baseline_rmse=baseline_rmse,arrival_context_rmse=full_rmse,
            gain=baseline_rmse-full_rmse,days_improved=int((daily>0).sum()),days=len(daily),
            tune_context_coverage=float(tune.arrival_gap_count.gt(0).mean()),
            median_context_gap_count=float(tune.arrival_gap_count.median()),
            correction_abs_mean=float(np.mean(np.abs(correction))),
            no_score_labels_read=True)
        print('TUNE_RESULT',fold,reports[fold],flush=True)
    common.write_json(OUT/'diagnostic.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),
        upstream_availability_sha256=common.sha256(auditpath),arrival_months=arrival_stats,folds=reports,
        limitation='Simpleconditionalmean screennotmatchedfullLGBcausaltest; sourcepublicationunknown; earlierstageownclockconventions alreadyinclude day/rounding fields.',
        outputs={p.name:common.sha256(p) for p in OUT.glob('*.parquet')}))


if __name__=='__main__':
    main()
