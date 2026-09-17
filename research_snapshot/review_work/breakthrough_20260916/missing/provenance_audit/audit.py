"""Schema, proxy plateaus and strictly-prior source-distribution signatures."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from common import ID,TARGET,MOVEMENT,read_json,write_json,sha256,object_hash,utc_now
from taxiout.paths import external_path
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT=external_path(ROOT/'private_runs/breakthrough_20260916/missing/provenance_audit')


def stats(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
        rmse=float(np.sqrt(np.mean(values**2))),mae=float(np.abs(values).mean()),
        q10=float(np.quantile(values,.1)),q90=float(np.quantile(values,.9))) if len(values) else dict(n=0)


def prior_matching_fraction(meta,width=3600):
    p=meta.proxy_sec.to_numpy(float)
    valid=np.isfinite(p)&(p>=0)&(p<=7200)
    bins=np.floor((np.where(valid,p,0)+30)/60).astype(int)
    times=pd.to_datetime(meta[MOVEMENT],utc=True)
    sec=times.dt.as_unit('ns').astype('int64').to_numpy()/1e9
    groups=pd.DataFrame({'airport':meta.ADEP_mvt.astype(str),'month':times.dt.strftime('%Y-%m')})
    count=np.zeros(len(meta),float)
    match=np.zeros(len(meta),float)
    mode_share=np.zeros(len(meta),float)
    for _,positions in groups.groupby(['airport','month'],observed=True).indices.items():
        positions=positions[np.argsort(sec[positions],kind='stable')]
        query=sec[positions]
        events=positions[valid[positions]]
        et=sec[events]
        left=np.searchsorted(et,query-width,side='left')
        right=np.searchsorted(et,query,side='left')
        count[positions]=right-left
        # Fixed finite proxyminute bins are observation summaries, not target bins.
        maxcount=np.zeros(len(positions))
        for minute in np.unique(bins[events]):
            bt=sec[events[bins[events]==minute]]
            amount=np.searchsorted(bt,query,side='left')-np.searchsorted(bt,query-width,side='left')
            use=valid[positions]&(bins[positions]==minute)
            match[positions[use]]=amount[use]
            maxcount=np.maximum(maxcount,amount)
        mode_share[positions]=np.divide(maxcount,count[positions],out=np.zeros(len(positions)),where=count[positions]>0)
    fraction=np.divide(match,count,out=np.zeros(len(match)),where=count>0)
    return pd.DataFrame({ID:meta[ID].to_numpy(),'prior_count':count,'ownminute_match_fraction':fraction,
        'prior_modal_fraction':mode_share,'proxy_minute':np.where(valid,bins,-1)})


def grouped_prediction(fit,query,keys):
    airport=fit.groupby('airport',observed=True).residual.mean()
    fallback=query.airport.map(airport).fillna(fit.residual.mean()).to_numpy(float)
    table=fit.groupby(keys,observed=True,dropna=False).residual.agg(['mean','size']).reset_index()
    joined=query[keys].merge(table,on=keys,how='left',validate='many_to_one')
    n=joined['size'].fillna(0).to_numpy()
    return n/(n+50)*joined['mean'].fillna(0).to_numpy()+(50/(n+50))*fallback


def correction(fit,query,keys,fit_prediction):
    residual=fit.residual.to_numpy()-fit_prediction
    data=fit[keys].copy()
    data['residual']=residual
    table=data.groupby(keys,observed=True,dropna=False).residual.agg(['mean','size']).reset_index()
    joined=query[keys].merge(table,on=keys,how='left',validate='many_to_one')
    n=joined['size'].fillna(0).to_numpy()
    return n/(n+50)*joined['mean'].fillna(0).to_numpy()


def main():
    if (OUT/'audit.json').exists():
        raise ValueError('Completedprovenanceaudit retained')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=utc_now(),script_sha256=sha256(__file__),
        objective='Inspectactualschema provenanceavailability, standardtaxiplateau hypotheses and priorwindowdistribution beyondownclockconventions.',
        source_url='https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf',
        source_limits='GenericADRR fallback possibility, exactchallengepipeline andperrecordindicator unproven; no suppliedNMtakeoff tovalidate exactconstant subtraction.',
        label_scope='OriginalF1/F3fit/tune only; rawYunchanged; no scorelabel rules.',
        new_context='Strictpriordeparturetakeoff window60min, sameairport/UTCmovementmonth, priorfiniteproxy0..7200 minutehistogram; ownminute fraction andmodalconcentration.',
        availability='Prior completedmovement timestamps but finalNMfields; publicationavailability isnotproven. Doesnotlookforward in suppliedtakeofftime.',
        frequency_rule='Airport fitproxy rounded60seconds; localpeak ratio count/(meanofadjacent4bins+1), candidateplateau count>=200,share>=.01,ratio>=1.5.',
        context_bins='match/modal fractions bins0,.05,.1,.2,.4,.6,.8,1; supportbins0,5,15,30,60,120,infinity.',
        matched_control='Fit-only rawresidualtable ownairport+order+gap+proxyminute, shrink50; additional ownplanningclock13bins then contextadditive residualmeans.',
        no_model_or_cache=True)
    write_json(OUT/'protocol.json',protocol)
    schemas=[]
    for path in sorted(common.RAW.glob('*.parquet')):
        file=pq.ParquetFile(path)
        schema=file.schema_arrow
        schemas.append(dict(file=path.name,rows=file.metadata.num_rows,rowgroups=file.metadata.num_row_groups,
            columns=[dict(name=f.name,type=str(f.type),metadata={k.decode():v.decode(errors='replace') for k,v in (f.metadata or {}).items()}) for f in schema],
            schema_metadata_keys=[key.decode() for key in (file.metadata.metadata or {})]))
    x,meta=common.load_data()
    temporal=prior_matching_fraction(meta)
    directory=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    marker=read_json(directory/'manifest.json')
    assert sha256(directory/'features.parquet')==marker['feature_sha256']
    names=['conv_clock_rank_signature','conv_nm_est_gap_bucket','conv_nm_second','conv_nm_minute_aligned']
    conv=pd.read_parquet(directory/'features.parquet',columns=[ID,*names]).set_index(ID)
    np.testing.assert_array_equal(conv.index,meta[ID])
    data=temporal.copy()
    for col in names:
        data[col]=conv[col].to_numpy()
    proxy=meta.proxy_sec.to_numpy(float)
    data['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    data['operator']=x.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy()
    data['residual']=meta[TARGET].to_numpy(float)-proxy
    data['y']=meta[TARGET].to_numpy(float)
    data['proxy']=proxy
    edges=[-np.inf,-86400,-21600,-7200,-3600,-1800,-300,300,1800,3600,7200,21600,86400,np.inf]
    for name,clock in [('initial','IOBT_flt'),('estimated','EOBT_1_flt'),('last','LOBT_flt'),('schedule','SCHED_TIME_UTC_mvt')]:
        value=x['takeoff_minus_'+clock].to_numpy(float)
        delta=proxy-value
        delta[(value==-999999)|~np.isfinite(value)]=np.nan
        data[name+'_gap']=pd.cut(delta,edges,labels=False)
    fractionedges=[-np.inf,0,.05,.1,.2,.4,.6,.8,np.inf]
    data['match_bin']=pd.cut(data.ownminute_match_fraction,fractionedges,labels=False)
    data['mode_bin']=pd.cut(data.prior_modal_fraction,fractionedges,labels=False)
    data['support_bin']=pd.cut(data.prior_count,[-np.inf,0,5,15,30,60,120,np.inf],labels=False)
    data=data.fillna(-1)
    reports={}
    ordinary=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        a=idx['fit'][ordinary[idx['fit']]]
        b=idx['tune'][ordinary[idx['tune']]]
        fit,tune=data.iloc[a],data.iloc[b]
        peaks=[]
        for airport,group in fit.groupby('airport'):
            counts=group.proxy_minute.value_counts().to_dict()
            totals=len(group)
            for minute,count in sorted(counts.items(),key=lambda pair:pair[1],reverse=True)[:10]:
                neighbors=np.mean([counts.get(minute+offset,0) for offset in [-2,-1,1,2]])
                ratio=count/(neighbors+1.)
                atom=bool(count>=200 and count/totals>=.01 and ratio>=1.5)
                fitgroup=group.loc[group.proxy_minute.eq(minute)]
                tunegroup=tune.loc[tune.airport.eq(airport)&tune.proxy_minute.eq(minute)]
                peaks.append(dict(airport=airport,proxy_minute=int(minute),fit_count=int(count),fit_share=count/totals,
                    local_peak_ratio=float(ratio),candidate_atom=atom,
                    fit_nm_minute_aligned_pct=float(fitgroup.conv_nm_minute_aligned.mean()*100),
                    fit_residual=stats(fitgroup.residual),fit_actualY=stats(fitgroup.y),
                    tune_count=len(tunegroup),tune_share=len(tunegroup)/max(1,int(tune.airport.eq(airport).sum())),
                    tune_residual=stats(tunegroup.residual)))
        keys=['airport','conv_clock_rank_signature','conv_nm_est_gap_bucket','proxy_minute']
        f=grouped_prediction(fit,fit,keys)
        t=grouped_prediction(fit,tune,keys)
        metrics={'own_proxyminute_order_gap':stats(t-tune.residual.to_numpy())}
        for column in ['initial_gap','estimated_gap','last_gap','schedule_gap','conv_nm_second','operator']:
            cf=correction(fit,fit,['airport',column],f)
            ct=correction(fit,tune,['airport',column],f)
            f+=cf
            t+=ct
            metrics['after_'+column]=stats(t-tune.residual.to_numpy())
        for label,keys in [('context_match',['airport','match_bin','support_bin']),
            ('context_mode',['airport','mode_bin','support_bin'])]:
            cf=correction(fit,fit,keys,f)
            ct=correction(fit,tune,keys,f)
            f+=cf
            t+=ct
            metrics['after_'+label]=stats(t-tune.residual.to_numpy())
        reports[fold]=dict(split=split,fit_rows=len(a),tune_rows=len(b),peaks=peaks,tune_metrics=metrics,
            fit_context=stats(fit.prior_count),tune_context=stats(tune.prior_count))
        print('PROVENANCE',fold,{k:v['rmse'] for k,v in metrics.items()},flush=True)
    write_json(OUT/'audit.json',dict(created_utc=utc_now(),protocol_sha256=sha256(OUT/'protocol.json'),
        schemas=schemas,reports=reports,source_conventions_manifest_sha256=sha256(directory/'manifest.json'),
        model_trained=False,feature_cache_written=False,score_labels_used=False))


if __name__=='__main__':
    main()
