"""NM flight identifier ordering feasibility, using observations before tune labels."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from common import ID, TARGET, MOVEMENT, read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import FLIGHT_ID, PHASE
from taxiout.paths import external_path
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT/'private_runs/breakthrough_20260916/missing/nm_id_context')
CLOCKS = ['AOBT_3_flt','EOBT_1_flt','IOBT_flt','ARVT_3_flt']
COLS = [ID,FLIGHT_ID,PHASE,MOVEMENT,'SCHED_TIME_UTC_mvt','ADEP_mvt','ADES_mvt',*CLOCKS]
BINS = [-np.inf,-86400,-21600,-3600,-900,900,3600,21600,86400,np.inf]


def seconds(series):
    parsed = pd.to_datetime(series,utc=True)
    result = parsed.dt.as_unit('ns').astype('int64').to_numpy(float)/1e9
    result[parsed.isna().to_numpy()] = np.nan
    return result


def stats(values):
    values = np.asarray(values,float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'n':0}
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
        mae=float(np.abs(values).mean()),rmse=float(np.sqrt(np.mean(values**2))),
        abs_q90=float(np.quantile(np.abs(values),.9)),abs_q99=float(np.quantile(np.abs(values),.99)),
        within60_pct=float(np.mean(np.abs(values)<=60)*100))


def rho(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[valid],b[valid]).statistic) if valid.sum()>5 and np.std(a[valid])>0 and np.std(b[valid])>0 else None


def neighbors(queries,context,width=8):
    # One row per distinct NM flight excludes every same-journey ARR/DEP duplicate.
    known=context.loc[context[FLIGHT_ID].notna()].copy()
    known['nm']=seconds(known.AOBT_3_flt)
    known['initial']=seconds(known.EOBT_1_flt)
    known['movement']=seconds(known[MOVEMENT])
    aggregate=known.groupby(FLIGHT_ID,sort=True)[['nm','initial','movement']].median()
    ids=aggregate.index.to_numpy(float)
    qid=queries[FLIGHT_ID].to_numpy(float)
    left=np.searchsorted(ids,qid,side='left')
    right=np.searchsorted(ids,qid,side='right')
    columns={}
    for name in aggregate:
        source=aggregate[name].to_numpy(float)
        samples=[]
        for shift in range(width):
            for positions in [left-shift-1,right+shift]:
                valid=np.isfinite(qid)&(positions>=0)&(positions<len(ids))
                safe=np.clip(positions,0,max(0,len(ids)-1))
                samples.append(np.where(valid,source[safe] if len(source) else np.nan,np.nan))
        matrix=np.column_stack(samples)
        count=np.isfinite(matrix).sum(axis=1)
        clean=np.where(np.isfinite(matrix),matrix,np.inf)
        ordered=np.sort(clean,axis=1)
        center=(ordered[np.arange(len(count)),np.maximum((count-1)//2,0)]+ordered[np.arange(len(count)),np.maximum(count//2,0)])/2
        center[count==0]=np.nan
        columns['nmid_peer_'+name+'_seconds']=center
        columns['nmid_peer_'+name+'_count']=count.astype(float)
    return pd.DataFrame(columns,index=queries.index)


def shared_flight(data):
    known=data.loc[data[FLIGHT_ID].notna()].copy()
    kinds=known.groupby(FLIGHT_ID)[PHASE].agg(lambda values:','.join(sorted(set(values))))
    shared=kinds.index[kinds.str.contains('ARR')&kinds.str.contains('DEP')]
    pairs=known.loc[known[FLIGHT_ID].isin(shared)]
    report=dict(shared_nm_flights=len(shared),shared_rows=len(pairs))
    for column in CLOCKS:
        values=pairs.assign(value=seconds(pairs[column])).groupby(FLIGHT_ID).value.agg(['min','max','count'])
        spread=values['max']-values['min']
        report[column]=dict(stats(spread),different_positive=int((spread>0).sum()),complete_repeated=int(values['count'].ge(2).sum()))
    return report


def summarize_month(data):
    dep=data.loc[data[PHASE].eq('DEP')].copy()
    peers=neighbors(dep,data)
    nm=seconds(dep.AOBT_3_flt)
    takeoff=seconds(dep[MOVEMENT])
    sched=seconds(dep.SCHED_TIME_UTC_mvt)
    flight=dep[FLIGHT_ID].to_numpy(float)
    proxy=takeoff-nm
    features=pd.DataFrame({ID:dep[ID].to_numpy(),
        'nmid_peer_nm_minus_own_nm':peers.nmid_peer_nm_seconds.to_numpy()-nm,
        'nmid_peer_initial_minus_own_nm':peers.nmid_peer_initial_seconds.to_numpy()-nm,
        'nmid_peer_nm_minus_takeoff':peers.nmid_peer_nm_seconds.to_numpy()-takeoff,
        'nmid_present':np.isfinite(flight).astype(float),
        'nm_peer_count':peers.nmid_peer_nm_count.to_numpy()})
    report=dict(rows=len(dep),known_flight_id=int(np.isfinite(flight).sum()),
        missing_nm=int((~np.isfinite(nm)).sum()),missing_nm_with_id=int((~np.isfinite(nm)&np.isfinite(flight)).sum()),
        id_integer_exact=bool(np.equal(flight[np.isfinite(flight)],np.floor(flight[np.isfinite(flight)])).all()),
        nm_id_min=float(np.nanmin(flight)),nm_id_max=float(np.nanmax(flight)),
        id_time_spearman={name:rho(flight,value) for name,value in [('takeoff',takeoff),('nm',nm),('schedule',sched)]},
        peer_nm_minus_own_nm=stats(features.nmid_peer_nm_minus_own_nm),
        peer_initial_minus_own_nm=stats(features.nmid_peer_initial_minus_own_nm),
        peer_nm_minus_takeoff=stats(features.nmid_peer_nm_minus_takeoff),
        shared_arr_dep=shared_flight(data),airports=[])
    for airport,positions in dep.groupby('ADEP_mvt',observed=True).indices.items():
        report['airports'].append(dict(airport=str(airport),rows=len(positions),
            id_nm_spearman=rho(flight[positions],nm[positions]),
            peer_nm_minus_own_nm=stats(features.iloc[positions].nmid_peer_nm_minus_own_nm)))
    return features,report


def grouped_predict(fit,tune,keys):
    airport=fit.groupby('airport',observed=True).residual.agg(['mean','size'])
    fallback=tune[['airport']].merge(airport,left_on='airport',right_index=True,how='left')['mean'].fillna(fit.residual.mean()).to_numpy()
    if keys==['airport']:
        return fallback
    table=fit.groupby(keys,observed=True,dropna=False).residual.agg(['mean','size']).reset_index()
    values=tune[keys].merge(table,on=keys,how='left',validate='many_to_one')
    count=values['size'].fillna(0).to_numpy()
    weight=count/(count+50.)
    return weight*values['mean'].fillna(0).to_numpy()+(1-weight)*fallback


def tune_feasibility(features,meta):
    np.testing.assert_array_equal(features[ID],meta[ID])
    proxy=meta.proxy_sec.to_numpy(float)
    context=features.copy()
    context['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    context['residual']=meta[TARGET].to_numpy(float)-proxy
    context['peer_bin']=pd.cut(context.nmid_peer_nm_minus_own_nm,BINS,labels=False).fillna(-1).astype(int)
    context['initial_peer_bin']=pd.cut(context.nmid_peer_initial_minus_own_nm,BINS,labels=False).fillna(-1).astype(int)
    reports={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        reports[fold]=dict(split=split,scoring='Original tune only; no score-label-driven rules',cohorts={})
        for name,mask in [('all_finite',np.isfinite(proxy)),('ordinary',np.isfinite(proxy)&(proxy>=0)&(proxy<=7200))]:
            fitrows=idx['fit'][mask[idx['fit']]]
            tunerows=idx['tune'][mask[idx['tune']]]
            fit=context.iloc[fitrows]
            tune=context.iloc[tunerows]
            models={}
            for method,keys in [('airport',['airport']),('peer_nm',['airport','peer_bin']),('peer_nm_initial',['airport','peer_bin','initial_peer_bin'])]:
                prediction=proxy[tunerows]+grouped_predict(fit,tune,keys)
                models[method]=stats(prediction-meta.iloc[tunerows][TARGET].to_numpy(float))
            reports[fold]['cohorts'][name]=dict(fit_rows=len(fitrows),tune_rows=len(tunerows),
                fit_id_hash=object_hash(meta.iloc[fitrows][ID].tolist()),tune_id_hash=object_hash(meta.iloc[tunerows][ID].tolist()),
                predictions=models,association={column:rho(tune[column],tune.residual) for column in ['nmid_peer_nm_minus_own_nm','nmid_peer_initial_minus_own_nm']})
    return reports


def main():
    if (OUT/'audit.json').exists():
        raise ValueError('Completed NMID audit preserved')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=read_json(ROOT/'private_runs/submission_v2/protocol.json')
    protocol=dict(created_utc=utc_now(),script_sha256=sha256(__file__),width_each_side=8,
        source='FLIGHT_ID_mvt NM flight identifier, distinct from airport MVT_ID',
        hypothesis='Empirical numeric order may reveal source record date/reliability; official semantics only unique flight ID.',
        features='Allairports within supplied month; median16 adjacent distinct NMflightIDs, excludes ALL same NMflight context rows.',
        availability='RETROSPECTIVE final supplied month; numeric ID neighbors may be temporally future and are not causal.',
        duplicate='One context row per NMflight, median source clocks across ARR/DEP; sameflight clock agreement separately audited.',
        targets='No target/BLOCK rawcolumns loaded; only original F1/F3 fit/tune labels accessed from audited metadata after features built.',
        rules='Fixed peerclock minus ownNM bins at +/-15min,1h,6h,24h; airportmean shrinkage50. No score-derived rules or clipping.',
        raw_columns=COLS,raw_hashes=frozen['raw_hashes'])
    write_json(OUT/'protocol.json',protocol)
    started=time.monotonic()
    frames=[]
    months=[]
    for path in sorted(common.RAW.glob('training*.parquet')):
        assert sha256(path)==frozen['raw_hashes'][path.name]
        raw=pq.read_table(path,columns=COLS,use_threads=False).to_pandas()
        frame,report=summarize_month(raw)
        frames.append(frame)
        months.append(dict(file=path.name,**report))
        print('NMID',path.name,report['id_time_spearman'],report['peer_nm_minus_own_nm'],flush=True)
    features=pd.concat(frames,ignore_index=True)
    audit=read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(path)==audit['artifacts'][path.name]
    meta=pd.read_parquet(path)
    features=features.set_index(ID).loc[meta[ID]].reset_index()
    result=dict(created_utc=utc_now(),protocol_sha256=sha256(OUT/'protocol.json'),months=months,
        tune_feasibility=tune_feasibility(features,meta),runtime_sec=time.monotonic()-started,
        feature_cache_written=False,model_trained=False,score_labels_used=False)
    write_json(OUT/'audit.json',result)
    print('DONE',OUT,flush=True)


if __name__=='__main__':
    main()
