"""Calendar-date batch audit using observed peer clocks, never target rules."""
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
from taxiout.schema import PHASE,BLOCK
from taxiout.paths import external_path
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT=external_path(ROOT/'private_runs/breakthrough_20260916/missing/source_date_audit')


def day_number(series):
    values=pd.to_datetime(series,utc=True)
    result=values.dt.as_unit('ns').astype('int64').to_numpy(float)/86400e9
    result=np.floor(result)
    result[values.isna().to_numpy()]=np.nan
    return result


def majority(matrix):
    ordered=np.sort(np.where(np.isfinite(matrix),matrix,np.inf),axis=1)
    n=np.isfinite(matrix).sum(axis=1)
    best=np.zeros(len(matrix),int)
    mode=np.full(len(matrix),np.nan)
    length=np.zeros(len(matrix),int)
    previous=np.full(len(matrix),np.nan)
    tied=np.zeros(len(matrix),bool)
    for column in ordered.T:
        length=np.where(column==previous,length+1,1)
        valid=np.isfinite(column)
        new=valid&(length>best)
        same=valid&(length==best)&(column!=mode)
        tied=np.where(new,False,tied|same)
        mode=np.where(new,column,mode)
        best=np.where(new,length,best)
        previous=column
    mode[tied|(n==0)]=np.nan
    return mode,np.divide(best,n,out=np.zeros(len(matrix),float),where=n>0),n


def date_neighbors(queries,events,width):
    result=pd.DataFrame(index=queries.index,columns=['day','purity','support','bracketed','max_id_gap'],dtype=float)
    eventgroups=events.groupby(['airport','month'],observed=True).indices
    for key,positions in queries.groupby(['airport','month'],observed=True).indices.items():
        if key not in eventgroups:
            continue
        e=events.iloc[eventgroups[key]].dropna(subset=['source_day']).sort_values(ID)
        ids=e[ID].to_numpy(float)
        dates=e.source_day.to_numpy(float)
        q=queries.iloc[positions][ID].to_numpy(float)
        left=np.searchsorted(ids,q,side='left')
        right=np.searchsorted(ids,q,side='right')
        samples=[]
        gaps=[]
        for offset in range(width):
            for index in [left-offset-1,right+offset]:
                valid=(index>=0)&(index<len(ids))
                safe=np.clip(index,0,max(len(ids)-1,0))
                samples.append(np.where(valid,dates[safe] if len(ids) else np.nan,np.nan))
                gaps.append(np.where(valid,np.abs(ids[safe]-q) if len(ids) else np.nan,np.nan))
        matrix=np.column_stack(samples)
        day,purity,support=majority(matrix)
        bracket=(left>=width)&(len(ids)-right>=width)
        gap=np.nanmax(np.column_stack(gaps),axis=1) if len(ids) else np.full(len(q),np.nan)
        result.iloc[positions]=np.column_stack([day,purity,support,bracket.astype(float),gap])
    return result


def summarize(prediction,takeoff_day,hidden_day,mask):
    valid=mask&np.isfinite(prediction)
    changed=valid&(prediction!=takeoff_day)
    targetchanged=mask&(hidden_day!=takeoff_day)
    return dict(rows=int(mask.sum()),supported=int(valid.sum()),
        baseline_takeoff_correct=int((mask&(takeoff_day==hidden_day)).sum()),
        date_estimate_correct=int((valid&(prediction==hidden_day)).sum()),
        changed_date_n=int(changed.sum()),changed_date_correct=int((changed&(prediction==hidden_day)).sum()),
        changed_date_wrong=int((changed&(prediction!=hidden_day)).sum()),
        true_B_other_day_n=int(targetchanged.sum()),otherday_found_correct=int((targetchanged&valid&(prediction==hidden_day)).sum()),
        date_difference_counts={str(int(v)):int(n) for v,n in zip(*np.unique((prediction-takeoff_day)[valid],return_counts=True))})


def main():
    if (OUT/'audit.json').exists():
        raise ValueError('Completeddateaudit retained')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=utc_now(),script_sha256=sha256(__file__),
        hypothesis='MVT-ID neighbor calendar-date majority may reveal source/extraction-day batches despite weak minuteorder.',
        windows_each_side=[8,32],sources=['dep_nm_date','arr_inblock_date'],
        confidence_grid=dict(min_purity=[.75,.9,1.],max_id_gap=[64.,512.,None],bracket_required=True,min_support='2*width'),
        availability='Retrospective same reportingairport and actualUTCmovementmonth, pooledsourcepacks; ownMVT_IDexcluded. DEP source=AOBT only; ARR source=observed inblock.',
        labelboundary='ReadrawBLOCK onlywith Arrowfilter PHASE=ARR; neverload DEP BLOCK. HiddenB date laterderived from auditedT-Y for originalF1/F3fit/tune evaluation only.',
        selection='Explanatoryaudit only, no fitteddate correction or rulesfrom13scoretails; allrawnegative/longlabels kept.',
        output='Aggregatesonly, no largefeaturecache or model.')
    write_json(OUT/'protocol.json',protocol)
    frozen=read_json(ROOT/'private_runs/submission_v2/protocol.json')
    depframes=[]
    arrframes=[]
    for path in sorted(common.RAW.glob('training*.parquet'))[:10]:
        assert sha256(path)==frozen['raw_hashes'][path.name]
        cols=[ID,PHASE,MOVEMENT,'AOBT_3_flt','ADEP_mvt','ADES_mvt']
        observed=pq.read_table(path,columns=cols,use_threads=False).to_pandas()
        dep=observed.loc[observed[PHASE].eq('DEP')].copy()
        dep['airport']=dep.ADEP_mvt.astype(str)
        dep['month']=pd.to_datetime(dep[MOVEMENT],utc=True).dt.strftime('%Y-%m')
        dep['source_day']=day_number(dep.AOBT_3_flt)
        depframes.append(dep[[ID,'airport','month','source_day']])
        arrivals=pq.read_table(path,columns=[ID,PHASE,MOVEMENT,BLOCK,'ADES_mvt'],filters=[(PHASE,'=','ARR')],use_threads=False).to_pandas()
        assert arrivals[PHASE].eq('ARR').all()
        arrivals['airport']=arrivals.ADES_mvt.astype(str)
        arrivals['month']=pd.to_datetime(arrivals[MOVEMENT],utc=True).dt.strftime('%Y-%m')
        arrivals['source_day']=day_number(arrivals[BLOCK])
        arrframes.append(arrivals[[ID,'airport','month','source_day']])
    dep=pd.concat(depframes,ignore_index=True)
    arr=pd.concat(arrframes,ignore_index=True)
    audited=read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(path)==audited['artifacts'][path.name]
    meta=pd.read_parquet(path)
    queries=dep.copy()
    query_meta=meta.set_index(ID).loc[queries[ID]]
    takeoff=pd.to_datetime(query_meta[MOVEMENT],utc=True)
    takeoff_day=day_number(takeoff)
    target=query_meta[TARGET].to_numpy(float)
    hidden_day=day_number(takeoff-pd.to_timedelta(target,unit='s'))
    missing=~np.isfinite(query_meta.proxy_sec.to_numpy(float))
    reports={}
    sources={'dep_nm_date':dep,'arr_inblock_date':arr}
    for source,events in sources.items():
        for width in protocol['windows_each_side']:
            features=date_neighbors(queries,events,width)
            key=f'{source}_w{width}'
            reports[key]=dict(coverage=dict(rows=len(features),date_available=int(features.day.notna().sum()),
                fully_bracketed=int(features.bracketed.eq(1).sum()),
                purity_quantiles=features.purity.dropna().quantile([0,.1,.5,.9,1]).tolist()),folds={})
            for fold in ['F1','F3']:
                idx,split,_=common.fold_data(meta,fold,full=True)
                stages={}
                for stage in ['fit','tune']:
                    stage_mask=np.isin(queries[ID],meta.iloc[idx[stage]][ID])
                    cohorts={'all':stage_mask,'missing':stage_mask&missing,
                        'missing_over12h':stage_mask&missing&(target>43200),
                        'missing_day_plus_2h':stage_mask&missing&(target>=86400)&(target<=93600)}
                    predictions={'unfiltered':features.day.to_numpy(float)}
                    for purity in [.75,.9,1.]:
                        for gap in [64.,512.,np.inf]:
                            use=features.purity.ge(purity)&features.support.ge(2*width)&features.bracketed.eq(1)&features.max_id_gap.le(gap)
                            predictions[f'purity{purity}_gap{gap}']=np.where(use,features.day.to_numpy(float),np.nan)
                    stages[stage]={name:{cohort:summarize(pred,takeoff_day,hidden_day,mask) for cohort,mask in cohorts.items()} for name,pred in predictions.items()}
                reports[key]['folds'][fold]=stages
            print('DATE_AUDIT',key,reports[key]['coverage'],flush=True)
    write_json(OUT/'audit.json',dict(created_utc=utc_now(),protocol_sha256=sha256(OUT/'protocol.json'),
        source_hashes={str(Path(__file__).relative_to(ROOT)):sha256(__file__)},reports=reports,
        score_labels_used=False,model_trained=False,feature_cache_written=False))


if __name__=='__main__':
    main()
