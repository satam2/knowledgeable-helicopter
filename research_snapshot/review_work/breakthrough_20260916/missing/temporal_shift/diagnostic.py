"""Original fit/tune-only clock-regime drift and fixed-window prior comparison."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/missing/temporal_shift'
CONV=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
KEYS=['airport','order','proxy_bin']
SHRINK=100.
BINS=[0,300,600,900,1200,1800,3600,7200]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def prior(reference, query):
    global_mean=float(reference.residual.mean())
    airport=reference.groupby('airport',observed=True).residual.agg(['sum','size'])
    airport['mean']=(airport['sum']+SHRINK*global_mean)/(airport['size']+SHRINK)
    groups=reference.groupby(KEYS,observed=True).residual.agg(['sum','size']).reset_index()
    groups['prior']=groups.airport.map(airport['mean']).fillna(global_mean)
    groups['correction']=(groups['sum']+SHRINK*groups.prior)/(groups['size']+SHRINK)
    merged=query[KEYS].merge(groups[KEYS+['correction','size']],on=KEYS,how='left',validate='many_to_one')
    fallback=query.airport.map(airport['mean']).fillna(global_mean).to_numpy()
    predicted=np.where(merged.correction.notna(),merged.correction.to_numpy(),fallback)
    return predicted,merged['size'].fillna(0).to_numpy(),groups


def score(y,p):
    error=np.asarray(p)-np.asarray(y)
    return dict(n=len(error),rmse=float(np.sqrt(np.mean(error**2))),bias=float(np.mean(error)),sse=float(np.sum(error**2)))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=dict(source_sha256=common.sha256(__file__),folds=['F1','F3'],score_labels_accessed=False,
        eligibility='Every original ordinary finite NM proxy 0..7200 fit/tune row; raw labels retained without clipping.',
        groups=KEYS,proxy_bins=BINS,shrink=SHRINK,recent_days=90,
        prior='Raw residual mean shrunk100 toward airport mean shrunk100 toward globalmean, separately for each referencewindow.',
        comparison='Expanding originalfit vs final90days before originaltune start; fixedwindow/shrink/bins, no grid.',
        purpose='Tune-only descriptive diagnosis; no score selection or fresh holdout claim.')
    common.write_json(OUT/'protocol.json',protocol)
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(path)==audit['artifacts'][path.name]
    meta=pq.read_table(path,columns=[ID,TIME,'FLIGHT_ID_mvt','ADEP_mvt','proxy_sec'],use_threads=False).to_pandas()
    meta[TARGET]=0.
    manifest=common.read_json(CONV/'manifest.json')
    assert common.sha256(CONV/'features.parquet')==manifest['feature_sha256']
    conv=pq.read_table(CONV/'features.parquet',columns=[ID,'conv_clock_rank_signature'],use_threads=False).to_pandas()
    np.testing.assert_array_equal(meta[ID],conv[ID])
    times=pd.to_datetime(meta[TIME],utc=True)
    proxy=meta.proxy_sec.to_numpy(float)
    ordinary=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    data=pd.DataFrame({ID:meta[ID], 'airport':meta.ADEP_mvt.astype('string').fillna('<missing>'),
        'order':conv.conv_clock_rank_signature.astype('string').fillna('<missing>'),
        'proxy_bin':pd.cut(proxy,BINS,include_lowest=True).astype('string').fillna('<missing>'),
        'proxy':proxy,'time':times,'month':times.dt.strftime('%Y-%m')})
    folds={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        fit_idx=idx['fit'][ordinary[idx['fit']]]
        tune_idx=idx['tune'][ordinary[idx['tune']]]
        allowed=np.r_[fit_idx,tune_idx]
        begin=pd.Timestamp(split['spec']['fit'][0],tz='UTC').to_pydatetime()
        end=pd.Timestamp(split['spec']['tune'][1],tz='UTC').to_pydatetime()
        labels=pq.read_table(path,columns=[ID,TARGET],filters=[(TIME,'>=',begin),(TIME,'<',end)],use_threads=False).to_pandas().set_index(ID)
        selected=data.iloc[allowed].copy()
        selected['residual']=labels.loc[selected[ID],TARGET].to_numpy()-selected.proxy.to_numpy()
        fit=selected.iloc[:len(fit_idx)].copy()
        tune=selected.iloc[len(fit_idx):].copy()
        assert fit.time.max()<tune.time.min()
        cutoff=pd.Timestamp(split['spec']['tune'][0],tz='UTC')-pd.Timedelta(days=90)
        recent=fit.loc[fit.time>=cutoff]
        assert len(recent)>0
        expanded,nexp,gexp=prior(fit,tune)
        recency,nrec,grec=prior(recent,tune)
        y=tune.residual.to_numpy()
        comparison=pd.DataFrame({ID:tune[ID].to_numpy(),'day':tune.time.dt.strftime('%Y-%m-%d').to_numpy(),
            'raw_residual_sec':y,'expanding_correction':expanded,'recent90_correction':recency,
            'expanding_group_n':nexp,'recent90_group_n':nrec})
        comparison.to_parquet(OUT/(fold+'_tune_predictions.parquet'),index=False)
        months=selected.groupby(['month',*KEYS],observed=True).residual.agg(n='size',mean='mean',std='std').reset_index()
        totals=months.groupby(['month','airport'],observed=True).n.transform('sum')
        months['airport_month_share']=months.n/totals
        months.to_parquet(OUT/(fold+'_monthly_regimes.parquet'),index=False)
        frequencies=selected.groupby(['month','airport','order'],observed=True).residual.agg(n='size',mean='mean').reset_index()
        frequencies['airport_month_share']=frequencies.n/frequencies.groupby(['month','airport'],observed=True).n.transform('sum')
        frequencies.to_parquet(OUT/(fold+'_monthly_order_frequencies.parquet'),index=False)
        a,b=(expanded-y)**2,(recency-y)**2
        day_rows=[]
        for day in sorted(comparison.day.unique()):
            choose=comparison.day.eq(day).to_numpy()
            day_rows.append(dict(day=day,n=int(choose.sum()),expanding_rmse=float(np.sqrt(a[choose].mean())),recent90_rmse=float(np.sqrt(b[choose].mean())),sse_gain=float(a[choose].sum()-b[choose].sum()),
                remove_day_delta=float(np.sqrt(b[~choose].mean())-np.sqrt(a[~choose].mean()))))
        diff=gexp[KEYS+['size','correction']].merge(grec[KEYS+['size','correction']],on=KEYS,suffixes=('_expanding','_recent90'),how='outer')
        diff['correction_change']=diff.correction_recent90-diff.correction_expanding
        diff.to_parquet(OUT/(fold+'_fit_prior_changes.parquet'),index=False)
        counts=tune.groupby(KEYS,observed=True).size().rename('tune_n').reset_index()
        drift=diff.merge(counts,on=KEYS,how='inner').dropna()
        supported=drift.loc[(drift.size_expanding>=100)&(drift.size_recent90>=100)]
        folds[fold]=dict(fit_rows=len(fit),recent90_rows=len(recent),tune_rows=len(tune),
            fit_id_hash=common.object_hash(fit[ID].tolist()),tune_id_hash=common.object_hash(tune[ID].tolist()),
            split=split,cutoff=str(cutoff),uncorrected=score(y,np.zeros(len(y))),
            expanding=score(y,expanded),recent90=score(y,recency),
            improvement=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
            days_improved=sum(d['sse_gain']>0 for d in day_rows),days=len(day_rows),day_results=day_rows,
            every_single_day_removal_improves=all(d['remove_day_delta']<0 for d in day_rows),
            recent90_unseen_groups=int((nrec==0).sum()),expanding_unseen_groups=int((nexp==0).sum()),
            supported_tune_rows=int(supported.tune_n.sum()),
            weighted_absolute_prior_change=float(np.average(np.abs(supported.correction_change),weights=supported.tune_n)),
            top_prior_changes=supported.assign(magnitude=lambda x:np.abs(x.correction_change)).sort_values('magnitude',ascending=False).head(15).to_dict('records'))
        print('TUNE',fold,folds[fold]['expanding'],folds[fold]['recent90'],folds[fold]['days_improved'],flush=True)
    result=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),folds=folds,
        labels='Only original fit and tune date intervals read per fold; score reference artifacts never opened.',
        diagnostic_only=True,outputs={p.name:common.sha256(p) for p in OUT.glob('*.parquet')})
    common.write_json(OUT/'diagnostic.json',result)


if __name__=='__main__':
    main()
