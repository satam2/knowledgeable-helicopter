"""Frozen fit-only source-transfer support audit; no predictive fitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT, ID, TARGET, TIME = v.ROOT, v.ID, v.TARGET, v.common.MOVEMENT
OUT = ROOT / 'private_runs/tail240_20260916/validation/normalized_transfer_support_v1'
REPORT = ROOT / 'output/tail240_20260916/validation/NORMALIZED_TRANSFER_SUPPORT.md'
CACHE = ROOT / 'private_runs/breakthrough_20260916/missing/domain_adaptation'
META = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
META_COLS = [ID, 'FLIGHT_ID_mvt', TIME, 'proxy_sec']
FEATURE_COLS = [ID, 'airport', 'schedule_bucket', 'schedule_proxy_sec', 'stand_missing', 'flight_prefix']
BASE_KEYS = ['airport', 'schedule_bucket', 'schedule_day', 'schedule_missing']
KEY_SETS = {'base': BASE_KEYS, 'stand_missing': BASE_KEYS + ['stand_missing'], 'flight_prefix': BASE_KEYS + ['flight_prefix']}
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    m = psutil.Process().memory_info()
    peak = max(m.rss, getattr(m, 'peak_wset', 0))
    assert peak < 2 * 1024**3, ('2GiB cap', peak)
    assert psutil.virtual_memory().available >= 8 * 1024**3, '8GiB host reserve'
    return peak


def declare():
    record = dict(source_sha256=v.sha256(__file__), cache_manifest_sha256=v.sha256(CACHE/'features_manifest.json'), metadata_sha256=v.sha256(META), metadata_columns=META_COLS, feature_columns=FEATURE_COLS,
        stages='Original purged F1 and F3 fit only. Metadata labels never loaded unfiltered; each fold label projection uses exact fit-ID predicate. F3 fit legitimately contains earlier June/July, but never treats those as tune/score observations in this audit.',
        groups=KEY_SETS, day='Signed floor(schedule_proxy_sec/86400), invalid/sentinel -> 0 plus separate missing flag. Existing schedule bucket unchanged. No learned bins.',
        support='For each missing fit row, known-source same-cell rows strictly before query UTC day and excluding every nonnull identical FLIGHT_ID. Report rows, independent days, and raw target>=7200/86400 support; no future/same-day rows.',
        means='Descriptive raw and normalized source group means, quantiles, tails and independent days. Mean differences use same-day cluster contributions with 1.96 normal approximation, not an exchangeability guarantee; few-day cells remain uncertain.',
        scale='sqrt(3600^2+(schedule_proxy-900)^2), invalid/sentinel=>3600. Unnormalized weight=s^2; ESS and relative masses invariant to fit normalization.',
        resources='1CPU,2GiB process current/historical peak,8GiB host reserve,start10GiB; noGPU/model/acquisition.',
        selection='No fit/tune model, no prediction/score/ranking reads, no outlier removal, no new thresholds/transfer ratios selected; all outputs descriptive.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert v.read_json(path) == record
    else:
        v.write_json(path, record)
    return record


def summary(g):
    y, z, w = g[TARGET].to_numpy(), g.z.to_numpy(), g.weight.to_numpy()
    day_weights = g.groupby('day').weight.sum()
    return dict(n=len(g), days=int(g.day.nunique()), raw_mean=float(y.mean()), z_mean=float(z.mean()), raw_quantiles=np.quantile(y, [0,.1,.5,.9,.99,1]).tolist(), z_quantiles=np.quantile(z, [0,.1,.5,.9,.99,1]).tolist(), ge7200=int((y>=7200).sum()), ge86400=int((y>=86400).sum()), ge7200_days=int(g.loc[g[TARGET].ge(7200),'day'].nunique()), ge86400_days=int(g.loc[g[TARGET].ge(86400),'day'].nunique()), weight_sum=float(w.sum()), max_weight=float(w.max()), ess=float(w.sum()**2/(w@w)), max_day_weight_fraction=float(day_weights.max()/w.sum()))


def mean_difference(g, column):
    missing, known = g.loc[g.missing], g.loc[~g.missing]
    if len(missing)==0 or len(known)==0:
        return None
    diff = float(missing[column].mean()-known[column].mean())
    # Shared UTC-day clusters preserve contemporaneous correlation across sources.
    terms=[]
    for part, sign in [(missing,1), (known,-1)]:
        s = part.groupby('day')[column].agg(['sum','count'])
        terms.append(sign*(s['sum']-s['count']*part[column].mean())/len(part))
    influence=terms[0].add(terms[1], fill_value=0)
    days=len(influence)
    se=float(np.sqrt((influence.to_numpy()**2).sum()*days/(days-1))) if days>1 else None
    return dict(missing_minus_known=diff, cluster_days=days, missing_days=int(missing.day.nunique()), known_days=int(known.day.nunique()), approximate_day_cluster_95=[diff-1.96*se,diff+1.96*se] if se is not None else None, limitation='Descriptive normal approximation; rare tails/few missing days can make this interval unreliable. No transportability claim.')


def prior_support(frame, keys):
    missing = frame.loc[frame.missing]
    known = frame.loc[~frame.missing]
    query_groups = missing.groupby(keys, observed=True, dropna=False)
    known_groups = known.groupby(keys, observed=True, dropna=False).indices
    records=[]
    for key, query in query_groups:
        positions=known_groups.get(key)
        if positions is None:
            for row in query.itertuples(index=False):
                records.append({ID:getattr(row,ID), 'known_prior_rows':0,'known_prior_days':0,'known_prior_ge7200':0,'known_prior_ge86400':0})
            continue
        history=known.iloc[positions]
        daily=history.groupby('day').agg(n=(ID,'size'),ge7200=('ge7200','sum'),ge86400=('ge86400','sum'))
        dates=daily.index.to_numpy()
        totals=daily.to_numpy(dtype=np.int64)
        query_flights=set(query.FLIGHT_ID_mvt.dropna().tolist())
        related=history.loc[history.FLIGHT_ID_mvt.isin(query_flights)]
        flight_counts={flight:part.groupby('day').agg(n=(ID,'size'),ge7200=('ge7200','sum'),ge86400=('ge86400','sum')).reindex(daily.index,fill_value=0).to_numpy(dtype=np.int64) for flight,part in related.groupby('FLIGHT_ID_mvt',observed=True)}
        for row in query.itertuples(index=False):
            stop=int(np.searchsorted(dates,row.day,side='left'))
            values=totals[:stop]
            same=flight_counts.get(row.FLIGHT_ID_mvt)
            if same is not None:
                values=values-same[:stop]
            assert (values>=0).all()
            sums=values.sum(axis=0)
            records.append({ID:getattr(row,ID),'known_prior_rows':int(sums[0]),'known_prior_days':int((values[:,0]>0).sum()),'known_prior_ge7200':int(sums[1]),'known_prior_ge86400':int(sums[2])})
        guard()
    result=pd.DataFrame(records).set_index(ID).loc[missing[ID]].reset_index()
    np.testing.assert_array_equal(result[ID],missing[ID])
    return result


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--declare-only',action='store_true')
    args=args.parse_args()
    protocol=declare()
    if args.declare_only:
        print('DECLARED',v.sha256(OUT/'protocol.json'),flush=True)
        return
    assert not (OUT/'summary.json').exists()
    assert psutil.virtual_memory().available>=10*1024**3
    marker=v.read_json(CACHE/'features_manifest.json')
    assert v.sha256(CACHE/'features.parquet')==marker['feature_sha256']
    assert v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][META.name]==protocol['metadata_sha256']
    meta=pd.read_parquet(META,columns=META_COLS)
    meta[TARGET]=np.nan
    cache=pd.read_parquet(CACHE/'features.parquet',columns=FEATURE_COLS)
    np.testing.assert_array_equal(cache[ID],meta[ID])
    records={}
    for fold in ['F1','F3']:
        idx,split,_=v.common.fold_data(meta,fold,full=True)
        prior=v.read_json(CACHE/f'models/{fold}_s20260916/manifest.json')
        assert len(idx['fit'])==prior['fit_ids']['fit']['n']
        assert v.object_hash(meta.iloc[idx['fit']][ID].tolist())==prior['fit_ids']['fit']['id_hash']
        frame=cache.iloc[idx['fit']].copy().reset_index(drop=True)
        m=meta.iloc[idx['fit']].reset_index(drop=True)
        # The Parquet predicate selects only original permitted fit IDs before labels materialize.
        labels=pq.read_table(META,columns=[ID,TARGET],filters=[(ID,'in',frame[ID].tolist())],use_threads=False).to_pandas()
        assert len(labels)==len(frame) and labels[ID].is_unique
        frame[TARGET]=labels.set_index(ID).loc[frame[ID],TARGET].to_numpy(float)
        assert np.isfinite(frame[TARGET]).all()
        frame['FLIGHT_ID_mvt']=m.FLIGHT_ID_mvt
        frame['missing']=~np.isfinite(m.proxy_sec.to_numpy(float))
        frame['day']=(pd.to_datetime(m[TIME],utc=True).dt.as_unit('ns').astype('int64')//(86400*10**9)).to_numpy()
        frame['month']=pd.to_datetime(m[TIME],utc=True).dt.strftime('%Y-%m').to_numpy()
        schedule=frame.schedule_proxy_sec.to_numpy(float)
        valid=np.isfinite(schedule)&(schedule!=-999999)
        frame['schedule_missing']=~valid
        frame['schedule_day']=np.floor(np.where(valid,schedule,0)/86400).astype('int32')
        frame['scale']=np.sqrt(3600.**2+np.where(valid,schedule-900.,0.)**2)
        frame['weight']=frame.scale**2
        frame['z']=(frame[TARGET]-900)/frame.scale
        frame['ge7200']=frame[TARGET].ge(7200)
        frame['ge86400']=frame[TARGET].ge(86400)
        frame['large_schedule']=valid&(np.abs(schedule)>=43200)
        guard()
        foldout=OUT/fold
        foldout.mkdir(exist_ok=False)
        tables=[]
        comparisons=[]
        for key,g in frame.groupby(BASE_KEYS,observed=True,dropna=False):
            values={k:(i.item() if isinstance(i,np.generic) else i) for k,i in zip(BASE_KEYS,key)}
            for flag,part in g.groupby('missing'):
                tables.append(dict(values,source='missing' if flag else 'known',**summary(part)))
            comparisons.append(dict(values,raw_difference=mean_difference(g,TARGET),z_difference=mean_difference(g,'z')))
        v.write_json(foldout/'cells.json',tables)
        v.write_json(foldout/'cell_mean_comparisons.json',comparisons)
        support={}
        for name,keys in KEY_SETS.items():
            result=prior_support(frame,keys)
            result.to_parquet(foldout/f'{name}_support.parquet',index=False)
            flags=frame.loc[frame.missing,['large_schedule','schedule_missing','schedule_day','ge7200','ge86400']].reset_index(drop=True)
            support[name]={}
            for label,mask in [('all',np.ones(len(flags),bool)),('large_schedule',flags.large_schedule.to_numpy()),('nonzero_schedule_day',flags.schedule_day.ne(0).to_numpy()),('raw_ge7200',flags.ge7200.to_numpy()),('raw_ge86400',flags.ge86400.to_numpy())]:
                q=result.loc[mask]
                support[name][label]=dict(n=len(q),no_prior_rows=int(q.known_prior_rows.eq(0).sum()),no_prior_days=int(q.known_prior_days.eq(0).sum()),under_two_prior_days=int(q.known_prior_days.lt(2).sum()),no_prior_ge7200=int(q.known_prior_ge7200.eq(0).sum()),no_prior_ge86400=int(q.known_prior_ge86400.eq(0).sum()),row_quantiles=np.quantile(q.known_prior_rows,[0,.1,.5,.9,1]).tolist() if len(q) else [],day_quantiles=np.quantile(q.known_prior_days,[0,.1,.5,.9,1]).tolist() if len(q) else [])
            guard()
        by_month=frame.groupby(['month','airport','missing'],observed=True).size().reset_index(name='n')
        by_month.to_parquet(foldout/'month_airport_counts.parquet',index=False)
        groups={name:summary(part) for name,part in [('all',frame),('missing',frame.loc[frame.missing]),('known',frame.loc[~frame.missing])]}
        groups['missing']['weight_mass_fraction']=groups['missing']['weight_sum']/groups['all']['weight_sum']
        records[fold]=dict(split_hash=v.object_hash(split),fit_rows=len(frame),fit_id_hash=v.object_hash(frame[ID].tolist()),fit_raw_label_hash=v.object_hash(frame[TARGET].tolist()),groups=groups,support=support,outputs={p.name:v.sha256(p) for p in foldout.iterdir()},peak_bytes=guard())
        v.write_json(foldout/'receipt.json',records[fold])
        print('FIT_SUPPORT_COMPLETE',fold,records[fold]['fit_rows'],support['base'],flush=True)
        del frame,m,labels,tables,comparisons,result,flags
        gc.collect()
    output=dict(status='complete',protocol_sha256=v.sha256(OUT/'protocol.json'),source_sha256=v.sha256(__file__),folds=records,peak_bytes=guard(),no_predictive_models=True,no_tune_or_score_role_labels=True)
    v.write_json(OUT/'summary.json',output)
    print('COMPLETE',OUT,output['peak_bytes'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
