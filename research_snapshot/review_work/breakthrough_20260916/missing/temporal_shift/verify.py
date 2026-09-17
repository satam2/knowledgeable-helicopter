"""Independent grouped-sum prior oracle and descriptive monthly shift summary."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
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
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    assert not (OUT/'verification.json').exists()
    report=common.read_json(OUT/'diagnostic.json')
    for name,digest in report['outputs'].items():
        assert common.sha256(OUT/name)==digest
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    conv_path=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions/features.parquet'
    conv=pq.read_table(conv_path,columns=[ID,'conv_clock_rank_signature'],use_threads=False).to_pandas().set_index(ID)
    results={}
    for fold in ['F1','F3']:
        record=report['folds'][fold]
        spec=record['split']['spec']
        start=pd.Timestamp(spec['fit'][0],tz='UTC').to_pydatetime()
        stop=pd.Timestamp(spec['tune'][1],tz='UTC').to_pydatetime()
        frame=pq.read_table(meta_path,columns=[ID,TARGET,TIME,'ADEP_mvt','proxy_sec'],filters=[(TIME,'>=',start),(TIME,'<',stop)],use_threads=False).to_pandas()
        frame=frame.loc[frame.proxy_sec.between(0,7200)&np.isfinite(frame.proxy_sec)].copy()
        frame['airport']=frame.ADEP_mvt.astype('string').fillna('<missing>')
        frame['order']=conv.loc[frame[ID],'conv_clock_rank_signature'].astype('string').fillna('<missing>').to_numpy()
        frame['bin']=pd.cut(frame.proxy_sec,[0,300,600,900,1200,1800,3600,7200],include_lowest=True).astype('string')
        frame['residual']=frame[TARGET]-frame.proxy_sec
        boundary=pd.Timestamp(spec['tune'][0],tz='UTC')
        fit=frame.loc[frame[TIME]<boundary]
        tune=frame.loc[frame[TIME]>=boundary]
        assert common.object_hash(fit[ID].tolist())==record['fit_id_hash']
        assert common.object_hash(tune[ID].tolist())==record['tune_id_hash']
        saved=pd.read_parquet(OUT/(fold+'_tune_predictions.parquet'))
        np.testing.assert_array_equal(saved[ID],tune[ID])
        np.testing.assert_array_equal(saved.raw_residual_sec,tune.residual)
        positions=np.unique(np.r_[0,len(tune)-1,np.random.default_rng(20260916).choice(len(tune),30,replace=False)])
        for name,reference in [('expanding',fit),('recent90',fit.loc[fit[TIME]>=boundary-pd.Timedelta(days=90)])]:
            global_mean=reference.residual.to_numpy().mean()
            for pos in positions:
                query=tune.iloc[pos]
                airport=reference.loc[reference.airport.eq(query.airport),'residual'].to_numpy()
                parent=(airport.sum()+100*global_mean)/(len(airport)+100)
                group=reference.loc[reference.airport.eq(query.airport)&reference.order.eq(query.order)&reference['bin'].eq(query['bin']),'residual'].to_numpy()
                expected=(group.sum()+100*parent)/(len(group)+100)
                np.testing.assert_allclose(saved.iloc[pos][name+'_correction'],expected,rtol=1e-12,atol=1e-10)
                assert saved.iloc[pos][name+'_group_n']==len(group)
            error=saved[name+'_correction']-saved.raw_residual_sec
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)),record[name]['rmse'],rtol=0,atol=1e-10)
        monthly=pd.read_parquet(OUT/(fold+'_monthly_order_frequencies.parquet'))
        first,last=monthly.month.min(),monthly.month.max()
        old=monthly.loc[monthly.month.eq(first),['airport','order','n','mean','airport_month_share']]
        new=monthly.loc[monthly.month.eq(last),['airport','order','n','mean','airport_month_share']]
        comparison=old.merge(new,on=['airport','order'],how='outer',suffixes=('_first','_tune')).fillna(0)
        comparison['absolute_frequency_change']=(comparison.airport_month_share_tune-comparison.airport_month_share_first).abs()
        tv=comparison.groupby('airport').absolute_frequency_change.sum()/2
        comparison.sort_values('absolute_frequency_change',ascending=False).to_parquet(OUT/(fold+'_first_to_tune_order_shift.parquet'),index=False)
        results[fold]=dict(independent_prior_queries=len(positions),reference_windows=2,
            all_tune_raw_residuals_exact=True,prior_oracle_tolerance=1e-10,
            first_month=first,tune_month=last,airport_order_total_variation=tv.to_dict())
        print('VERIFIED_PRIOR',fold,len(positions)*2,'maxAirportOrderTV',tv.max(),flush=True)
    common.write_json(OUT/'verification.json',dict(status='passed',source_sha256=common.sha256(__file__),
        diagnostic_sha256=common.sha256(OUT/'diagnostic.json'),folds=results,
        limitation='All original purges are zero for these folds; verifier confirms exact fit/tune ID hashes. No scorelabel/artifact access.'))


if __name__=='__main__':
    main()
