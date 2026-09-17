"""Verify diagnostic error budgets and label-free neighbor selection separately."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import validate_candidate as validation

ROOT,ID,TARGET=validation.ROOT,validation.ID,validation.TARGET
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as producer
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
BASE=ROOT/'private_runs/tail240_20260916/state/scaled_forensics/v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/scaled_forensics_v1'


def main():
    summary=read(BASE/'summary.json')
    protocol=read(BASE/'protocol.json')
    assert summary['status']=='complete' and summary['protocol_sha256']==sha(BASE/'protocol.json')
    assert protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/state/scaled_forensics/audit.py')
    for name,digest in summary['outputs'].items():
        assert sha(BASE/name)==digest
    x,meta=producer.shared.load_missing()
    indexed=meta.set_index(ID)
    results={}
    for fold in ('F1','F3'):
        report=read(BASE/f'{fold}.json')
        idx,split,_=producer.common.fold_data(meta,fold,full=True)
        assert oh(report['split'])==oh(split)
        missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
        ids={stage:meta.iloc[idx[stage][missing[idx[stage]]]][ID] for stage in ('fit','tune')}
        originaldir=producer.OUT/fold/'observed_schedule_scale'
        assert report['producer_manifest_sha256']==sha(originaldir/'manifest.json')
        original=pd.read_parquet(originaldir/'tune.parquet').set_index(ID)
        actual=pd.read_parquet(BASE/f'{fold}_error_rows.parquet').set_index(ID)
        assert set(actual.index)==set(ids['tune'])
        np.testing.assert_array_equal(actual.raw_Y,indexed.loc[actual.index,TARGET])
        np.testing.assert_array_equal(actual.prediction,original.loc[actual.index,'prediction_sec'])
        squared=(actual.prediction.to_numpy()-actual.raw_Y.to_numpy())**2
        np.testing.assert_array_equal(actual.squared_error,squared)
        assert np.all(squared[:-1]>=squared[1:])
        for k in (1,5,10,20,50):
            np.testing.assert_allclose(squared[:k].sum()/squared.sum(),report['error_budget'][str(k)]['sse_share'],rtol=1e-14)
            np.testing.assert_allclose(np.sqrt((squared.sum()-squared[:k].sum())/len(squared)),report['error_budget'][str(k)]['remaining_rmse_oracle'],rtol=1e-12)
        saved=joblib.load(originaldir/'model.joblib')
        unseen=~x.loc[actual.index[:20],'flight'].astype('string').isin(saved['encoder'].categories['flight'])
        assert int(unseen.sum())==report['top20']['unseen_flight']
        counterparts=pd.read_parquet(BASE/f'{fold}_counterparts.parquet')
        for (query_id,stage),pairs in counterparts.groupby(['query_id','stage']):
            assert query_id in actual.index[:20]
            q=x.loc[query_id]
            candidates=x.loc[ids[stage]]
            mask=candidates.airport.eq(q.airport)&candidates.source_record_regime.eq(q.source_record_regime)&(candidates.index!=query_id)
            flight=indexed.loc[query_id,'FLIGHT_ID_mvt']
            if pd.notna(flight):
                mask &= indexed.loc[candidates.index,'FLIGHT_ID_mvt'].ne(flight).to_numpy()
            candidates=candidates.loc[mask]
            distance=np.zeros(len(candidates))
            for field,scale in [('schedule_proxy_sec',3600.),('idctx_peer_age_w8_sec',3600.),('schedule_calendar_day_delta',1.)]:
                right=pd.to_numeric(candidates[field]).replace(-999999,np.nan).to_numpy(float)
                left=float(q[field])
                if left==-999999 or not np.isfinite(left):
                    distance+=np.isfinite(right)
                else:
                    distance+=np.where(np.isfinite(right),np.log1p(np.abs(right-left)/scale),1.)
            for field in ('flight','stand','runway','destination'):
                distance+=candidates[field].astype(str).to_numpy()!=str(q[field])
            order=np.argsort(distance,kind='stable')
            nearest=candidates.index[order[:20]]
            for row in pairs.itertuples(index=False):
                if row.role.startswith('nearest'):
                    assert row.neighbor_id==nearest[int(row.role[-1])-1]
                else:
                    assert row.role=='best_error_among_nearest20' and stage=='tune'
                    assert row.neighbor_id==actual.loc[nearest,'squared_error'].idxmin()
                assert row.query_Y==float(indexed.loc[query_id,TARGET]) and row.neighbor_Y==float(indexed.loc[row.neighbor_id,TARGET])
                assert row.distance==distance[candidates.index.get_loc(row.neighbor_id)]
        results[fold]={'tune_rows':len(actual),'top1_missing_sse_share':report['error_budget']['1']['sse_share'],
            'top20_missing_sse_share':report['top20']['sse_share'],'top20_unseen_flights':int(unseen.sum()),
            'counterpart_rows_verified':len(counterparts),'all_neighbor_distance_and_selected_roles_verified':True}
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'receipt.json',{'status':'passed','source_sha256':sha(__file__),'producer_summary_sha256':sha(BASE/'summary.json'),
        'folds':results,'scope':'Fit/tune diagnostic only. Top-error query selection and best-error-neighbor role target-defined and explicitly posthoc; candidate distance/ranking independently label-free. Zero-error remainingRMSE is an oracle budget, not achievable prediction.'})
    print(results,flush=True)


if __name__=='__main__':
    main()
