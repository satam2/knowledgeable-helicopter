"""Independent normalized-target replay, raw-loss equivalence and tune stability."""
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

ROOT=validation.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as producer
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
ID,TARGET=validation.ID,validation.TARGET
OUT=ROOT/'private_runs/tail240_20260916/validation/normalized_missing_tune_v1'


def main():
    protocol=read(producer.OUT/'protocol.json')
    summary=read(producer.OUT/'summary.json')
    assert summary['status']=='complete' and sha(producer.__file__)==protocol['source_sha256']
    for path,digest in protocol['sources'].items():
        assert sha(ROOT/path)==digest
    x,meta=producer.shared.load_missing()
    oldcolumns=read(ROOT/'private_runs/tail240_20260916/state/models_v2/control/F1/manifest.json')['feature_columns']
    assert list(x)==oldcolumns and len(x.columns)==76
    binding=read(validation.BINDING)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    results={}
    for fold in ('F1','F3'):
        idx,split,_=producer.common.fold_data(meta,fold,full=True)
        selected={stage:idx[stage][missing[idx[stage]]] for stage in ('fit','tune')}
        ids={stage:meta.iloc[rows][ID] for stage,rows in selected.items()}
        expected={stage:x.loc[values] for stage,values in ids.items()}
        predictions={}
        receipts={}
        target=meta.iloc[selected['tune']][TARGET].to_numpy(float)
        dates=pd.to_datetime(meta.iloc[selected['tune']][producer.TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        for arm in protocol['arms']:
            folder=producer.OUT/fold/arm
            record=read(folder/'manifest.json')
            assert record['status']=='complete' and record['source_sha256']==sha(producer.__file__)
            assert record['protocol_sha256']==sha(producer.OUT/'protocol.json')
            assert oh(record['split'])==binding['folds'][fold]['split_hash']
            for stage in ('fit','tune'):
                assert record['fit_ids'][stage]==binding['folds'][fold]['cohorts']['missing_nm'][stage]
            for path,digest in record['outputs'].items():
                assert sha(folder/path)==digest
            output=pd.read_parquet(folder/'tune.parquet')
            np.testing.assert_array_equal(output[ID],ids['tune'])
            np.testing.assert_array_equal(output.raw_target_sec,target)
            scales={}
            for stage,frame in expected.items():
                schedule=frame.schedule_proxy_sec.to_numpy(float)
                good=np.isfinite(schedule)&(schedule!=-999999)
                scales[stage]=np.ones(len(frame)) if arm=='unscaled' else np.sqrt(3600**2+np.where(good,schedule-900,0)**2)
            norm=float(np.mean(scales['fit']**2))
            assert norm==record['normalizer']
            weights={stage:scale**2/norm for stage,scale in scales.items()}
            assert all(np.isfinite(value).all() and (value>0).all() for value in weights.values())
            np.testing.assert_array_equal(output.scale,scales['tune'])
            np.testing.assert_array_equal(output.weight,weights['tune'])
            np.testing.assert_array_equal(output.transformed_target,(target-900)/scales['tune'])
            fitted=joblib.load(folder/'model.joblib')
            assert fitted['encoder'].columns==oldcolumns
            independently_fit=producer.FrameEncoder().fit(expected['fit'])
            assert fitted['encoder'].categories==independently_fit.categories
            z=fitted['model'].predict(fitted['encoder'].transform(expected['tune']),num_iteration=record['steps'],num_threads=2)
            pred=900+scales['tune']*z
            np.testing.assert_array_equal(pred,output.prediction_sec)
            scaled_error=weights['tune']*(z-output.transformed_target.to_numpy())**2
            np.testing.assert_allclose(scaled_error,(pred-target)**2/norm,rtol=1e-9,atol=1e-7)
            rmse=float(np.sqrt(np.mean((pred-target)**2)))
            assert rmse==record['rmse']==summary['results'][fold][arm]
            ess=float(weights['fit'].sum()**2/np.sum(weights['fit']**2))
            assert ess==record['fit_weight_ess']
            predictions[arm]=pred
            receipts[arm]={'manifest_sha256':sha(folder/'manifest.json'),'native_replay_max_abs_delta':0.,'rmse':rmse,'positive_weight_rows':len(weights['fit']),'fit_weight_ess':ess,'steps':record['steps']}
        historical_dir=ROOT/f'private_runs/tail240_20260916/state/models_v2/control/{fold}'
        marker=read(historical_dir/'manifest.json')
        assert sha(historical_dir/'tune_predictions.parquet')==marker['outputs']['tune_predictions.parquet']
        historical=pd.read_parquet(historical_dir/'tune_predictions.parquet')
        np.testing.assert_array_equal(historical[ID],ids['tune'])
        predictions['historical_template']=historical.prediction_sec.to_numpy(float)
        comparisons={name:validation.paired(target,predictions[name],predictions['observed_schedule_scale'],dates) for name in ('unscaled','historical_template')}
        point=all(item['delta_rmse']<0 for item in comparisons.values())
        daystable=all(item['all_day_removals_improve'] for item in comparisons.values())
        union_dir=ROOT/f'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        marker=read(union_dir/'manifest.json')
        path=union_dir/'tune_predictions.parquet'
        assert sha(path)==marker['outputs'][path.name]
        union=pd.read_parquet(path).set_index(ID)
        fullmeta=meta.iloc[idx['tune']]
        finite=~missing[idx['tune']]
        assert set(union.index)==set(fullmeta.loc[finite,ID])
        finite_error=(union.loc[fullmeta.loc[finite,ID],'prediction_sec'].to_numpy()-fullmeta.loc[finite,TARGET].to_numpy())**2
        composition={name:float(np.sqrt((finite_error.sum()+np.sum((p-target)**2))/len(fullmeta))) for name,p in predictions.items()}
        results[fold]={'models':receipts,'missing_comparisons':comparisons,'point_gate':point,'every_day_removal_gate':daystable,
            'full_tune_composition':{'reference':'Saved union FIT-model TUNE predictions on finite rows; historical/unscaled/normalized alternatives on missing rows. This is not global9 or a score prediction.',
                'all_rows':len(fullmeta),'missing_rows':len(target),'finite_rows':int(finite.sum()),'rmse':composition,
                'full_tune_mse_reduction_vs_historical':float(np.sum((predictions['historical_template']-target)**2-(predictions['observed_schedule_scale']-target)**2)/len(fullmeta))}}
        validation.guard()
    OUT.mkdir(parents=True,exist_ok=False)
    receipt={'status':'passed_validation','source_sha256':sha(__file__),'producer_summary_sha256':sha(producer.OUT/'summary.json'),
        'folds':results,'both_point_gates':all(v['point_gate'] for v in results.values()),
        'both_day_removal_gates':all(v['every_day_removal_gate'] for v in results.values()),
        'peak_rss_bytes':validation.guard(),'scope':'Fit/tune only. No score predictions/labels evaluated. WeightedESS is objective concentration, not row removal. FixedL2 regularization is not invariant to response parameterization.'}
    write(OUT/'receipt.json',receipt)
    print({fold:{'point':r['point_gate'],'daystable':r['every_day_removal_gate'],'full_tune':r['full_tune_composition']} for fold,r in results.items()},flush=True)


if __name__=='__main__':
    main()
