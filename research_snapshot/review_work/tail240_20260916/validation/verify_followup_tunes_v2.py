"""Independent native replay and day-stability gates for CatBoost followups."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm
import argparse
import importlib.util
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import validate_candidate as validation

ROOT,ID,TARGET=validation.ROOT,validation.ID,validation.TARGET
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as common_source
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['rotation','catboost','linear'],required=True)
    family=parser.parse_args().family
    x,meta=common_source.shared.load_missing()
    binding=read(validation.BINDING)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    if family=='catboost':
        import normalized_catboost_tune as producer
        root=producer.OUT
        arms=producer.ARMS
        definition=read(root/'protocol.json')
        sources=definition['imported_sources']
    elif family=='linear':
        import normalized_linear_tune as producer
        root=producer.OUT
        arms=producer.ARMS
        definition=read(root/'protocol.json')
        sources=definition['source_hashes']
    else:
        sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/forensics'))
        import run_alias_rotation_models_v2 as producer
        root=producer.OUT
        arms=producer.ARMS
        definition=read(root/'protocol.json')['declaration']
        sources=definition['dependency_sources']
    summary=read(root/'summary.json')
    assert summary['status']=='complete' and definition['source_sha256']==sha(producer.__file__)
    for path,digest in sources.items():
        assert sha(ROOT/path)==digest
    if family=='rotation':
        assert sha(root/'features100.parquet')==summary['feature_sha256']
        cached=pd.read_parquet(root/'features100.parquet').set_index(ID)
        pd.testing.assert_frame_equal(cached.loc[:,list(x)],x,check_index_type=False)
        rawpath=producer.BASE/'training_features.parquet'
        pilotpath=producer.PILOT/'features.parquet'
        assert sha(rawpath)==read(producer.BASE/'manifest.json')['outputs'][rawpath.name]
        assert sha(pilotpath)==read(producer.PILOT/'manifest.json')['outputs'][pilotpath.name]
        extension=producer.extensions(x.index,meta.set_index(ID).loc[x.index,'ADEP_mvt'],pd.read_parquet(rawpath).set_index(ID),pd.read_parquet(pilotpath).set_index(ID))
        pd.testing.assert_frame_equal(cached,pd.concat([x,extension],axis=1),check_index_type=False)
    results={}
    for fold in ('F1','F3'):
        idx,_,_=validation.common.fold_data(meta,fold,full=True)
        rows={stage:idx[stage][missing[idx[stage]]] for stage in ('fit','tune')}
        tune_meta=meta.iloc[rows['tune']]
        ids=tune_meta[ID]
        target=tune_meta[TARGET].to_numpy(float)
        predictions={}
        model_records={}
        for arm in arms:
            directory=root/fold/arm if family in ('catboost','linear') else root/fold
            record=read(directory/'manifest.json')
            assert record['status']=='complete' and oh(record['split'])==binding['folds'][fold]['split_hash']
            for stage in ('fit','tune'):
                assert record['ids'][stage]==binding['folds'][fold]['cohorts']['missing_nm'][stage]
            for name,digest in record['outputs'].items():
                assert sha(directory/name)==digest
            if family=='linear':
                data=x.loc[ids]
                assert list(data)==record['feature_columns'] and len(data.columns)==76
                model=joblib.load(directory/'model.joblib')
                encoder=common_source.FrameEncoder().fit(x.loc[meta.iloc[rows['fit']][ID]])
                assert encoder.categories==model['encoder'].categories
                fit=encoder.transform(x.loc[meta.iloc[rows['fit']][ID]])
                numeric=fit[encoder.numeric].to_numpy(float)
                means=np.array([np.mean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in numeric.T])
                stds=np.array([np.std(col[np.isfinite(col)]) if np.isfinite(col).any() else 1. for col in numeric.T])
                stds=np.where(stds>1e-6,stds,1.)
                np.testing.assert_array_equal(means,model['scaler']['means'])
                np.testing.assert_array_equal(stds,model['scaler']['stds'])
                transformed=model['encoder'].transform(data)
                transformed[encoder.numeric]=(transformed[encoder.numeric].to_numpy(float)-means)/stds
                scale=common_source.scale_of(data)
                normalizer=float(np.mean(common_source.scale_of(x.loc[meta.iloc[rows['fit']][ID]])**2))
                assert normalizer==record['normalizer']
                z=model['model'].predict(transformed,num_threads=2)
                pred=900+scale*z
                output=pd.read_parquet(directory/'tune.parquet')
                np.testing.assert_array_equal(output.scale,scale)
                np.testing.assert_array_equal(output.raw_target_sec,target)
                assert record['params']['linear_tree']==(arm=='linear')
            elif family=='catboost':
                data=x.loc[ids]
                assert list(data)==record['feature_columns'] and len(data.columns)==76
                scale=np.ones(len(data)) if arm=='unscaled' else common_source.scale_of(data)
                fit_scale=np.ones(len(rows['fit'])) if arm=='unscaled' else common_source.scale_of(x.loc[meta.iloc[rows['fit']][ID]])
                assert float(np.mean(fit_scale**2))==record['normalizer']
                model=CatBoostRegressor().load_model(str(directory/'model.cbm'))
                z=model.predict(data,thread_count=2)
                pred=900+scale*z
                output=pd.read_parquet(directory/'tune.parquet')
                np.testing.assert_array_equal(output.scale,scale)
                np.testing.assert_array_equal(output.raw_target_sec,target)
                weighted=scale**2/record['normalizer']*(z-(target-900)/scale)**2
                np.testing.assert_allclose(weighted,(pred-target)**2/record['normalizer'],rtol=1e-9,atol=1e-7)
            else:
                columns=record['results'][arm]['features']
                data=cached.loc[ids,columns]
                times=pd.Series(pd.to_datetime(tune_meta[common_source.TIME],utc=True).to_numpy(),index=data.index)
                model=joblib.load(directory/f'{arm}.joblib')
                pred=producer.prior.predict_arm(model,data,times)
                output=pd.read_parquet(directory/f'{arm}_tune.parquet')
                np.testing.assert_array_equal(output[TARGET],target)
                if arm=='control76':
                    previousdir=ROOT/f'private_runs/tail240_20260916/state/models_v2/control/{fold}'
                    assert sha(previousdir/'tune_predictions.parquet')==read(previousdir/'manifest.json')['outputs']['tune_predictions.parquet']
                    previous=pd.read_parquet(previousdir/'tune_predictions.parquet')
                    np.testing.assert_array_equal(previous[ID],ids)
                    np.testing.assert_array_equal(previous.prediction_sec,pred)
            np.testing.assert_array_equal(output[ID],ids)
            delta=float(np.max(np.abs(pred-output.prediction_sec.to_numpy())))
            assert delta<=1e-9
            predictions[arm]=pred
            model_records[arm]={'replay_max_abs_delta':delta,'rmse':float(np.sqrt(np.mean((pred-target)**2))),'manifest_sha256':sha(directory/'manifest.json')}
            validation.guard()
        if family in ('catboost','linear'):
            previousdir=common_source.OUT/fold/'observed_schedule_scale'
            if family=='catboost':
                assert sha(previousdir/'manifest.json')==definition['references'][fold]
            assert sha(previousdir/'tune.parquet')==read(previousdir/'manifest.json')['outputs']['tune.parquet']
            previous=pd.read_parquet(previousdir/'tune.parquet')
            np.testing.assert_array_equal(previous[ID],ids)
            own_control='unscaled' if family=='catboost' else 'constant'
            candidate_arm='observed_schedule_scale' if family=='catboost' else 'linear'
            controls={own_control:predictions[own_control],'scaled_lightgbm':previous.prediction_sec.to_numpy()}
            candidate=predictions[candidate_arm]
            mask=np.ones(len(target),bool)
        else:
            controls={arm:predictions[arm] for arm in ('control76','raw84')}
            candidate=predictions['alias100']
            mask=tune_meta.ADEP_mvt.eq('LIRF').to_numpy()
        dates=tune_meta[common_source.TIME].dt.strftime('%Y-%m-%d').to_numpy()
        comparisons={name:validation.paired(target[mask],control[mask],candidate[mask],dates[mask]) for name,control in controls.items()}
        gate=all(item['delta_rmse']<0 and item['all_day_removals_improve'] for item in comparisons.values())
        results[fold]={'models':model_records,'gate_rows':int(mask.sum()),'comparisons':comparisons,'gate_passed':gate}
    advance=all(record['gate_passed'] for record in results.values())
    assert advance==summary['gate_passed' if family=='rotation' else 'advance']
    out=ROOT/f'private_runs/tail240_20260916/validation/{family}_followup_tune_v1'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'receipt.json',{'status':'passed_validation','source_sha256':sha(__file__),'producer_summary_sha256':sha(root/'summary.json'),
        'family':family,'folds':results,'advance':advance,'peak_rss_bytes':validation.guard(),
        'scope':'Original complete missingfit/tune cohorts and rawtunelabels verified; savednative predictions replayed for allmissing tune rows. Rotationcomposites rebuilt fromboundcaches, originalcontrol exact. Gate, day/row removal andbootstrap computed independently. No score predictions.'})
    print(family,{fold:{'gate':r['gate_passed'],'deltas':{name:c['delta_rmse'] for name,c in r['comparisons'].items()}} for fold,r in results.items()},flush=True)


if __name__=='__main__':
    main()
