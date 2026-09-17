"""Native context ablation replay and independent raw-MSE stability gate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm
import importlib.util
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import validate_candidate as validation

ROOT=validation.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/normalized_context'))
spec=importlib.util.spec_from_file_location('normalized_context_source',ROOT/'review_work/tail240_20260916/state/normalized_context/run.py')
producer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
BASE=producer.OUT
OUT=ROOT/'private_runs/tail240_20260916/validation/normalized_context_v1'
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
ID,TARGET=validation.ID,validation.TARGET


def main():
    protocol=read(BASE/'protocol.json')
    for path,digest in protocol['source_hashes'].items():
        assert sha(ROOT/path)==digest
    cache=read(BASE/'cache_manifest.json')
    assert cache['status']=='complete' and cache['protocol_sha256']==sha(BASE/'protocol.json')
    assert sha(BASE/'context.parquet')==cache['feature_sha256']
    context=pd.read_parquet(BASE/'context.parquet').set_index(ID)
    x,meta=producer.base.shared.load_missing()
    np.testing.assert_array_equal(context.index,x.index)
    assert len(x)==cache['rows']==22470 and oh(x.index.tolist())==cache['original_missing_ids_hash']
    assert list(context)==protocol['context_columns'] and list(x)==protocol['base_columns']
    sampled=x.index[np.unique(np.linspace(0,len(x)-1,96,dtype=int))]
    compared=0
    for source in cache['sources']:
        path=Path(source['path'])
        assert sha(path)==source['sha256']
        names=source['columns']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas().set_index(ID)
            frame=frame.loc[frame.index.isin(sampled)]
            if len(frame)==0:
                continue
            for name in names:
                actual=context.loc[frame.index,name]
                if pd.api.types.is_numeric_dtype(frame[name]):
                    expected=pd.to_numeric(frame[name]).replace([np.inf,-np.inf],np.nan).astype('float32')
                    np.testing.assert_array_equal(actual.to_numpy(),expected.to_numpy())
                else:
                    pd.testing.assert_series_equal(actual.astype('string'),frame[name].astype('string'),check_names=False,check_index_type=False)
                compared+=len(frame)
    assert compared==96*375
    binding=read(validation.BINDING)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    results={}
    for fold in ('F1','F3'):
        idx,_,_=producer.common.fold_data(meta,fold,full=True)
        rows={stage:idx[stage][missing[idx[stage]]] for stage in ('fit','tune')}
        frames={stage:x.loc[meta.iloc[pos][ID]] for stage,pos in rows.items()}
        target=meta.iloc[rows['tune']][TARGET].to_numpy(float)
        scale=producer.base.scale_of(frames['tune'])
        norm=float(np.mean(producer.base.scale_of(frames['fit'])**2))
        predictions={}
        modelinfo={}
        for arm in protocol['arms']:
            directory=BASE/fold/arm
            record=read(directory/'manifest.json')
            assert record['status']=='complete' and record['protocol_sha256']==sha(BASE/'protocol.json')
            assert record['cache_manifest_sha256']==sha(BASE/'cache_manifest.json')
            assert oh(record['split'])==binding['folds'][fold]['split_hash']
            for stage in ('fit','tune'):
                assert record['fit_ids'][stage]==binding['folds'][fold]['cohorts']['missing_nm'][stage]
                assert record['label_hashes'][stage]==oh(meta.iloc[rows[stage]][TARGET].to_numpy(float).tolist())
            for name,digest in record['outputs'].items():
                assert sha(directory/name)==digest
            data={stage:frame if arm=='control76' else pd.concat([frame,context.loc[frame.index]],axis=1) for stage,frame in frames.items()}
            assert list(data['fit'])==record['feature_columns']
            model=joblib.load(directory/'model.joblib')
            independent_encoder=producer.base.FrameEncoder().fit(data['fit'])
            assert independent_encoder.categories==model['encoder'].categories
            z=model['model'].predict(model['encoder'].transform(data['tune']),num_iteration=record['steps'],num_threads=2)
            pred=900+scale*z
            output=pd.read_parquet(directory/'tune.parquet')
            np.testing.assert_array_equal(output[ID],data['tune'].index)
            np.testing.assert_array_equal(output.raw_target_sec,target)
            np.testing.assert_array_equal(output.prediction_sec,pred)
            np.testing.assert_array_equal(output.scale,scale)
            np.testing.assert_array_equal(output.weight,scale**2/norm)
            np.testing.assert_array_equal(output.transformed_target,(target-900)/scale)
            np.testing.assert_allclose(output.weight*(z-output.transformed_target)**2,(pred-target)**2/norm,rtol=1e-9,atol=1e-7)
            if arm=='control76':
                olddir=producer.base.OUT/fold/'observed_schedule_scale'
                assert sha(olddir/'manifest.json')==record['baseline_manifest_sha256']
                old=pd.read_parquet(olddir/'tune.parquet')
                np.testing.assert_array_equal(pred,old.prediction_sec)
                assert record['steps']==read(olddir/'manifest.json')['steps']
            predictions[arm]=pred
            modelinfo[arm]={'manifest_sha256':sha(directory/'manifest.json'),'replay_max_abs_delta':0.,'rmse':float(np.sqrt(np.mean((pred-target)**2))),'steps':record['steps']}
        days=meta.iloc[rows['tune']][producer.TIME].dt.strftime('%Y-%m-%d').to_numpy()
        comparison=validation.paired(target,predictions['control76'],predictions['context451'],days)
        reported=read(BASE/fold/'context451/metrics.json')
        np.testing.assert_allclose(comparison['delta_rmse'],-reported['rmse_improvement'],rtol=0,atol=1e-10)
        assert comparison['all_day_removals_improve']==reported['all_day_removals_improve']
        results[fold]={'models':modelinfo,'comparison':comparison,'gate_passed':comparison['delta_rmse']<0 and comparison['all_day_removals_improve']}
        validation.guard()
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'receipt.json',{'status':'passed_validation','source_sha256':sha(__file__),'producer_protocol_sha256':sha(BASE/'protocol.json'),
        'context_raw_sample_cells_verified':compared,'folds':results,'advance':all(r['gate_passed'] for r in results.values()),
        'peak_rss_bytes':validation.guard(),'scope':'Tuneonly; exactoriginalcohorts/rawlabels and all4nativepredictions replayed, originalcontrol exact. Independent96rowx375column rawcache spotoracle; fullcacheID/hash checked. Day/row sensitivity and2000paired-daybootstrap independently computed.'})
    print({fold:{'delta':r['comparison']['delta_rmse'],'daystable':r['comparison']['all_day_removals_improve'],'gate':r['gate_passed']} for fold,r in results.items()},flush=True)


if __name__=='__main__':
    main()
