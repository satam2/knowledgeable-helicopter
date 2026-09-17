"""Full contracts and128-row independent CPU replay for combined225 tree families."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import lightgbm
import argparse
import gc
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models/verification_id_template'))
import common
from audit import comparison,metric
from taxiout.metrics import season_score
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
BASE=ROOT/'private_runs/breakthrough_20260916'
OUT=BASE/'missing/combined_family_audit'
FEATURES=ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def guard():
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    assert peak<2*1024**3,peak
    return peak


def selected(path,columns,ids):
    parquet=pq.ParquetFile(path)
    pieces=[]
    for i in range(parquet.num_row_groups):
        key=parquet.read_row_group(i,columns=[ID],use_threads=False).column(0)
        mask=pc.is_in(key,value_set=pa.array(ids.to_numpy(),type=key.type))
        if pc.any(mask).as_py():
            pieces.append(parquet.read_row_group(i,columns=[ID,*columns],use_threads=False).filter(mask).to_pandas())
        guard()
    result=pd.concat(pieces,ignore_index=True).set_index(ID)
    assert result.index.is_unique
    return result.loc[ids]


def query_features(ids,dates,desired):
    pieces=[]
    sources=[]
    months=pd.to_datetime(dates,utc=True).dt.strftime('%Y-%m').to_numpy()
    for month in sorted(set(months)):
        path=next(FEATURES.glob('training_'+month+'-01_*.parquet'))
        marker=common.read_json(path.with_suffix('.json'))
        assert common.sha256(path)==marker['sha256']
        cols=[c for c in pq.read_schema(path).names if c!=ID]
        pieces.append(selected(path,cols,ids[months==month]))
        sources.append(dict(path=str(path),sha256=marker['sha256']))
    frame=pd.concat(pieces).loc[ids]
    for directory,filename in [('private_runs/mechanism_20260916/information/retrospective_v2','training_features.parquet'),
            ('private_runs/breakthrough_20260916/batch_context','training_features.parquet'),
            ('private_runs/breakthrough_20260916/missing/source_conventions','features.parquet'),
            ('private_runs/breakthrough_20260916/geometry_v2','training_features.parquet'),
            ('private_runs/breakthrough_20260916/weather','training_features.parquet')]:
        folder=ROOT/directory
        marker=common.read_json(folder/'manifest.json')
        path=folder/filename
        expected=marker['feature_sha256'] if filename=='features.parquet' else marker['outputs'][filename]
        assert common.sha256(path)==expected
        columns=[c for c in desired if c in pq.read_schema(path).names and c not in frame]
        extra=selected(path,columns,ids)
        for col in columns:
            if pd.api.types.is_numeric_dtype(extra[col]):
                extra[col]=extra[col].replace([np.inf,-np.inf],np.nan).fillna(-999999).astype('float32')
            else:
                extra[col]=extra[col].astype('string').fillna('MISSING').astype('category')
        frame=pd.concat([frame,extra],axis=1)
        sources.append(dict(path=str(path),sha256=expected))
        guard()
    frame=frame[desired]
    assert frame.shape==(len(ids),225)
    return frame,sources


def independent_transform(encoder,frame):
    data={}
    for name in encoder.columns:
        if name in encoder.categories:
            values=frame[name].astype('string')
            code=values.map(encoder.categories[name]).fillna(1).to_numpy(np.int32)
            code[values.isna().to_numpy()]=0
            data[name]=pd.Categorical(code,categories=range(len(encoder.categories[name])+2))
        else:
            v=frame[name].to_numpy(np.float32,na_value=np.nan).copy()
            v[(v==-999999)|~np.isfinite(v)]=np.nan
            data[name]=v
    independent=pd.DataFrame(data,index=frame.index)
    pd.testing.assert_frame_equal(independent,encoder.transform(frame))
    return independent


def checked(folder):
    rec=common.read_json(folder/'manifest.json')
    assert rec['status']=='complete',str(folder)
    for name,digest in rec['outputs'].items():
        assert common.sha256(folder/name)==digest
    key=rec['family']+'_aobt_allfinite_s20260916'
    snap=folder.parent/'source_snapshots'/key
    for name,digest in rec['source_hashes'].items():
        assert common.sha256(snap/Path(name).name)==digest
    assert common.sha256(folder.parent/'protocols'/(key+'.json'))==rec['protocol_sha256']
    assert rec['fit']['steps']==rec['refit']['steps']
    assert rec['fit']['rows']==rec['fit_ids']['fit']['n']
    assert rec['refit']['rows']==rec['fit_ids']['refit']['n']
    assert rec['reload_max_abs_delta']==0
    return rec


def run_fold(family,fold,meta):
    folder=(BASE/'combined_catboost'/f'catboost_combined_aobt_allfinite_{fold}_s20260916' if family=='catboost'
        else BASE/'combined_xgb'/f'xgb_aobt_allfinite_{fold}_s20260916')
    destination=OUT/family/(fold+'.json')
    assert not destination.exists()
    rec=checked(folder)
    controls={'leaf63':BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
        'tabm':BASE/'full_neural'/f'tabm_combined_standard_aobt_allfinite_{fold}_s20260916'}
    markers={name:common.read_json(path/'manifest.json') for name,path in controls.items()}
    assert len(rec['feature_columns'])==225 and rec['seed']==20260916 and rec['threads']==2
    for marker in markers.values():
        assert marker['status']=='complete'
        assert rec['feature_columns']==marker['feature_columns']
        assert rec['fit_ids']==marker['fit_ids']
        assert rec['anchor']['feature_receipts']==marker['anchor']['feature_receipts']
        assert rec['split']==marker['split']
    for receipt in rec['anchor']['feature_receipts']:
        path=Path(receipt.get('path',receipt.get('manifest')))
        assert common.sha256(path)==receipt.get('sha256',receipt.get('manifest_sha256'))
    idx,split,_=common.fold_data(meta,fold,full=True)
    finite=np.isfinite(meta.proxy_sec.to_numpy(float))
    assert common.object_hash(split)==common.object_hash(rec['split'])
    for stage,pos in idx.items():
        chosen=pos[finite[pos]]
        assert rec['fit_ids'][stage]==dict(n=len(chosen),hash=common.object_hash(meta.iloc[chosen][ID].tolist()))
    ref,refrec=common.reference(fold)
    np.testing.assert_array_equal(ref[ID],meta.iloc[idx['score']][ID])
    np.testing.assert_array_equal(ref[TARGET],meta.iloc[idx['score']][TARGET])
    frame=pd.read_parquet(folder/'candidate.parquet')
    np.testing.assert_array_equal(frame[ID],ref[ID])
    np.testing.assert_array_equal(frame[TARGET],ref[TARGET])
    missing=~np.isfinite(ref.proxy_sec.to_numpy(float))
    np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing],ref.prediction_sec.to_numpy()[missing])
    positions=np.flatnonzero(~missing)
    chosen=positions[np.linspace(0,len(positions)-1,128,dtype=int)]
    query=ref.iloc[chosen]
    features,feature_sources=query_features(pd.Index(query[ID]),query[TIME],rec['feature_columns'])
    replay={}
    for stage,filename in [('refit','model.joblib'),('fit','fit_model.joblib')]:
        model=joblib.load(folder/filename)
        assert model['encoder'].columns==rec['feature_columns'] and model['encoder'].neural is False
        assert model['steps']==rec[stage]['steps']
        encoded=independent_transform(model['encoder'],features)
        if family=='catboost':
            assert model['estimator'].get_cat_feature_indices()==[i for i,c in enumerate(rec['feature_columns']) if c in model['encoder'].categories]
            values=model['estimator'].predict(encoded,ntree_end=model['steps'],thread_count=2,task_type='CPU')
        else:
            model['estimator'].set_params(device='cpu',n_jobs=2)
            values=model['estimator'].predict(encoded,iteration_range=(0,model['steps']))
        values=np.asarray(values,float)+query.proxy_sec.to_numpy(float)
        assert np.isfinite(values).all()
        if stage=='refit':
            delta=float(np.max(np.abs(values-frame.iloc[chosen].prediction_sec.to_numpy(float))))
            assert delta<=1e-4,delta
            replay[stage]=dict(rows=128,max_abs_delta=delta,CPU=True,threshold=1e-4)
        else:
            replay[stage]=dict(encoder_loaded=True,query_transform_exact=True,CPU_prediction_finite=True,
                note='Score-row fit predictions have no saved counterpart; only refit is independently replayed.')
        del model,encoded
        gc.collect()
        guard()
    result=dict(status='passed',family=family,fold=fold,source_sha256=common.sha256(__file__),
        manifest_sha256=common.sha256(folder/'manifest.json'),protocol_sha256=rec['protocol_sha256'],
        same225_schema_IDs_receipts_as_leaf63_and_TabM=True,full_original_score_IDs_labels=True,
        missing_V2_exact=True,fit_refit_rows_and_stopping_receipts_match=True,native_categories_and_numeric_transform_verified=True,
        model_replay=replay,query_IDs=query[ID].tolist(),query_ID_hash=common.object_hash(query[ID].tolist()),
        feature_sources=feature_sources,metrics=metric(ref[TARGET].to_numpy(float),frame.prediction_sec.to_numpy(float)),
        comparisons={},peak_rss_bytes=guard(),GPU_used=False,
        limitation='CPU replay covers128 deterministically spaced finite score rows; full score replay is the saved producer receipt. Fit-only encoding verified from hashed source and serialized stage encoders, not exhaustive independent vocabulary reconstruction.')
    for name,path in controls.items():
        marker=markers[name]
        assert common.sha256(path/'candidate.parquet')==marker['outputs']['candidate.parquet']
        control=pd.read_parquet(path/'candidate.parquet')
        np.testing.assert_array_equal(control[ID],ref[ID])
        result['comparisons'][name]=comparison(ref[TARGET].to_numpy(float),control.prediction_sec.to_numpy(float),frame.prediction_sec.to_numpy(float),ref.day.to_numpy())
    common.write_json(destination,result)
    print('FAMILY_AUDIT',family,fold,result['metrics'],replay,'peak',guard(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['catboost','xgb'],required=True)
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    args=parser.parse_args()
    (OUT/args.family).mkdir(parents=True,exist_ok=True)
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    with threadpool_limits(2):
        for fold in args.folds:
            run_fold(args.family,fold,meta)


if __name__=='__main__':
    main()
