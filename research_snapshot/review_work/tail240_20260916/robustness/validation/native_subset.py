"""Independent CPU source-to-native replay on predeclared spread-out row samples."""
import os
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
if '--family' in sys.argv and sys.argv[sys.argv.index('--family')+1]=='ple':
    import torch
    torch.set_num_threads(1)
from materialize_cohorts import ROOT,ID,sha,read,guard
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BASE=ROOT/'private_runs/tail240_20260916/robustness/chronological_v1'
OUT=ROOT/'private_runs/tail240_20260916/robustness/validation/native_subset'
BASE_CATEGORIES={'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt',
                 'WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','airport_stand','airport_runway'}


def main(fold,family,phase):
    folder=BASE/fold/family/phase;marker=read(folder/'manifest.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==sha(BASE/'protocol.json')
    protocol=read(BASE/'protocol.json')
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest
    checked=read(ROOT/f'private_runs/tail240_20260916/robustness/validation/experts/{fold}_{family}_{phase}.json')
    assert checked['status']=='passed' and checked['expert_manifest_sha256']==sha(folder/'manifest.json')
    samples=[]
    for stage,rec in marker['predictions'].items():
        path=folder/(stage+'.parquet');assert sha(path)==rec['sha256']
        frame=pd.read_parquet(path)
        indices=np.arange(min(32,len(frame))) if family=='ple' else np.unique(np.linspace(0,len(frame)-1,17,dtype=np.int64))
        selected=frame.iloc[indices].copy();selected['stage']=stage;samples.append(selected)
    sample=pd.concat(samples,ignore_index=True);ids=pd.Index(sample[ID]);assert ids.is_unique
    columns=marker['features'];coverage=read(folder/'source_coverage.json')
    values={};seen={name:np.zeros(len(sample),bool) for name in columns};categories=set()
    for source in coverage['sources']:
        path=Path(source['path']);assert sha(path)==source['sha256']
        names=source['columns'];schema=pq.read_schema(path)
        for name in names:
            dtype=schema.field(name).type
            if pa.types.is_dictionary(dtype):dtype=dtype.value_type
            numeric=pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)
            if name not in values:values[name]=np.empty(len(sample),dtype=np.float32 if numeric else object)
            if not numeric:categories.add(name)
        frame=ds.dataset(path,format='parquet').to_table(columns=[ID,*names],filter=ds.field(ID).isin(ids.to_numpy()),use_threads=False).to_pandas()
        positions=ids.get_indexer(frame[ID]);assert (positions>=0).all() and frame[ID].is_unique
        for name in names:
            assert not seen[name][positions].any();seen[name][positions]=True
            if name in categories:
                value=frame[name].astype('string')
                if name not in BASE_CATEGORIES:value=value.fillna('MISSING')
                values[name][positions]=value.to_numpy(na_value=None)
            else:
                value=pd.to_numeric(frame[name]).to_numpy(dtype=np.float32,na_value=np.nan)
                value[~np.isfinite(value)]=np.nan
                if source['fill']:value[np.isnan(value)]=-999999.
                values[name][positions]=value
        guard()
    assert all(mask.all() for mask in seen.values())
    frame=pd.DataFrame({name:pd.Categorical(values[name]) if name in categories else values[name] for name in columns},index=ids)
    sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
    sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
    if family=='ple':
        import tabm_ple_gpu as adapter
        import tabm_gpu as trainer
    else:
        path=ROOT/('review_work/breakthrough_20260916/deeper_lgb/adapter.py' if family=='lgb' else
                   'review_work/breakthrough_20260916/models_retrieval/combined_catboost/adapter.py')
        spec=importlib.util.spec_from_file_location('robustness_'+family,path);adapter=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    assert sha(folder/'model.joblib')==marker['model_sha256']
    model=joblib.load(folder/'model.joblib')
    if family=='lgb':model['estimator'].set_params(n_jobs=1)
    else:model['threads']=1
    if family=='ple':
        assert all(value.device.type=='cpu' for value in model['estimator'].parameters())
        model['estimator'].eval()
        prediction=trainer.infer(model['estimator'],trainer.tensors(model['encoder'],frame),'cpu')*model['y_scale']+model['y_mean']
        prediction+=sample.proxy_sec.to_numpy(float)
    else:prediction=adapter.predict(model,frame)+sample.proxy_sec.to_numpy(float)
    delta=np.abs(prediction-sample.prediction_sec.to_numpy(float))
    assert np.isfinite(prediction).all()
    tolerance=.01 if family=='ple' else 1e-6
    np.testing.assert_allclose(prediction,sample.prediction_sec,rtol=0,atol=tolerance)
    result=dict(status='passed',verifier_sha256=sha(__file__),expert_manifest_sha256=sha(folder/'manifest.json'),
        fold=fold,family=family,phase=phase,rows=len(sample),sample_rule=('First32 rows per stage' if family=='ple' else '17 evenly spread positions per prediction stage')+', fixed before reading prediction values',
        tolerance_sec=tolerance,inference_device='cpu',gpu_used=False,
        max_abs_delta_sec=float(delta.max()),per_stage={stage:dict(rows=int((sample.stage==stage).sum()),
            max_abs_delta_sec=float(delta[sample.stage==stage].max())) for stage in sample.stage.unique()},
        every_sample_feature_rebuilt_from_hash_bound_sources=True,cached_matrix_used=False,
        source_encoder_fitted_again=False,ranking_labels_read=False,any_labels_read=False,fit_used=False,peak_bytes=guard())
    OUT.mkdir(parents=True,exist_ok=True);dest=OUT/f'{fold}_{family}_{phase}.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3'])
    parser.add_argument('--family',required=True,choices=['lgb','catboost','ple']);parser.add_argument('--phase',required=True,choices=['select','refit'])
    args=parser.parse_args();main(args.fold,args.family,args.phase)
