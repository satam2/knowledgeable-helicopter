"""Lean CPU-only XGB replay; fixed strict threshold remains unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
OUT=ROOT/'private_runs/breakthrough_20260916/missing/combined_family_audit/xgb'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    fixture=OUT/(args.fold+'_fixture')
    receipt=json.loads((fixture/'manifest.json').read_text())
    folder=Path(receipt['model_folder'])
    rec=json.loads((folder/'manifest.json').read_text())
    assert sha(folder/'manifest.json')==receipt['model_manifest_sha256']
    assert sha(folder/'model.joblib')==rec['outputs']['model.joblib']
    for name,digest in receipt['outputs'].items():
        assert sha(fixture/name)==digest
    features=pd.read_parquet(fixture/'features.parquet')
    expected=pd.read_parquet(fixture/'expected.parquet')
    np.testing.assert_array_equal(features.index,expected.MVT_ID_mvt)
    model=joblib.load(folder/'model.joblib')
    model['estimator'].set_params(device='cpu',n_jobs=2)
    encoder=model['encoder']
    transformed=encoder.transform(features)
    assert encoder.columns==rec['feature_columns']
    data={}
    for name in encoder.columns:
        if name in encoder.categories:
            v=features[name].astype('string')
            codes=v.map(encoder.categories[name]).fillna(1).to_numpy(np.int32)
            codes[v.isna().to_numpy()]=0
            data[name]=pd.Categorical(codes,categories=range(len(encoder.categories[name])+2))
        else:
            v=features[name].to_numpy(np.float32,na_value=np.nan).copy()
            v[(v==-999999)|~np.isfinite(v)]=np.nan
            data[name]=v
    pd.testing.assert_frame_equal(transformed,pd.DataFrame(data,index=features.index))
    values=np.asarray(model['estimator'].predict(transformed,iteration_range=(0,model['steps'])),float)+expected.proxy_sec.to_numpy(float)
    repeated=np.asarray(model['estimator'].predict(transformed,iteration_range=(0,model['steps'])),float)+expected.proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(values,repeated)
    delta=values-expected.prediction_sec.to_numpy(float)
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    assert peak<2*1024**3,peak
    y=expected.TAXITIME_SEC_mvt.to_numpy(float)
    result=dict(status='strict_parity_failed' if np.max(np.abs(delta))>1e-4 else 'passed',
        source_sha256=sha(__file__),fixture_manifest_sha256=sha(fixture/'manifest.json'),
        model_manifest_sha256=sha(folder/'manifest.json'),fold=args.fold,rows=len(values),GPU_used=False,
        max_abs_delta=float(np.max(np.abs(delta))),mean_abs_delta=float(np.mean(np.abs(delta))),
        delta_rmse=float(np.sqrt(np.mean(delta**2))),n_over_original_threshold=int(np.sum(np.abs(delta)>1e-4)),
        original_threshold=1e-4,cpu_repeat_delta=0.,native_transform_independently_exact=True,
        saved_sample_rmse=float(np.sqrt(np.mean((expected.prediction_sec.to_numpy()-y)**2))),
        CPU_sample_rmse=float(np.sqrt(np.mean((values-y)**2))),peak_rss_bytes=peak,
        caveat='No waiver,promotion,or exactcrossdeviceclaim. Immutableproducerfullreloadreceipt is separate.')
    output=OUT/(args.fold+'_CPU_parity_lean.json')
    assert not output.exists()
    expected['CPU_prediction']=values
    expected['delta']=delta
    expected.to_parquet(output.with_suffix('.parquet'),index=False)
    result['rows_sha256']=sha(output.with_suffix('.parquet'))
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
