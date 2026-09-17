"""Diagnose CPU/GPU prediction parity without changing the frozen strict gate."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('combined_family_frozen',Path(__file__).with_name('audit.py'))
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
import argparse
import gc
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    folder=frozen.BASE/'combined_xgb'/f'xgb_aobt_allfinite_{args.fold}_s20260916'
    rec=frozen.checked(folder)
    out=frozen.OUT/'xgb'/(args.fold+'_CPU_parity.json')
    assert not out.exists()
    reference,_=frozen.common.reference(args.fold)
    frame=pd.read_parquet(folder/'candidate.parquet')
    np.testing.assert_array_equal(frame[frozen.ID],reference[frozen.ID])
    np.testing.assert_array_equal(frame[frozen.TARGET],reference[frozen.TARGET])
    finite=np.flatnonzero(np.isfinite(reference.proxy_sec.to_numpy(float)))
    chosen=finite[np.linspace(0,len(finite)-1,128,dtype=int)]
    query=reference.iloc[chosen]
    features,sources=frozen.query_features(pd.Index(query[frozen.ID]),query[frozen.TIME],rec['feature_columns'])
    model=joblib.load(folder/'model.joblib')
    model['estimator'].set_params(device='cpu',n_jobs=2)
    encoded=frozen.independent_transform(model['encoder'],features)
    values=np.asarray(model['estimator'].predict(encoded,iteration_range=(0,model['steps'])),float)+query.proxy_sec.to_numpy(float)
    repeated=np.asarray(model['estimator'].predict(encoded,iteration_range=(0,model['steps'])),float)+query.proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(values,repeated)
    saved=frame.iloc[chosen].prediction_sec.to_numpy(float)
    delta=values-saved
    y=query[frozen.TARGET].to_numpy(float)
    result=dict(status='strict_parity_failed' if np.max(np.abs(delta))>1e-4 else 'passed',
        source_sha256=frozen.common.sha256(__file__),helper_source_sha256=frozen.common.sha256(Path(__file__).with_name('audit.py')),
        manifest_sha256=frozen.common.sha256(folder/'manifest.json'),fold=args.fold,
        query_ID_hash=frozen.common.object_hash(query[frozen.ID].tolist()),rows=len(chosen),
        max_abs_delta=float(np.max(np.abs(delta))),mean_abs_delta=float(np.mean(np.abs(delta))),
        delta_rmse=float(np.sqrt(np.mean(delta**2))),signed_mean_delta=float(np.mean(delta)),
        n_over_original_threshold=int(np.sum(np.abs(delta)>1e-4)),original_threshold=1e-4,
        cpu_repeat_max_abs_delta=0.,saved_sample_rmse=float(np.sqrt(np.mean((saved-y)**2))),
        CPU_sample_rmse=float(np.sqrt(np.mean((values-y)**2))),peak_rss_bytes=frozen.guard(),
        feature_sources=sources,GPU_used=False,
        limitation='No tolerance waiver or promotion. Same128spacedfinitequerycanary as frozen audit; CPU/GPU parity discrepancy explicit. Model artifact unchanged.')
    details=query[[frozen.ID,frozen.TARGET,'proxy_sec']].copy()
    details['saved_GPU_prediction']=saved
    details['CPU_prediction']=values
    details['delta']=delta
    detail=out.with_suffix('.parquet')
    details.to_parquet(detail,index=False)
    result['details_sha256']=frozen.common.sha256(detail)
    frozen.common.write_json(out,result)
    print('XGB_CPU_PARITY',result,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
