"""Verify the finished normalized component while the independent ET fit runs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import lightgbm
from preflight import ROOT, OUT, ID, read, sha
import json
import joblib
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import encoders


def main():
    target=OUT/'normalized_partial_receipt.json';assert not target.exists()
    root=ROOT/'private_runs/tail240_20260916/forensics/final_missing/v1';folder=root/'models'
    prep=read(OUT/'missing_preflight_receipt.json')
    assert prep['status']=='passed' and sha(root/'protocol.json')==prep['protocol_sha256']
    for name,digest in read(root/'preparation.json')['outputs'].items():assert sha(root/name)==digest
    x=pd.read_parquet(root/'training_features.parquet');xr=pd.read_parquet(root/'ranking_features.parquet')
    encoder=joblib.load(folder/'normalized_encoder.joblib')
    for name,mapping in encoder.categories.items():assert mapping=={value:i+2 for i,value in enumerate(sorted(x[name].astype('string').dropna().unique()))}
    schedule=x.schedule_proxy_sec.to_numpy(float)
    scale=np.sqrt(3600.**2+np.where(np.isfinite(schedule)&(schedule!=-999999),schedule-900.,0.)**2)
    evidence=read(folder/'normalized_fit.json')
    assert evidence['normalizer']==float((scale**2).mean()) and evidence['steps']==111
    model=lightgbm.Booster(model_file=str(folder/'normalized.txt'));assert model.num_trees()==111
    schedule=xr.schedule_proxy_sec.to_numpy(float)
    scale=np.sqrt(3600.**2+np.where(np.isfinite(schedule)&(schedule!=-999999),schedule-900.,0.)**2)
    values=900.+scale*model.predict(encoder.transform(xr),num_threads=1)
    expected=pd.read_parquet(folder/'normalized_ranking.parquet');np.testing.assert_array_equal(expected[ID],xr.index)
    np.testing.assert_array_equal(values,expected.prediction_sec)
    result=dict(status='passed',scope='Normalized component only; ET and complete missing manifest pending',
        protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(__file__),rows=len(xr),training_rows=len(x),steps=111,
        exact_fit_vocabulary_scale_and_native_predictions=True,max_abs_delta_sec=0.,
        outputs={name:sha(folder/name) for name in ['normalized.txt','normalized_encoder.joblib','normalized_ranking.parquet','normalized_fit.json']},
        gpu_used=False,model_fitting_used=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
