"""Post-hoc saved-model ablation; no fitting or promotion of base-only output."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import torch
import lightgbm
import gc
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import verify_attention_f1 as verification

v,ROOT,ID,TARGET,TIME=verification.v,verification.ROOT,verification.ID,verification.TARGET,verification.TIME
subject=verification.subject
BASE=verification.FOLDER
OUT=ROOT/'private_runs/tail240_20260916/validation/attention_base_diagnostic_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    native=v.read_json(verification.OUT/'receipt.json');assert native['status']=='passed'
    manifest=v.read_json(BASE/'manifest.json')
    assert v.sha256(BASE/'manifest.json')==native['producer_manifest_sha256']
    assert v.sha256(BASE/'fit_model.joblib')==manifest['outputs']['fit_model.joblib']
    saved=pd.read_parquet(BASE/'tune_predictions.parquet')
    assert v.sha256(BASE/'tune_predictions.parquet')==manifest['outputs']['tune_predictions.parquet']
    model=joblib.load(BASE/'fit_model.joblib');encoder=model['encoder']
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',TIME,'proxy_sec']);meta[TARGET]=np.nan
    idx,_,_=v.common.fold_data(meta,'F1',full=True)
    fit=meta.iloc[idx['fit']].loc[lambda x:np.isfinite(x.proxy_sec)]
    fitids=pd.Index(fit[ID]);nfit=len(fit)
    assert v.object_hash(fit[ID].tolist())==manifest['fit_ids_hash']
    columns=encoder.columns
    vocab={name:set() for name in encoder.categories}
    for source in manifest['feature_receipts']:
        selected=[n for n in source['columns'] if n in vocab]
        if not selected:continue
        assert v.sha256(source['path'])==source['sha256']
        for batch in pq.ParquetFile(source['path']).iter_batches(batch_size=8192,columns=[ID,*selected],use_threads=False):
            frame=batch.to_pandas();keep=fitids.get_indexer(frame[ID])>=0
            for name in selected:vocab[name].update(frame.loc[keep,name].dropna().astype(str))
    vocab={name:sorted(words) for name,words in vocab.items()}
    matrix=np.memmap(verification.OUT/'matrix.float32',mode='r',dtype='float32',shape=(nfit+len(saved),len(columns)),order='F')
    xt=subject.context.decode_frame(matrix[nfit:],vocab,columns)
    torch.cuda.reset_peak_memory_stats()
    network=model['estimator'].base.cuda()
    baseline=subject.adapter.frozen.infer(network,subject.adapter.frozen.tensors(encoder,xt),'cuda')*model['y_scale']+model['y_mean']+saved.proxy_sec.to_numpy(float)
    network.cpu();torch.cuda.empty_cache()
    gpupeak=torch.cuda.max_memory_allocated()
    y=saved[TARGET].to_numpy(float);full=saved.prediction_sec.to_numpy(float);control=saved.control387_prediction_sec.to_numpy(float);days=saved[TIME].dt.floor('D').to_numpy()
    reports={}
    for name,mask in [('all_finite',np.ones(len(saved),bool)),('ordinary',saved.proxy_sec.between(0,7200).to_numpy())]:
        reports[name]=dict(original387=v.metrics(y[mask],control[mask]),jointly_trained_base_only=v.metrics(y[mask],baseline[mask]),full_attention=v.metrics(y[mask],full[mask]),full_vs_joint_base=v.paired(y[mask],baseline[mask],full[mask],days[mask]),joint_base_vs_original387=v.paired(y[mask],control[mask],baseline[mask],days[mask]))
    delta=full-baseline
    correction=dict(mean=float(delta.mean()),std=float(delta.std()),quantiles=np.quantile(delta,[0,.01,.1,.5,.9,.99,1]).tolist())
    output=saved[[ID,TARGET,'proxy_sec']].copy()
    output['full_attention_prediction_sec']=full
    output['jointly_trained_base_only_prediction_sec']=baseline
    output['original387_prediction_sec']=control
    output.to_parquet(OUT/'predictions.parquet',index=False)
    receipt=dict(status='complete_diagnostic',source_sha256=v.sha256(__file__),native_receipt_sha256=v.sha256(verification.OUT/'receipt.json'),producer_manifest_sha256=v.sha256(BASE/'manifest.json'),prediction_sha256=v.sha256(OUT/'predictions.parquet'),rows=len(saved),metrics=reports,correction=correction,gpu_peak_bytes=gpupeak,peak_bytes=verification.guard(),no_training=True,no_promotion=True,limitation='Post-hoc base-only inference from jointly trained attention model. It is not a separately trained matched static architecture and cannot causally isolate attention or justify another candidate. Original387 differs in training trajectory; no feature/algorithm change or F3 experiment.')
    v.write_json(OUT/'receipt.json',receipt)
    print({name:{k:d[k]['rmse'] for k in ['original387','jointly_trained_base_only','full_attention']} for name,d in reports.items()},flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):main()
