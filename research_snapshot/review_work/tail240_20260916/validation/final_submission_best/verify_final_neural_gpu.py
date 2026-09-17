"""Native original-8192-batch GPU replay with approved six-GiB host allowance."""
import torch
import lightgbm
from verify_final_neural import ROOT, OUT, ID, read, sha, inputs, folder_for, tabm_gpu
import argparse
import gc
import json
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits


def main(name):
    target=OUT/f'{name}_native_receipt.json';assert not target.exists()
    checked=OUT/f'{name}_cpu';receipt=read(checked/'receipt.json')
    assert receipt['status']=='passed'
    folder=folder_for(name);marker,protocol,train,ranking=inputs(folder,name)
    assert sha(folder/'manifest.json')==receipt['manifest_sha256']
    assert sha(checked/'ranking_frame.parquet')==receipt['ranking_frame_sha256']
    frame=pd.read_parquet(checked/'ranking_frame.parquet');np.testing.assert_array_equal(frame.index,ranking[ID])
    model=joblib.load(folder/'model.joblib')
    model['estimator'].eval().cuda();torch.cuda.reset_peak_memory_stats()
    predicted=np.empty(len(ranking))
    with torch.inference_mode():
        for start in range(0,len(ranking),8192):
            stop=min(start+8192,len(ranking))
            numeric,categorical=tabm_gpu.tensors(model['encoder'],frame.iloc[start:stop])
            raw=model['estimator'](numeric.cuda(),categorical.cuda()).mean(dim=1).cpu().numpy().astype(float)
            predicted[start:stop]=raw*model['y_scale']+model['y_mean']+ranking.proxy_sec.iloc[start:stop].to_numpy(float)
            memory=psutil.Process().memory_info()
            assert memory.peak_wset<6*1024**3 and psutil.virtual_memory().available>=8*1024**3
    saved=pd.read_parquet(folder/'ranking_predictions.parquet');np.testing.assert_array_equal(saved[ID],ranking[ID])
    np.testing.assert_array_equal(predicted,saved.prediction_sec)
    gpu_peak=torch.cuda.max_memory_allocated();model['estimator'].cpu();del model;gc.collect();torch.cuda.empty_cache()
    result=dict(receipt,cpu_receipt_sha256=sha(checked/'receipt.json'),gpu_verifier_sha256=sha(__file__),
        gpu_used=True,native_max_abs_delta_sec=0.,cpu_threads=2,native_batch_size=8192,
        GPU_batch_policy='Same complete ranking ID sequence,original8192 chunks including remainder; fit encoder applied perchunk without changed arithmetic',
        gpu_peak_bytes=gpu_peak,peak_bytes=psutil.Process().memory_info().peak_wset,
        gpu_live_allocated_after_release=torch.cuda.memory_allocated())
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['ple387','tabm225'],required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):main(args.name)
