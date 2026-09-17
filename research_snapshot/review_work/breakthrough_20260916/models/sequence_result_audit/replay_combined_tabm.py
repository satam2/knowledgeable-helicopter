"""Independent CPU replay on bounded raw feature slices from both score months."""
import torch
import lightgbm
import sys
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import psutil
from threadpoolctl import threadpool_limits
import prepare_fit_canary as fixture
import audit

ROOT=fixture.ROOT
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import tabm_gpu


def main():
    torch.set_num_threads(2)
    results={}
    base=ROOT/'private_runs/breakthrough_20260916'
    for fold,month,nextmonth in [('F1','07','08'),('F3','11','12')]:
        folder=base/'full_neural'/f'tabm_combined_standard_aobt_allfinite_{fold}_s20260916'
        record=audit.checked(folder)
        control=audit.checked(base/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916')
        assert record['feature_columns']==control['feature_columns']
        assert record['fit_ids']==control['fit_ids']
        assert record['anchor']['feature_receipts']==control['anchor']['feature_receipts']
        desired=record['feature_columns']
        saved=pd.read_parquet(folder/'candidate.parquet')
        finite=np.flatnonzero(np.isfinite(saved.proxy_sec.to_numpy()))
        positions=finite[np.linspace(0,len(finite)-1,128,dtype=int)]
        subset=saved.iloc[positions].set_index(audit.ID)
        ids=subset.index
        path=ROOT/f'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-{month}-01_2025-{nextmonth}-01.parquet'
        assert fixture.sha256(path)==fixture.read_json(path.with_suffix('.json'))['sha256']
        columns=[name for name in pq.read_schema(path).names if name!=audit.ID]
        x=fixture.selected_frame(path,columns,ids)
        for receipt in record['anchor']['feature_receipts']:
            root=Path(receipt.get('path',receipt.get('manifest'))).parent
            filename='features.parquet' if receipt.get('block')=='conventions' else 'training_features.parquet'
            path=root/filename
            source=fixture.read_json(root/'manifest.json')
            expected=source['feature_sha256'] if filename=='features.parquet' else source['outputs'][filename]
            assert fixture.sha256(path)==expected
            columns=[name for name in desired if name in pq.read_schema(path).names and name not in x]
            extra=fixture.selected_frame(path,columns,ids)
            for column in columns:
                if pd.api.types.is_numeric_dtype(extra[column]):
                    extra[column]=extra[column].replace([np.inf,-np.inf],np.nan).fillna(-999999).astype('float32')
                else:
                    extra[column]=extra[column].astype('string').fillna('MISSING').astype('category')
            x=pd.concat([x,extra],axis=1)
        x=x[desired]
        model=joblib.load(folder/'model.joblib')
        assert next(model['estimator'].parameters()).device.type=='cpu'
        value=tabm_gpu.infer(model['estimator'],tabm_gpu.tensors(model['encoder'],x),'cpu',batch_size=128)
        prediction=value*model['y_scale']+model['y_mean']+subset.proxy_sec.to_numpy()
        delta=np.abs(prediction-subset.prediction_sec.to_numpy())
        assert np.isfinite(prediction).all() and delta.max()<.05
        ref,_=audit.common.reference(fold)
        missing=~np.isfinite(saved.proxy_sec.to_numpy())
        np.testing.assert_array_equal(saved.prediction_sec.to_numpy()[missing],ref.prediction_sec.to_numpy()[missing])
        result={'status':'passed','replay_rows':len(ids),'id_hash':fixture.object_hash(ids.tolist()),
                'cpu_gpu_max_abs_delta_sec':float(delta.max()),'cpu_gpu_mean_abs_delta_sec':float(delta.mean()),
                'declared_cross_device_tolerance_sec':.05,'original_full_gpu_saved_replay_delta':record['reload_max_abs_delta'],
                'selected_epochs':record['fit']['steps'],'versus_leaf63':audit.compare(pd.read_parquet(base/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'/'candidate.parquet'),saved),
                'manifest_sha256':fixture.sha256(folder/'manifest.json'),'model_sha256':fixture.sha256(folder/'model.joblib')}
        results[fold]=result
        print('COMBINED_TABM_CPU_REPLAY',fold,result['replay_rows'],result['cpu_gpu_max_abs_delta_sec'],flush=True)
        del model,x,extra,saved
        gc.collect()
    fixture.write_json(audit.OUT/'combined_tabm_replay.json',{'source_sha256':fixture.sha256(__file__),'folds':results,
        'cuda_initialized':torch.cuda.is_initialized(),'peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
        'scope':'Only128existing scorepredictions per fold independently replayedonCPU; no fitting or GPU; fullscore hash/cohort/replayreceipts alsoverified'})
    assert not torch.cuda.is_initialized()


if __name__=='__main__':
    with threadpool_limits(2):
        main()
