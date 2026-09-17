"""Independently reconstruct full fit/refit categorical vocabularies in batches."""
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
import pyarrow.parquet as pq
import psutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import common
BASE=ROOT/'private_runs/breakthrough_20260916'
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def peak():
    info=psutil.Process().memory_info()
    value=getattr(info,'peak_wset',info.rss)
    assert value<2*1024**3,value
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['catboost','xgb'],required=True)
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    folder=(BASE/'combined_catboost'/f'catboost_combined_aobt_allfinite_{args.fold}_s20260916' if args.family=='catboost'
        else BASE/'combined_xgb'/f'xgb_aobt_allfinite_{args.fold}_s20260916')
    rec=common.read_json(folder/'manifest.json')
    assert rec['status']=='complete'
    out=BASE/'missing/combined_family_audit'/args.family/(args.fold+'_vocabulary.json')
    assert not out.exists()
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    idx,split,_=common.fold_data(meta,args.fold,full=True)
    assert split==rec['split']
    finite=np.isfinite(meta.proxy_sec.to_numpy(float))
    reports={}
    inputs={}
    for stage,filename in [('fit','fit_model.joblib'),('refit','model.joblib')]:
        assert common.sha256(folder/filename)==rec['outputs'][filename]
        model=joblib.load(folder/filename)
        encoder=model['encoder']
        positions=idx[stage][finite[idx[stage]]]
        ids=meta.iloc[positions][ID].to_numpy()
        assert rec['fit_ids'][stage]==dict(n=len(ids),hash=common.object_hash(ids.tolist()))
        allowed=set(ids.tolist())
        vocabulary={name:set() for name in encoder.categories}
        base_columns=[name for name in vocabulary if not name.startswith('conv_')]
        conv_columns=[name for name in vocabulary if name.startswith('conv_')]
        files=[(p,base_columns,False) for p in sorted((ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418').glob('training_*.parquet'))]
        files.append((BASE/'missing/source_conventions/features.parquet',conv_columns,True))
        counts={'base':0,'conventions':0}
        for path,columns,fill in files:
            if not columns:
                continue
            expected=(common.read_json(path.parent/'manifest.json')['feature_sha256'] if fill
                else common.read_json(path.with_suffix('.json'))['sha256'])
            if str(path) not in inputs:
                assert common.sha256(path)==expected
                inputs[str(path)]=expected
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16384,columns=[ID,*columns],use_threads=False):
                frame=batch.to_pandas()
                mask=frame[ID].map(allowed.__contains__).to_numpy(bool)
                counts['conventions' if fill else 'base']+=int(mask.sum())
                if mask.any():
                    for name in columns:
                        values=frame.loc[mask,name].astype('string')
                        if fill:
                            values=values.fillna('MISSING')
                        vocabulary[name].update(values.dropna().unique().tolist())
                peak()
        assert counts=={'base':len(ids),'conventions':len(ids)},counts
        for name,values in vocabulary.items():
            expected={value:i+2 for i,value in enumerate(sorted(values))}
            assert expected==encoder.categories[name],(stage,name)
        reports[stage]=dict(rows=len(ids),all_category_maps_exact=True,
            category_sizes={name:len(values) for name,values in vocabulary.items()},cohort_ID_hash=common.object_hash(ids.tolist()))
        print('VOCABULARY',args.family,args.fold,stage,len(ids),'exact','peak',peak(),flush=True)
        del model,encoder,allowed,vocabulary,frame
        gc.collect()
    common.write_json(out,dict(status='passed',source_sha256=common.sha256(__file__),
        manifest_sha256=common.sha256(folder/'manifest.json'),family=args.family,fold=args.fold,
        stages=reports,feature_source_hashes=inputs,peak_rss_bytes=peak(),GPU_used=False,
        scope='Every serialized category map independently reconstructed from all and only original eligible fit/refit IDs, using batch categorical-only reads. Numeric values and models not fitted.'))


if __name__=='__main__':
    main()
