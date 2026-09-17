"""Independent normalized finite replay, scales, weights and fit vocabularies."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import lightgbm as lgb
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/models/scaled_finite_tune_v1'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    m=psutil.Process().memory_info()
    peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def main(fold):
    assert psutil.virtual_memory().available>=10*1024**3
    folder=BASE/fold
    out=ROOT/'private_runs/tail240_20260916/validation'/('scaled_finite_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    marker=v.read_json(folder/'manifest.json')
    protocol=v.read_json(BASE/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert marker['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/scaled_finite_tune.py')
    for name,digest in marker['outputs'].items():assert v.sha256(folder/name)==digest
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec'])
    idx,split,_=v.common.fold_data(meta,fold,full=True)
    assert v.object_hash(split)==v.object_hash(marker['split'])
    rows={s:meta.iloc[idx[s]].loc[lambda f:np.isfinite(f.proxy_sec)] for s in ['fit','tune']}
    assert {s:dict(n=len(f),hash=v.object_hash(f[ID].tolist())) for s,f in rows.items()}==marker['ids']
    fitids=pd.Index(rows['fit'][ID]);tune=rows['tune'];ids=pd.Index(tune[ID])
    encoder=v.read_json(folder/'encoder.json');columns=encoder['columns']
    assert columns==protocol['columns'] and len(columns)==387
    matrix=np.full((len(ids),len(columns)),np.nan,dtype='float32',order='F')
    counts=np.zeros(len(columns),dtype=int)
    rebuilt={name:set() for name in encoder['vocab']}
    locations={name:i for i,name in enumerate(columns)}
    for receipt in marker['feature_receipts']:
        path=Path(receipt['path'])
        assert v.sha256(path)==receipt['sha256']
        names=receipt['columns'];seen=np.zeros(len(ids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            f=batch.to_pandas();p=ids.get_indexer(f[ID]);keep=p>=0;positions=p[keep]
            assert not seen[positions].any() and len(np.unique(positions))==len(positions)
            seen[positions]=True
            fit=fitids.get_indexer(f[ID])>=0
            for name in names:
                if name in rebuilt:
                    rebuilt[name].update(f.loc[fit,name].dropna().astype(str).tolist())
                    mapping={word:i+2 for i,word in enumerate(encoder['vocab'][name])}
                    values=f.loc[keep,name]
                    values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(f.loc[keep,name]).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if 'screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix():values[np.isnan(values)]=-999999.
                matrix[positions,locations[name]]=values
        for name in names:counts[locations[name]]+=int(seen.sum())
        guard()
    assert np.all(counts==len(ids))
    assert {name:sorted(words) for name,words in rebuilt.items()}==encoder['vocab']
    gap_name,missing_name='conv_nm_minus_sched_sec','conv_nm_minus_sched_missing'
    fit_clock=np.full((len(fitids),2),np.nan,dtype='float32')
    fit_counts=np.zeros(2,int)
    for receipt in marker['feature_receipts']:
        chosen=[n for n in [gap_name,missing_name] if n in receipt['columns']]
        if not chosen:continue
        for batch in pq.ParquetFile(receipt['path']).iter_batches(batch_size=8192,columns=[ID,*chosen],use_threads=False):
            f=batch.to_pandas();pos=fitids.get_indexer(f[ID]);keep=pos>=0
            for name in chosen:
                j=[gap_name,missing_name].index(name)
                fit_clock[pos[keep],j]=pd.to_numeric(f.loc[keep,name]).to_numpy('float32',na_value=np.nan)
                fit_counts[j]+=int(keep.sum())
    assert np.all(fit_counts==len(fitids))
    def scale(gap,missing):
        gap=np.asarray(gap,float);missing=np.asarray(missing,float)
        return np.hypot(900.,np.where(np.isfinite(gap)&(gap!=-999999.)&(missing==0),gap,0.))
    scales={'fit':scale(fit_clock[:,0],fit_clock[:,1]),'tune':scale(matrix[:,columns.index(gap_name)],matrix[:,columns.index(missing_name)])}
    normalizer=float(np.mean(scales['fit']**2))
    assert normalizer==marker['normalizer']
    for stage,s in scales.items():
        w=s**2/normalizer
        diag=marker['scale_diagnostics'][stage]
        np.testing.assert_array_equal(np.quantile(s,[0,.5,.9,.99,.999,1]),diag['scale_quantiles'])
        np.testing.assert_allclose(w.sum()**2/(w@w),diag['weight_ess'],rtol=1e-13)
        np.testing.assert_allclose(w.max()/w.sum(),diag['max_weight_share'],rtol=1e-13)
    model=lgb.Booster(model_file=str(folder/'model.txt'))
    assert model.feature_name()==columns
    z=model.predict(matrix,num_threads=1)
    pred=tune.proxy_sec.to_numpy(float)+scales['tune']*z
    raw_y=tune[TARGET].to_numpy(float)
    transformed=(raw_y-tune.proxy_sec.to_numpy(float))/scales['tune']
    np.testing.assert_allclose(scales['tune']**2/normalizer*(z-transformed)**2,(pred-raw_y)**2/normalizer,rtol=1e-8,atol=1e-6)
    saved=pd.read_parquet(folder/'tune.parquet')
    np.testing.assert_array_equal(saved[ID],ids)
    np.testing.assert_array_equal(saved.prediction_sec,pred)
    np.testing.assert_array_equal(saved.scale,scales['tune'])
    old=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    oldmarker=v.read_json(old/'manifest.json')
    assert oldmarker['feature_columns']==columns[:387] and oldmarker['fit']['params']==protocol['params']
    assert v.sha256(old/'tune_predictions.parquet')==oldmarker['outputs']['tune_predictions.parquet']
    control=pd.read_parquet(old/'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID],ids)
    y=tune[TARGET].to_numpy(float);days=tune[TIME].dt.floor('D').to_numpy()
    matched=v.paired(y,control.prediction_sec.to_numpy(),pred,days)
    ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    ensemble=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=v.read_json(ensemble/'preparation.json')['folds'][fold]
    alignedpath=ensemble/f'{fold}_aligned_tune.parquet'
    assert v.sha256(alignedpath)==prep['aligned_tune_sha256']
    w=v.read_json(ensemble/f'{fold}_weights.json')
    assert w==prep['weights']
    aligned=pd.read_parquet(alignedpath)
    np.testing.assert_array_equal(aligned[ID],ids[ordinary])
    np.testing.assert_array_equal(aligned.lgb63_union,control.prediction_sec.to_numpy()[ordinary])
    baseline=aligned[w['experts']].to_numpy()@np.asarray(w['global'])
    blend=.75*baseline+.25*pred[ordinary]
    comparison=v.paired(y[ordinary],baseline,blend,days[ordinary])
    np.testing.assert_allclose(-comparison['delta_rmse'],marker['ordinary_fixed25']['reference_rmse']-marker['ordinary_fixed25']['rmse'],rtol=1e-11,atol=1e-10)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),manifest_sha256=v.sha256(folder/'manifest.json'),protocol_sha256=v.sha256(BASE/'protocol.json'),cohorts=marker['ids'],all_native_predictions_exact=True,all_fit_vocabularies_exact=True,all_fit_tune_scales_and_weight_diagnostics_exact=True,raw_loss_identity_verified=True,control_matches_original_union_and_global9=True,matched=matched,ordinary_fixed25=comparison,peak_bytes=guard(),limitation='All original387 fields, fit category maps and fit/tune scale diagnostics independently reconstructed. Original control vector hash/parity checked; control model not rerun here.')
    v.write_json(out/'receipt.json',receipt)
    print('VERIFIED',fold,'matched_gain',-matched['delta_rmse'],'fixed25gain',-comparison['delta_rmse'],'peak',receipt['peak_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3'],required=True)
    with threadpool_limits(1):main(parser.parse_args().fold)
