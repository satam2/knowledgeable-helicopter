"""Independent full finite state399 replay with fit vocabulary reconstruction."""
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
BASE=ROOT/'private_runs/tail240_20260916/models/ordinary_state_tune_v1'
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
    out=ROOT/'private_runs/tail240_20260916/validation'/('ordinary_state_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    marker=v.read_json(folder/'manifest.json')
    protocol=v.read_json(BASE/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert marker['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/ordinary_state_tune.py')
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
    assert columns==protocol['columns'] and len(columns)==399
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
    state_root=ROOT/'private_runs/tail240_20260916/state/cache_v1'
    assert v.sha256(state_root/'manifest.json')==protocol['cache_manifest_sha256']
    cache=v.read_json(state_root/'manifest.json')
    assert v.object_hash(cache['folds'][fold]['split'])==v.object_hash(split)
    state=pd.read_parquet(state_root/f'{fold}_tune.parquet').set_index(ID).loc[ids]
    assert list(state)==columns[-12:]
    expected=state.to_numpy('float32');expected[~np.isfinite(expected)]=-999999.
    np.testing.assert_array_equal(matrix[:,-12:],expected)
    model=lgb.Booster(model_file=str(folder/'model.txt'))
    assert model.feature_name()==columns
    pred=model.predict(matrix,num_threads=1)+tune.proxy_sec.to_numpy(float)
    saved=pd.read_parquet(folder/'tune.parquet')
    np.testing.assert_array_equal(saved[ID],ids)
    np.testing.assert_array_equal(saved.prediction_sec,pred)
    old=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    oldmarker=v.read_json(old/'manifest.json')
    assert oldmarker['feature_columns']==columns[:387] and oldmarker['fit']['params']==marker['params']==protocol['params']
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
    np.testing.assert_allclose(-comparison['delta_rmse'],marker['ordinary_fixed25']['gain'],rtol=1e-11,atol=1e-10)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),manifest_sha256=v.sha256(folder/'manifest.json'),protocol_sha256=v.sha256(BASE/'protocol.json'),cohorts=marker['ids'],all_native_predictions_exact=True,all_fit_vocabularies_exact=True,all_twelve_state_features_exact=True,control_matches_original_union_and_global9=True,matched=matched,ordinary_fixed25=comparison,peak_bytes=guard(),limitation='State cache chronology independently audited previously; this run rebuilds exact stage joins, fit vocabularies and native candidate predictions. Original control vector hash/parity checked; control model not rerun here.')
    v.write_json(out/'receipt.json',receipt)
    print('VERIFIED',fold,'matched_gain',-matched['delta_rmse'],'fixed25gain',-comparison['delta_rmse'],'peak',receipt['peak_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3'],required=True)
    with threadpool_limits(1):main(parser.parse_args().fold)
