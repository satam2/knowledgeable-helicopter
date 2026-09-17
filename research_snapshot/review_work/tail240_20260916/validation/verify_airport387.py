"""Independent airport387 bank replay and current387 ensemble composition."""
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
BASE=ROOT/'private_runs/tail240_20260916/forensics/airport387/tune_v2'
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
    out=ROOT/'private_runs/tail240_20260916/validation'/('airport387_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    marker=v.read_json(folder/'manifest.json')
    protocol=v.read_json(BASE/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert marker['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/airport387/run_v2.py')
    recovery=v.read_json(BASE/'recovery_protocol.json')
    assert recovery['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/airport387/recover_f1_v1.py')
    assert recovery['protocol_sha256']==v.sha256(BASE/'protocol.json')
    original=BASE.parent/'tune_v1'/fold
    assert recovery['original_progress_sha256']==v.sha256(original/'airport_progress.json')
    assert v.read_json(original/'airport_progress.json')==marker['bank']
    assert v.read_json(folder/'recovery_receipt.json')['no_refit'] is True
    for airport,item in marker['bank'].items():
        assert v.sha256(original/item['model_file'])==v.sha256(folder/item['model_file'])==recovery['original_models'][airport]
    for name,digest in marker['outputs'].items():assert v.sha256(folder/name)==digest
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME,'ADEP_mvt',TARGET,'proxy_sec'])
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
    old=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    oldmarker=v.read_json(old/'manifest.json')
    assert v.sha256(old/'manifest.json')==protocol['original_controls'][fold]
    assert oldmarker['feature_columns']==columns and oldmarker['fit']['params']==marker['params']==protocol['params']
    assert v.sha256(old/'tune_predictions.parquet')==oldmarker['outputs']['tune_predictions.parquet']
    control=pd.read_parquet(old/'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID],ids)
    fresh=ROOT/'private_runs/tail240_20260916/models'/('following_groups_tune_v1' if fold=='F1' else 'following_groups_tune_v2')/fold/'control387'
    freshmarker=v.read_json(fresh/'manifest.json')
    assert v.sha256(fresh/'manifest.json')==protocol['fresh_controls'][fold]
    assert v.sha256(fresh/'tune.parquet')==freshmarker['outputs']['tune.parquet']
    np.testing.assert_array_equal(pd.read_parquet(fresh/'tune.parquet').prediction_sec,control.prediction_sec)
    pred=control.prediction_sec.to_numpy().copy()
    assigned=np.zeros(len(tune),bool)
    fa=rows['fit'].ADEP_mvt.astype('string').fillna('__MISSING__').astype(str).to_numpy()
    ta=tune.ADEP_mvt.astype('string').fillna('__MISSING__').astype(str).to_numpy()
    assert sorted(marker['bank'])==sorted(set(fa))
    bankchecks={}
    for number,airport in enumerate(sorted(set(fa))):
        fi=np.flatnonzero(fa==airport);ti=np.flatnonzero(ta==airport)
        item=marker['bank'][airport]
        assert len(fi)==item['fit_n'] and len(ti)==item['tune_n']
        assert v.object_hash(fitids[fi].tolist())==item['fit_id_hash']
        assert v.object_hash(ids[ti].tolist())==item['tune_id_hash']
        params=dict(protocol['params'])
        if not len(ti):params['n_estimators']=freshmarker['steps']
        assert params==item['params'] and item['fixed_global_rounds_no_tune']==(not bool(len(ti)))
        assert item['model_file']==f'airport_{number:02d}.txt'
        assert v.sha256(folder/item['model_file'])==item['model_sha256']
        model=lgb.Booster(model_file=str(folder/item['model_file']))
        assert model.feature_name()==columns
        if len(ti):
            values=model.predict(np.asarray(matrix[ti],dtype='float32',order='C'),num_threads=1)+tune.proxy_sec.to_numpy(float)[ti]
            pred[ti]=values;assigned[ti]=True
        bankchecks[airport]=dict(fit_n=len(fi),tune_n=len(ti),native_replayed=bool(len(ti)))
        del model
        guard()
    np.testing.assert_array_equal(assigned,np.isin(ta,np.unique(fa)))
    assert int((~assigned).sum())==marker['global_fallback_rows']
    saved=pd.read_parquet(folder/'tune.parquet')
    np.testing.assert_array_equal(saved[ID],ids)
    np.testing.assert_array_equal(saved.prediction_sec,pred)
    np.testing.assert_array_equal(saved.global_fallback,~assigned)
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
    neuralroot=ROOT/'private_runs/tail240_20260916/state/neural_context'/('v1' if fold=='F1' else 'v3')/fold
    assert v.sha256(neuralroot/'manifest.json')==protocol['neural_controls'][fold]
    neuralmarker=v.read_json(neuralroot/'manifest.json')
    assert v.sha256(neuralroot/'tune_predictions.parquet')==neuralmarker['outputs']['tune_predictions.parquet']
    neural=pd.read_parquet(neuralroot/'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET],aligned[TARGET])
    baseline+=w['global'][w['experts'].index('tabm_ple8')]*(neural.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    blend=.75*baseline+.25*pred[ordinary]
    ordinarysaved=pd.read_parquet(folder/'ordinary_fixed25.parquet')
    np.testing.assert_array_equal(ordinarysaved[ID],ids[ordinary])
    np.testing.assert_array_equal(ordinarysaved.baseline_sec,baseline)
    np.testing.assert_array_equal(ordinarysaved.prediction_sec,blend)
    comparison=v.paired(y[ordinary],baseline,blend,days[ordinary])
    np.testing.assert_allclose(-comparison['delta_rmse'],marker['results']['ordinary_fixed25']['gain'],rtol=1e-11,atol=1e-10)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),manifest_sha256=v.sha256(folder/'manifest.json'),protocol_sha256=v.sha256(BASE/'protocol.json'),cohorts=marker['ids'],all_native_predictions_exact=True,all_fit_vocabularies_exact=True,all_airport_models_and_routes_exact=True,bank=bankchecks,current387_composition_exact=True,control_matches_original_union_and_global9=True,matched=matched,ordinary_fixed25=comparison,peak_bytes=guard(),limitation='This run rebuilds complete globalfit vocabulary and387tune inputs, then replays every airport model and exact fallback/current387blend. Original control vector hash/parity checked; control model not rerun here.')
    v.write_json(out/'receipt.json',receipt)
    print('VERIFIED',fold,'matched_gain',-matched['delta_rmse'],'fixed25gain',-comparison['delta_rmse'],'peak',receipt['peak_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3'],required=True)
    with threadpool_limits(1):main(parser.parse_args().fold)
