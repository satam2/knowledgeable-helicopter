"""Independent hash sampling, selected vocabularies, matrices and saved models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import lightgbm as lgb
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT,read,sha,write,guard,object_hash
import validate_candidate as v
from verify_linear_finite import check_result

BASE=ROOT/'private_runs/tail240_20260916/state/learning_curve_design/v3'
ID,TIME,TARGET='MVT_ID_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt'


def main():
    folder=BASE/'F1';marker=read(folder/'manifest.json');protocol=read(BASE/'protocol.json');selection=read(BASE/'selection.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==selection['protocol_sha256']==sha(BASE/'protocol.json')
    assert marker['selection_sha256']==sha(BASE/'selection.json') and selection['outcome_columns_read']==[]
    assert sha(BASE/'selected_ids.parquet')==selection['selected_ids_file_sha256']
    for relative,digest in protocol['sources'].items():assert sha(ROOT/relative)==digest
    for name,digest in marker['outputs'].items():assert sha(folder/name)==digest
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet';assert sha(metadata)==protocol['metadata_sha256']
    parts=[]
    for batch in pq.ParquetFile(metadata).iter_batches(batch_size=8192,columns=[ID,TIME,'ADEP_mvt','proxy_sec'],use_threads=False):
        frame=batch.to_pandas();parts.append(frame.loc[frame[TIME].ge(pd.Timestamp('2025-01-01',tz='UTC'))&frame[TIME].lt(pd.Timestamp('2025-07-01',tz='UTC'))&np.isfinite(frame.proxy_sec)])
    meta=pd.concat(parts,ignore_index=True);fit=meta.loc[meta[TIME].lt(pd.Timestamp('2025-06-01',tz='UTC'))].reset_index(drop=True)
    tune=meta.loc[meta[TIME].ge(pd.Timestamp('2025-06-01',tz='UTC'))].reset_index(drop=True)
    assert len(fit)==816060 and len(tune)==180640
    assert object_hash(fit[ID].tolist())==selection['full_fit_ids_hash'] and object_hash(tune[ID].tolist())==selection['tune_ids_hash']
    assert not any(marker['split']['purged_related_departures'].values())
    ids=fit[ID].to_numpy();assert np.isfinite(ids).all() and np.equal(ids,np.floor(ids)).all() and len(np.unique(ids))==len(ids)
    digests=np.asarray([hashlib.sha256(('prc2026-uniform-half-v1|20260916|'+str(int(value))).encode('ascii')).hexdigest() for value in ids])
    ranking=np.lexsort((ids,digests));selected=np.sort(ranking[:408030]);chosen=fit.iloc[selected]
    np.testing.assert_array_equal(chosen[ID],pd.read_parquet(BASE/'selected_ids.parquet')[ID])
    assert object_hash(chosen[ID].tolist())==selection['selected_fit_ids_hash']
    assert chosen[TIME].dt.strftime('%Y-%m').value_counts().sort_index().to_dict()==selection['by_month']
    assert chosen.ADEP_mvt.value_counts().sort_index().to_dict()==selection['by_airport']
    del parts,meta,digests,ranking
    gc.collect()
    fullids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True));halfids=pd.Index(pd.concat([chosen[ID],tune[ID]],ignore_index=True))
    nfit=len(fit);nhalf=len(chosen)
    native_folder=ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    assert sha(native_folder/'manifest.json')==protocol['native_manifest_sha256']
    native_marker=read(native_folder/'manifest.json')
    for name in ('model.txt','encoder.json'):assert sha(native_folder/name)==native_marker['outputs'][name]
    fullenc=read(native_folder/'encoder.json');halfenc=read(folder/'encoder.json');cols=protocol['columns']
    assert fullenc['columns']==halfenc['columns']==cols
    prior=read(ROOT/'private_runs/tail240_20260916/validation/union387_source_audit_v1/protocol.json')
    assert marker['feature_receipts']==prior['sources']
    full_vocab,half_vocab=fullenc['vocab'],halfenc['vocab'];known={name:set() for name in half_vocab}
    tune_full=np.empty((len(tune),len(cols)),dtype='float32',order='F');tune_half=np.empty_like(tune_full,order='F')
    counts_full={name:0 for name in cols};counts_half={name:0 for name in cols}
    for receipt in marker['feature_receipts']:
        path=Path(receipt['path']);assert sha(path)==receipt['sha256'];names=receipt['columns']
        fill='screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        full=np.memmap(folder/'matrix.float32',mode='r',dtype='float32',shape=(len(fullids),len(cols)),order='F')
        half=np.memmap(folder/'half_matrix.float32',mode='r',dtype='float32',shape=(len(halfids),len(cols)),order='F')
        seen=np.zeros(len(fullids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas();positions=fullids.get_indexer(frame[ID]);take=positions>=0;positions=positions[take];frame=frame.loc[take]
            assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
            hp=halfids.get_indexer(frame[ID]);hk=hp>=0
            for name in names:
                j=cols.index(name);values=frame[name]
                if name in full_vocab:
                    known[name].update(values.iloc[np.flatnonzero((hp>=0)&(hp<nhalf))].dropna().astype(str).tolist())
                    raw=values.astype('string');mapping={value:i+2 for i,value in enumerate(full_vocab[name])}
                    candidate_map={value:i+2 for i,value in enumerate(half_vocab[name])}
                    fv=raw.map(mapping).fillna(1).where(raw.notna(),0).to_numpy('float32')
                    hv=raw.map(candidate_map).fillna(1).where(raw.notna(),0).to_numpy('float32')
                else:
                    fv=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan);fv[~np.isfinite(fv)]=np.nan
                    if fill:fv[np.isnan(fv)]=-999999.
                    hv=fv
                np.testing.assert_array_equal(fv,full[positions,j]);np.testing.assert_array_equal(hv[hk],half[hp[hk],j])
                tunes=positions>=nfit;tune_full[positions[tunes]-nfit,j]=fv[tunes];tune_half[positions[tunes]-nfit,j]=hv[tunes]
                counts_full[name]+=len(positions);counts_half[name]+=int(hk.sum())
            assert guard()<4*1024**3
        assert int(seen.sum())==receipt['rows']
        del full,half
        gc.collect()
    assert all(c==len(fullids) for c in counts_full.values()) and all(c==len(halfids) for c in counts_half.values())
    assert {name:sorted(values) for name,values in known.items()}==half_vocab
    label_ids=pd.concat([chosen[ID],tune[ID]],ignore_index=True).to_numpy()
    labels=pq.read_table(metadata,columns=[ID,TARGET],filters=[(ID,'in',label_ids)],use_threads=False).to_pandas().set_index(ID)[TARGET]
    yfit=labels.loc[chosen[ID]].to_numpy(float)-chosen.proxy_sec.to_numpy(float);y=labels.loc[tune[ID]].to_numpy(float)
    assert object_hash(yfit.tolist())==marker['selected_fit_label_hash'] and object_hash(y.tolist())==marker['tune_label_hash']
    baseline=lgb.Booster(model_file=str(native_folder/'model.txt')).predict(tune_full,num_threads=1)+tune.proxy_sec.to_numpy(float)
    candidate_model=lgb.Booster(model_file=str(folder/'model.txt'));assert candidate_model.current_iteration()==marker['trees']==2001
    assert candidate_model.feature_name()==cols and marker['parameters']==protocol['parameters']
    prediction=candidate_model.predict(tune_half,num_threads=1)+tune.proxy_sec.to_numpy(float)
    control=ROOT/'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916'
    assert sha(control/'manifest.json')==protocol['control_manifest_sha256']
    assert sha(control/'tune_predictions.parquet')==read(control/'manifest.json')['outputs']['tune_predictions.parquet']
    savedfull=pd.read_parquet(control/'tune_predictions.parquet');savedhalf=pd.read_parquet(folder/'tune.parquet')
    np.testing.assert_array_equal(savedfull[ID],tune[ID]);np.testing.assert_array_equal(savedhalf[ID],tune[ID])
    np.testing.assert_array_equal(savedfull.prediction_sec,baseline);np.testing.assert_array_equal(savedhalf.prediction_sec,prediction)
    ordinary=tune.proxy_sec.between(0,7200).to_numpy();days=tune[TIME].dt.floor('D').to_numpy()
    metrics=dict(all_finite=check_result(y,prediction,baseline,days,marker['results']['all_finite']),
        ordinary=check_result(y[ordinary],prediction[ordinary],baseline[ordinary],days[ordinary],marker['results']['ordinary']))
    out=ROOT/'private_runs/tail240_20260916/validation/half_curve_F1_v3';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',source_sha256=sha(Path(__file__)),manifest_sha256=sha(folder/'manifest.json'),
        protocol_sha256=sha(BASE/'protocol.json'),selection_sha256=sha(BASE/'selection.json'),
        independently_recomputed_hash_membership_exact=True,all_full_and_half_fit_tune_cells_exact=True,
        selected_only15_vocabularies_exact=True,original_labels_and_cohorts_exact=True,
        full_native_max_abs_delta=0.,half_native_max_abs_delta=0.,metrics=metrics,peak_bytes=guard(),no_GPU_or_fit=True,
        limitation='Singleuniformhalfversusfull,F1only,capacity2001chosenonpreviouslyexposedJune. No all-yearforecast or evidence for a third sample size.')
    v.write_json(out/'receipt.json',result);print('HALF_CURVE_VERIFIED',marker['results'],'peak',result['peak_bytes'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
