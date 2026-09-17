"""Independent full387 input/category reconstruction and CPU-only old CB225 replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm
import catboost
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
from encoders import FrameEncoder
BASE=ROOT/'private_runs/tail240_20260916/models/catboost_union387_v2'
OUT=ROOT/'private_runs/tail240_20260916/validation/catboost387_inputs_F1_v1'
PROTOCOL='3066f9f0ec71ff2ecb53d99b8b7ce199b3f46dd148a38086c3085eb9873bd647'
BASE_CATEGORIES={'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt',
    'AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','airport_stand','airport_runway'}


def guard():
    info=psutil.Process().memory_info();peak=max(info.rss,info.peak_wset)
    assert peak<10*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def decode(matrix,columns,vocab):
    values={}
    for j,name in enumerate(columns):
        if name in vocab:
            missing=np.nan if name in BASE_CATEGORIES else 'MISSING'
            labels=np.asarray([missing,'__UNSEEN_CONTEXT_VALUE__',*vocab[name]],dtype=object)
            values[name]=pd.Categorical(labels[matrix[:,j].astype(np.int32)])
        else:values[name]=matrix[:,j]
    return pd.DataFrame(values,copy=False)


def main():
    assert psutil.virtual_memory().available>=18*1024**3
    OUT.mkdir(parents=True,exist_ok=False)
    assert v.sha256(BASE/'protocol.json')==PROTOCOL
    protocol=v.read_json(BASE/'protocol.json')
    for path,digest in protocol['source_hashes'].items():assert v.sha256(ROOT/path)==digest
    assert protocol['preserved_v1_protocol_sha256']==v.sha256(BASE.parent/'catboost_union387_v1/protocol.json')
    binding=protocol['controls']['F1'];columns=protocol['columns'];assert len(columns)==387
    oldroot=ROOT/'private_runs/breakthrough_20260916/combined_catboost/catboost_combined_aobt_allfinite_F1_s20260916'
    assert v.sha256(oldroot/'manifest.json')==binding['manifest_sha256']
    oldmarker=v.read_json(oldroot/'manifest.json');assert oldmarker['feature_columns']==columns[:225]
    assert oldmarker['fit']['params']==protocol['parameters']
    for name in ['fit_model.joblib','tune_predictions.parquet']:assert v.sha256(oldroot/name)==binding['outputs'][name]
    original=ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    originalmarker=v.read_json(original/'manifest.json');receipts=originalmarker['feature_receipts']
    originalencoder=v.read_json(original/'encoder.json');vocab=originalencoder['vocab'];assert originalencoder['columns']==columns
    linear_audit=ROOT/'private_runs/tail240_20260916/validation/linear_finite_F1_v1/receipt.json'
    linear=v.read_json(linear_audit);assert linear['status']=='passed'
    linear_manifest=ROOT/'private_runs/tail240_20260916/models/linear_finite_tune_v1/F1/manifest.json'
    assert linear['producer_manifest_sha256']==v.sha256(linear_manifest)
    linear_arm=linear_manifest.parent/'constant/manifest.json'
    assert v.read_json(linear_manifest)['outputs']['constant/manifest.json']==v.sha256(linear_arm)
    assert v.read_json(linear_arm)['feature_receipts']==receipts
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec'])
    idx,split,_=v.common.fold_data(meta,'F1',full=True)
    assert v.object_hash(split)==v.object_hash(oldmarker['split'])
    rows={s:meta.iloc[idx[s]].loc[lambda f:np.isfinite(f.proxy_sec)].copy() for s in ['fit','tune']}
    fit,tune=rows['fit'],rows['tune'];nfit=len(fit)
    neuralroot=ROOT/'private_runs/tail240_20260916/state/neural_context/v1/F1'
    assert v.sha256(neuralroot/'manifest.json')==binding['neural_manifest_sha256']
    neural=v.read_json(neuralroot/'manifest.json')
    for stage,frame in rows.items():
        assert dict(n=len(frame),hash=v.object_hash(frame[ID].tolist()))==binding['fit_ids'][stage]
        assert v.object_hash(frame[TARGET].tolist())==neural[stage+'_label_hash']
        assert np.isfinite(frame[TARGET]).all()
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True));assert ids.is_unique
    del meta,idx,rows;gc.collect();guard()
    matrix=np.memmap(OUT/'matrix.float32',mode='w+',dtype='float32',shape=(len(ids),len(columns)),order='F')
    counts=np.zeros(len(columns),int);rebuilt={name:set() for name in vocab};flags=[]
    for receipt in receipts:
        path=Path(receipt['path']);assert v.sha256(path)==receipt['sha256']
        fill='screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        names=receipt['columns'];seen=np.zeros(len(ids),bool);flags.append(dict(path=str(path),fill=fill))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas();pos=ids.get_indexer(frame[ID]);keep=pos>=0;where=pos[keep]
            assert not seen[where].any() and len(np.unique(where))==len(where)
            seen[where]=True;fitmask=(pos>=0)&(pos<nfit)
            for name in names:
                values=frame.loc[keep,name]
                if name in vocab:
                    rebuilt[name].update(frame.loc[fitmask,name].dropna().astype(str).tolist())
                    mapping={word:i+2 for i,word in enumerate(vocab[name])}
                    values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if fill:values[np.isnan(values)]=-999999.
                matrix[where,columns.index(name)]=values
        assert int(seen.sum())==receipt['rows']
        for name in names:counts[columns.index(name)]+=int(seen.sum())
        print('SOURCE',path.parent.name,len(names),guard(),flush=True)
    assert np.all(counts==len(ids)) and {name:sorted(values) for name,values in rebuilt.items()}==vocab
    matrix.flush()
    xf=decode(matrix[:nfit],columns,vocab)
    encoder=FrameEncoder().fit(xf)
    assert encoder.columns==columns and len(encoder.numeric)==372 and len(encoder.categories)==15
    for name,mapping in encoder.categories.items():
        expected={word:i+2 for i,word in enumerate(sorted(xf[name].astype('string').dropna().unique()))}
        assert mapping==expected
    joblib.dump(encoder,OUT/'encoder387.joblib')
    old=joblib.load(oldroot/'fit_model.joblib')
    oldencoder=old['encoder'];assert oldencoder.columns==columns[:225]
    assert oldencoder.numeric==[name for name in encoder.numeric if name in columns[:225]]
    assert oldencoder.categories=={name:mapping for name,mapping in encoder.categories.items() if name in columns[:225]}
    del xf;gc.collect();guard()
    xt=decode(matrix[nfit:],columns,vocab)
    encoded=encoder.transform(xt)
    assert all(pd.api.types.is_float_dtype(encoded[name]) for name in encoder.numeric)
    for name in encoder.numeric:
        assert not np.isinf(encoded[name].to_numpy()).any()
        assert not (encoded[name].to_numpy()==-999999.).any()
    oldencoded=oldencoder.transform(xt[columns[:225]])
    pd.testing.assert_frame_equal(encoded[columns[:225]],oldencoded)
    prediction=old['estimator'].predict(oldencoded,ntree_end=old['steps'],thread_count=2,task_type='CPU')+tune.proxy_sec.to_numpy(float)
    stored=pd.read_parquet(oldroot/'tune_predictions.parquet')
    assert stored[ID].is_unique and len(stored)==len(tune) and set(stored[ID])==set(tune[ID])
    saved=stored.set_index(ID).loc[tune[ID],'prediction_sec'].to_numpy(float)
    delta=float(np.max(np.abs(prediction-saved)));assert delta<=1e-7
    assert np.isfinite(prediction).all()
    record=dict(status='passed',source_sha256=v.sha256(__file__),protocol_sha256=PROTOCOL,
        old_model_sha256=binding['outputs']['fit_model.joblib'],prior_linear_audit_sha256=v.sha256(linear_audit),
        fit_rows=nfit,tune_rows=len(tune),fit_ids_hash=v.object_hash(fit[ID].tolist()),tune_ids_hash=v.object_hash(tune[ID].tolist()),
        fit_raw_label_hash=v.object_hash(fit[TARGET].tolist()),tune_raw_label_hash=v.object_hash(tune[TARGET].tolist()),
        all387_fit_tune_source_inputs_rebuilt=True,all_fit_category_vocabularies_exact=True,
        original225_numeric_names_and_category_mappings_exact=True,original225_transformed_tune_frame_exact=True,
        old225_cpu_native_prediction_max_abs_delta=delta,source_receipts=receipts,source_fill_flags=flags,
        encoder387_sha256=v.sha256(OUT/'encoder387.joblib'),no_fit=True,no_gpu=True,peak_bytes=guard(),
        limitation='Prefit evidence only: full original inputs and fit-only mappings reconstructed; old225 native CPU inference replayed. No candidate training, candidate replay, new model gate or score-stage result.')
    v.write_json(OUT/'receipt.json',record)
    del matrix,xt,encoded,oldencoded,old;gc.collect();(OUT/'matrix.float32').unlink()
    print('VERIFIED_CB387_INPUTS',nfit,len(tune),'old_native_delta',delta,'peak',record['peak_bytes'],flush=True)


if __name__=='__main__':
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(2):main()
