"""Independent selector preprocessing, strict native state and late metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import torch
import lightgbm
import sys
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT,read,sha,write,guard,object_hash
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/forensics/current_selector/v3'))
import run_selector as subject
from verify_current_calibration import check_comparison,check_metrics,solve_simplex


def main():
    protocol=subject.declare();folder=subject.OUT/'F1';marker=read(folder/'manifest.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==sha(subject.OUT/'protocol.json')
    for name,digest in marker['outputs'].items():assert sha(folder/name)==digest
    prior=read(ROOT/'private_runs/tail240_20260916/validation/current_calibration_F1_v1/receipt.json')
    assert prior['status']=='passed' and prior['producer_manifest_sha256']==protocol['folds']['F1']['manifest.json']
    frame,savedpre=subject.inputs('F1',protocol)
    labels=pd.read_parquet(subject.old.OUT/'F1/inputs.parquet',columns=[subject.ID,subject.TARGET])
    np.testing.assert_array_equal(frame[subject.ID],labels[subject.ID]);y=labels[subject.TARGET].to_numpy(float)
    fit=frame.fit.to_numpy(bool);late=frame.late.to_numpy(bool)
    order=np.r_[np.flatnonzero(fit),np.flatnonzero(~fit)];inverse=np.argsort(order);nfit=int(fit.sum())
    ids=pd.Index(frame.iloc[order][subject.ID]);cols=protocol['columns']
    early_encoder=read(subject.old.OUT/'F1/encoder.json');vocab=early_encoder['vocab']
    stored=np.memmap(folder/'matrix.float32',mode='r',dtype='float32',shape=(len(ids),len(cols)),order='F')
    counts=np.zeros(len(cols),int);known={name:set() for name in vocab}
    for receipt in savedpre['source_receipts']:
        path=Path(receipt['path']);assert sha(path)==receipt['sha256']
        names=receipt['columns'];seen=np.zeros(len(ids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[subject.ID,*names],use_threads=False):
            data=batch.to_pandas();positions=ids.get_indexer(data[subject.ID]);keep=positions>=0;positions=positions[keep];data=data.loc[keep]
            assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
            for name in names:
                values=data[name]
                if name in vocab:
                    known[name].update(values.iloc[np.flatnonzero(positions<nfit)].dropna().astype(str).tolist())
                    mapping={word:i+2 for i,word in enumerate(vocab[name])}
                    raw=values.astype('string');values=raw.map(mapping).fillna(1).where(raw.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan);values[~np.isfinite(values)]=np.nan
                    if receipt['fill']:values[values==-999999.]=np.nan
                np.testing.assert_array_equal(values,stored[positions,cols.index(name)],err_msg=name)
            guard()
        for name in names:counts[cols.index(name)]+=int(seen.sum())
    assert (counts==len(ids)).all() and {name:sorted(values) for name,values in known.items()}==vocab
    model=joblib.load(folder/'model.joblib');encoder=model['encoder']
    assert encoder.columns==cols+protocol['contrasts'] and len(encoder.numeric)==381 and len(encoder.categories)==15
    p=frame[protocol['experts']].to_numpy(float);ordered_p=p[order]
    contrasts=(ordered_p-ordered_p.mean(axis=1,keepdims=True)).astype('float32')
    for index,name in enumerate(encoder.numeric):
        values=np.array(stored[:nfit,cols.index(name)] if name in cols else contrasts[:nfit,protocol['contrasts'].index(name)],copy=True)
        values[(values==-999999.)|~np.isfinite(values)]=np.nan
        valid=np.isfinite(values);median=np.float32(np.median(values[valid])) if valid.any() else np.float32(0.)
        filled=np.where(valid,values,median);mean=np.float32(filled.mean(dtype=np.float64));scale=filled.std(dtype=np.float64)
        scale=np.float32(scale if scale>1e-6 else 1.)
        assert median==encoder.medians[index] and mean==encoder.means[index] and scale==encoder.scales[index],name
    base_categories=set(cols[:11]);lookups={}
    for name,mapping in encoder.categories.items():
        labels_array=np.asarray([None if name in base_categories else 'MISSING','__UNSEEN_CONTEXT_VALUE__',*vocab[name]],object)
        observed=pd.Series(labels_array[np.asarray(stored[:nfit,cols.index(name)],dtype=int)]).astype('string')
        expected={word:i+2 for i,word in enumerate(sorted(observed.dropna().unique()))}
        assert mapping==expected,name
        lookups[name]=np.asarray([0 if value is None else mapping.get(value,1) for value in labels_array],dtype=np.int64)
    constant=solve_simplex(y[fit],p[fit]);np.testing.assert_allclose(constant,model['weights'],rtol=0,atol=1e-10)
    initial=np.maximum(constant,1e-6);initial/=initial.sum();np.testing.assert_allclose(initial,model['initial_weights'],rtol=0,atol=1e-10)
    assert model['evidence']['epochs']==20 and len(model['evidence']['history'])==20 and model['evidence']['batch_size']==4096
    network=subject.selector.Network(**model['dimensions'],initial_weights=model['initial_weights'])
    network.load_state_dict(torch.load(folder/'state.pt',map_location='cpu',weights_only=True),strict=True);network.eval()
    prediction=np.empty(len(ids));weights=np.empty((len(ids),9))
    with torch.inference_mode():
        for start in range(0,len(ids),8192):
            stop=min(start+8192,len(ids));raw=np.empty((stop-start,381),np.float32)
            for j,name in enumerate(encoder.numeric):
                raw[:,j]=stored[start:stop,cols.index(name)] if name in cols else contrasts[start:stop,protocol['contrasts'].index(name)]
            missing=(raw==-999999.)|~np.isfinite(raw)
            numbers=np.concatenate([(np.where(missing,encoder.medians,raw)-encoder.means)/encoder.scales,missing.astype('float32')],axis=1)
            cats=np.column_stack([lookups[name][np.asarray(stored[start:stop,cols.index(name)],dtype=int)] for name in encoder.categories])
            w=network(torch.from_numpy(np.ascontiguousarray(numbers)),torch.from_numpy(cats)).numpy().astype(float)
            w/=w.sum(axis=1,keepdims=True);weights[start:stop]=w
            mean=ordered_p[start:stop].mean(axis=1)
            prediction[start:stop]=mean+(w*(ordered_p[start:stop]-mean[:,None])).sum(axis=1)
            assert guard()<3*1024**3
    prediction,weights=prediction[inverse],weights[inverse]
    saved=pd.read_parquet(folder/'predictions.parquet')
    np.testing.assert_array_equal(saved[subject.ID],frame[subject.ID])
    np.testing.assert_array_equal(saved.prediction_sec,prediction)
    np.testing.assert_array_equal(saved[['weight_'+name for name in protocol['experts']]],weights)
    assert (weights>=0).all();np.testing.assert_allclose(weights.sum(axis=1),1.,rtol=0,atol=1e-12)
    assert (prediction>=p.min(axis=1)-1e-8).all() and (prediction<=p.max(axis=1)+1e-8).all()
    baseline=p@model['weights'];np.testing.assert_array_equal(saved.early_constant_sec,baseline)
    metrics=read(folder/'metrics.json');assert metrics==marker['metrics']
    check_comparison(y[late],prediction[late],baseline[late],frame.loc[late,subject.old.TIME],metrics['primary'])
    check_comparison(y[late],prediction[late],frame.loc[late,'frozen_current_sec'].to_numpy(),frame.loc[late,subject.old.TIME],metrics['frozen_current_diagnostic'])
    check_metrics(y[fit],prediction[fit],metrics['early_training']['candidate']);check_metrics(y[fit],baseline[fit],metrics['early_training']['constant'])
    watch=read(folder/'watchdog.json');assert watch['returncode']==0 and watch['failure'] is None
    assert len(watch['per_process_peaks'])>=2 and max(watch['per_process_peaks'].values())>1024**3 and watch['peak_bytes']<4*1024**3
    assert not torch.cuda.is_initialized()
    out=ROOT/'private_runs/tail240_20260916/validation/current_selector_F1_v3'
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',source_sha256=sha(Path(__file__)),producer_manifest_sha256=sha(folder/'manifest.json'),
        protocol_sha256=sha(subject.OUT/'protocol.json'),all387_source_values_and15early_vocab_exact=True,
        all381_neural_fit_statistics_exact=True,all9_contrasts_exact=True,native_prediction_and_weight_delta=0.,
        simplex_delta=float(np.max(np.abs(constant-model['weights']))),convex_hull_all_rows=True,
        all_metrics_day_bootstrap_and_influences_exact=True,primary=metrics['primary'],fit_rows=nfit,late_rows=int(late.sum()),
        watcher_covers_actual_worker=True,watchdog_sha256=sha(folder/'watchdog.json'),no_GPU_initialized=True,no_model_fit=True,
        peak_bytes=guard(),limitation='Exposed-developmentlateordinaryonly;upstreamstoppingusesfulltune. F1negativeholdsF3;no complete-scoreclaim.')
    write(out/'receipt.json',result);print('VERIFIED',metrics['primary'],'peak',result['peak_bytes'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):main()
