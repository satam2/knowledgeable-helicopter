"""Independent full-source preprocessing, native replay and attention endpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import torch
import lightgbm
import gc
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from threadpoolctl import threadpool_limits
import verify_neural_context_v4 as core
import validate_candidate as v

ROOT=core.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/neural_attention'))
import run_attention as subject
ID,TARGET,TIME=subject.ID,subject.TARGET,subject.TIME
FOLDER=subject.OUT/'F1'
OUT=ROOT/'private_runs/tail240_20260916/validation/neural_attention_F1_v1'


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<6*1024**3,('6GiB validation ceiling',peak)
    assert psutil.virtual_memory().available>=8*1024**3
    return peak


def token_stats(matrix,nfit,columns):
    counts=np.zeros(24,np.int64);sums=np.zeros(24);squares=np.zeros(24)
    ownidx=[columns.index(n) for n in subject.adapter.OWN_CLOCKS]
    tokenidx=[[columns.index(p+f) for f in subject.adapter.FIELDS] for p in subject.adapter.PREFIXES]
    def clean(values):
        values=np.asarray(values,dtype=np.float64).copy()
        values[(values==-999999)|~np.isfinite(values)]=np.nan
        return values
    for start in range(0,nfit,65536):
        part=matrix[start:min(start+65536,nfit)]
        own=clean(part[:,ownidx])
        for slot,indices in enumerate(tokenidx):
            raw=clean(part[:,indices]);present=raw[:,13]==0
            duration=own-raw[:,:5];stamp=duration-raw[:,8:9]
            valid=present[:,None]&np.isfinite(own)&np.isfinite(raw[:,:5])&np.isfinite(raw[:,8:9])
            if slot>=4:valid[:]=False
            duration[~valid]=np.nan;stamp[~valid]=np.nan
            values=np.column_stack([raw,duration,stamp])
            valid=np.isfinite(values)&present[:,None]
            values=np.where(valid,values,0.)
            counts+=valid.sum(axis=0);sums+=values.sum(axis=0);squares+=(values**2).sum(axis=0)
        guard()
    means=sums/np.maximum(counts,1);variance=np.maximum(squares/np.maximum(counts,1)-means**2,0.)
    scales=np.where(variance>1e-12,np.sqrt(variance),1.)
    return counts,means.astype('float32'),scales.astype('float32')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(subject.OUT/'protocol.json');manifest=v.read_json(FOLDER/'manifest.json')
    assert manifest['status']=='complete' and manifest['protocol_sha256']==v.sha256(subject.OUT/'protocol.json')
    for path,digest in protocol['source_hashes'].items():assert v.sha256(ROOT/path)==digest
    for name,digest in manifest['outputs'].items():assert v.sha256(FOLDER/name)==digest
    canary=v.read_json(ROOT/'private_runs/tail240_20260916/validation/neural_attention_canary_F1/receipt.json')
    assert canary['status']=='passed'
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec'])
    idx,split,_=v.common.fold_data(meta,'F1',full=True)
    assert v.object_hash(split)==v.object_hash(manifest['split'])
    parts={s:meta.iloc[idx[s]].loc[lambda f:np.isfinite(f.proxy_sec)].copy() for s in ['fit','tune']}
    fit,tune=parts['fit'],parts['tune']
    for stage,frame in parts.items():
        assert len(frame)==manifest[stage+'_rows']
        assert v.object_hash(frame[ID].tolist())==manifest[stage+'_ids_hash']
        assert v.object_hash(frame[TARGET].tolist())==manifest[stage+'_label_hash']
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True))
    del meta,idx,parts;gc.collect()
    columns=protocol['feature_columns'];assert len(columns)==387
    core.guard=guard;core.producer.risk.guard=guard
    matrix,vocab,receipts=core.bounded_load(ids,len(fit),columns,OUT)
    assert receipts==manifest['feature_receipts']
    for receipt in receipts:assert v.sha256(receipt['path'])==receipt['sha256']
    model=joblib.load(FOLDER/'fit_model.joblib');encoder=model['encoder'];network=model['estimator']
    assert encoder.columns==columns
    residual=(fit[TARGET]-fit.proxy_sec).to_numpy(float)
    assert model['y_mean']==float(residual.mean()) and model['y_scale']==max(float(residual.std()),1.)
    bins=[];embedded=[]
    for j,name in enumerate(encoder.numeric):
        values=np.asarray(matrix[:len(fit),columns.index(name)]).copy()
        values[(values==-999999)|~np.isfinite(values)]=np.nan
        median=np.float32(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.float32(0)
        filled=np.where(np.isfinite(values),values,median)
        mean=np.float32(filled.mean(dtype=np.float64));std=filled.std(dtype=np.float64)
        scale=np.float32(std if std>1e-6 else 1.)
        assert median==encoder.medians[j] and mean==encoder.means[j] and scale==encoder.scales[j],name
        numbers=(filled-mean)/scale
        if np.any(numbers!=numbers[0]):
            embedded.append(j)
            bins.extend(subject.adapter.ple.rtdl_num_embeddings.compute_bins(torch.from_numpy(numbers[:,None]),n_bins=48))
        if j%75==0:print('FIT_STATS',j,guard(),flush=True)
    for name,mapping in encoder.categories.items():
        series=subject.context.decode_frame(matrix[:len(fit),[columns.index(name)]],{name:vocab[name]},[name])[name].astype('string')
        assert mapping=={word:i+2 for i,word in enumerate(sorted(series.dropna().unique()))}
    passthrough=[i for i in range(2*len(encoder.numeric)) if i not in set(embedded)]
    np.testing.assert_array_equal(network.base.embedded,embedded)
    np.testing.assert_array_equal(network.base.passthrough,passthrough)
    rebuilt=subject.adapter.ple.rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
    left,right=rebuilt.impl.state_dict(),network.base.numeric_embeddings.impl.state_dict()
    assert set(left)==set(right)
    for key in left:assert torch.equal(left[key],right[key]),key
    counts,means,scales=token_stats(matrix,len(fit),columns)
    evidence=v.read_json(FOLDER/'fit_evidence.json');stats=evidence['token_statistics']
    np.testing.assert_array_equal(counts,stats['count'])
    np.testing.assert_array_equal(means,stats['means'])
    np.testing.assert_array_equal(scales,stats['scales'])
    np.testing.assert_array_equal(network.shared_means,means)
    np.testing.assert_array_equal(network.shared_scales,scales)
    np.testing.assert_array_equal(network.raw_means,encoder.means)
    np.testing.assert_array_equal(network.raw_scales,encoder.scales)
    expected_tokens=[[encoder.numeric.index(p+f) for f in subject.adapter.FIELDS] for p in subject.adapter.PREFIXES]
    np.testing.assert_array_equal(network.token_columns,expected_tokens)
    np.testing.assert_array_equal(network.own_columns,[encoder.numeric.index(n) for n in subject.adapter.OWN_CLOCKS])
    query=[i for i,n in enumerate(encoder.numeric) if not n.startswith('flat_')]
    np.testing.assert_array_equal(network.query_columns,query+[i+len(encoder.numeric) for i in query])
    np.testing.assert_array_equal(network.phase,[0]*4+[1]*4)
    assert sum(p.numel() for p in network.parameters())==evidence['parameter_count']
    del bins,rebuilt,left,right,values,filled,numbers,series,residual;gc.collect();guard()
    xt=subject.context.decode_frame(matrix[len(fit):],vocab,columns)
    saved=pd.read_parquet(FOLDER/'tune_predictions.parquet')
    for name in [ID,TARGET,'proxy_sec']:np.testing.assert_array_equal(saved[name],tune[name])
    torch.cuda.reset_peak_memory_stats()
    prediction=subject.adapter.predict(model,xt)+tune.proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(prediction,saved.prediction_sec)
    binding=protocol['controls']['F1'];oldpath=subject.CONTROLS['F1']/'fit_model.joblib'
    assert v.sha256(oldpath)==binding['fit_model_sha256']
    controlmodel=joblib.load(oldpath)
    control=subject.adapter.frozen.predict(controlmodel,xt)+tune.proxy_sec.to_numpy(float)
    old=pd.read_parquet(subject.CONTROLS['F1']/'tune_predictions.parquet')
    assert v.sha256(subject.CONTROLS['F1']/'tune_predictions.parquet')==binding['tune_predictions_sha256']
    np.testing.assert_array_equal(old[ID],saved[ID])
    np.testing.assert_array_equal(control,old.prediction_sec)
    np.testing.assert_array_equal(control,saved.control387_prediction_sec)
    gpu_peak=torch.cuda.max_memory_allocated();torch.cuda.empty_cache()
    print('NATIVE_REPLAY_EXACT',len(tune),'GPU_PEAK',gpu_peak,flush=True)
    y=tune[TARGET].to_numpy(float);days=tune[TIME].dt.floor('D').to_numpy();ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    compare={'all_finite':(y,prediction,control,days),'ordinary':(y[ordinary],prediction[ordinary],control[ordinary],days[ordinary])}
    weights=v.read_json(subject.ENSEMBLE/'F1_weights.json')
    assert v.sha256(subject.ENSEMBLE/'F1_weights.json')==binding['global9_weights_sha256']
    assert v.sha256(subject.ENSEMBLE/'F1_aligned_tune.parquet')==binding['global9_tune_sha256']
    aligned=pd.read_parquet(subject.ENSEMBLE/'F1_aligned_tune.parquet').set_index(ID)
    own=saved.set_index(ID).loc[aligned.index]
    assert set(aligned.index)==set(tune.loc[ordinary,ID])
    np.testing.assert_array_equal(own[TARGET],aligned[TARGET])
    baseline=aligned[weights['experts']].to_numpy()@np.asarray(weights['global'])
    coefficient=weights['global'][weights['experts'].index('tabm_ple8')]
    candidate=baseline+coefficient*(own.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    current=baseline+coefficient*(own.control387_prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    blend=.75*baseline+.25*own.prediction_sec.to_numpy()
    truth=aligned[TARGET].to_numpy();dates=aligned[TIME].dt.floor('D').to_numpy()
    compare.update(primary_replacement=(truth,candidate,baseline,dates),secondary_blend25=(truth,blend,baseline,dates),incremental_current387=(truth,candidate,current,dates))
    recorded=v.read_json(FOLDER/'metrics.json');reports={}
    for name,(target,pred,ref,dates) in compare.items():
        a,b=v.metrics(target,pred),v.metrics(target,ref)
        paired=v.paired(target,ref,pred,dates)
        if name in recorded:
            np.testing.assert_allclose(a['rmse'],recorded[name]['candidate_rmse'],atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(b['rmse'],recorded[name]['control_rmse'],atol=1e-10,rtol=1e-12)
            assert paired['all_day_removals_improve']==recorded[name]['all_day_removals_improve']
        reports[name]=dict(candidate=a,control=b,gain=b['rmse']-a['rmse'],paired=paired)
    best=min(evidence['history'],key=lambda e:e['tune_mse_sec2'])
    assert best['epoch']==model['steps']==manifest['selected_epochs']==10
    np.testing.assert_allclose(best['tune_mse_sec2'],reports['all_finite']['candidate']['mse'],atol=1e-7,rtol=1e-12)
    result=dict(status='passed',source_sha256=v.sha256(__file__),producer_manifest_sha256=v.sha256(FOLDER/'manifest.json'),protocol_sha256=manifest['protocol_sha256'],fit_rows=len(fit),tune_rows=len(tune),native_candidate_and_control_exact=True,fit_numeric_stats_exact=len(encoder.numeric),fit_categories_exact=len(encoder.categories),all_fullfit_bin_tensors_exact=True,all_shared_token_stats_exact=True,architecture_index_and_scaling_buffers_exact=True,selected_epochs=model['steps'],gpu_peak_bytes=gpu_peak,peak_bytes=guard(),metrics=reports,decision='F1 incremental current387 is negative. Old225 primary is positive; its two-month gate was not completed and is not retroactively declared failed. Parent stops F3/fullscore on incremental failure.')
    v.write_json(OUT/'receipt.json',result)
    print('COMPLETE',{name:r['gain'] for name,r in reports.items()},flush=True)


if __name__=='__main__':
    pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(2)
    with threadpool_limits(2):main()
