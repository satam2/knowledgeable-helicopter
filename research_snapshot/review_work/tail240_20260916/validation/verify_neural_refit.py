"""Independent full-refit preprocessing, native score and protected-route oracle."""
import torch
import argparse
import gc
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import verify_neural_context_v4 as core

v,ROOT,producer=core.v,core.ROOT,core.producer
ID,TARGET,common=core.ID,core.TARGET,core.common
BASE=ROOT/'private_runs/tail240_20260916/state/neural_context/refit_score_v3'


def main(fold):
    folder=BASE/fold;out=ROOT/f'private_runs/tail240_20260916/validation/neural_refit_v3_{fold}'
    out.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');marker=v.read_json(folder/'manifest.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert marker['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/state/neural_context/refit_score_v3.py')
    assert protocol['baseline_binding_sha256']==v.sha256(v.BINDING)
    for name,digest in marker['outputs'].items():assert v.sha256(folder/name)==digest
    bound=v.read_json(v.BINDING)['folds'][fold]
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    assert v.object_hash(split)==bound['split_hash']==v.object_hash(marker['split'])
    refit=meta.iloc[idx['refit']].loc[lambda x:np.isfinite(x.proxy_sec)].copy()
    full=meta.iloc[idx['score']].copy();finite=np.isfinite(full.proxy_sec).to_numpy();score=full.loc[finite].copy()
    assert v.object_hash(refit[ID].tolist())==marker['refit_ids_hash']==bound['cohorts']['finite_nm']['refit']['hash']
    assert v.object_hash(refit[TARGET].tolist())==marker['refit_label_hash']
    assert v.object_hash(full[ID].tolist())==marker['score_ids_hash']==bound['score_id_hash']
    assert v.object_hash(full[TARGET].tolist())==marker['score_label_hash']==bound['score_target_hash']
    assert v.object_hash(score[ID].tolist())==bound['cohorts']['finite_nm']['score']['hash']
    assert len(refit)==marker['refit_rows'] and len(full)==marker['full_score_rows'] and len(score)==marker['finite_score_rows']
    del meta,idx;gc.collect()
    desired=protocol['feature_columns'];assert desired==marker['feature_columns'] and len(desired)==387
    ids=pd.Index(pd.concat([refit[ID],score[ID]],ignore_index=True))
    matrix,vocab,receipts=core.bounded_load(ids,len(refit),desired,out);assert receipts==marker['feature_receipts']
    model=joblib.load(folder/'model.joblib');encoder=model['encoder'];assert encoder.columns==desired
    residual=(refit[TARGET]-refit.proxy_sec).to_numpy(float)
    assert model['y_mean']==float(residual.mean()) and model['y_scale']==max(float(residual.std()),1.)
    bins=[];embedded=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for j,name in enumerate(encoder.numeric):
            values=np.asarray(matrix[:len(refit),desired.index(name)]).copy();values[(values==-999999)|~np.isfinite(values)]=np.nan
            median=np.float32(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.float32(0)
            filled=np.where(np.isfinite(values),values,median);mean=np.float32(filled.mean(dtype=np.float64))
            scale=filled.std(dtype=np.float64);scale=np.float32(scale if scale>1e-6 else 1.)
            assert median==encoder.medians[j] and mean==encoder.means[j] and scale==encoder.scales[j],name
            numbers=(filled-mean)/scale
            if np.any(numbers!=numbers[0]):
                embedded.append(j);bins.extend(producer.adapter.rtdl_num_embeddings.compute_bins(torch.from_numpy(numbers[:,None]),n_bins=48))
            if j%50==0:print('REFIT_COLUMN',fold,j,core.guard(),flush=True)
        for name,mapping in encoder.categories.items():
            labels=producer.decode_frame(matrix[:len(refit),[desired.index(name)]],{name:vocab[name]},[name])[name].astype('string')
            assert {value:i+2 for i,value in enumerate(sorted(labels.dropna().unique()))}==mapping
        passthrough=[i for i in range(2*len(encoder.numeric)) if i not in set(embedded)]
        np.testing.assert_array_equal(model['estimator'].embedded.numpy(),embedded)
        np.testing.assert_array_equal(model['estimator'].passthrough.numpy(),passthrough)
        rebuilt=producer.adapter.rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
        left=rebuilt.impl.state_dict();right=model['estimator'].numeric_embeddings.impl.state_dict()
        assert list(left)==list(right)
        for name in left:assert torch.equal(left[name],right[name]),name
    del rebuilt,bins,labels,values,filled,numbers,residual;gc.collect();core.guard()
    expected_epochs=protocol['folds'][fold]['epochs'];assert model['steps']==marker['fixed_epochs']==expected_epochs
    evidence=v.read_json(folder/'refit_evidence.json')
    assert evidence['steps']==expected_epochs and evidence['rows']==len(refit) and len(evidence['history'])==expected_epochs
    assert all('tune_mse_sec2' not in row for row in evidence['history'])
    xs=producer.decode_frame(matrix[len(refit):],vocab,desired)
    prediction=producer.adapter.predict(model,xs)+score.proxy_sec.to_numpy(float)
    saved=pd.read_parquet(folder/'finite_score_predictions.parquet')
    for name in [ID,TARGET,'proxy_sec']:np.testing.assert_array_equal(saved[name],score[name])
    np.testing.assert_array_equal(prediction,saved.prediction_sec)
    old_folder=producer.control_folder(fold);old_marker=v.read_json(old_folder/'manifest.json')
    assert v.sha256(old_folder/'manifest.json')==protocol['folds'][fold]['original225_manifest_sha256']
    for name in ['model.joblib','candidate.parquet']:assert v.sha256(old_folder/name)==old_marker['outputs'][name]
    old_model=joblib.load(old_folder/'model.joblib')
    old_prediction=producer.adapter.predict(old_model,xs[desired[:225]])+score.proxy_sec.to_numpy(float)
    old_full=pd.read_parquet(old_folder/'candidate.parquet');np.testing.assert_array_equal(old_full[ID],full[ID])
    np.testing.assert_array_equal(old_prediction,old_full.prediction_sec.to_numpy()[finite]);np.testing.assert_array_equal(old_prediction,saved.control225_prediction_sec)
    print('GPU_REPLAY_COMPLETE',fold,0,0,flush=True)
    assert v.sha256(bound['prediction_path'])==bound['prediction_sha256']
    baseline=pd.read_parquet(bound['prediction_path']);np.testing.assert_array_equal(baseline[ID],full[ID]);np.testing.assert_array_equal(baseline[TARGET],full[TARGET])
    weights_path=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'/f'{fold}_weights.json'
    assert v.sha256(weights_path)==protocol['folds'][fold]['weights_sha256'];weights=v.read_json(weights_path)
    weight=weights['global'][weights['experts'].index('tabm_ple8')]
    ordinary=full.proxy_sec.between(0,7200).to_numpy();raw=np.full(len(full),np.nan);raw[finite]=prediction
    b=baseline.prediction_sec.to_numpy(float);old=old_full.prediction_sec.to_numpy(float)
    replacement=b.copy();replacement[ordinary]=b[ordinary]+weight*(raw[ordinary]-old[ordinary])
    blend=b.copy();blend[ordinary]=.75*b[ordinary]+.25*raw[ordinary]
    reports={};producer_metrics=v.read_json(folder/'metrics.json')
    y=full[TARGET].to_numpy(float);dates=full[common.MOVEMENT].dt.floor('D').to_numpy()
    for name,values in [('replacement',replacement),('blend25',blend)]:
        table=pd.read_parquet(folder/(name+'.parquet'));np.testing.assert_array_equal(table[ID],full[ID]);np.testing.assert_array_equal(table[TARGET],full[TARGET])
        np.testing.assert_array_equal(table.prediction_sec,values);np.testing.assert_array_equal(values[~ordinary],b[~ordinary])
        metrics=v.metrics(y,values);assert abs(metrics['rmse']-producer_metrics[name]['metrics']['rmse_sec'])<1e-10
        reports[name]={'candidate':metrics,'control':v.metrics(y,b),'paired':v.paired(y,b,values,dates),'protected_rows':int((~ordinary).sum()),'protected_exact':True}
    for name,mask in [('matched_allfinite',np.ones(len(score),bool)),('matched_ordinary',score.proxy_sec.between(0,7200).to_numpy())]:
        target=score[TARGET].to_numpy(float)[mask];p=prediction[mask];old=old_prediction[mask]
        metrics=v.metrics(target,p);assert abs(metrics['rmse']-producer_metrics[name]['metrics']['rmse_sec'])<1e-10
        reports[name]={'candidate':metrics,'control':v.metrics(target,old),'paired':v.paired(target,old,p,score[common.MOVEMENT].dt.floor('D').to_numpy()[mask])}
    v.write_json(out/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),'producer_manifest_sha256':v.sha256(folder/'manifest.json'),
        'fold':fold,'refit_rows':len(refit),'finite_score_rows':len(score),'full_score_rows':len(full),'fixed_epochs':expected_epochs,
        'new_native_max_abs_delta_sec':0.,'old225_native_max_abs_delta_sec':0.,'refit_numeric_statistics_exact':len(encoder.numeric),
        'refit_category_mappings_exact':len(encoder.categories),'official_bin_implementation_exact':True,'embedded_numeric_columns':len(embedded),
        'reports':reports,'peak_rss_bytes':core.guard(),'scope':'Original exposed score folds, full original labels/cohorts; no training, model selection or new weights.'})
    print('COMPLETE',fold,{name:value['candidate']['rmse'] for name,value in reports.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3']);args=parser.parse_args()
    core.pa.set_cpu_count(1);core.pa.set_io_thread_count(1);torch.set_num_threads(2)
    with core.threadpool_limits(2):main(args.fold)
