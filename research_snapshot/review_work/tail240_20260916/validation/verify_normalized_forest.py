"""Saved forest replay, preprocessing reconstruction and full-tune composition."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import lightgbm
import argparse
import gc
import importlib.util
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
HERE=ROOT/'review_work/tail240_20260916/state/normalized_forest'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('normalized_forest_subject',HERE/'run.py')
subject=importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
forest=subject.forest


def guard():
    m=psutil.Process().memory_info()
    peak=max(m.rss,getattr(m,'peak_wset',0))
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def main(fold):
    assert psutil.virtual_memory().available>=10*1024**3
    root=subject.OUT if fold=='F1' else subject.OUT.parent/'v2'
    folder=root/fold
    out=ROOT/'private_runs/tail240_20260916/validation'/('normalized_forest_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    marker=v.read_json(folder/'manifest.json')
    protocol=v.read_json(root/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(root/'protocol.json')
    assert protocol['source_sha256']==v.sha256(HERE/'run.py')
    if fold=='F3':
        assert protocol['wrapper_sha256']==v.sha256(HERE/'run_f3_v2.py')
        assert protocol['helper_sha256']==v.sha256(HERE/'parallel_fit.py')
    for path,digest in protocol['dependency_hashes'].items():
        assert v.sha256(ROOT/path)==digest
    for name,digest in marker['outputs'].items():
        assert v.sha256(folder/name)==digest
    subject.normalized.shared.state.guard=guard
    x,meta=subject.normalized.shared.load_missing()
    idx,split,_=v.common.fold_data(meta,fold,full=True)
    assert v.object_hash(split)==v.object_hash(marker['split'])
    frames={}
    targets={}
    for stage in ['fit','tune']:
        m=meta.iloc[idx[stage]].loc[lambda q:~np.isfinite(q.proxy_sec)]
        frames[stage]=x.loc[m[ID]].copy()
        frames[stage][forest.TIME_COLUMN]=m[TIME].to_numpy()
        targets[stage]=m[TARGET].to_numpy(float)
        assert {'n':len(m),'hash':v.object_hash(m[ID].tolist())}==marker['fit_ids'][stage]
    del x
    fit=forest.input_frame(frames['fit'])
    times=pd.Series(pd.to_datetime(frames['fit'][forest.TIME_COLUMN],utc=True).to_numpy(),index=fit.index)
    prior=forest.missing.HistoricalTemplate().fit(fit,targets['fit'],times)
    train=pd.concat([fit,forest.missing.crossfit_templates(fit,targets['fit'],times)],axis=1)
    rebuilt=forest.encoder(train)
    values=rebuilt.fit_transform(train)
    raw=forest.input_frame(frames['tune'])
    tt=pd.Series(pd.to_datetime(frames['tune'][forest.TIME_COLUMN],utc=True).to_numpy(),index=raw.index)
    tune=pd.concat([raw,prior.transform(raw,tt)],axis=1)
    encoded=rebuilt.transform(tune)
    saved=pd.read_parquet(folder/'missing_tune.parquet')
    np.testing.assert_array_equal(saved[ID],frames['tune'].index)
    np.testing.assert_array_equal(saved.raw_target_sec,targets['tune'])
    raw_predictions={}
    scale={}
    for stage in ['fit','tune']:
        s=frames[stage].schedule_proxy_sec.to_numpy(float)
        scale[stage]=np.sqrt(3600.**2+np.where(np.isfinite(s)&(s!=-999999),s-900.,0.)**2)
    for name,path in [('original',subject.original_folder(fold)/'fit_model.joblib'),('raw',folder/'raw_control.joblib'),('scaled',folder/'scaled_model.joblib')]:
        model=joblib.load(path)
        assert model['feature_columns']==list(train) and len(train.columns)==81
        assert model['columns']==list(frames['fit'])
        assert model['prior'].history_n==prior.history_n and model['prior'].global_mean==prior.global_mean and model['prior'].last_fit==prior.last_fit
        for (ka,ta),(kb,tb) in zip(model['prior'].tables,prior.tables):
            assert ka==kb
            pd.testing.assert_frame_equal(ta,tb)
        transform=model['encoder']
        np.testing.assert_array_equal(transform.get_feature_names_out(),rebuilt.get_feature_names_out())
        diff=transform.transform(train)-values
        assert diff.nnz==0
        assert (transform.transform(tune)-encoded).nnz==0
        estimator=model['estimator']
        params=estimator.get_params()
        assert params['n_estimators']==300 and params['max_features']==.7 and params['min_samples_leaf']==1 and params['criterion']=='squared_error' and not params['bootstrap']
        estimator.set_params(n_jobs=1)
        pred=estimator.predict(encoded)
        if name=='scaled':
            normalizer=float(np.mean(scale['fit']**2))
            assert model['fit_scale_squared_normalizer']==normalizer
            weights=scale['fit']**2/normalizer
            np.testing.assert_allclose(weights.sum()**2/(weights@weights),marker['fit_weight_ess'],rtol=1e-14)
            pred=900.+scale['tune']*pred
            np.testing.assert_array_equal(pred,saved.prediction_sec)
        else:
            np.testing.assert_array_equal(pred,saved.raw_control_prediction_sec)
        raw_predictions[name]=pred
        del model,estimator,transform
        gc.collect()
        guard()
    np.testing.assert_array_equal(raw_predictions['original'],raw_predictions['raw'])
    old=pd.read_parquet(subject.original_folder(fold)/'tune_predictions.parquet')
    np.testing.assert_array_equal(old[ID],saved[ID])
    np.testing.assert_array_equal(old.prediction_sec,raw_predictions['raw'])
    normpath=subject.norm_folder(fold)
    assert v.sha256(normpath/'manifest.json')==protocol['folds'][fold]['normalized_lgb_manifest_sha256']
    normmarker=v.read_json(normpath/'manifest.json')
    assert v.sha256(normpath/'tune.parquet')==normmarker['outputs']['tune.parquet']
    norm=pd.read_parquet(normpath/'tune.parquet')
    np.testing.assert_array_equal(norm[ID],saved[ID])
    np.testing.assert_array_equal(norm.prediction_sec,saved.normalized_lgb_prediction_sec)
    full=meta.iloc[idx['tune']].copy()
    finite=np.isfinite(full.proxy_sec).to_numpy()
    ordinary=full.proxy_sec.between(0,7200).to_numpy()
    unionpath=subject.risk.union_folder(fold)
    assert v.sha256(unionpath/'manifest.json')==protocol['folds'][fold]['union_manifest_sha256']
    unionmarker=v.read_json(unionpath/'manifest.json')
    assert v.sha256(unionpath/'tune_predictions.parquet')==unionmarker['outputs']['tune_predictions.parquet']
    union=pd.read_parquet(unionpath/'tune_predictions.parquet')
    np.testing.assert_array_equal(union[ID],full.loc[finite,ID])
    weights=v.read_json(subject.ENSEMBLE/f'{fold}_weights.json')
    aligned=pd.read_parquet(subject.ENSEMBLE/f'{fold}_aligned_tune.parquet')
    assert v.sha256(subject.ENSEMBLE/f'{fold}_weights.json')==protocol['folds'][fold]['global9_weights_sha256']
    assert v.sha256(subject.ENSEMBLE/f'{fold}_aligned_tune.parquet')==protocol['folds'][fold]['global9_aligned_tune_sha256']
    np.testing.assert_array_equal(aligned[ID],full.loc[ordinary,ID])
    reference=np.empty(len(full))
    reference[finite]=union.prediction_sec
    reference[ordinary]=aligned[weights['experts']].to_numpy()@np.asarray(weights['global'])
    reference[~finite]=norm.prediction_sec
    candidate=reference.copy()
    candidate[~finite]=.75*reference[~finite]+.25*raw_predictions['scaled']
    produced=pd.read_parquet(folder/'full_tune.parquet')
    np.testing.assert_array_equal(produced[ID],full[ID])
    np.testing.assert_array_equal(produced[TARGET],full[TARGET])
    np.testing.assert_array_equal(produced.prediction_sec,candidate)
    np.testing.assert_array_equal(produced.reference_prediction_sec,reference)
    y=full[TARGET].to_numpy(float)
    dates=full[TIME].dt.floor('D').to_numpy()
    comparison=v.paired(y,reference,candidate,dates)
    reports=v.read_json(folder/'metrics.json')
    np.testing.assert_allclose(-comparison['delta_rmse'],reports['full_tune_fixed25']['gain'],atol=1e-12)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),protocol_sha256=v.sha256(root/'protocol.json'),manifest_sha256=v.sha256(folder/'manifest.json'),fit_rows=len(train),tune_rows=len(tune),full_tune_rows=len(full),native_original_raw_scaled_exact=True,fit_encoder_rebuilt_exact=True,raw_target_priors_rebuilt_exact=True,full_route_composition_exact=True,missing_raw_rmse=v.metrics(targets['tune'],raw_predictions['raw'])['rmse'],missing_scaled_rmse=v.metrics(targets['tune'],raw_predictions['scaled'])['rmse'],missing_normalized_lgb_rmse=v.metrics(targets['tune'],norm.prediction_sec)['rmse'],full_tune=comparison,peak_bytes=guard(),no_model_fit=True)
    v.write_json(out/'receipt.json',receipt)
    print({k:value for k,value in receipt.items() if k!='full_tune'},flush=True)
    print('FULL_TUNE_GAIN',-comparison['delta_rmse'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',required=True,choices=['F1','F3'])
    args=parser.parse_args()
    with threadpool_limits(1):
        main(args.fold)
