"""Verify final missing preprocessors, native predictions, and nested arithmetic."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import lightgbm
from preflight import ROOT, OUT, ID, TARGET, read, sha
import gc
import importlib.util
import json
import joblib
import sys
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits


def main():
    output=OUT/'missing_native_receipt.json';assert not output.exists()
    code=ROOT/'review_work/tail240_20260916/forensics/final_missing/run.py'
    sys.path.insert(0,str(code.parent))
    spec=importlib.util.spec_from_file_location('missing_result_review',code)
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    folder=subject.OUT/'models';marker=read(folder/'manifest.json');protocol=read(subject.OUT/'protocol.json')
    preflight=read(OUT/'missing_preflight_receipt.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==preflight['protocol_sha256']==sha(subject.OUT/'protocol.json')
    assert marker['preparation_sha256']==preflight['preparation_sha256']==sha(subject.OUT/'preparation.json')
    for filename,digest in marker['outputs'].items():assert sha(folder/filename)==digest
    for relative,digest in protocol['source_hashes'].items():assert sha(ROOT/relative)==digest
    for filename,digest in read(subject.OUT/'preparation.json')['outputs'].items():assert sha(subject.OUT/filename)==digest
    x=pd.read_parquet(subject.OUT/'training_features.parquet');xr=pd.read_parquet(subject.OUT/'ranking_features.parquet')
    meta=pd.read_parquet(subject.OUT/'training_meta.parquet');ranking=pd.read_parquet(subject.OUT/'ranking_missing_meta.parquet').set_index(ID)
    np.testing.assert_array_equal(x.index,meta.index);np.testing.assert_array_equal(xr.index,ranking.index)
    encoder=joblib.load(folder/'normalized_encoder.joblib')
    for name,mapping in encoder.categories.items():
        assert mapping=={value:i+2 for i,value in enumerate(sorted(x[name].astype('string').dropna().unique()))}
    assert encoder.numeric==[name for name in x if name not in encoder.categories]
    schedule=x.schedule_proxy_sec.to_numpy(float)
    scale=np.sqrt(3600.**2+np.where(np.isfinite(schedule)&(schedule!=-999999),schedule-900.,0.)**2)
    normalizer=float((scale**2).mean());evidence=read(folder/'normalized_fit.json')
    assert normalizer==evidence['normalizer'] and evidence['steps']==111
    nmodel=lightgbm.Booster(model_file=str(folder/'normalized.txt'))
    assert nmodel.num_trees()==111 and nmodel.feature_name()==list(x)
    queryschedule=xr.schedule_proxy_sec.to_numpy(float)
    queryscale=np.sqrt(3600.**2+np.where(np.isfinite(queryschedule)&(queryschedule!=-999999),queryschedule-900.,0.)**2)
    npred=900.+queryscale*nmodel.predict(encoder.transform(xr),num_threads=1)
    saved=pd.read_parquet(folder/'normalized_ranking.parquet');np.testing.assert_array_equal(saved[ID],xr.index)
    np.testing.assert_array_equal(npred,saved.prediction_sec)
    forest=joblib.load(folder/'forest.joblib')
    assert forest['family']=='extratrees' and forest['steps']==1 and len(forest['estimator'].estimators_)==300
    assert forest['estimator'].get_params()['min_samples_leaf']==1
    ex=x.copy();ex[subject.forest.TIME_COLUMN]=meta[subject.TIME].to_numpy()
    er=xr.copy();er[subject.forest.TIME_COLUMN]=ranking[subject.TIME].to_numpy()
    raw=subject.forest.input_frame(ex)
    # Independent native prior-table reconstruction from every full-year missing label.
    tables=forest['prior'].tables
    assert forest['prior'].history_n==len(x)==22470 and forest['prior'].shrinkage==10
    assert forest['prior'].global_mean==float(meta[TARGET].mean())
    assert forest['prior'].last_fit==meta[subject.TIME].max()
    assert ranking[subject.TIME].min()>forest['prior'].last_fit
    data=raw.copy();data['_target']=meta[TARGET].to_numpy(float);data['_time']=pd.to_datetime(meta[subject.TIME].to_numpy(),utc=True)
    for keys,table in tables:
        wanted=data.sort_values('_time').groupby(keys,observed=True,dropna=False).agg(mean=('_target','mean'),n=('_target','size'),last=('_target','last'),time=('_time','max')).reset_index()
        pd.testing.assert_frame_equal(wanted,table)
    cross=subject.shared.base.crossfit_templates(raw,meta[TARGET].to_numpy(float),meta[subject.TIME])
    full=pd.concat([raw,cross],axis=1)
    transform=forest['encoder']
    for name,sub,cols in transform.transformers_:
        if name=='numeric':
            for j,col in enumerate(cols):
                values=full[col].to_numpy(dtype=sub.statistics_.dtype);values=values[np.isfinite(values)]
                expected=float(np.median(values)) if len(values) else 0.
                assert sub.statistics_[j]==expected,(col,sub.statistics_[j],expected)
        elif name=='category':
            categories=sub.named_steps['onehot'].categories_
            imputed=sub.named_steps['impute'].transform(full[cols])
            for j,col in enumerate(cols):np.testing.assert_array_equal(categories[j],np.unique(imputed[:,j]))
    forest['estimator'].set_params(n_jobs=1)
    epred=subject.forest.predict(forest,er)
    expected=pd.read_parquet(folder/'forest_ranking.parquet');np.testing.assert_array_equal(expected[ID],xr.index)
    delta=float(np.max(np.abs(epred-expected.prediction_sec.to_numpy(float))))
    assert delta<=1e-9
    v2=pd.read_parquet(ROOT/'private_runs/submission_v2/ranking_predictions.parquet').set_index(ID).loc[xr.index,'prediction_sec'].to_numpy(float)
    # Use retained native predictions for exact producer-order composition; native one-thread difference is bounded separately.
    composed=.75*(.75*v2+.25*expected.prediction_sec.to_numpy(float))+.25*npred
    final=pd.read_parquet(folder/'missing_ranking.parquet');np.testing.assert_array_equal(final[ID],xr.index)
    np.testing.assert_array_equal(composed,final.prediction_sec)
    receipt=dict(status='passed',manifest_sha256=sha(folder/'manifest.json'),protocol_sha256=sha(subject.OUT/'protocol.json'),
        verifier_sha256=sha(__file__),normalized_native_max_abs_delta=0.,forest_native_max_abs_delta=delta,
        prediction_sha256=sha(folder/'missing_ranking.parquet'),training_rows=len(x),ranking_rows=len(xr),
        normalized_fit_categories_and_scale_exact=True,forest_fit_templates_medians_categories_exact=True,
        template_query_chronology_passed=True,nested_composition_exact=True,peak_bytes=psutil.Process().memory_info().peak_wset,
        cpu_threads=1,gpu_used=False,model_training_used=False)
    assert receipt['peak_bytes']<3*1024**3
    output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
