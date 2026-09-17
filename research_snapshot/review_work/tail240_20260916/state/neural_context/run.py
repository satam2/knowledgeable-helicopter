"""Exact PLE8 architecture with union387 information, full finite tune-only study."""
import torch
import lightgbm
import argparse
import gc
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import tabm_gpu
import tabm_ple_gpu as adapter
import encoders
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/risk'))
import run_risk as risk
common, ID, TARGET = risk.common, risk.ID, risk.TARGET
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/neural_context/v1')


def guard():
    rss = psutil.Process().memory_info().rss
    assert rss < 20*1024**3, f'RSS budget exceeded: {rss}'
    assert psutil.virtual_memory().available >= 8*1024**3, 'Host reserve below8GiB'
    return int(rss)


def control_folder(fold):
    return ROOT/'private_runs/breakthrough_20260916/full_neural'/f'tabm_combined_ple8_aobt_allfinite_{fold}_s20260916'


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    control = common.read_json(control_folder('F1')/'manifest.json')
    full = common.read_json(risk.union_folder('F1')/'manifest.json')['feature_columns']
    assert len(full)==387 and full[:225] == control['feature_columns']
    for name, path in [('tabm_gpu.py',tabm_gpu.__file__),('encoders.py',encoders.__file__),
        ('review_work\\breakthrough_20260916\\models\\tabm_ple_gpu.py',adapter.__file__)]:
        assert common.sha256(path) == control['source_hashes'][name]
    record = dict(source_sha256=common.sha256(__file__),
        adapter_sha256=common.sha256(adapter.__file__), trainer_sha256=common.sha256(tabm_gpu.__file__),
        encoder_sha256=common.sha256(encoders.__file__), loader_sha256=common.sha256(risk.__file__),
        params=control['fit']['params'], numeric_bins=48, numeric_embedding_dim=16, members=8,
        feature_columns=full, control_columns=full[:225], folds=['F1','F3'], stages=['fit','tune'],
        scope='All original purged finiteNM fit/tune rows, rawY-P residual; no score/refit/ranking predictions. Missingroute untouched.',
        preprocessing='Frozenfit-onlyFrameEncoder median/mean/std/missingindicators plusPLE48bins/16dims, categories; exactfirst225controlreplay beforefit.',
        training='Unmodified frozenPLE8adapter/TabMtrainer; batch4096/40epochcap/5patience, rawMSEaffinescale, seed20260916,2CPU, no sampling/grid.',
        control='Reusescomplete savedPLE8combined225 fit-model andfulltune predictions withhash/cohort/source/nativeGPUparity; no new225fit. SameGPUtolerance1e-6seconds beforeadvance.',
        evaluation='Fullfinite andordinaryproxy0..7200. CandidatevsPLE225. Global9ordinary currentfixedweights replacementofitsPLE225component andfixed25candidate/global9blend; no weightrefit. Same-tuneensembleweights caveat explicit.',
        advancement='No automatic scoreadvance. Bothmonthsgain plusday-removalstability and>=2secondseasonalpointmaterialityagainstcomparisonrequired; parentmustseparatelydeclare anyrefit/score.',
        novelty='Boundedsources/manifests show existingstandard/PLE8 neuralcombined225; ordered337/449 andunion387 trainedLightGBM only. ExistingGRU32event architectureisnot samePLE8informationtest.',
        resources='ExclusiveCUDA GPU, unchangedbatch4096;2CPU;sampled20GiBRSSlimit and8GiBhostreserve. GuardsnotOSenforced.')
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


def decode_frame(matrix,vocab,columns):
    values = {}
    for j,name in enumerate(columns):
        if name in vocab:
            known = vocab[name]
            assert '__UNSEEN_CONTEXT_VALUE__' not in known
            base_categories={'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt',
                'AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','airport_stand','airport_runway'}
            missing=np.nan if name in base_categories else 'MISSING'
            labels = np.asarray([missing,'__UNSEEN_CONTEXT_VALUE__',*known],dtype=object)
            values[name] = pd.Categorical(labels[matrix[:,j].astype(np.int32)])
        else:
            values[name] = matrix[:,j]
    return pd.DataFrame(values,copy=False)


def paired(y,candidate,control,dates):
    a,b=(y-candidate)**2,(y-control)**2
    codes,days=pd.factorize(dates,sort=True)
    weights=np.random.default_rng(20260916).multinomial(len(days),np.full(len(days),1/len(days)),size=300)
    removes=[]
    for k,day in enumerate(days):
        mask=codes!=k
        removes.append(dict(day=str(day),gain=float(np.sqrt(b[mask].mean())-np.sqrt(a[mask].mean()))))
    influence={}
    ranked=np.argsort(-(b-a),kind='stable')
    for k in [1,5,10]:
        keep=np.ones(len(y),bool);keep[ranked[:k]]=False
        influence[str(k)]=float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))
    return dict(candidate_rmse=float(np.sqrt(a.mean())),control_rmse=float(np.sqrt(b.mean())),
        gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())),
        mse_gain_ci95=risk.bootstrap_interval(b-a,np.ones(len(y)),codes,weights),
        day_removals=removes,all_day_removals_improve=all(row['gain']>0 for row in removes),
        remove_top_beneficial_rows_gain=influence)


def run(fold):
    protocol=freeze()
    folder=common.external_path(OUT/fold)
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    risk.guard=guard
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    parts={stage:meta.iloc[idx[stage][np.isfinite(meta.iloc[idx[stage]].proxy_sec)]].copy() for stage in ['fit','tune']}
    fit,tune=parts['fit'],parts['tune']
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True))
    desired=protocol['feature_columns']
    del meta,parts
    gc.collect()
    matrix,vocab,receipts=risk.load_matrix(ids,len(fit),desired,folder)
    x=decode_frame(matrix,vocab,desired)
    x.index=ids
    xf,xt=x.iloc[:len(fit)],x.iloc[len(fit):]
    reference_folder=control_folder(fold)
    control=common.read_json(reference_folder/'manifest.json')
    assert control['feature_columns']==desired[:225]
    for name in ['fit_model.joblib','tune_predictions.parquet']:
        assert common.sha256(reference_folder/name)==control['outputs'][name]
    saved=pd.read_parquet(reference_folder/'tune_predictions.parquet').set_index(ID)
    assert saved.index.is_unique and set(saved.index)==set(tune[ID])
    assert len(fit)==control['fit']['rows']
    base_model=joblib.load(reference_folder/'fit_model.joblib')
    base_pred=tabm_gpu.predict(base_model,xt[desired[:225]])+tune.proxy_sec.to_numpy(float)
    old_pred=saved.loc[tune[ID],'prediction_sec'].to_numpy()
    delta=float(np.max(np.abs(base_pred-old_pred)))
    assert delta<=1e-6, f'Control225 fulltuneGPUparity failed: {delta}'
    common.write_json(folder/'control_replay.json',dict(status='passed',rows=len(tune),max_abs_delta_sec=delta,
        control_manifest_sha256=common.sha256(reference_folder/'manifest.json'),
        control_fit_model_sha256=common.sha256(reference_folder/'fit_model.joblib')))
    del base_model,saved
    gc.collect()
    guard()
    rf=(fit[TARGET]-fit.proxy_sec).to_numpy(float)
    rt=(tune[TARGET]-tune.proxy_sec).to_numpy(float)
    model,evidence=adapter.fit(xf,rf,(xt,rt),seed=20260916,threads=2)
    guard()
    pred=adapter.predict(model,xt)+tune.proxy_sec.to_numpy(float)
    joblib.dump(model,folder/'fit_model.joblib')
    reload=joblib.load(folder/'fit_model.joblib')
    replay=adapter.predict(reload,xt)+tune.proxy_sec.to_numpy(float)
    assert np.array_equal(pred,replay)
    outputs=tune[[ID,common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec']].copy()
    outputs['prediction_sec']=pred
    outputs['control225_prediction_sec']=old_pred
    outputs.to_parquet(folder/'tune_predictions.parquet',index=False)
    y=tune[TARGET].to_numpy(float)
    dates=tune[common.MOVEMENT].dt.floor('D').to_numpy()
    ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    reports={name:paired(y[mask],pred[mask],old_pred[mask],dates[mask]) for name,mask in
        [('all_finite',np.ones(len(y),bool)),('ordinary_proxy',ordinary)]}
    ensemble=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=common.read_json(ensemble/'preparation.json')['folds'][fold]
    aligned_path=ensemble/(fold+'_aligned_tune.parquet')
    assert common.sha256(aligned_path)==prep['aligned_tune_sha256']
    weight=common.read_json(ensemble/(fold+'_weights.json'))
    assert weight==prep['weights']
    aligned=pd.read_parquet(aligned_path).set_index(ID)
    assert set(aligned.index)==set(tune.loc[ordinary,ID])
    own=outputs.set_index(ID).loc[aligned.index]
    assert np.array_equal(own[TARGET],aligned[TARGET])
    global9=aligned[weight['experts']].to_numpy()@np.asarray(weight['global'])
    replacement=global9+weight['global'][weight['experts'].index('tabm_ple8')]*(own.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    blend=.25*own.prediction_sec.to_numpy()+.75*global9
    ensemble_dates=aligned[common.MOVEMENT].dt.floor('D').to_numpy()
    reports['global9_replacement']=paired(aligned[TARGET].to_numpy(),replacement,global9,ensemble_dates)
    reports['global9_blend25']=paired(aligned[TARGET].to_numpy(),blend,global9,ensemble_dates)
    reports['global9_limitation']='Frozenweightswerefitonsametunelabels;replacement/blend diagnosticnotfreshcalibrationholdout. Noensembleweightupdated.'
    common.write_json(folder/'fit_evidence.json',evidence)
    common.write_json(folder/'metrics.json',reports)
    common.write_json(folder/'manifest.json',dict(status='complete',fold=fold,source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),feature_columns=desired,feature_receipts=receipts,split=split,
        fit_ids_hash=common.object_hash(fit[ID].tolist()),tune_ids_hash=common.object_hash(tune[ID].tolist()),
        fit_label_hash=common.object_hash(fit[TARGET].tolist()),tune_label_hash=common.object_hash(tune[TARGET].tolist()),
        full_finite_fit_rows=len(fit),full_finite_tune_rows=len(tune),selected_epochs=model['steps'],
        native_reload_max_delta_sec=0.,sampled_rss_bytes=guard(),runtime_sec=time.monotonic()-started,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
        outputs={name:common.sha256(folder/name) for name in ['control_replay.json','fit_model.joblib','tune_predictions.parquet','fit_evidence.json','metrics.json']}))
    del model,reload,x,xf,xt,matrix
    gc.collect()
    (folder/'matrix.float32').unlink()
    print('COMPLETE',fold,{key:value['gain'] for key,value in reports.items() if isinstance(value,dict)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    pa.set_cpu_count(2);pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare_only:
            print(freeze())
        else:
            assert args.fold
            run(args.fold)
