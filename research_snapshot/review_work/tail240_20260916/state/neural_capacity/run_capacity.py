"""Full387 PLE32 fit/tune with separate no-optimizer canary and current387 endpoint."""
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
import capacity_adapter as adapter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/state/neural_context'))
import run as context
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import schema_discovery
common, risk = context.common, context.risk
ID, TARGET, TIME = context.ID, context.TARGET, common.MOVEMENT
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/neural_capacity/v1')
CONTROLS = {f:ROOT / 'private_runs/tail240_20260916/state/neural_context' / v / f for f,v in [('F1','v1'),('F3','v3')]}
ENSEMBLE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'


def guard(fold):
    info = psutil.Process().memory_info()
    cap = 12 if fold == 'F1' else 20
    assert max(info.rss, info.peak_wset) < cap*1024**3, f'{cap}GiB process budget {info}'
    available = psutil.virtual_memory().available
    assert available >= 8*1024**3, 'Host reserve below8GiB'
    return dict(rss_bytes=info.rss, peak_wset_bytes=info.peak_wset, available_bytes=available)


def declare():
    files = [Path(__file__), Path(adapter.__file__), Path(adapter.frozen.__file__), Path(adapter.ple.__file__),
        Path(sys.modules['encoders'].__file__), Path(context.__file__), Path(risk.__file__), Path(schema_discovery.__file__),
        Path(__file__).with_name('test_capacity.py')]
    controls = {}
    columns = None
    for fold, folder in CONTROLS.items():
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete'
        if columns is None:
            columns = marker['feature_columns']
        assert marker['feature_columns'] == columns and len(columns)==387
        controls[fold] = dict(manifest_sha256=common.sha256(folder / 'manifest.json'),
            model_sha256=marker['outputs']['fit_model.joblib'], predictions_sha256=marker['outputs']['tune_predictions.parquet'],
            fit_ids_hash=marker['fit_ids_hash'], tune_ids_hash=marker['tune_ids_hash'],
            fit_label_hash=marker['fit_label_hash'], tune_label_hash=marker['tune_label_hash'],
            fit_rows=marker['full_finite_fit_rows'], tune_rows=marker['full_finite_tune_rows'],
            global9_weights_sha256=common.sha256(ENSEMBLE / f'{fold}_weights.json'),
            global9_tune_sha256=common.sha256(ENSEMBLE / f'{fold}_aligned_tune.parquet'))
    value = dict(source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in files}, controls=controls,
        feature_columns=columns, folds=['F1','F3'], stages=['fit','tune'],
        model_change='ONLY PLE static ensemble members8to32. Originalple.Network3x256/dropout.1/48bins/embed16; no attention or new input; freshseed20260916.',
        training='Originalfullfinitepurged fit/tune IDs, rawY-P, fit-onlyFrameEncoder and allfitofficialPLEbins grouped16columns; frozenAdamW.001/.0001/batch4096/rawMSE/clipnorm10/max40/patience5/tuneepochselection;2CPU',
        controls_native='Every savedPLE8union387 tune prediction <=1e-6seconds beforecanary/fit; candidate native reload exact',
        primary='Everyordinarytunerow: global9+wPLE*(PLE32-old225) versus CURRENT387ensemble=global9+wPLE*(savedPLE8_387-old225), frozenoriginalweights',
        secondary='Matched32vs8 on fullfiniteandordinary; standalonewinnotrequiredwhenprimarycomplementaritypasses; noalternateendpointpromotion',
        gate='Primary current387ensemble gain>0bothmonths/allleaveonedayout and>=2seasonalordinaryRMSEseconds. Negative=>noscore. No refit/ranking stageauthorized.',
        season_weights=[192122/344841,152719/344841],
        limitations='Exposeddevelopment tune labels; global9weights learnedonsametune; current387usedsamevalidationforearlystop. Notfreshholdout orofficialscore.',
        canary='Separate --canary action fitsallfit preprocessing andnativeold8control then8192inference+4096backward with32members,finiteoutput/grads,nooptimizer;<12GiBallocatedGPU;savepreparation+hashes thenexit; --fit requirescompletedboundcanaryforthatfold.',
        resources='ExclusiveGPUaftercentralallocation;2CPU;F1currentRSS/historicalpeak12GiB and>=20GiBstartup;F3cap20GiB/>=28GiBstartup laterseparateallocation;8GiBhostreserve. GuardsperiodicnotOSenforced.')
    OUT.mkdir(parents=True, exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==value, 'Frozen source/protocol changed'
    else:
        common.write_json(path,value)
    return value


def data(fold, folder, protocol):
    required = 20 if fold=='F1' else 28
    assert psutil.virtual_memory().available >= required*1024**3
    folder.mkdir(parents=True, exist_ok=False)
    common.write_json(folder/'launch.json',dict(created_utc=common.utc_now(),resources=guard(fold),protocol_sha256=common.sha256(OUT/'protocol.json')))
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(metadata)=='503df9cb08f3ed7260fd7d969a3505d7b58434e20bccc80f1526d517a292e976'
    meta=pd.read_parquet(metadata,columns=[ID,'FLIGHT_ID_mvt',TIME,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    parts={s:meta.iloc[idx[s][np.isfinite(meta.iloc[idx[s]].proxy_sec)]].copy() for s in ['fit','tune']}
    binding=protocol['controls'][fold]
    for stage,frame in parts.items():
        assert len(frame)==binding[stage+'_rows']
        assert common.object_hash(frame[ID].tolist())==binding[stage+'_ids_hash']
        assert common.object_hash(frame[TARGET].tolist())==binding[stage+'_label_hash']
    fit,tune=parts['fit'],parts['tune']
    del meta,parts
    gc.collect()
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True))
    original=risk.feature_sources
    oldguard=risk.guard
    def cached(columns):
        sources,receipt=schema_discovery.cached_discovery(original,columns)
        common.write_json(folder/'discovery.json',receipt)
        guard(fold)
        return sources
    risk.feature_sources=cached
    risk.guard=lambda:guard(fold)
    try:
        matrix,vocab,receipts=risk.load_matrix(ids,len(fit),protocol['feature_columns'],folder)
    finally:
        risk.feature_sources=original
        risk.guard=oldguard
    frame=context.decode_frame(matrix,vocab,protocol['feature_columns'])
    frame.index=ids
    xf,xt=frame.iloc[:len(fit)],frame.iloc[len(fit):]
    source=CONTROLS[fold]
    for name,key in [('manifest.json','manifest_sha256'),('fit_model.joblib','model_sha256'),('tune_predictions.parquet','predictions_sha256')]:
        assert common.sha256(source/name)==binding[key]
    old=joblib.load(source/'fit_model.joblib')
    stored=pd.read_parquet(source/'tune_predictions.parquet').set_index(ID).loc[tune[ID]]
    np.testing.assert_array_equal(stored[TARGET],tune[TARGET])
    replay=adapter.predict(old,xt)+tune.proxy_sec.to_numpy(float)
    delta=float(np.max(np.abs(replay-stored.prediction_sec.to_numpy())))
    assert delta<=1e-6
    common.write_json(folder/'control_replay.json',dict(rows=len(tune),max_abs_delta_sec=delta,model_sha256=binding['model_sha256']))
    del old,replay
    gc.collect()
    guard(fold)
    return fit,tune,xf,xt,matrix,receipts,split,stored.prediction_sec.to_numpy()


def canary(fold):
    protocol=declare()
    folder=common.external_path(OUT/'canary'/fold)
    fit,tune,xf,xt,matrix,receipts,split,old=data(fold,folder,protocol)
    preparation=adapter.prepare(xf)
    guard(fold)
    joblib.dump(preparation,folder/'preparation.joblib')
    result=adapter.canary(xf,preparation)
    result.update(status='passed',fold=fold,protocol_sha256=common.sha256(OUT/'protocol.json'),
        fit_rows=len(fit),fit_ids_hash=protocol['controls'][fold]['fit_ids_hash'],
        tune_rows=len(tune),fit_label_hash=protocol['controls'][fold]['fit_label_hash'],
        control_receipt_sha256=common.sha256(folder/'control_replay.json'),
        preparation_sha256=common.sha256(folder/'preparation.joblib'),
        bin_hash=common.object_hash([v.tolist() for v in preparation['bins']]),
        canary_ids_hash=common.object_hash(fit[ID].iloc[:8192].tolist()),
        input_hash=common.object_hash([v.tolist() for v in adapter.frozen.tensors(preparation['encoder'],xf.iloc[:8192])]),
        resources=guard(fold),feature_receipts=receipts,split=split)
    common.write_json(folder/'receipt.json',result)
    print('CANARY_COMPLETE',fold,result['gpu_peak_allocated_bytes'],result['resources'],flush=True)


def fit_fold(fold):
    protocol=declare()
    canarypath=OUT/'canary'/fold/'receipt.json'
    canaryrecord=common.read_json(canarypath)
    assert canaryrecord['status']=='passed' and canaryrecord['optimizer_updates']==0
    assert canaryrecord['protocol_sha256']==common.sha256(OUT/'protocol.json')
    assert canaryrecord['gpu_peak_allocated_bytes']<12*1024**3
    folder=common.external_path(OUT/'models'/fold)
    started=time.monotonic()
    fit,tune,xf,xt,matrix,receipts,split,old=data(fold,folder,protocol)
    original_infer=adapter.frozen.infer
    def guarded_infer(*args,**kwargs):
        guard(fold)
        values=original_infer(*args,**kwargs)
        guard(fold)
        return values
    adapter.frozen.infer=guarded_infer
    try:
        model,evidence=adapter.fit(xf,(fit[TARGET]-fit.proxy_sec).to_numpy(float),
            (xt,(tune[TARGET]-tune.proxy_sec).to_numpy(float)),seed=20260916,threads=2)
        predicted=adapter.predict(model,xt)+tune.proxy_sec.to_numpy(float)
        joblib.dump(model,folder/'fit_model.joblib')
        reloaded=joblib.load(folder/'fit_model.joblib')
        np.testing.assert_array_equal(predicted,adapter.predict(reloaded,xt)+tune.proxy_sec.to_numpy(float))
    finally:
        adapter.frozen.infer=original_infer
    output=tune[[ID,TIME,'ADEP_mvt',TARGET,'proxy_sec']].copy()
    output['prediction_sec']=predicted
    output['control8_prediction_sec']=old
    output.to_parquet(folder/'tune_predictions.parquet',index=False)
    ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    days=tune[TIME].dt.floor('D').to_numpy()
    metrics={name:context.paired(tune[TARGET].to_numpy()[selected],predicted[selected],old[selected],days[selected])
        for name,selected in [('all_finite',np.ones(len(tune),bool)),('ordinary',ordinary)]}
    binding=protocol['controls'][fold]
    for name,key in [(f'{fold}_weights.json','global9_weights_sha256'),(f'{fold}_aligned_tune.parquet','global9_tune_sha256')]:
        assert common.sha256(ENSEMBLE/name)==binding[key]
    weights=common.read_json(ENSEMBLE/f'{fold}_weights.json')
    aligned=pd.read_parquet(ENSEMBLE/f'{fold}_aligned_tune.parquet').set_index(ID)
    assert set(aligned.index)==set(tune.loc[ordinary,ID])
    own=output.set_index(ID).loc[aligned.index]
    np.testing.assert_array_equal(own[TARGET],aligned[TARGET])
    global9=aligned[weights['experts']].to_numpy()@np.asarray(weights['global'])
    w=weights['global'][weights['experts'].index('tabm_ple8')]
    current387=global9+w*(own.control8_prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    candidate=global9+w*(own.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    metrics['primary_current387']=context.paired(aligned[TARGET].to_numpy(),candidate,current387,aligned[TIME].dt.floor('D').to_numpy())
    common.write_json(folder/'metrics.json',metrics)
    common.write_json(folder/'fit_evidence.json',evidence)
    common.write_json(folder/'manifest.json',dict(status='complete',fold=fold,
        protocol_sha256=common.sha256(OUT/'protocol.json'),canary_receipt_sha256=common.sha256(canarypath),
        feature_columns=protocol['feature_columns'],feature_receipts=receipts,split=split,
        fit_rows=len(fit),tune_rows=len(tune),fit_ids_hash=binding['fit_ids_hash'],tune_ids_hash=binding['tune_ids_hash'],
        fit_label_hash=binding['fit_label_hash'],tune_label_hash=binding['tune_label_hash'],
        selected_epochs=model['steps'],native_reload_max_delta_sec=0.,resources=guard(fold),
        runtime_sec=time.monotonic()-started,
        outputs={n:common.sha256(folder/n) for n in ['control_replay.json','fit_model.joblib','tune_predictions.parquet','metrics.json','fit_evidence.json']}))
    print('COMPLETE',fold,{k:v['gain'] for k,v in metrics.items()},flush=True)


def assess():
    protocol=declare()
    folds={}
    for fold in FOLDS:
        folder=OUT/'models'/fold
        marker=common.read_json(folder/'manifest.json')
        assert marker['status']=='complete' and marker['protocol_sha256']==common.sha256(OUT/'protocol.json')
        for name,digest in marker['outputs'].items():
            assert common.sha256(folder/name)==digest
        folds[fold]=common.read_json(folder/'metrics.json')
    results={}
    for endpoint in ['all_finite','ordinary','primary_current387']:
        candidate=float(np.sqrt(sum(w*folds[f][endpoint]['candidate_rmse']**2 for f,w in zip(FOLDS,protocol['season_weights']))))
        control=float(np.sqrt(sum(w*folds[f][endpoint]['control_rmse']**2 for f,w in zip(FOLDS,protocol['season_weights']))))
        results[endpoint]=dict(candidate_rmse=candidate,control_rmse=control,gain=control-candidate,
            both_months_positive=all(folds[f][endpoint]['gain']>0 for f in FOLDS),
            all_day_removals_positive=all(folds[f][endpoint]['all_day_removals_improve'] for f in FOLDS))
    primary=results['primary_current387']
    report=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),seasonal=results,
        gate_passed=primary['gain']>=2 and primary['both_months_positive'] and primary['all_day_removals_positive'],
        no_score_authorization=True,folds=folds)
    path=OUT/'assessment.json'
    assert not path.exists()
    common.write_json(path,report)
    print('ASSESSMENT',report['gate_passed'],results,flush=True)


FOLDS=['F1','F3']
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--declare-only',action='store_true')
    mode.add_argument('--canary',action='store_true')
    mode.add_argument('--fit',action='store_true')
    mode.add_argument('--assess',action='store_true')
    parser.add_argument('--fold',choices=FOLDS)
    args=parser.parse_args()
    torch.set_num_threads(2)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare_only:
            declare()
            print(common.sha256(OUT/'protocol.json'),flush=True)
        elif args.assess:
            assess()
        else:
            assert args.fold
            (canary if args.canary else fit_fold)(args.fold)
