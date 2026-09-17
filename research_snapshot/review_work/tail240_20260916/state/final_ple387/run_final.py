"""Frozen PLE8 union387 full2025 fit and ranking component, fixed19 epochs."""
import torch
import lightgbm
import argparse
import gc
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from threadpoolctl import threadpool_limits
import sources

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/neural_context'))
import run_f3_v3 as memory
context=memory.frozen
common,risk,adapter= context.common,context.risk,context.adapter
ID,TARGET=context.ID,context.TARGET
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/state/final_ple387/model_v1')
REFERENCE=ROOT/'private_runs/tail240_20260916/state/neural_context/v1/F1'


def fixed_epochs(values):
    return int(np.median(values))


def guard():
    p=psutil.Process().memory_info()
    assert max(p.rss,p.peak_wset)<24*1024**3, f'24GiB process cap {p}'
    assert psutil.virtual_memory().available>=8*1024**3, '8GiB host reserve'
    return dict(rss_bytes=p.rss,peak_bytes=p.peak_wset,available_bytes=psutil.virtual_memory().available)


def declare():
    refit=common.read_json(ROOT/'private_runs/tail240_20260916/state/neural_context/refit_score_v3/protocol.json')
    epochs=[refit['folds'][f]['epochs'] for f in ['F1','F3']]
    assert epochs==[20,19] and fixed_epochs(epochs)==19
    for name,module in [('adapter',adapter),('trainer',context.tabm_gpu),('encoder',context.encoders),('loader',risk)]:
        assert common.sha256(module.__file__)==refit['source_chain'][name]
    assert common.sha256(memory.prior.__file__)==refit['source_chain']['grouped_bins']
    paths=[Path(__file__),Path(sources.__file__),Path(context.__file__),Path(memory.prior.__file__),
           Path(adapter.__file__),Path(context.tabm_gpu.__file__),Path(context.encoders.__file__),Path(risk.__file__),
           Path(__file__).with_name('test_final.py')]
    record=dict(sources={str(p.relative_to(ROOT)):common.sha256(p) for p in paths},
        source_refit_protocol_sha256=common.sha256(ROOT/'private_runs/tail240_20260916/state/neural_context/refit_score_v3/protocol.json'),
        reference_manifest_sha256=common.sha256(REFERENCE/'manifest.json'),
        ranking_cache_manifest_sha256=common.sha256(sources.PREP/'manifest.json'),
        columns=refit['feature_columns'],fold_epochs=epochs,final_epochs=19,
        epoch_rule='Inherited release int(np.median([20,19]))=19; explicitly authorized by parent, no score-based adjustment.',
        model='Exact originalPLE8 3x256/dropout.1/48bins/embed16,seed20260916,AdamW.001/.0001,batch4096,clipnorm10;19fixedepochs,noearlystop/tune/score/weightsearch.',
        fit='Every original2025finiteNM row2062577; rawY-minus-NM target, no clipping/sampling. Categories, medians, scaling and bins fittedonlyonall2025training. Originalsource-specificsentinelinterpretation.',
        ranking='All339551finiteNM ranking rows. No hiddenrankinglabels/DEPblock. Parent composesonlyordinaryroute withoriginalweightedensemble; output is unroundedcomponent.',
        parity='Replay savedoriginalPLE387 F1fitmodel on all180640June inputs fromfull-yearcache beforetraining, then serialize/native replay allrankingpredictions exact.',
        resources='ExclusiveGPU,2CPU,<24GiBcurrent/historicalprocesspeak,start>=32GiBavailable,>=8GiBhostreserve;guardload/prepare and every200optimizersteps. No otherfit beforeparentallocation.',
        authority='Userauthorized full-year best268.662991recipe and one newofficialsubmission; parentownscomposition/upload, no externalrawdata transfer.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record,'Frozen final protocol changed'
    else:common.write_json(path,record)
    return record


def run():
    protocol=declare()
    assert psutil.virtual_memory().available>=32*1024**3
    dest=OUT/'fit';dest.mkdir(exist_ok=False)
    started=time.monotonic()
    common.write_json(dest/'launch.json',dict(created_utc=common.utc_now(),protocol_sha256=common.sha256(OUT/'protocol.json'),resources=guard()))
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(metadata)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    train=pd.read_parquet(metadata,columns=[ID,common.MOVEMENT,'proxy_sec',TARGET])
    assert len(train)==2085047 and train[common.MOVEMENT].dt.year.eq(2025).all()
    train=train.loc[np.isfinite(train.proxy_sec)].copy();assert len(train)==2062577
    rank_path=ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    binding=common.read_json(ROOT/'private_runs/submission_v2/ranking_inputs.json')
    assert common.sha256(rank_path)==binding['files'][rank_path.name]
    ranking=pd.read_parquet(rank_path,columns=[ID,common.MOVEMENT,'proxy_sec'])
    assert len(ranking)==344841
    ranking=ranking.loc[np.isfinite(ranking.proxy_sec)].copy();assert len(ranking)==339551
    ids=pd.Index(pd.concat([train[ID],ranking[ID]],ignore_index=True));assert ids.is_unique
    nfit=len(train)
    risk.feature_sources=sources.feature_sources
    risk.guard=lambda:guard()['rss_bytes']
    matrix,vocab,receipts=risk.load_matrix(ids,nfit,protocol['columns'],dest)
    x=context.decode_frame(matrix,vocab,protocol['columns']);x.index=ids
    xf,xr=x.iloc[:nfit],x.iloc[nfit:]
    guard()
    ref=common.read_json(REFERENCE/'manifest.json')
    assert common.sha256(REFERENCE/'manifest.json')==protocol['reference_manifest_sha256']
    for name in ['fit_model.joblib','tune_predictions.parquet']:assert common.sha256(REFERENCE/name)==ref['outputs'][name]
    expected=pd.read_parquet(REFERENCE/'tune_predictions.parquet')
    assert expected[ID].isin(train[ID]).all() and len(expected)==180640
    old=joblib.load(REFERENCE/'fit_model.joblib')
    old_prediction=adapter.predict(old,xf.loc[expected[ID]])+expected.proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(old_prediction,expected.prediction_sec.to_numpy())
    common.write_json(dest/'reference_replay.json',dict(rows=len(expected),max_abs_delta_sec=0.,reference_manifest_sha256=protocol['reference_manifest_sha256']))
    del old,old_prediction,expected
    gc.collect();torch.cuda.empty_cache();guard()
    adapter.fit_bins=memory.prior.grouped_fit_bins
    original_step=torch.optim.AdamW.step
    updates=[0]
    def guarded_step(self,*args,**kwargs):
        if updates[0]%200==0:guard()
        result=original_step(self,*args,**kwargs);updates[0]+=1
        return result
    torch.optim.AdamW.step=guarded_step
    try:
        model,evidence=adapter.fit(xf,(train[TARGET]-train.proxy_sec).to_numpy(float),steps=19,seed=20260916,threads=2)
    finally:torch.optim.AdamW.step=original_step
    assert model['steps']==19 and evidence['rows']==2062577 and len(evidence['history'])==19
    guard();joblib.dump(model,dest/'model.joblib')
    predicted=adapter.predict(model,xr)+ranking.proxy_sec.to_numpy(float)
    reloaded=joblib.load(dest/'model.joblib')
    replay=adapter.predict(reloaded,xr)+ranking.proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(predicted,replay);assert np.isfinite(predicted).all()
    ranking.assign(prediction_sec=predicted).to_parquet(dest/'ranking_predictions.parquet',index=False)
    common.write_json(dest/'fit_evidence.json',evidence)
    common.write_json(dest/'encoder_categories.json',dict(columns=protocol['columns'],vocab=vocab))
    result=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),
        training_rows=nfit,ranking_finite_rows=len(ranking),epochs=19,training_ids_hash=common.object_hash(train[ID].tolist()),
        training_label_hash=common.object_hash(train[TARGET].tolist()),ranking_ids_hash=common.object_hash(ranking[ID].tolist()),
        feature_columns=protocol['columns'],feature_receipts=receipts,native_reload_max_abs_delta_sec=0.,
        training_updates=updates[0],resources=guard(),elapsed_seconds=time.monotonic()-started,
        outputs={name:common.sha256(dest/name) for name in ['model.joblib','ranking_predictions.parquet','fit_evidence.json','encoder_categories.json','reference_replay.json']})
    common.write_json(dest/'manifest.json',result)
    print('FINAL_PLE_COMPLETE',nfit,len(ranking),result['elapsed_seconds'],result['resources'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);pa.set_cpu_count(2);pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare_only:declare();print(common.sha256(OUT/'protocol.json'))
        else:run()
