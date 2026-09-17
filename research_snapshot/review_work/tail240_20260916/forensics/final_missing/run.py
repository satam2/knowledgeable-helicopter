"""Exact full-2025 missing route transfer, preparation separated from fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '2'
import lightgbm as lgb
import argparse
from pathlib import Path
import sys
import importlib.util
import gc
import threading
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as normalized
import features
common, shared = normalized.common, normalized.shared
ID,TARGET,TIME = normalized.ID,normalized.TARGET,normalized.TIME
spec = importlib.util.spec_from_file_location('final_missing_frozen_forest',ROOT/'review_work/breakthrough_20260916/models/missing_forest/adapter.py')
forest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forest)
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/forensics/final_missing/v1')
RAW = ROOT/'data/09-15-2026-18-55-03_files_list'
V2 = ROOT/'private_runs/submission_v2'
ET = ROOT/'private_runs/breakthrough_20260916/models/missing_forest'
FIELDS = [ID,'PHASE_mvt',TIME,'AOBT_3_flt','SCHED_TIME_UTC_mvt','ADEP_mvt','ADES_mvt',
          'STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt','FLIGHT_RULE_mvt']
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def guard(cap=4):
    memory = psutil.Process().memory_info()
    peak = max(memory.rss,getattr(memory,'peak_wset',memory.rss))
    if peak >= cap*1024**3 or psutil.virtual_memory().available < 8*1024**3:
        raise MemoryError(f'Final missing process {cap}GiB / reserve8GiB guard')
    return peak


def declare():
    steps, et_steps, bindings = {}, {}, {}
    for fold in ['F1','F3']:
        nfolder = normalized.OUT/fold/'observed_schedule_scale'
        efolder = ET/f'extratrees_missing_template_idcontext_{fold}_s20260916'
        n,e = common.read_json(nfolder/'manifest.json'),common.read_json(efolder/'manifest.json')
        assert n['params'] == normalized.PARAMS and n['status'] == e['status'] == 'complete'
        assert e['fit']['params'] == e['refit']['params'] == dict(n_estimators=300,max_features=.7,min_samples_leaf=1,criterion='squared_error',n_jobs=2,bootstrap=False)
        steps[fold],et_steps[fold] = n['steps'],e['fit']['steps']
        bindings[fold] = dict(normalized_manifest_sha256=common.sha256(nfolder/'manifest.json'),
                              forest_manifest_sha256=common.sha256(efolder/'manifest.json'))
    assert steps == {'F1':77,'F3':146} and et_steps == {'F1':1,'F3':1}
    base = common.read_json(V2/'submission_ready.json')
    assert base['passed'] and base['rows'] == 344841
    assert common.sha256(V2/'ranking_predictions.parquet') == base['ranking_predictions_sha256']
    files = [Path(__file__),Path(features.__file__),Path(__file__).with_name('test_features.py'),
             Path(normalized.__file__),Path(shared.__file__),Path(shared.base.__file__),Path(shared.identity.__file__),
             Path(forest.__file__),ROOT/'review_work/breakthrough_20260916/models/encoders.py',
             ROOT/'review_work/mechanism_20260916/id_neighborhood.py']
    record = dict(source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in files},
        raw_hashes=common.read_json(V2/'protocol.json')['raw_hashes'],
        v2_ready_sha256=common.sha256(V2/'submission_ready.json'),v2_prediction_sha256=base['ranking_predictions_sha256'],
        id_training_audit_sha256=common.sha256(shared.identity.CACHE/'audit.json'),
        ranking_input_manifest_sha256=common.sha256(V2/'ranking_inputs.json'),
        feature_columns=common.read_json(ROOT/'private_runs/tail240_20260916/state/models_v2/control/F1/manifest.json')['feature_columns'],
        frozen_folds=bindings,normalized_steps=111,normalized_fold_steps=steps,
        transfer_rule='int(median(frozen F1/F3 tune selections)); inherited final-release integer-median policy; no new tuning',
        normalized_params=normalized.PARAMS,forest_leaf=1,forest_trees=300,forest_fold_leaves=et_steps,
        train='All22470 full2025 missing-NM rows, original raw targets, no filtering/clipping/sampling; encoders fit fulltraining only',
        ranking='Exactly5290 missing-NM rows among344841rankingDEP; observed2026Jan/Jul only; no ranking target/block reads',
        features='Same76 airport+ID-context fields; ranking peer rules exactly original sameairport+UTCmonth IDsort,selfexcluded widths2/8/32',
        forest='Frozen ExtraTrees300/minleaf1/maxfeatures.7/seed20260916/2CPU; earliermonth crossfit templates on train,full2025templatefor2026ranking',
        normalized='Frozen scale sqrt3600^2+(schedule-900)^2; Z=(Y-900)/scale; weights=scale^2/fitmean;111rounds,no evalset;raw reconstruction900+scale*z',
        composition='missing only: V4=.75V2+.25ET; selected=.75V4+.25normalized =.5625V2+.1875ET+.25normalized; no weight fitting',
        replay='Beforefit, both original normalized/ET tune vectors on exact originalmissing IDs; normalizedexact,ETtol1e-9; finalreloadnormalizedexact/ET1e-9',
        resources='Preparation4GiB/start12;fit6GiB/start14;reserve8;2CPU;worker-local100mscurrent+OSpeak monitor;noGPU')
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path) == record,'Frozen declaration changed'
    else:
        common.write_json(path,record)
    return record


def prepare():
    assert psutil.virtual_memory().available >= 12*1024**3
    protocol = declare()
    assert not (OUT/'preparation.json').exists()
    started = time.monotonic()
    x,meta = shared.load_missing()
    assert list(x) == protocol['feature_columns'] and len(x)==22470
    train_meta = meta.set_index(ID).loc[x.index,[TARGET,TIME,'proxy_sec']]
    jan = meta.loc[pd.to_datetime(meta[TIME],utc=True).dt.month.eq(1),[ID,'ADEP_mvt',TIME]]
    peer = features.record_peers(jan)
    cached = pd.read_parquet(shared.identity.CACHE/'features.parquet').set_index(ID).loc[peer.index]
    pd.testing.assert_frame_equal(peer,cached,check_dtype=True)
    del peer,cached,jan
    fold_replay = {}
    for fold in ['F1','F3']:
        idx,_,_ = common.fold_data(meta,fold,full=True)
        rows = idx['tune'][~np.isfinite(meta.iloc[idx['tune']].proxy_sec.to_numpy(float))]
        tx = x.loc[meta.iloc[rows][ID]]
        nfolder = normalized.OUT/fold/'observed_schedule_scale'
        nm = common.read_json(nfolder/'manifest.json')
        assert common.sha256(nfolder/'model.joblib') == nm['outputs']['model.joblib']
        saved = joblib.load(nfolder/'model.joblib')
        npred = normalized.reconstruct(saved['model'].predict(saved['encoder'].transform(tx)),normalized.scale_of(tx))
        expected = pd.read_parquet(nfolder/'tune.parquet')
        np.testing.assert_array_equal(expected[ID],tx.index)
        np.testing.assert_array_equal(npred,expected.prediction_sec)
        del saved
        efolder = ET/f'extratrees_missing_template_idcontext_{fold}_s20260916'
        em = common.read_json(efolder/'manifest.json')
        for name in ['fit_model.joblib','tune_predictions.parquet']:
            assert common.sha256(efolder/name) == em['outputs'][name]
        saved = joblib.load(efolder/'fit_model.joblib')
        forest_frame = tx.copy()
        forest_frame[forest.TIME_COLUMN] = meta.iloc[rows][TIME].to_numpy()
        epred = forest.predict(saved,forest_frame)
        expected = pd.read_parquet(efolder/'tune_predictions.parquet')
        np.testing.assert_array_equal(expected[ID],tx.index)
        delta = float(np.max(np.abs(epred-expected.prediction_sec.to_numpy())))
        assert delta <= 1e-9
        fold_replay[fold] = dict(rows=len(tx),normalized_max_abs_delta=0.,forest_max_abs_delta=delta)
        del saved,forest_frame
        gc.collect()
        guard()
    del meta
    path = RAW/'ranking.parquet'
    assert common.sha256(path) == protocol['raw_hashes'][path.name]
    raw = pq.read_table(path,columns=FIELDS,use_threads=False).to_pandas(strings_to_categorical=True)
    dep = raw.loc[raw.PHASE_mvt.eq('DEP')].copy()
    ranking_meta = pd.read_parquet(V2/'ranking_meta.parquet')
    ranking_inputs = common.read_json(V2/'ranking_inputs.json')
    assert common.sha256(V2/'ranking_meta.parquet') == ranking_inputs['files']['ranking_meta.parquet']
    np.testing.assert_array_equal(dep[ID],ranking_meta[ID])
    mask = dep.AOBT_3_flt.isna().to_numpy()
    np.testing.assert_array_equal(mask,~np.isfinite(ranking_meta.proxy_sec.to_numpy(float)))
    assert len(dep)==344841 and mask.sum()==5290
    peers = features.record_peers(dep[[ID,'ADEP_mvt',TIME]])
    xr = shared.base.airport_features(dep.loc[mask])
    times = pd.Series(dep.loc[mask,TIME].to_numpy(),index=xr.index)
    xr = pd.concat([xr,shared.identity.id_context_features(xr,peers.loc[xr.index],times)],axis=1)
    assert list(xr)==list(x)
    x.to_parquet(OUT/'training_features.parquet')
    train_meta.to_parquet(OUT/'training_meta.parquet')
    xr.to_parquet(OUT/'ranking_features.parquet')
    ranking_meta.loc[mask].to_parquet(OUT/'ranking_missing_meta.parquet',index=False)
    peers.to_parquet(OUT/'ranking_peer_features.parquet')
    outputs = {name:common.sha256(OUT/name) for name in ['training_features.parquet','training_meta.parquet','ranking_features.parquet','ranking_missing_meta.parquet','ranking_peer_features.parquet']}
    common.write_json(OUT/'preparation.json',dict(status='passed',protocol_sha256=common.sha256(OUT/'protocol.json'),
        training_rows=len(x),ranking_missing_rows=len(xr),ranking_full_rows=len(dep),feature_columns=list(x),
        training_ids_hash=common.object_hash(x.index.tolist()),ranking_missing_ids_hash=common.object_hash(xr.index.tolist()),
        original_january_peer_rows_reconstructed=153706,original_january_peer_exact=True,fold_replay=fold_replay,
        ranking_input_columns=FIELDS,ranking_labels_read=False,outputs=outputs,peak_bytes=guard(),elapsed_seconds=time.monotonic()-started))
    print('PREPARED',fold_replay,guard(),flush=True)


def monitor(stop,folder):
    while not stop.wait(.1):
        try:
            guard(6)
        except Exception as exc:
            common.write_json(folder/'resource_failure.json',dict(status='resource_failed',error=repr(exc)))
            os._exit(87)


def run():
    protocol=declare()
    assert psutil.virtual_memory().available>=14*1024**3
    prep=common.read_json(OUT/'preparation.json')
    assert prep['status']=='passed' and prep['protocol_sha256']==common.sha256(OUT/'protocol.json')
    for name,digest in prep['outputs'].items():
        assert common.sha256(OUT/name)==digest
    dest=common.external_path(OUT/'models')
    dest.mkdir(exist_ok=False)
    stop=threading.Event()
    watcher=threading.Thread(target=monitor,args=(stop,dest),daemon=True)
    watcher.start()
    common.write_json(dest/'launch.json',dict(pid=os.getpid(),utc=common.utc_now(),protocol_sha256=common.sha256(OUT/'protocol.json')))
    started=time.monotonic()
    try:
        x=pd.read_parquet(OUT/'training_features.parquet')
        meta=pd.read_parquet(OUT/'training_meta.parquet')
        xr=pd.read_parquet(OUT/'ranking_features.parquet')
        rm=pd.read_parquet(OUT/'ranking_missing_meta.parquet').set_index(ID)
        np.testing.assert_array_equal(x.index,meta.index)
        np.testing.assert_array_equal(xr.index,rm.index)
        y=meta[TARGET].to_numpy(float)
        encoder=normalized.FrameEncoder().fit(x)
        scale=normalized.scale_of(x)
        normalizer=float(np.mean(scale**2))
        train=lgb.Dataset(encoder.transform(x),label=normalized.transformed(y,scale),weight=scale**2/normalizer,categorical_feature=list(encoder.categories))
        model=lgb.train(normalized.PARAMS,train,num_boost_round=protocol['normalized_steps'],callbacks=[lambda env:guard(6)])
        n_pred=normalized.reconstruct(model.predict(encoder.transform(xr)),normalized.scale_of(xr))
        model.save_model(str(dest/'normalized.txt'))
        joblib.dump(encoder,dest/'normalized_encoder.joblib')
        reloaded=lgb.Booster(model_file=str(dest/'normalized.txt'))
        n_replay=normalized.reconstruct(reloaded.predict(joblib.load(dest/'normalized_encoder.joblib').transform(xr)),normalized.scale_of(xr))
        np.testing.assert_array_equal(n_pred,n_replay)
        pd.DataFrame({ID:xr.index,'prediction_sec':n_pred}).to_parquet(dest/'normalized_ranking.parquet',index=False)
        common.write_json(dest/'normalized_fit.json',dict(steps=111,normalizer=normalizer,params=normalized.PARAMS,reload_max_abs_delta=0.,fit_rows=len(x),ranking_rows=len(xr)))
        del train,model,reloaded,encoder
        gc.collect()
        ex,er=x.copy(),xr.copy()
        ex[forest.TIME_COLUMN]=meta[TIME].to_numpy()
        er[forest.TIME_COLUMN]=rm[TIME].to_numpy()
        forest.FAMILY='extratrees'
        bundle,evidence=forest.fit(ex,y,tuning=None,steps=1,seed=20260916,threads=2)
        joblib.dump(bundle,dest/'forest.joblib')
        e_pred=forest.predict(bundle,er)
        del bundle
        gc.collect()
        saved=joblib.load(dest/'forest.joblib')
        e_replay=forest.predict(saved,er)
        delta=float(np.max(np.abs(e_pred-e_replay)))
        assert delta<=1e-9
        pd.DataFrame({ID:xr.index,'prediction_sec':e_pred}).to_parquet(dest/'forest_ranking.parquet',index=False)
        baseline=pd.read_parquet(V2/'ranking_predictions.parquet',columns=[ID,'prediction_sec'])
        assert common.sha256(V2/'ranking_predictions.parquet')==protocol['v2_prediction_sha256']
        loc=baseline.set_index(ID).loc[xr.index,'prediction_sec'].to_numpy(float)
        v4=.75*loc+.25*e_pred
        prediction=.75*v4+.25*n_pred
        assert np.isfinite(prediction).all()
        pd.DataFrame({ID:xr.index,'prediction_sec':prediction,'v2_sec':loc,'forest_sec':e_pred,'normalized_sec':n_pred}).to_parquet(dest/'missing_ranking.parquet',index=False)
        common.write_json(dest/'manifest.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),
            preparation_sha256=common.sha256(OUT/'preparation.json'),training_rows=len(x),ranking_missing_rows=len(xr),
            normalized_steps=111,forest=evidence,forest_reload_max_abs_delta=delta,normalized_reload_max_abs_delta=0.,
            composition='.5625V2+.1875ET+.25normalized, implemented as nested original25percent blends',
            peak_bytes=guard(6),elapsed_seconds=time.monotonic()-started,
            outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}))
        print('COMPLETE',len(prediction),guard(6),flush=True)
    finally:
        stop.set()
        watcher.join()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['declare','prepare','fit'],required=True)
    args=parser.parse_args()
    {'declare':declare,'prepare':prepare,'fit':run}[args.stage]()
