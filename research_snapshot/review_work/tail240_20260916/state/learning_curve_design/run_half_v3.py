"""One outcome-frozen uniform-half F1 fit at the saved full model's capacity."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='2'
import argparse
import gc
from pathlib import Path
import sys
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from sampling import selected_positions, selected_vocab, PREFIX
from reference import reference_predictions

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import following_groups_tune as shared
import schema_discovery
import linear_finite_tune as metric
common,risk,ID,TARGET=shared.common,shared.risk,shared.ID,shared.TARGET
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/state/learning_curve_design/v3')
CONTROL=risk.union_folder('F1')
NATIVE=ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
META=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def guard(_env=None):
    p=psutil.Process().memory_info()
    peak=max(p.rss,p.peak_wset)
    assert peak<8*1024**3, f'8GiB process cap: {peak}'
    assert psutil.virtual_memory().available>=8*1024**3, '8GiB host reserve'
    return int(peak)


def declare():
    control=common.read_json(CONTROL/'manifest.json')
    assert control['fit']['rows']==816060 and control['fit']['steps']==2001
    assert len(control['feature_columns'])==387
    params=dict(control['fit']['params'],n_estimators=2001)
    paths=[Path(__file__),Path(__file__).with_name('sampling.py'),Path(__file__).with_name('test_sampling.py'),
           Path(risk.__file__),Path(schema_discovery.__file__),Path(metric.__file__),
           Path(__file__).with_name('reference.py'),Path(__file__).with_name('test_reference.py')]
    record=dict(sources={str(p.relative_to(ROOT)):common.sha256(p) for p in paths},
        control_manifest_sha256=common.sha256(CONTROL/'manifest.json'),
        native_manifest_sha256=common.sha256(NATIVE/'manifest.json'),
        metadata_sha256=common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][META.name],
        columns=control['feature_columns'],parameters=params,full_fit_rows=816060,selected_fit_rows=408030,tune_rows=180640,
        sampling=f'SHA256 ASCII {PREFIX}<base10 integral movement ID>; ascending digest then ID; retain lowest408030, restore original row order; no outcomes used',
        split='Original F1 make_fold flight-ID purges before finite-NM route and sample. Original full June tune; no refit/score/ranking.',
        category_ownership='Native full-fit vocabulary for exact saved-control replay only. Candidate vocabulary comprises only codes observed among selected fit rows, sorted as original; missing0, unseen1. Unselected categories cannot remain known.',
        numeric='Original raw387 cache-specific finite sentinel/NaN treatment unchanged. No scaling, imputation, physicalmask, target cleanup or context resampling.',
        target='Raw Y-minus-finite NM proxy, add same proxy; raw squared loss, no weighting, clipping or early stopping. Exactly2001 trees, no eval_set for training.',
        comparison='Uniform half vs saved full387 model on identical full finite June and ordinary June; diagnostic two-point learning curve, no model promotion or official forecast.',
        resources='2CPU, noGPU; process current/historical peak<8GiB, startup>=16GiB, host reserve>=8GiB; stage and every-tree checks.',
        launch='Declare and freeze outcome-free ID selection first. Run requires independent preflight and separate parent CPU allocation; never auto-run.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record,'Frozen protocol changed'
    else:common.write_json(path,record)
    return record


def selection():
    protocol=declare()
    assert common.sha256(META)==protocol['metadata_sha256']
    # Read only identity, timing and observed proxy until membership is frozen.
    meta=pd.read_parquet(META,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt','proxy_sec'])
    idx,split=common.make_fold(meta,common.load_config('configs/folds.yaml')['F1'])
    parts={s:meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy() for s,p in idx.items() if s in ['fit','tune']}
    control=common.read_json(CONTROL/'manifest.json')
    for stage,frame in parts.items():
        assert len(frame)==control['fit_ids'][stage]['n']
        assert common.object_hash(frame[ID].tolist())==control['fit_ids'][stage]['hash']
    selected=selected_positions(parts['fit'][ID].to_numpy(),408030)
    chosen=parts['fit'].iloc[selected]
    receipt=dict(status='frozen_without_outcomes',protocol_sha256=common.sha256(OUT/'protocol.json'),split=split,
        full_fit_ids_hash=common.object_hash(parts['fit'][ID].tolist()),tune_ids_hash=common.object_hash(parts['tune'][ID].tolist()),
        selected_fit_ids_hash=common.object_hash(chosen[ID].tolist()),selected_rows=len(chosen),
        by_month=chosen[common.MOVEMENT].dt.strftime('%Y-%m').value_counts().sort_index().to_dict(),
        by_airport=chosen.ADEP_mvt.value_counts().sort_index().to_dict(),outcome_columns_read=[],peak_bytes=guard())
    path=OUT/'selection.json'
    if path.exists():
        old=common.read_json(path)
        ignored={'peak_bytes','selected_ids_file_sha256'}
        assert common.object_hash({k:v for k,v in old.items() if k not in ignored})==common.object_hash({k:v for k,v in receipt.items() if k not in ignored})
        np.testing.assert_array_equal(pd.read_parquet(OUT/'selected_ids.parquet')[ID],chosen[ID])
    else:
        chosen[[ID]].to_parquet(OUT/'selected_ids.parquet',index=False)
        receipt['selected_ids_file_sha256']=common.sha256(OUT/'selected_ids.parquet')
        common.write_json(path,receipt)
    return protocol,parts,selected,split


def run():
    assert psutil.virtual_memory().available>=16*1024**3
    protocol,parts,selected,split=selection()
    assert (OUT/'selection.json').exists()
    selection_record=common.read_json(OUT/'selection.json')
    assert common.sha256(OUT/'selected_ids.parquet')==selection_record['selected_ids_file_sha256']
    dest=common.external_path(OUT/'F1')
    dest.mkdir(exist_ok=False)
    started=time.monotonic()
    common.write_json(dest/'launch.json',dict(protocol_sha256=common.sha256(OUT/'protocol.json'),selection_sha256=common.sha256(OUT/'selection.json'),resources=guard()))
    sources,discovery=schema_discovery.cached_discovery(risk.feature_sources,protocol['columns'])
    risk.feature_sources=lambda columns:sources if columns==protocol['columns'] else (_ for _ in ()).throw(ValueError('Schema changed'))
    risk.guard=guard
    ids=pd.Index(pd.concat([parts['fit'][ID],parts['tune'][ID]],ignore_index=True))
    nfit=len(parts['fit']); ntune=len(parts['tune'])
    matrix,vocab,receipts=risk.load_matrix(ids,nfit,protocol['columns'],dest)
    native_marker=common.read_json(NATIVE/'manifest.json')
    assert common.sha256(NATIVE/'manifest.json')==protocol['native_manifest_sha256']
    for name in ['model.txt','encoder.json']:assert common.sha256(NATIVE/name)==native_marker['outputs'][name]
    assert common.read_json(NATIVE/'encoder.json')==dict(columns=protocol['columns'],vocab=vocab)
    control_marker=common.read_json(CONTROL/'manifest.json')
    assert common.sha256(CONTROL/'manifest.json')==protocol['control_manifest_sha256']
    assert common.sha256(CONTROL/'tune_predictions.parquet')==control_marker['outputs']['tune_predictions.parquet']
    stored=pd.read_parquet(CONTROL/'tune_predictions.parquet')
    reference_predictions(stored,parts['tune'][ID])
    native=lgb.Booster(model_file=str(NATIVE/'model.txt'))
    replay=native.predict(matrix[nfit:],num_threads=2)+parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(replay,stored.prediction_sec.to_numpy())
    del native,replay
    half=np.memmap(dest/'half_matrix.float32',mode='w+',dtype='float32',shape=(len(selected)+ntune,len(protocol['columns'])),order='F')
    half_vocab={}
    for j,name in enumerate(protocol['columns']):
        if name in vocab:
            half_vocab[name],mapping=selected_vocab(matrix[:nfit,j],selected,vocab[name])
            half[:len(selected),j]=mapping[matrix[selected,j].astype(np.int64)]
            half[len(selected):,j]=mapping[matrix[nfit:,j].astype(np.int64)]
        else:
            half[:len(selected),j]=matrix[selected,j]
            half[len(selected):,j]=matrix[nfit:,j]
    half.flush()
    del matrix
    gc.collect()
    common.write_json(dest/'encoder.json',dict(columns=protocol['columns'],vocab=half_vocab))
    # Selection has been persisted before the first outcome read in this script.
    eligible_label_ids=pd.concat([parts['fit'].iloc[selected][ID],parts['tune'][ID]],ignore_index=True).to_numpy()
    labels=pq.read_table(META,columns=[ID,TARGET],filters=[(ID,'in',eligible_label_ids)],use_threads=False).to_pandas().set_index(ID)[TARGET]
    assert len(labels)==len(eligible_label_ids) and labels.index.is_unique
    fit=parts['fit'].iloc[selected]
    yfit=labels.loc[fit[ID]].to_numpy(float)-fit.proxy_sec.to_numpy(float)
    ytune=labels.loc[parts['tune'][ID]].to_numpy(float)
    assert np.isfinite(yfit).all() and np.isfinite(ytune).all()
    del labels
    guard()
    model=lgb.LGBMRegressor(**protocol['parameters'])
    model.fit(half[:len(selected)],yfit,categorical_feature=[protocol['columns'].index(c) for c in half_vocab],
              feature_name=protocol['columns'],callbacks=[guard])
    assert model.n_estimators_==2001
    prediction=model.predict(half[len(selected):])+parts['tune'].proxy_sec.to_numpy(float)
    model.booster_.save_model(str(dest/'model.txt'))
    reloaded=lgb.Booster(model_file=str(dest/'model.txt'))
    replay=reloaded.predict(half[len(selected):],num_threads=2)+parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(prediction,replay)
    ordinary=parts['tune'].proxy_sec.between(0,7200).to_numpy()
    days=parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    results={key:metric.compare(ytune[mask],prediction[mask],stored.prediction_sec.to_numpy()[mask],days[mask])
             for key,mask in [('all_finite',np.ones(ntune,bool)),('ordinary',ordinary)]}
    pd.DataFrame({ID:parts['tune'][ID],'prediction_sec':prediction}).to_parquet(dest/'tune.parquet',index=False)
    report=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),selection_sha256=common.sha256(OUT/'selection.json'),
        selected_rows=len(selected),tune_rows=ntune,trees=2001,parameters=protocol['parameters'],
        control_native_max_delta_sec=0.,candidate_reload_max_delta_sec=0.,results=results,split=split,
        selected_fit_label_hash=common.object_hash(yfit.tolist()),tune_label_hash=common.object_hash(ytune.tolist()),
        label_metadata_sha256=protocol['metadata_sha256'],reference_schema='ID and prediction only; labels bound to original hashed metadata with selected-fit/tune ID predicate',
        feature_receipts=receipts,discovery=discovery,elapsed_seconds=time.monotonic()-started,peak_bytes=guard(),
        outputs={name:common.sha256(dest/name) for name in ['model.txt','encoder.json','tune.parquet']})
    common.write_json(dest/'manifest.json',report)
    print('COMPLETE',results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--declare-only',action='store_true')
    mode.add_argument('--freeze-selection',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.declare_only:declare();print(common.sha256(OUT/'protocol.json'))
    elif args.freeze_selection:selection();print(common.sha256(OUT/'selection.json'))
    else:run()




