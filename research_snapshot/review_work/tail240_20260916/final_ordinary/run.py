"""Full-year release fits of the frozen best ensemble's ordinary experts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
if '--gpu' in sys.argv:
    import torch
import lightgbm
import argparse
import gc
import importlib.util
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
for path in ['review_work/campaign_20260916', 'review_work/breakthrough_20260916/models',
             'review_work/tail240_20260916/state/risk',
             'review_work/tail240_20260916/state/final_ple387',
             'review_work/tail240_20260916/forensics/source_contract/v2']:
    sys.path.insert(0, str(ROOT/path))
import common
import run_risk as risk
import sources
import contract

ID, TARGET = common.ID, common.TARGET
BASE = ROOT/'private_runs/breakthrough_20260916'
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/final_ordinary/v2')
REGISTRY = {
    'lgb225': ('deeper_lgb/combined', 'lightgbm_leaf63', 225),
    'lgb449': ('deeper_sequence8', 'lightgbm_leaf63_sequence8', 449),
    'lgb387': ('deeper_context_union', 'lightgbm_leaf63_sequence', 387),
    'tabm225': ('full_neural', 'tabm_combined_standard', 225),
    'cat225': ('combined_catboost', 'catboost_combined', 225),
}


def folder(name, fold):
    parent, family, _ = REGISTRY[name]
    return BASE/parent/f'{family}_aobt_allfinite_{fold}_s20260916'


def adapter_for(name):
    if name == 'tabm225':
        import tabm_gpu
        return tabm_gpu
    relative = ('deeper_lgb/adapter.py' if name.startswith('lgb')
                else 'models_retrieval/combined_catboost/adapter.py')
    spec = importlib.util.spec_from_file_location('release_'+name, ROOT/'review_work/breakthrough_20260916'/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard():
    state = dict(rss=psutil.Process().memory_info().rss, available=psutil.virtual_memory().available)
    if state['rss'] > 24*1024**3 or state['available'] < 8*1024**3:
        raise MemoryError(state)
    return state


def monitor(stop, dest):
    while not stop.wait(2):
        try:
            guard()
        except MemoryError as exc:
            common.write_json(dest/'resource_failure.json', dict(error=str(exc), utc=common.utc_now()))
            os._exit(87)


def decode(matrix, vocab, columns):
    values = {}
    base_categories = {'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','airport_stand','airport_runway'}
    for j, name in enumerate(columns):
        if name in vocab:
            assert '__UNSEEN_CONTEXT_VALUE__' not in vocab[name]
            labels = np.asarray([np.nan if name in base_categories else 'MISSING',
                                 '__UNSEEN_CONTEXT_VALUE__', *vocab[name]], dtype=object)
            values[name] = pd.Categorical(labels[matrix[:,j].astype(np.int32)])
        else:
            values[name] = matrix[:,j]
    return pd.DataFrame(values, copy=False)


def declare(name):
    adapter = adapter_for(name)
    controls = {f:common.read_json(folder(name,f)/'manifest.json') for f in ['F1','F3']}
    columns = controls['F1']['feature_columns']
    assert len(columns) == REGISTRY[name][2]
    assert columns == controls['F3']['feature_columns']
    assert all(m['status']=='complete' for m in controls.values())
    steps = [m['fit']['steps'] for m in controls.values()]
    paths = [Path(__file__), Path(adapter.__file__), Path(sources.__file__), Path(contract.__file__), Path(risk.__file__)]
    import encoders
    import lgbm_adapter
    paths.extend([Path(encoders.__file__), Path(lgbm_adapter.__file__)])
    record = dict(name=name, columns=columns, fold_steps=steps, final_steps=int(np.median(steps)),
        policy='Inherited int(median(F1,F3 tune-selected steps)); all eligible 2025 finite NM rows; raw residual Y-P; no tuning, sampling, clipping or ranking labels.',
        sources={str(p.relative_to(ROOT)):common.sha256(p) for p in paths},
        controls={f:common.sha256(folder(name,f)/'manifest.json') for f in controls},
        params={f:m['fit']['params'] for f,m in controls.items()}, seed=20260916, threads=2,
        preserved_v1_source_sha256=common.sha256(Path(__file__).with_name('run_v1_frozen.py')),
        resource='30GiB startup;24GiB current RSS cap,8GiB host reserve,2CPU; one GPU job at a time')
    dest = OUT/name
    dest.mkdir(parents=True, exist_ok=True)
    file = dest/'protocol.json'
    if file.exists():
        assert common.read_json(file)==record, 'Release source/protocol changed'
    else:
        common.write_json(file,record)
    return record, adapter, dest


def data():
    path = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    train = pd.read_parquet(path, columns=[ID,TARGET,'proxy_sec','FLIGHT_ID_mvt',common.MOVEMENT])
    rankpath = ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    marker = common.read_json(rankpath.parent/'ranking_inputs.json')
    assert common.sha256(rankpath)==marker['files'][rankpath.name]
    rank = pd.read_parquet(rankpath)
    assert len(train)==2085047 and len(rank)==344841
    assert train[ID].is_unique and rank[ID].is_unique
    assert not set(train[ID]).intersection(rank[ID])
    assert not set(train.FLIGHT_ID_mvt.dropna()).intersection(rank.FLIGHT_ID_mvt.dropna())
    assert train[common.MOVEMENT].dt.year.eq(2025).all()
    assert rank[common.MOVEMENT].dt.year.eq(2026).all()
    return train.loc[np.isfinite(train.proxy_sec)].copy(), rank.loc[np.isfinite(rank.proxy_sec)].copy()


def run(name):
    protocol, adapter, dest = declare(name)
    assert not (dest/'manifest.json').exists(), 'Preserve completed fit'
    assert psutil.virtual_memory().available >= 30*1024**3, 'Full-year fit starts with30GiB available'
    stop = threading.Event()
    threading.Thread(target=monitor,args=(stop,dest),daemon=True).start()
    started = time.monotonic()
    try:
        train, rank = data()
        ids = pd.Index(pd.concat([train[ID],rank[ID]],ignore_index=True))
        columns = protocol['columns']
        matrix,vocab,loaded,checked = contract.load_checked(risk,sources.feature_sources(columns),ids,len(train),columns,dest,guard=guard)
        x = decode(matrix,vocab,columns)
        x.index = ids
        common.write_json(dest/'feature_receipt.json',checked)
        y = (train[TARGET]-train.proxy_sec).to_numpy(float)
        assert np.isfinite(y).all()
        print('FIT_START',name,len(train),len(columns),protocol['final_steps'],flush=True)
        model,evidence = adapter.fit(x.iloc[:len(train)],y,steps=protocol['final_steps'],seed=20260916,threads=2)
        assert model['steps']==protocol['final_steps']
        joblib.dump(model,dest/'model.joblib')
        if name.startswith('lgb'):
            model['estimator'].booster_.save_model(str(dest/'native.txt'))
        elif name=='cat225':
            model['estimator'].save_model(str(dest/'native.cbm'))
        prediction = adapter.predict(model,x.iloc[len(train):])+rank.proxy_sec.to_numpy(float)
        del model
        gc.collect()
        model = joblib.load(dest/'model.joblib')
        replay = adapter.predict(model,x.iloc[len(train):])+rank.proxy_sec.to_numpy(float)
        np.testing.assert_array_equal(prediction,replay)
        assert np.isfinite(prediction).all()
        rank[[ID,'proxy_sec']].assign(prediction_sec=prediction).to_parquet(dest/'ranking_predictions.parquet',index=False)
        common.write_json(dest/'fit_evidence.json',evidence)
        common.write_json(dest/'manifest.json',dict(status='complete',name=name,training_rows=len(train),ranking_rows=len(rank),
            train_id_hash=common.object_hash(train[ID].tolist()),ranking_id_hash=common.object_hash(rank[ID].tolist()),
            train_label_hash=common.object_hash(train[TARGET].tolist()),native_reload_max_abs_delta=0.,steps=model['steps'],
            protocol_sha256=common.sha256(dest/'protocol.json'),resources=guard(),runtime_sec=time.monotonic()-started,
            outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='matrix.float32'}))
        print('COMPLETE',name,'seconds',time.monotonic()-started,flush=True)
    finally:
        stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=list(REGISTRY),required=True)
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    if args.name in ['tabm225','cat225'] and not args.gpu:
        raise ValueError('GPU adapters require --gpu and CUDA runtime')
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.declare_only:
            print(declare(args.name)[0])
        else:
            run(args.name)
