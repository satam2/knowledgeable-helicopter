"""Prepared matched combined225 versus combined225+flatten112 CPU comparison."""
import lightgbm
import argparse
import gc
import importlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916'))
import run_full as core
import run_information as information
import run_augmented
import lgbm_adapter

CACHE=ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten'
BASE=ROOT/'private_runs/breakthrough_20260916/information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
OUT=CACHE/'models'
BLOCKS=['conventions','geometry','source_past','source_twosided','surface_T','trajectory','weather_T']


def sources():
    paths=[Path(__file__),HERE/'build.py',HERE/'verify.py',HERE/'test_flatten.py',
        Path(lgbm_adapter.__file__),Path(information.__file__),Path(run_augmented.__file__),
        Path(run_augmented.feature_screen.__file__)]
    return {str(path.relative_to(ROOT)):path for path in paths}


def checked(args):
    cache=core.read_json(CACHE/'manifest.json')
    verify=core.read_json(CACHE/'verification.json')
    assert cache['status']=='complete' and verify['status']=='passed'
    assert verify['manifest_sha256']==core.sha256(CACHE/'manifest.json')
    assert cache['source_sha256']==core.sha256(HERE/'build.py')
    assert core.sha256(CACHE/'training_features.parquet')==cache['outputs']['training_features.parquet']
    baseline={}
    for fold in args.folds:
        directory=BASE/f'lightgbm_aobt_allfinite_{fold}_s20260916'
        record=core.read_json(directory/'manifest.json')
        assert record['status']=='complete'
        assert record['seed']==args.seed and record['threads']==args.threads
        assert record['source_hashes']['lgbm_adapter.py']==core.sha256(lgbm_adapter.__file__)
        for name in ['candidate.parquet','tune_predictions.parquet']:
            assert core.sha256(directory/name)==record['outputs'][name]
        baseline[fold]=record
    first=baseline[args.folds[0]]
    for record in baseline.values():
        assert record['feature_columns']==first['feature_columns']
        assert record['fit']['params']==first['fit']['params']
        assert record['anchor']['feature_receipts']==first['anchor']['feature_receipts']
    assert len(first['feature_columns'])==225 and len(cache['features'])==112
    return cache,baseline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    cache,baseline=checked(args)
    first=baseline[args.folds[0]]
    original=core.source_hashes
    core.source_hashes=lambda:{**original(),**{name:core.sha256(path) for name,path in sources().items()}}
    core.importlib=SimpleNamespace(import_module=lambda name:lgbm_adapter if name=='tabm_gpu' else importlib.import_module(name))
    core.OUT=core.external_path(OUT)
    args.family='lightgbm'
    args.formulation='aobt_allfinite'
    protocol=core.declare(args)
    detail=dict(source_hashes=core.source_hashes(),cache_manifest_sha256=core.sha256(CACHE/'manifest.json'),
        cache_verification_sha256=core.sha256(CACHE/'verification.json'),
        baseline={fold:dict(manifest_sha256=core.sha256(BASE/f'lightgbm_aobt_allfinite_{fold}_s20260916'/'manifest.json'),
            candidate_sha256=record['outputs']['candidate.parquet'],fit_ids=record['fit_ids']) for fold,record in baseline.items()},
        baseline_features=first['feature_columns'],added_features=cache['features'],features=337,
        statistical_and_compute_params=first['fit']['params'],
        matched='SamefrozenLGB600adapter,seed,4threads,originalfinitefit/tune/refit/scoreIDs; originalfulltunestopping andfreshrefit; only112featuresadded.',
        labels='AllfiniteNMproxy includingnegative/long; allraweligiblelabels retained; missingNM V2exact; originalfullscore.',
        availability='Reuseverifiedsameairport/movementmonth strictprior60min tokens,no self/sameflight; finalNMpublicationtimes unknown.',
        selection='Adaptive exposeddevelopment; onefixedlast4DEP+4ARR, no grid or freshholdout.',
        resources='NoGPU.12GiBfreebeforefullfeatureload; existing8GiBbeforefit. Added112float32matrix~0.87GiB; encoder/slicing/trainingcopies needseveralGiB.')
    path=OUT/'matched_protocol.json'
    if path.exists():
        assert core.read_json(path)==detail
    else:
        core.write_json(path,detail)
    snapshot=OUT/'source_snapshots/lightgbm_aobt_allfinite_s20260916'
    for name,source in sources().items():
        destination=snapshot/source.name
        if destination.exists():
            assert core.sha256(destination)==core.sha256(source)
        else:
            shutil.copyfile(source,destination)
    if args.prepare_only:
        print('PREPARED_FLATTEN',337,path,flush=True)
        return
    if psutil.virtual_memory().available<12*1024**3:
        raise MemoryError('Flatten337 load requires12GiBavailablehostRAM')
    x,meta=core.common.load_data()
    standard=[b for b in BLOCKS if b in ['trajectory',*run_augmented.BATCH_PATTERNS]]
    x,receipts=run_augmented.augment(x,standard)
    x,extra=information.additional(x,[b for b in BLOCKS if b not in standard])
    receipts+=extra
    assert list(x)==first['feature_columns']
    assert receipts==first['anchor']['feature_receipts']
    flat=pd.read_parquet(CACHE/'training_features.parquet').set_index(core.ID)
    np.testing.assert_array_equal(flat.index,x.index)
    assert list(flat)==cache['features'] and not set(flat).intersection(x.columns)
    x=pd.concat([x,flat],axis=1)
    del flat
    gc.collect()
    assert len(x.columns)==337
    x,proxy,eligible,info=core.anchors(x,meta,args.formulation)
    info.update(feature_receipts=receipts,flatten_manifest_sha256=detail['cache_manifest_sha256'],
        flatten_verification_sha256=detail['cache_verification_sha256'],matched_protocol_sha256=core.sha256(path))
    records={}
    for fold in args.folds:
        idx,split,_=core.common.fold_data(meta,fold,full=True)
        actual={stage:dict(n=int(eligible[pos].sum()),hash=core.object_hash(meta.iloc[pos[eligible[pos]]][core.ID].tolist())) for stage,pos in idx.items()}
        assert actual==baseline[fold]['fit_ids']
        record=core.run(args,fold,x,meta,proxy,eligible,info,protocol)
        assert record['fit']['params']==baseline[fold]['fit']['params']
        records[fold]=record
        gc.collect()
    if set(records)=={'F1','F3'}:
        summary={name:core.season_score(records['F1']['reports'][name]['metrics']['overall'],records['F3']['reports'][name]['metrics']['overall']) for name in ['candidate','blend25']}
        core.write_json(OUT/'summary.json',dict(seasonal_rmse=summary,matched_protocol_sha256=core.sha256(path)))
        print('SEASONAL',summary,flush=True)


if __name__=='__main__':
    main()
