"""Isolated external-information controls using frozen family adapters/evaluator."""
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

if '--family' in sys.argv and sys.argv[sys.argv.index('--family')+1]=='tabm':
    import torch
import lightgbm
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'models'))
import run_full as core
import run_augmented


def additional(x,blocks):
    receipts=[]
    for block in blocks:
        directory={'geometry':'geometry_v2','weather_T':'weather','weather_N':'weather',
                   'conventions':'missing/source_conventions'}[block]
        root=core.ROOT/'private_runs/breakthrough_20260916'/directory
        manifest=core.read_json(root/'manifest.json')
        filename='features.parquet' if block=='conventions' else 'training_features.parquet'
        expected=manifest['feature_sha256'] if block=='conventions' else manifest['outputs'][filename]
        path=root/filename
        assert core.sha256(path)==expected
        if block=='geometry':
            verified=core.read_json(root/'verification.json')
            assert verified['status']=='passed'
        ext=pd.read_parquet(path).set_index(core.ID)
        assert ext.index.is_unique and np.array_equal(ext.index,x.index)
        if block.startswith('weather_'):
            ext=ext[[column for column in ext if column.startswith(block+'_')]]
        assert len(ext.columns)>0 and not set(x).intersection(ext)
        for column in ext:
            if pd.api.types.is_numeric_dtype(ext[column]):
                ext[column]=ext[column].replace([np.inf,-np.inf],np.nan).fillna(-999999).astype('float32')
            else:
                ext[column]=ext[column].astype('string').fillna('MISSING').astype('category')
        x=pd.concat([x,ext],axis=1)
        receipts.append({'block':block,'manifest':str(root/'manifest.json'),
            'manifest_sha256':core.sha256(root/'manifest.json'),'features_sha256':expected,'columns':list(ext)})
    return x,receipts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['lightgbm','tabm','xgb'],required=True)
    p.add_argument('--formulation',choices=['direct','aobt_allfinite','multi_anchor'],required=True)
    p.add_argument('--features',nargs='+',choices=['geometry','weather_T','weather_N','conventions','trajectory',*run_augmented.BATCH_PATTERNS],required=True)
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260916)
    args=p.parse_args()
    assert len(args.features)==len(set(args.features))
    args.features=sorted(args.features)
    core.OUT=core.external_path(core.ROOT/'private_runs/breakthrough_20260916/information_models'/'__'.join(args.features))
    original_hashes=core.source_hashes
    extra_paths=[Path(__file__),Path(run_augmented.__file__),Path(run_augmented.feature_screen.__file__)]
    if args.family=='lightgbm':
        import lgbm_adapter
        extra_paths.append(Path(lgbm_adapter.__file__))
        core.importlib=SimpleNamespace(import_module=lambda name:lgbm_adapter if name=='tabm_gpu' else importlib.import_module(name))
    core.source_hashes=lambda:{**original_hashes(),**{path.name:core.sha256(path) for path in extra_paths}}
    x,meta=core.common.load_data()
    standard=[b for b in args.features if b in ['trajectory',*run_augmented.BATCH_PATTERNS]]
    others=[b for b in args.features if b not in standard]
    x,receipts=run_augmented.augment(x,standard)
    x,extra=additional(x,others)
    receipts+=extra
    x,offset,eligible,info=core.anchors(x,meta,args.formulation)
    info={**info,'feature_blocks':args.features,'feature_receipts':receipts}
    protocol=core.declare(args)
    payload={'blocks':args.features,'receipts':receipts,'columns':list(x),'source_hashes':core.source_hashes()}
    feature_protocol=protocol.with_name(protocol.stem+'_features.json')
    if feature_protocol.exists():
        assert core.read_json(feature_protocol)==payload
    else:
        core.write_json(feature_protocol,payload)
    snapshot=core.OUT/'source_snapshots'/f'{args.family}_{args.formulation}_s{args.seed}'
    for path in extra_paths:
        shutil.copyfile(path,snapshot/path.name)
    records={}
    for fold in args.folds:
        records[fold]=core.run(args,fold,x,meta,offset,eligible,info,protocol)
        gc.collect()
    if set(records)=={'F1','F3'}:
        scores={v:core.season_score(records['F1']['reports'][v]['metrics']['overall'],records['F3']['reports'][v]['metrics']['overall'])
                for v in ('candidate','blend25')}
        core.write_json(core.OUT/f'{args.family}_{args.formulation}_s{args.seed}_summary.json',{'seasonal_rmse':scores,'features':payload})
        print('SEASONAL',scores,flush=True)


if __name__=='__main__':
    main()
