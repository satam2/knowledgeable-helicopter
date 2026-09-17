"""Matched ordinary information extension for allfinite NM residual experts."""
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import lightgbm
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'models'))
sys.path.insert(0,str(HERE.parent))
import run_full as core
import run_information
import run_augmented
import lgbm_adapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--combined',action='store_true')
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260916)
    args=p.parse_args()
    args.family='lightgbm_identity';args.formulation='aobt_allfinite'
    root=core.ROOT/'private_runs/breakthrough_20260916/flight_identity'
    core.OUT=core.external_path(root/'models'/('combined' if args.combined else 'base'))
    paths=[Path(__file__),HERE/'build.py',Path(run_information.__file__),Path(run_augmented.__file__),Path(lgbm_adapter.__file__)]
    original_hashes=core.source_hashes
    core.source_hashes=lambda:{**original_hashes(),**{str(path.relative_to(core.ROOT)):core.sha256(path) for path in paths}}
    core.importlib=SimpleNamespace(import_module=lambda name:lgbm_adapter if name=='tabm_gpu' else importlib.import_module(name))
    x,meta=core.common.load_data()
    receipts=[]
    if args.combined:
        x,receipts=run_augmented.augment(x,['source_past','source_twosided','surface_T','trajectory'])
        x,extra=run_information.additional(x,['conventions','geometry','weather_T'])
        receipts+=extra
    marker=core.read_json(root/'manifest.json')
    assert marker['status']=='complete'
    path=root/'training_features.parquet'
    assert core.sha256(path)==marker['outputs'][path.name]
    ext=pd.read_parquet(path).set_index(core.ID)
    assert np.array_equal(ext.index,x.index) and not set(ext).intersection(x)
    for col in ext.select_dtypes('number'):
        ext[col]=ext[col].replace([np.inf,-np.inf],np.nan).fillna(-999999).astype('float32')
    x=pd.concat([x,ext],axis=1)
    receipts.append({'manifest':str(root/'manifest.json'),'sha256':core.sha256(root/'manifest.json'),'features_sha256':core.sha256(path)})
    x,offset,eligible,info=core.anchors(x,meta,args.formulation)
    info.update(feature_receipts=receipts)
    protocol=core.declare(args)
    detail=protocol.with_name(protocol.stem+'_identity.json')
    payload={'columns':list(x),'receipts':receipts,'source_hashes':core.source_hashes()}
    if detail.exists():
        assert core.read_json(detail)==payload
    else:
        core.write_json(detail,payload)
    snapshot=core.OUT/'source_snapshots'/f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path,snapshot/path.name)
    records={}
    for fold in args.folds:
        records[fold]=core.run(args,fold,x,meta,offset,eligible,info,protocol)
        gc.collect()
    if set(records)=={'F1','F3'}:
        scores={v:core.season_score(records['F1']['reports'][v]['metrics']['overall'],records['F3']['reports'][v]['metrics']['overall']) for v in ('candidate','blend25')}
        core.write_json(core.OUT/f'{args.family}_summary.json',{'seasonal_rmse':scores,'source_hashes':core.source_hashes()})
        print('SEASONAL',scores,flush=True)


if __name__=='__main__':
    main()
