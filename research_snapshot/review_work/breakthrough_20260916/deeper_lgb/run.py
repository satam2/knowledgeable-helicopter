"""Declared capacity follow-up on the new, combined information set."""
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'models'))
sys.path.insert(0,str(HERE.parent))
import run_full as core
import run_information
import run_augmented
import adapter

BLOCKS=['conventions','geometry','source_past','source_twosided','surface_T','trajectory','weather_T']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preset',choices=['leaf63','leaf127'],required=True)
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--base-features',action='store_true')
    p.add_argument('--declare-only',action='store_true')
    args=p.parse_args()
    args.family='lightgbm_'+args.preset
    args.formulation='aobt_allfinite'
    adapter.PRESET=args.preset
    core.OUT=core.external_path(core.ROOT/'private_runs/breakthrough_20260916/deeper_lgb'/('base' if args.base_features else 'combined'))
    paths=[Path(__file__),Path(adapter.__file__),Path(run_information.__file__),Path(run_augmented.__file__),
           Path(run_augmented.feature_screen.__file__)]
    original_hashes=core.source_hashes
    core.source_hashes=lambda:{**original_hashes(),**{str(path.relative_to(core.ROOT)):core.sha256(path) for path in paths}}
    core.importlib=SimpleNamespace(import_module=lambda name:adapter if name=='tabm_gpu' else importlib.import_module(name))
    x,meta=core.common.load_data()
    receipts=[]
    if not args.base_features:
        x,receipts=run_augmented.augment(x,['source_past','source_twosided','surface_T','trajectory'])
        x,extra=run_information.additional(x,['conventions','geometry','weather_T'])
        receipts+=extra
    x,offset,eligible,info=core.anchors(x,meta,args.formulation)
    info.update(feature_receipts=receipts,preset=args.preset)
    protocol=core.declare(args)
    detail=protocol.with_name(protocol.stem+'_capacity.json')
    payload={'params':adapter.PARAMS[args.preset],'features':list(x),'receipts':receipts,'source_hashes':core.source_hashes(),
        'rationale':'Adaptive capacity follow-up after combined-information standalone600tree result287.15665sec vs matched base298.02459sec; exposeddevelopment.',
        'selection':'Tune earlystop150, original fullrefit, fixed25percentblend; no scoreweights.'}
    if detail.exists():
        assert core.read_json(detail)==payload
    else:
        core.write_json(detail,payload)
    snapshot=core.OUT/'source_snapshots'/f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path,snapshot/path.name)
    if args.declare_only:
        print('DECLARED',detail,flush=True)
        return
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
