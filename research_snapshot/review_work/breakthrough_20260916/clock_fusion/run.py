"""Centrally scheduled raw-mean clock fusion pilot."""
import torch
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'models'))
sys.path.insert(0,str(HERE.parent))
import run_full as core
import run_information
import adapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['convex','affine'],required=True)
    p.add_argument('--conventions',action='store_true')
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260916)
    args=p.parse_args()
    args.family='clockfusion_'+args.mode
    args.formulation='finite_rawmean'
    adapter.MODE=args.mode
    core.OUT=core.external_path(core.ROOT/'private_runs/breakthrough_20260916/clock_fusion'/('conventions' if args.conventions else 'base'))
    paths=[Path(__file__),Path(adapter.__file__),Path(run_information.__file__),HERE/'DESIGN.md']
    original_hashes=core.source_hashes
    core.source_hashes=lambda:{**original_hashes(),**{str(p.relative_to(core.ROOT)):core.sha256(p) for p in paths}}
    core.importlib=SimpleNamespace(import_module=lambda name:adapter if name=='tabm_gpu' else importlib.import_module(name))
    x,meta=core.common.load_data()
    receipts=[]
    if args.conventions:
        x,receipts=run_information.additional(x,['conventions'])
    eligible=np.isfinite(meta.proxy_sec.to_numpy(float))
    info={'rule':'Full finite-NM raw taxi target; missing source keeps frozen reference','mode':args.mode,'feature_receipts':receipts,
        'mixing':'Five observed clocks plus learned direct expert, observed-source mask, optional learned additive correction'}
    protocol=core.declare(args)
    feature_protocol=protocol.with_name(protocol.stem+'_features.json')
    feature_payload={'columns':list(x),'receipts':receipts,'mode':args.mode,'source_hashes':core.source_hashes()}
    if feature_protocol.exists():
        assert core.read_json(feature_protocol)==feature_payload
    else:
        core.write_json(feature_protocol,feature_payload)
    snapshot=core.OUT/'source_snapshots'/f'{args.family}_{args.formulation}_s{args.seed}'
    for path in paths:
        shutil.copyfile(path,snapshot/path.name)
    records={}
    for fold in args.folds:
        records[fold]=core.run(args,fold,x,meta,np.zeros(len(x)),eligible,info,protocol)
        gc.collect()
    if set(records)=={'F1','F3'}:
        scores={v:core.season_score(records['F1']['reports'][v]['metrics']['overall'],records['F3']['reports'][v]['metrics']['overall']) for v in ('candidate','blend25')}
        core.write_json(core.OUT/f'{args.family}_summary.json',{'seasonal_rmse':scores,'source_hashes':core.source_hashes()})
        print('SEASONAL',scores,flush=True)


if __name__=='__main__':
    main()
