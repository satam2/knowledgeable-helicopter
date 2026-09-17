"""Frozen LightGBM full-cohort controls and isolated information additions."""
import lightgbm
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'models'))
import run_full as core
import run_augmented
import lgbm_adapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--formulation',choices=['direct','aobt_allfinite','multi_anchor'],required=True)
    p.add_argument('--features',nargs='*',default=[],choices=['trajectory',*run_augmented.BATCH_PATTERNS])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260916)
    args=p.parse_args()
    args.family='lightgbm'
    args.features=sorted(args.features)
    core.OUT=core.external_path(core.ROOT/'private_runs/breakthrough_20260916/cpu'/('__'.join(args.features) or 'base'))
    original_hashes=core.source_hashes
    core.source_hashes=lambda:{**original_hashes(),'run_cpu.py':core.sha256(__file__),
        'run_augmented.py':core.sha256(run_augmented.__file__),'lgbm_adapter.py':core.sha256(lgbm_adapter.__file__),
        'feature_screen.py':core.sha256(run_augmented.feature_screen.__file__)}
    core.importlib=SimpleNamespace(import_module=lambda name:lgbm_adapter if name=='tabm_gpu' else importlib.import_module(name))
    x,meta=core.common.load_data()
    x,receipts=run_augmented.augment(x,args.features)
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
    for path in [Path(__file__),Path(lgbm_adapter.__file__),Path(run_augmented.__file__),Path(run_augmented.feature_screen.__file__)]:
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
