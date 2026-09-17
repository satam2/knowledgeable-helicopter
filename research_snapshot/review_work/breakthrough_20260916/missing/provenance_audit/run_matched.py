"""Matched ordinary-source conventions with/without prior proxy-minute support."""
import lightgbm
import argparse
import gc
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
import run_full as core
import lgbm_adapter
OUT=core.external_path(ROOT/'private_runs/breakthrough_20260916/missing/provenance_audit/matched_models')


def add_cache(x,directory,filename,prefix=None,verified=False):
    marker=core.read_json(directory/'manifest.json')
    expected=marker.get('feature_sha256') or marker['outputs'][filename]
    assert core.sha256(directory/filename)==expected
    receipt=dict(manifest_sha256=core.sha256(directory/'manifest.json'),feature_sha256=expected)
    if verified:
        check=core.read_json(directory/'verification.json')
        assert check['status']=='passed' and check['manifest_sha256']==receipt['manifest_sha256']
        receipt['verification_sha256']=core.sha256(directory/'verification.json')
    ext=pd.read_parquet(directory/filename).set_index(core.ID)
    if not ext.index.is_unique or not np.array_equal(ext.index,x.index):
        raise ValueError('ContextIDorder differs')
    if prefix:
        ext=ext[[c for c in ext if c.startswith(prefix)]]
    assert not set(ext).intersection(x.columns)
    receipt['columns']=list(ext)
    return pd.concat([x,ext],axis=1),receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    x,meta=core.common.load_data()
    x,conventions=add_cache(x,ROOT/'private_runs/breakthrough_20260916/missing/source_conventions','features.parquet')
    x,context=add_cache(x,ROOT/'private_runs/breakthrough_20260916/missing/provenance_audit/context_cache','training_features.parquet',verified=True)
    sources=core.source_hashes
    extra=[Path(__file__),Path(lgbm_adapter.__file__)]
    core.source_hashes=lambda:{**sources(),**{str(p.relative_to(ROOT)):core.sha256(p) for p in extra}}
    core.importlib=SimpleNamespace(import_module=lambda name:lgbm_adapter if name=='tabm_gpu' else importlib.import_module(name))
    proxy=meta.proxy_sec.to_numpy(float)
    eligible=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    declaration=dict(created_utc=core.utc_now(),source_hashes=core.source_hashes(),seed=args.seed,threads=args.threads,
        folds=args.folds,conventions=conventions,context=context,
        params='Frozen CPUlightgbm adapter:600max31leavesdepth8lr.05L2=5minchild30,tune60roundstop,freshfullrefit.',
        eligibility='AlloriginalordinaryfiniteNMproxy0..7200; everyeligible rawlabel retained. Nonordinary+missing V2routes exact.',
        comparison='Sameownbase30+conventions85 control vs +three priorproxyfrequencyfields; onlycontext difference.',
        interpretation='Notfallbackprovenance or sourcequalitylabel. Airportstrictprior60min frequency features with finalobservedNM clocks.',
        outputs='Fulloriginalscore cohorts andfixed25percentblend; exposeddevelopmentfolds, noofficialscore/submission.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        old=core.read_json(path)
        comparable={k:v for k,v in declaration.items() if k!='created_utc'}
        assert {k:v for k,v in old.items() if k!='created_utc'}==comparable
    else:
        core.write_json(path,declaration)
    for arm in ['control','context']:
        args.family='lightgbm'
        args.formulation='ordinary_source_residual'
        core.OUT=core.external_path(OUT/arm)
        protocol=core.declare(args)
        data=x.drop(columns=[c for c in x if c.startswith('prov_')]) if arm=='control' else x
        info=dict(matched_protocol_sha256=core.sha256(path),arm=arm,conventions=conventions,context=context,
            rule='Ordinarysource rawY-NMproxy correction; unsupportedroute retainsreference')
        if not args.declare_only:
            for fold in args.folds:
                core.run(args,fold,data,meta,proxy,eligible,info,protocol)
                gc.collect()
    print('MATCHED_DECLARED' if args.declare_only else 'MATCHED_COMPLETE',x.shape,flush=True)


if __name__=='__main__':
    main()
