"""Separate information ablations, keeping the initial family runner frozen."""
import argparse
import gc
import lightgbm
import numpy as np
import pandas as pd
import run
from common import OUT, HERE, ID, MOVEMENT, SEED, load_data, read_json, sha256, write_json, utc_now


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['arrival','aviation','physical','physical_base'],required=True)
    p.add_argument('--family',default='lightgbm',choices=['lightgbm','xgboost','catboost'])
    p.add_argument('--target',default='correction',choices=['correction','direct'])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--full',action='store_true')
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    args.features=args.mode
    x,meta=load_data()
    if args.mode!='physical_base':
        path=OUT/'aviation'/'features.parquet'
        manifest=read_json(OUT/'aviation'/'manifest.json')
        for name,digest in manifest['sources'].items():
            assert sha256(HERE/'aviation'/name)==digest
        # Cache marker verified against the feature producer before training.
        expected=manifest.get('feature_sha256') or manifest.get('sha256') or manifest.get('feature_cache_sha256') or manifest.get('outputs',{}).get(path.name)
        assert expected and sha256(path)==expected
        ext=pd.read_parquet(path)
        if ID in ext:
            ext=ext.set_index(ID)
        assert np.array_equal(ext.index,x.index)
        cols=[c for c in ext if c.startswith('arr_') or c.startswith('surface_')] if args.mode=='arrival' else list(ext)
        x=pd.concat([x,ext[cols]],axis=1)
    if args.mode.startswith('physical'):
        import domain_adapter
        drop=[c for c in x if c.startswith(('takeoff_minus_','precision_')) or c in ['nm_actual_minus_estimated','last_minus_initial','proxy_missing']]
        x=x.drop(columns=drop)
        x['__movement_ns']=pd.to_datetime(meta[MOVEMENT],utc=True).dt.as_unit('ns').astype('int64').to_numpy()
        args.family='lightgbm'
        args.target='direct'
        run.adapter=lambda family:(domain_adapter.fit,domain_adapter.predict,domain_adapter)
    declare=OUT/'domain_declarations'/f'{args.family}_{args.target}_{args.mode}_{"full" if args.full else "s200k"}.json'
    if not declare.exists():
        write_json(declare,dict(created_utc=utc_now(),mode=args.mode,features=list(x),
            source_hashes={str(p.relative_to(HERE)):sha256(p) for p in [HERE/'run_domain.py',HERE/'domain_adapter.py',HERE/'lgbm_adapter.py',HERE/'aviation/arrival_features.py']},
            design='Separate information/formulation test after identical-information family screen. Complete score cohorts.'))
    for fold in args.folds:
        run.run_one(args,x,meta,fold)
        gc.collect()
