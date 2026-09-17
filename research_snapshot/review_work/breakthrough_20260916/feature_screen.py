"""Explicit feature-information screens using the previous fixed model adapter."""
import lightgbm
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'campaign_20260916'))
import common
import run as harness

OUT=common.external_path(common.WORKSPACE/'private_runs/breakthrough_20260916/feature_screens')


def augmented(x,mode):
    roots=[]
    if mode in ['trajectory','linkage','trajectory_linkage']:
        root=common.WORKSPACE/'private_runs/mechanism_20260916/information/retrospective_v2'
        manifest=common.read_json(root/'manifest.json')
        path=root/'training_features.parquet'
        assert common.sha256(path)==manifest['outputs'][path.name]
        ext=pd.read_parquet(path).set_index(common.ID)
        assert np.array_equal(ext.index,x.index)
        if mode=='trajectory':
            cols=[c for c in ext if c.startswith(('retro_own_','source_'))]
        elif mode=='linkage':
            cols=[c for c in ext if c.startswith(('retro_exact_','retro_broad_'))]
        else:
            cols=list(ext)
        roots.append((ext[cols],str(root/'manifest.json'),common.sha256(root/'manifest.json')))
    if mode=='id_neighbors':
        root=common.WORKSPACE/'private_runs/mechanism_20260916/id_neighbors'
        manifest=common.read_json(root/'audit.json')
        path=root/'features.parquet'
        assert common.sha256(path)==manifest['feature_sha256']
        ext=pd.read_parquet(path).set_index(common.ID)
        assert np.array_equal(ext.index,x.index)
        roots.append((ext,str(root/'audit.json'),common.sha256(root/'audit.json')))
    for ext,_,_ in roots:
        x=pd.concat([x,ext.replace([np.inf,-np.inf],np.nan).fillna(-999999).astype('float32')],axis=1)
    return x,[{'path':p,'sha256':h} for _,p,h in roots]


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features',required=True,choices=['base','trajectory','linkage','trajectory_linkage','id_neighbors'])
    p.add_argument('--target',default='correction',choices=['correction','direct'])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    p.add_argument('--family',default='lightgbm')
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    args.full=True
    x,meta=common.load_data()
    x,receipts=augmented(x,args.features)
    if not (OUT/'protocol.json').exists():
        common.write_json(OUT/'protocol.json',dict(created_utc=common.utc_now(),
            design='Fixed previous LightGBM600depth8leaves31 full-data control, isolated retrospective information blocks.',
            availability='Owntrajectory final provided M3 fields retrospective; linkage may use future ARR+adjacentmonth. Explicit separate contract from causal control.',
            labels='Original raw labels unchanged; all score rows retained; original temporal flightIDpurges.',
            blending='Fixed25percent candidate plus75reference; no score-fit weights.',
            folds=['F1','F3'],status='Declared before first new feature score inspection'))
    common.write_json(OUT/'declarations'/f'{args.features}_{args.target}_s{args.seed}.json',dict(created_utc=common.utc_now(),
        features=list(x),source_sha256=common.sha256(__file__),features_receipts=receipts))
    harness.OUT=OUT
    for fold in args.folds:
        harness.run_one(args,x,meta,fold)
