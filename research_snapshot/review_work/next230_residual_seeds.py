"""Repeat the main residual refit at fixed tuning-selected tree counts."""

import argparse
import numpy as np

from next230_common import OUT, load_data, load_reference, start_run, finish_run, fit, reload_parity, same_split
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import TARGET


def execute(fold,seed,data):
    x,meta,labels=data
    reference,baseline,idx,split=load_reference(fold,meta)
    original=OUT/'models'/f'capacity_d8_5000_{fold}_s20260910'
    source=read_json(original/'manifest.json')
    if source['status']!='complete' or not same_split(source['split'],split):
        raise ValueError('Seed-10 capacity run not complete')
    for file,digest in source['outputs'].items():
        if sha256(original/file)!=digest:
            raise ValueError('Capacity seed source changed')
    config={**source['config'],'seed':seed,'seed_policy':'reuse seed-10 tuning-selected tree count'}
    path,record,reused=start_run('capacity_d8_5000',fold,seed,config,reference,split)
    if reused:
        return
    status=proxy_status(meta.proxy_sec,config)
    sr=idx['refit'][status[idx['refit']]=='present']
    target=labels[TARGET].to_numpy(float)-meta.proxy_sec.to_numpy(float)
    print(f'FIXED MAIN SEED {seed} {fold}: {len(sr)} rows, {source["trees"]} trees',flush=True)
    model,training=fit(x.iloc[sr],target[sr],config,trees=source['trees'])
    values,delta=reload_parity(model,path/'component.cbm',x.iloc[idx['score']])
    changed=baseline.route.isin(['residual','residual_long_proxy']).to_numpy()
    predictions=baseline.copy()
    predictions.loc[changed,'prediction_sec']=meta.iloc[idx['score']].proxy_sec.to_numpy()[changed]+values[changed]
    schema=read_json(original/'component_schema.json')
    write_json(path/'component_schema.json',schema)
    finish_run(path,record,baseline,predictions,labels.iloc[idx['score']],changed,
        {'trees':source['trees'],'training':{'refit':training},'refit_id_hash':object_hash(x.index[sr].tolist()),
         'runner_sha256':sha256(__file__),'reload_max_abs_delta':delta,
         'tree_selection_source':str(original),'tree_selection_source_manifest_sha256':sha256(original/'manifest.json'),
         'supplemental_protocol_sha256':sha256(OUT/'residual_seed_protocol.json'),
         'selection':'Fixed seeds 20260910/11/12. Tree counts chosen by seed-10 tuning, then held fixed to isolate refit seed sensitivity.'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[20260911,20260912])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    args=p.parse_args()
    file=OUT/'residual_seed_protocol.json'
    protocol={'runner_sha256':sha256(__file__),'seeds':[20260910,20260911,20260912],
              'tree_policy':'use completed seed-10 tune-selected counts for every repeat',
              'selection':'Declared before repeated main residual scores; all rows, features, other routes fixed; no best-seed selection.'}
    if file.exists() and read_json(file)!=protocol:
        raise ValueError('Residual seed protocol changed')
    write_json(file,protocol)
    data=load_data()
    for seed in args.seeds:
        for fold in args.folds:
            execute(fold,seed,data)
