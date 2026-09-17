"""Perturb only the residual refit seed in the selected full pipeline."""

import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from next230_common import OUT,load_data,load_reference,start_run,finish_run,same_split
from next230_clock_gate import gate_features
from next230_features import apply_blend
from taxiout.artifacts import read_json,sha256,write_json
from taxiout.schema import ID


def check(path,filename='manifest.json'):
    r=read_json(path/filename)
    if r['status']!='complete':
        raise ValueError('Seed check dependency incomplete')
    for f,d in r['outputs'].items():
        if sha256(path/f)!=d:
            raise ValueError('Seed check dependency corrupt')
    return r


def execute(fold,seed,data):
    x,meta,labels=data
    reference,baseline,idx,split=load_reference(fold,meta)
    residual_path=OUT/'models'/f'capacity_d8_5000_{fold}_s{seed}'
    final_path=OUT/'models'/f'clock_and_rome_ensemble_{fold}_s20260910'
    gate_path=OUT/'models'/f'clock_learned_gate_{fold}_s20260910'
    direct_path=OUT/'clock_cpu_experts'/fold
    for dependency in [residual_path,final_path,gate_path]:
        if not same_split(check(dependency)['split'],split):
            raise ValueError('Seed check split mismatch')
    check(direct_path,'expert.json')
    path,record,reused=start_run('clock_and_rome_seedcheck',fold,seed,
        {'residual_seed':seed,'other_components_seed':20260910,'rome_seeds':[20260910,20260911,20260912]},reference,split)
    if reused:
        return
    residual=pd.read_parquet(residual_path/'score_predictions.parquet')
    selected=pd.read_parquet(final_path/'score_predictions.parquet')
    direct=pd.read_parquet(direct_path/'score_direct.parquet')
    if not all(np.array_equal(baseline[ID],f[ID]) for f in [residual,selected,direct]):
        raise ValueError('Seed check score IDs mismatch')
    model=CatBoostRegressor()
    model.load_model(str(gate_path/'gate.cbm'))
    features=gate_features(x.iloc[idx['score']],residual.prediction_sec,direct.direct)
    weights=np.clip(model.predict(features,thread_count=4),0,1)
    original=selected.prediction_sec.to_numpy().copy()
    use=baseline.route.eq('residual').to_numpy()
    original[use]=residual.loc[use,'prediction_sec']
    predictions=selected.copy()
    predictions['prediction_sec']=apply_blend(original,direct.direct,baseline.route,weights)
    changed=baseline.route.isin(['residual','rome_schedule_residual']).to_numpy()
    deps={'residual':str(residual_path),'selected':str(final_path),'gate':str(gate_path),'direct':str(direct_path)}
    finish_run(path,record,baseline,predictions,labels.iloc[idx['score']],changed,
        {'seed_sensitivity':deps,'dependency_hashes':{k:sha256(p/('expert.json' if k=='direct' else 'manifest.json'))
             for k,p in [('residual',residual_path),('selected',final_path),('gate',gate_path),('direct',direct_path)]},
         'runner_sha256':sha256(__file__),'reload_max_abs_delta':0.,
         'scope':'Only residual refit seed varied. Direct expert, learned gate and three-head Rome ensemble held fixed. Not a full-pipeline retraining seed repeat.'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[20260911,20260912])
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    a=p.parse_args()
    data=load_data()
    for seed in a.seeds:
        for fold in a.folds:
            execute(fold,seed,data)
