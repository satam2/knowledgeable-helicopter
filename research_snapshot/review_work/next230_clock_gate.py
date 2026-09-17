"""Fit a small convex clock gate on out-of-sample tuning expert predictions."""

import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from next230_common import OUT, load_data, load_reference, start_run, finish_run, same_split
from next230_features import fit_blend, blend_weights, apply_blend
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.schema import ID, TARGET

PARAMS={'iterations':300,'depth':3,'learning_rate':.03,'l2_leaf_reg':30,
        'loss_function':'RMSE','random_seed':20260910,'thread_count':4,
        'verbose':100,'allow_writing_files':False,'bootstrap_type':'No'}


def gate_targets(y,residual,direct):
    y,residual,direct=[np.asarray(a,float) for a in [y,residual,direct]]
    if not all(np.isfinite(a).all() for a in [y,residual,direct]):
        raise ValueError('Nonfinite gate training values')
    difference=direct-residual
    eligible=np.abs(difference)>=1
    weights=difference[eligible]**2
    scale=float(weights.mean())
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Insufficient expert disagreement')
    return eligible,(y[eligible]-residual[eligible])/difference[eligible],weights/scale,scale


def gate_features(x,residual,direct):
    out=x.copy()
    out['gate_residual_prediction']=np.asarray(residual,dtype='float32')
    out['gate_direct_prediction']=np.asarray(direct,dtype='float32')
    out['gate_expert_difference']=np.asarray(direct,dtype='float32')-np.asarray(residual,dtype='float32')
    return out


def checked(path,filename):
    record=read_json(path/filename)
    if record['status']!='complete':
        raise ValueError('Gate needs completed experts')
    for f,digest in record['outputs'].items():
        if sha256(path/f)!=digest:
            raise ValueError('Expert hash mismatch')
    return record


def execute(fold,data):
    x,meta,labels=data
    reference,baseline,idx,split=load_reference(fold,meta)
    direct_path=OUT/'clock_cpu_experts'/fold
    residual_path=OUT/'models'/f'capacity_d8_5000_{fold}_s20260910'
    direct_record=checked(direct_path,'expert.json')
    residual_record=checked(residual_path,'manifest.json')
    if not same_split(split,direct_record['split']) or not same_split(split,residual_record['split']):
        raise ValueError('Gate split mismatch')
    td=pd.read_parquet(direct_path/'tune_direct.parquet')
    tr=pd.read_parquet(residual_path/'tune_predictions.parquet')
    if not np.array_equal(td[ID],tr[ID]):
        raise ValueError('Tuning expert IDs mismatch')
    locations=pd.Index(meta[ID]).get_indexer(td[ID])
    if (locations<0).any() or not np.isin(locations,idx['tune']).all():
        raise ValueError('Gate rows are outside tuning month')
    if not np.array_equal(td[TARGET],labels.iloc[locations][TARGET]):
        raise ValueError('Gate labels differ from original')
    residual=meta.iloc[locations].proxy_sec.to_numpy()+tr.prediction.to_numpy()
    direct=td.direct.to_numpy()
    eligible,target,weights,scale=gate_targets(td[TARGET],residual,direct)
    features=gate_features(x.iloc[locations],residual,direct)
    path,record,reused=start_run('clock_learned_gate',fold,20260910,PARAMS,reference,split)
    if reused:
        return
    model=CatBoostRegressor(**PARAMS)
    model.fit(features.iloc[np.flatnonzero(eligible)],target,sample_weight=weights,
              cat_features=list(features.select_dtypes('category').columns))
    model.save_model(str(path/'gate.cbm'))
    score_direct=pd.read_parquet(direct_path/'score_direct.parquet')
    score_residual=pd.read_parquet(residual_path/'score_predictions.parquet')
    if not np.array_equal(baseline[ID],score_direct[ID]) or not np.array_equal(baseline[ID],score_residual[ID]):
        raise ValueError('Score expert IDs mismatch')
    score_features=gate_features(x.iloc[idx['score']],score_residual.prediction_sec,score_direct.direct)
    gate=np.clip(model.predict(score_features,thread_count=4),0,1)
    saved=CatBoostRegressor()
    saved.load_model(str(path/'gate.cbm'))
    delta=float(np.max(np.abs(gate-np.clip(saved.predict(score_features,thread_count=4),0,1))))
    if delta>1e-9:
        raise ValueError('Gate serialization mismatch')
    changed=baseline.route.eq('residual').to_numpy()
    original=baseline.prediction_sec.to_numpy().copy()
    original[changed]=score_residual.loc[changed,'prediction_sec']
    predictions=baseline.copy()
    predictions['prediction_sec']=apply_blend(original,score_direct.direct,baseline.route,gate)
    predictions['learned_direct_weight']=gate
    write_json(path/'gate_schema.json',{'columns':list(features),'dtypes':{c:str(features[c].dtype) for c in features},
        'calibration_id_hash':object_hash(td.loc[eligible,ID].tolist()),'weight_scale':scale})
    finish_run(path,record,baseline,predictions,labels.iloc[idx['score']],changed,
        {'learned_gate':True,'direct_expert_path':str(direct_path),'residual_expert_path':str(residual_path),
         'expert_manifest_sha256':sha256(direct_path/'expert.json'),
         'residual_manifest_sha256':sha256(residual_path/'manifest.json'),
         'runner_sha256':sha256(__file__),'reload_max_abs_delta':delta,
         'supplemental_protocol_sha256':sha256(OUT/'learned_clock_protocol.json'),
         'gate_training_id_hash':object_hash(td.loc[eligible,ID].tolist()),'gate_training_rows':int(eligible.sum())})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    args=p.parse_args()
    protocol={'runner_sha256':sha256(__file__),'params':PARAMS,'minimum_observed_expert_difference_sec':1,
              'probability_bounds':[0,1],
              'selection':'Declared after simple clock F1 gain; fixed tiny gate fits only tune-month OOS expert predictions with exact weighted MSE objective.'}
    file=OUT/'learned_clock_protocol.json'
    if file.exists() and read_json(file)!=protocol:
        raise ValueError('Learned clock protocol changed')
    write_json(file,protocol)
    data=load_data()
    for fold in args.folds:
        execute(fold,data)
