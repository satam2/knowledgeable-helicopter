"""Train clock-source trust, rather than an unconstrained seconds correction."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'campaign_20260916'))
from common import WORKSPACE, ID, TARGET, SEED, load_data, fold_data, reference, read_json, write_json, sha256, external_path, utc_now, object_hash
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from taxiout.metrics import evaluate, paired_stability

HERE=Path(__file__).resolve().parent
OUT=external_path(WORKSPACE/'private_runs/mechanism_20260916/source_gate')
PARAMS=dict(iterations=600,depth=4,learning_rate=.05,l2_leaf_reg=20,
            loss_function='RMSE',random_seed=20260916,thread_count=4,
            allow_writing_files=False,verbose=200)


def clocks(x):
    p=x['takeoff_minus_AOBT_3_flt'].to_numpy(float)
    q=x['takeoff_minus_EOBT_1_flt'].to_numpy(float)
    valid=np.isfinite(p)&np.isfinite(q)&(p!= -999999)&(q!=-999999)&(p>=0)&(p<=7200)
    gap=q-p
    return p,q,gap,valid


def declaration():
    proposal=dict(created_utc=utc_now(),params=PARAMS,source_sha256=sha256(__file__),
        hypothesis='Explicit trust in estimated versus actual NM off-block clock can extrapolate correction with observed clock disagreement.',
        target='(Y-P)/(Q-P), weights=(Q-P)^2 normalized by fit mean; no target clipping.',
        fit_eligibility='ordinary actual proxy in0..7200, both clocks finite, |Q-P|>=60sec',
        inference_eligibility='same observed fields, |Q-P|>=600sec; otherwise frozen V2 unchanged',
        tuning='Choose among50,100,...600 trees using raw clipped-prediction RMSE on eligible tune rows only, then fresh full eligible refit.',
        prediction='P + clip(predicted trust,0,1)*(Q-P)',
        variants=['candidate','blend25'],blending='fixed25%; no score calibration',
        folds=['F1','F3'],caveat='Hypothesis prompted by label-aware diagnostic; exposed development evidence, not fresh confirmation.')
    path=OUT/'protocol.json'
    if path.exists():
        assert read_json(path)['source_sha256']==proposal['source_sha256']
    else:
        write_json(path,proposal)


def run(fold,x,meta):
    path=OUT/fold
    if path.exists():
        assert read_json(path/'manifest.json')['status']=='complete'
        return
    path.mkdir(parents=True)
    idx,split,_=fold_data(meta,fold,full=True)
    ref,old=reference(fold)
    assert object_hash(split)==object_hash(old['split'])
    p,q,gap,valid=clocks(x)
    train=valid&(np.abs(gap)>=60)
    eligible=valid&(np.abs(gap)>=600)
    y=meta[TARGET].to_numpy(float)
    sf=idx['fit'][train[idx['fit']]]
    st=idx['tune'][eligible[idx['tune']]]
    sr=idx['refit'][train[idx['refit']]]
    ss=idx['score'][eligible[idx['score']]]
    target=(y-p)/np.where(np.abs(gap)>=60,gap,1.)
    cats=list(x.select_dtypes('category').columns)
    start=time.monotonic()
    record=dict(status='running',fold=fold,split=split,created_utc=utc_now(),
                protocol_sha256=sha256(OUT/'protocol.json'),rows={'fit':len(sf),'tune':len(st),'refit':len(sr),'score_eligible':len(ss)},
                id_hashes={k:object_hash(meta.iloc[v][ID].tolist()) for k,v in [('fit',sf),('tune',st),('refit',sr),('score',ss)]})
    write_json(path/'manifest.json',record)
    weights=gap[sf]**2
    model=CatBoostRegressor(**PARAMS)
    model.fit(x.iloc[sf],target[sf],sample_weight=weights/weights.mean(),cat_features=cats)
    tuning=[]
    for n in range(50,601,50):
        trust=np.clip(model.predict(x.iloc[st],ntree_end=n,thread_count=4),0,1)
        pred=p[st]+trust*gap[st]
        tuning.append({'trees':n,'mse_sec2':float(np.mean((pred-y[st])**2))})
    trees=min(tuning,key=lambda r:r['mse_sec2'])['trees']
    model.save_model(str(path/'fit_model.cbm'))
    model=CatBoostRegressor(**{**PARAMS,'iterations':trees})
    weights=gap[sr]**2
    model.fit(x.iloc[sr],target[sr],sample_weight=weights/weights.mean(),cat_features=cats)
    model.save_model(str(path/'model.cbm'))
    trust=np.clip(model.predict(x.iloc[ss],thread_count=4),0,1)
    loaded=CatBoostRegressor().load_model(str(path/'model.cbm'))
    assert np.array_equal(trust,np.clip(loaded.predict(x.iloc[ss],thread_count=4),0,1))
    candidate=ref.prediction_sec.to_numpy().copy()
    mask=eligible[idx['score']]
    candidate[mask]=p[ss]+trust*gap[ss]
    assert np.array_equal(candidate[~mask],ref.prediction_sec.to_numpy()[~mask])
    reports={}
    for variant,pred in [('candidate',candidate),('blend25',ref.prediction_sec.to_numpy()+.25*(candidate-ref.prediction_sec.to_numpy()))]:
        frame=ref.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
        frame['prediction_sec']=pred
        metrics,errors=evaluate(frame,meta.iloc[idx['score']][[ID,TARGET]])
        errors.to_parquet(path/f'{variant}.parquet',index=False)
        reports[variant]={'metrics':metrics,'stability':paired_stability(ref,errors,repetitions=500)}
    record.update(status='complete',completed_utc=utc_now(),runtime_sec=time.monotonic()-start,
        selected_trees=trees,tuning_curve=tuning,reports=reports,reload_exact=True)
    record['outputs']={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name!='manifest.json'}
    write_json(path/'manifest.json',record)
    print('SOURCE_GATE',fold,{k:r['metrics']['overall']['rmse_sec'] for k,r in reports.items()},flush=True)


if __name__=='__main__':
    declaration()
    x,meta=load_data()
    for fold in ['F1','F3']:
        run(fold,x,meta)
