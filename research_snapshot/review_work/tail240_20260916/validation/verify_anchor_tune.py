"""Independent tune-cohort, identity, probability and native replay checks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import validate_candidate as validator

ROOT=validator.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/forensics'))
import run_anchor_mixture_v3 as producer
BASE=producer.OUT
OUT=ROOT/'private_runs/tail240_20260916/validation/anchor_tune_v3'
read_json,sha,object_hash,write_json=validator.read_json,validator.sha256,validator.object_hash,validator.write_json
ID,TARGET=validator.ID,validator.TARGET


def main():
    summary=read_json(BASE/'summary.json')
    protocol=read_json(BASE/'protocol.json')['declaration']
    assert summary['status']=='complete' and sha(producer.__file__)==protocol['source_sha256']
    for path,digest in protocol['dependency_source_hashes'].items():
        assert sha(ROOT/path)==digest
    OUT.mkdir(parents=True,exist_ok=False)
    binding=read_json(validator.BINDING)
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(meta_path)==read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path)
    results={}
    for fold in ('F1','F3'):
        directory=BASE/fold
        record=read_json(directory/'manifest.json')
        assert record['status']=='complete'
        assert record['protocol_sha256']==sha(BASE/'protocol.json')
        assert object_hash(record['split'])==binding['folds'][fold]['split_hash']
        for name,digest in record['outputs'].items():
            assert sha(directory/name)==digest
        idx,_,_=producer.prior.common.fold_data(meta,fold,full=True)
        missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
        frames,anchors,targets={},{},{}
        for stage in ('fit','tune'):
            assert record['ids'][stage]==binding['folds'][fold]['cohorts']['missing_nm'][stage]
            expected=meta.iloc[idx[stage][missing[idx[stage]]]]
            frames[stage]=pd.read_parquet(directory/f'{stage}_features.parquet').set_index(ID)
            data=pd.read_parquet(directory/f'{stage}_anchors.parquet').set_index(ID)
            np.testing.assert_array_equal(data.index,expected[ID])
            np.testing.assert_array_equal(frames[stage].index,data.index)
            np.testing.assert_array_equal(data[TARGET],expected[TARGET])
            anchors[stage]=data[[f'anchor_{j}' for j in range(4)]].to_numpy(float)
            targets[stage]=data[TARGET].to_numpy(float)
            q,residual,active=producer.encode(targets[stage],anchors[stage])
            np.testing.assert_allclose(np.sum(q*np.nan_to_num(anchors[stage]),axis=1)+residual,targets[stage],atol=1e-9,rtol=0)
            np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12,rtol=0)
            assert np.all(q[~active]==0) and np.all(q>=0)
        residual=joblib.load(directory/'residual_model.joblib').predict(frames['tune'],thread_count=2)
        previous_path=producer.LEXICAL/'lexical/models'/f'historical_template_{fold}_s20260916'/'tune_predictions.parquet'
        previous=pd.read_parquet(previous_path)
        np.testing.assert_array_equal(previous[ID],frames['tune'].index)
        previous_mse=float(np.mean((previous.prediction_sec.to_numpy()-targets['tune'])**2))
        assert previous_mse==record['previous_lexical_mse']
        arms={}
        for arm in ('three_anchor','dst_four_anchor'):
            model=joblib.load(directory/f'{arm}.joblib')
            observed=anchors['tune'].copy()
            if arm=='three_anchor':
                observed[:,1]=np.nan
            active=producer.collapse_anchors(observed)
            stages=[]
            for step in producer.STEPS:
                probabilities=model.predict_proba(frames['tune'],ntree_end=step,thread_count=2)
                assert np.isfinite(probabilities).all() and np.all(probabilities>=0)
                np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-12,rtol=0)
                filled=np.zeros_like(observed)
                filled[:,model.classes_.astype(int)]=probabilities
                filled[~active]=0
                empty=filled.sum(axis=1)==0
                filled[empty,0]=1
                filled/=filled.sum(axis=1,keepdims=True)
                np.testing.assert_allclose(filled.sum(axis=1),1,atol=1e-12,rtol=0)
                mean=np.sum(filled*np.nan_to_num(observed),axis=1)
                prediction=mean+residual
                output=pd.read_parquet(directory/f'{arm}_{step}_tune.parquet')
                np.testing.assert_array_equal(output[ID],frames['tune'].index)
                np.testing.assert_array_equal(output[TARGET],targets['tune'])
                delta=float(np.max(np.abs(output.prediction_sec.to_numpy()-prediction)))
                assert delta<=1e-9
                mse=float(np.mean((prediction-targets['tune'])**2))
                reported=next(item for item in record['reports'][arm]['stages'] if item['trees']==step)
                assert abs(mse-reported['mse'])<=1e-8
                stages.append({'trees':step,'rmse':float(np.sqrt(mse)),'mse':mse,'replay_max_abs_delta':delta,'empty_probability_fallback_rows':int(empty.sum())})
            best=min(stages,key=lambda item:(item['mse'],item['trees']))
            assert best['trees']==record['reports'][arm]['best']['trees']
            arms[arm]={'stages':stages,'best':best}
            del model
        passed=arms['dst_four_anchor']['best']['mse']<min(arms['three_anchor']['best']['mse'],previous_mse)
        assert passed==record['gate_passed']
        results[fold]={'arms':arms,'previous_lexical_rmse':float(np.sqrt(previous_mse)),'gate_passed':bool(passed),'manifest_sha256':sha(directory/'manifest.json')}
        validator.guard()
    passed=all(record['gate_passed'] for record in results.values())
    assert passed==summary['gate_passed']
    receipt={'status':'passed_validation','gate_passed':passed,'source_sha256':sha(__file__),'producer_summary_sha256':sha(BASE/'summary.json'),
        'folds':results,'peak_rss_bytes':validator.guard(),'scope':'Tune predictions only. Full metadata read to verify original cohort labels. No score predictions computed. Observable anchors source-reviewed; saved input/model replay separately verified.'}
    write_json(OUT/'receipt.json',receipt)
    print({fold:{'gate':r['gate_passed'],'previous':r['previous_lexical_rmse'],'three':r['arms']['three_anchor']['best']['rmse'],'dst':r['arms']['dst_four_anchor']['best']['rmse']} for fold,r in results.items()},flush=True)


if __name__=='__main__':
    main()
