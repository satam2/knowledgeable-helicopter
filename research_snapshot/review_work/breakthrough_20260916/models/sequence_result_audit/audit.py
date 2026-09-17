"""Small saved-prediction audit; no model reload, feature load, or GPU work."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models/stacking'))
import common
from run_simplex_v2 import day_sensitivity
from taxiout.artifacts import read_json,write_json,sha256,object_hash,utc_now
from taxiout.schema import ID,TARGET
from taxiout.metrics import season_score

BASE=ROOT/'private_runs/breakthrough_20260916'
OUT=BASE/'models/sequence_result_audit'


def checked(folder,complete=True):
    record=read_json(folder/'manifest.json')
    if complete:
        assert record['status']=='complete'
        for name,digest in record['outputs'].items():
            assert sha256(folder/name)==digest,name
        assert record['fit']['steps']==record['refit']['steps']
        assert record['fit']['rows']==record['fit_ids']['fit']['n']
        assert record['refit']['rows']==record['fit_ids']['refit']['n']
        assert record['reload_max_abs_delta']==0.
    return record


def compare(left,right):
    np.testing.assert_array_equal(left[ID],right[ID])
    np.testing.assert_array_equal(left[TARGET],right[TARGET])
    y=left[TARGET].to_numpy(float)
    a=(left.prediction_sec.to_numpy()-y)**2
    b=(right.prediction_sec.to_numpy()-y)**2
    assert np.isfinite(a).all() and np.isfinite(b).all()
    def metric(keep):
        return {'n':int(keep.sum()),'control_rmse':float(np.sqrt(a[keep].mean())),
                'candidate_rmse':float(np.sqrt(b[keep].mean())),'sse_gain':float((a[keep]-b[keep]).sum())}
    gain=a-b
    order=np.argsort(gain)[::-1]
    removed={}
    for n in [1,2,5,10]:
        keep=np.ones(len(y),bool)
        keep[order[:n]]=False
        removed[str(n)]=metric(keep)
    return {'overall':metric(np.ones(len(y),bool)),'day_removals':day_sensitivity(left,right),
            'remove_top_gain_rows':removed,'label_slices':{'at_most7200':metric(y<=7200),'over7200':metric(y>7200),
                                                        'over86400':metric(y>86400)},
            'top_gain_rows':[{'id':str(left.iloc[i][ID]),'label':float(y[i]),'gain':float(gain[i])} for i in order[:10]]}


def contract(control,other,seed_difference=False):
    assert control['fit_ids']==other['fit_ids']
    assert object_hash(control['split'])==object_hash(other['split'])
    assert control['formulation']==other['formulation']=='aobt_allfinite'
    assert control['threads']==other['threads']
    a,b=control['fit']['params'],other['fit']['params']
    differences={key:[a.get(key),b.get(key)] for key in set(a)|set(b) if a.get(key)!=b.get(key)}
    assert set(differences)==({'random_state'} if seed_difference else set()),differences
    assert other['seed']==(20260917 if seed_difference else control['seed'])
    return differences


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=BASE/'missing/sequence_flatten'
    marker=read_json(cache/'manifest.json')
    verification=read_json(cache/'verification.json')
    assert marker['status']=='complete' and verification['status']=='passed'
    assert verification['manifest_sha256']==sha256(cache/'manifest.json')
    assert sha256(cache/'training_features.parquet')==marker['outputs']['training_features.parquet']
    assert len(marker['features'])==112
    upstream=BASE/'sequence_context/manifest.json'
    assert sha256(upstream)==marker['source_cache_manifest_sha256']
    assert sha256(BASE/'missing/sequence_independent_audit/verification.json')==marker['availability_receipt_sha256']
    flatroot=cache/'models'
    flatprotocol=read_json(flatroot/'matched_protocol.json')
    assert flatprotocol['cache_manifest_sha256']==sha256(cache/'manifest.json')
    assert flatprotocol['cache_verification_sha256']==sha256(cache/'verification.json')
    result={'created_utc':utc_now(),'source_sha256':sha256(__file__),'flatten_cache':{
        'manifest_sha256':sha256(cache/'manifest.json'),'verification_sha256':sha256(cache/'verification.json'),
        'all_rows':verification['all_rows'],'oracle_queries':verification['oracle_queries'],
        'source_policy':read_json(upstream)['policy']},'folds':{}}
    for fold in ['F1','F3']:
        small=BASE/'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'/f'lightgbm_aobt_allfinite_{fold}_s20260916'
        flat=flatroot/f'lightgbm_aobt_allfinite_{fold}_s20260916'
        large=BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'
        replica=BASE/'deeper_replication/s20260917'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260917'
        a,b,c,d=[checked(folder) for folder in [small,flat,large,replica]]
        contract(a,b)
        differences=contract(c,d,True)
        assert b['feature_columns']==a['feature_columns']+marker['features']
        assert len(a['feature_columns'])==225 and len(b['feature_columns'])==337
        assert c['feature_columns']==d['feature_columns']==a['feature_columns']
        assert a['anchor']['feature_receipts']==b['anchor']['feature_receipts']==c['anchor']['feature_receipts']==d['anchor']['feature_receipts']
        assert b['anchor']['flatten_manifest_sha256']==sha256(cache/'manifest.json')
        snapshot=BASE/'deeper_replication/s20260917/source_snapshots/lightgbm_leaf63_aobt_allfinite_s20260917'
        for name,digest in d['source_hashes'].items():
            assert sha256(snapshot/Path(name).name)==digest
        reference,_=common.reference(fold)
        frames=[pd.read_parquet(folder/'candidate.parquet') for folder in [small,flat,large,replica]]
        missing=~np.isfinite(reference.proxy_sec.to_numpy())
        for frame in frames:
            np.testing.assert_array_equal(frame[ID],reference[ID])
            np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
            np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing],reference.prediction_sec.to_numpy()[missing])
        row={'flatten_vs_225':compare(frames[0],frames[1]),'replication_vs_original':compare(frames[2],frames[3]),
             'seed_parameter_difference':differences,'steps':{'small225':a['fit']['steps'],'small337':b['fit']['steps'],
                                                              'original_leaf63':c['fit']['steps'],'replica_leaf63':d['fit']['steps']},
             'manifest_receipts':{str(folder):sha256(folder/'manifest.json') for folder in [small,flat,large,replica]},
             'full_replay_receipts':{'flatten':b['reload_max_abs_delta'],'replication':d['reload_max_abs_delta']},
             'replay_scope':'Frozenrun_full.py predictsall eligible score rows and reloads saved model for allsame rows; audit verifiedreceipt/hash/code, didnotreloadmodels.'}
        deep=BASE/'deeper_sequence'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        if (deep/'manifest.json').exists():
            live=checked(deep,False)
            assert live['fit_ids']==c['fit_ids']
            assert live['feature_columns']==c['feature_columns']+marker['features']
            assert live['seed']==c['seed'] and live['threads']==c['threads']
            assert live['anchor']['flattened_manifest_sha256']==sha256(cache/'manifest.json')
            assert live['anchor']['feature_receipts']==c['anchor']['feature_receipts']
            row['deep_sequence_current']={'status':live['status'],'cohorts_and_schema_verified':True}
            if live['status']=='complete':
                checked(deep)
                contract(c,live)
                deepframe=pd.read_parquet(deep/'candidate.parquet')
                row['deep_sequence_current']['versus225']=compare(frames[2],deepframe)
                np.testing.assert_array_equal(deepframe.prediction_sec.to_numpy()[missing],reference.prediction_sec.to_numpy()[missing])
        else:
            row['deep_sequence_current']={'status':'not_started_at_audit'}
        result['folds'][fold]=row
        print('AUDIT',fold,'flatten',row['flatten_vs_225']['overall'],'seed',row['replication_vs_original']['overall'],
              'deep',row['deep_sequence_current']['status'],flush=True)
    result['limits']='Adaptive exposeddevelopment. Twoexpertseeds only; not fullpipeline replication. Current originalleaf63 historicalencoderhashomission unchanged.'
    write_json(OUT/'audit.json',result)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
