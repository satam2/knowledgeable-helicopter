"""Independent complete-cohort337-versus225 leaf63 audit without fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models/verification_id_template'))
import common
from audit import comparison,metric
from taxiout.metrics import season_score
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
BASE=ROOT/'private_runs/breakthrough_20260916'
NEW=BASE/'deeper_sequence_v2'
OLD=BASE/'deeper_lgb/combined'
CACHE=BASE/'missing/sequence_flatten'
OUT=BASE/'missing/sequence_deep_audit'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def checked(directory):
    manifest=common.read_json(directory/'manifest.json')
    if manifest['status']!='complete':
        raise ValueError('Wait for complete immutable run: '+str(directory))
    for name,digest in manifest['outputs'].items():
        assert common.sha256(directory/name)==digest
    key=manifest['family']+'_aobt_allfinite_s20260916'
    snapshot=directory.parent/'source_snapshots'/key
    for name,digest in manifest['source_hashes'].items():
        assert common.sha256(snapshot/Path(name).name)==digest
    assert common.sha256(directory.parent/'protocols'/(key+'.json'))==manifest['protocol_sha256']
    return manifest


def fold_audit(fold,meta):
    resultpath=OUT/(fold+'.json')
    if resultpath.exists():
        receipt=common.read_json(resultpath)
        assert receipt['audit_source_sha256']==common.sha256(__file__)
        assert receipt['new_manifest_sha256']==common.sha256(NEW/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'/'manifest.json')
        return receipt
    olddir=OLD/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'
    newdir=NEW/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    old,new=checked(olddir),checked(newdir)
    assert old['fit']['params']==new['fit']['params']
    assert old['fit_ids']==new['fit_ids']
    assert old['feature_columns']==new['feature_columns'][:225]
    cm=common.read_json(CACHE/'manifest.json')
    cv=common.read_json(CACHE/'verification.json')
    assert cm['status']=='complete' and cv['status']=='passed'
    assert common.sha256(CACHE/'manifest.json')==cv['manifest_sha256']==new['anchor']['flattened_manifest_sha256']
    assert common.sha256(CACHE/'verification.json')==new['anchor']['flattened_verification_sha256']
    assert common.sha256(CACHE/'training_features.parquet')==cm['outputs']['training_features.parquet']
    assert new['feature_columns'][225:]==cm['features'] and len(new['feature_columns'])==337
    assert new['anchor']['feature_receipts']==old['anchor']['feature_receipts']
    for receipt in new['anchor']['feature_receipts']:
        path=Path(receipt.get('path',receipt.get('manifest')))
        assert common.sha256(path)==receipt.get('sha256',receipt.get('manifest_sha256'))
    assert any(str(key).endswith('lgbm_adapter.py') for key in new['source_hashes'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    finite=np.isfinite(meta.proxy_sec.to_numpy(float))
    assert common.object_hash(split)==common.object_hash(old['split'])==common.object_hash(new['split'])
    for stage,positions in idx.items():
        chosen=positions[finite[positions]]
        assert new['fit_ids'][stage]==dict(n=len(chosen),hash=common.object_hash(meta.iloc[chosen][ID].tolist()))
    reference,refrec=common.reference(fold)
    assert common.object_hash(split)==common.object_hash(refrec['split'])
    np.testing.assert_array_equal(reference[ID],meta.iloc[idx['score']][ID])
    np.testing.assert_array_equal(reference[TARGET],meta.iloc[idx['score']][TARGET])
    predictions={}
    tunes={}
    for name,directory,record in [('deep225',olddir,old),('deep337',newdir,new)]:
        frame=pd.read_parquet(directory/'candidate.parquet')
        np.testing.assert_array_equal(frame[ID],reference[ID])
        np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
        predictions[name]=frame.prediction_sec.to_numpy(float)
        assert np.isfinite(predictions[name]).all()
        tune=pd.read_parquet(directory/'tune_predictions.parquet')
        np.testing.assert_array_equal(tune[ID],meta.iloc[idx['tune'][finite[idx['tune']]]][ID])
        tunes[name]=tune.prediction_sec.to_numpy(float)
        assert record['fit']['steps']==record['refit']['steps']
        assert record['fit']['rows']==record['fit_ids']['fit']['n'] and record['refit']['rows']==record['fit_ids']['refit']['n']
        assert record['reload_max_abs_delta']==0.
    a,b=predictions['deep225'],predictions['deep337']
    y=reference[TARGET].to_numpy(float)
    proxy=meta.iloc[idx['score']].proxy_sec.to_numpy(float)
    missing=~np.isfinite(proxy)
    np.testing.assert_array_equal(a[missing],reference.prediction_sec.to_numpy()[missing])
    np.testing.assert_array_equal(b[missing],a[missing])
    day=reference.day.to_numpy()
    gain=(a-y)**2-(b-y)**2
    order=np.argsort(gain)[::-1]
    removal={}
    for count in [1,2,5,10]:
        mask=np.ones(len(y),bool)
        mask[order[:count]]=False
        ma,mb=metric(y,a,mask),metric(y,b,mask)
        removal[str(count)]=dict(deep225=ma,deep337=mb,gain_seconds=ma['rmse_sec']-mb['rmse_sec'])
    groups={'missing':missing,'finite_source_gap_over1800':~missing&(np.abs(y-proxy)>1800),
        'finite_source_gap_le1800':~missing&(np.abs(y-proxy)<=1800),
        'negative_proxy':~missing&(proxy<0),'ordinary_proxy':~missing&(proxy>=0)&(proxy<=7200),'long_proxy':~missing&(proxy>7200)}
    for airport in sorted(reference.ADEP_mvt.unique()):
        groups['airport_'+str(airport)]=reference.ADEP_mvt.eq(airport).to_numpy()
    attribution={name:dict(n=int(mask.sum()),deep225=metric(y,a,mask),deep337=metric(y,b,mask),sse_gain=float(gain[mask].sum())) for name,mask in groups.items()}
    rows=reference[[ID,TARGET,'ADEP_mvt','day','proxy_sec']].iloc[order[:20]].copy()
    rows['deep225']=a[order[:20]]
    rows['deep337']=b[order[:20]]
    rows['sse_gain']=gain[order[:20]]
    rows.to_parquet(OUT/(fold+'_top20_gain.parquet'),index=False)
    comparison_record=comparison(y,a,b,day)
    first,second=metric(y,a),metric(y,b)
    assert abs(second['rmse_sec']-new['reports']['candidate']['metrics']['overall']['rmse_sec'])<1e-10
    fit_tune_labels=meta.iloc[idx['tune'][finite[idx['tune']]]][TARGET].to_numpy(float)
    result=dict(status='passed',audit_source_sha256=common.sha256(__file__),fold=fold,
        old_manifest_sha256=common.sha256(olddir/'manifest.json'),new_manifest_sha256=common.sha256(newdir/'manifest.json'),
        old=first,new=second,gain_seconds=first['rmse_sec']-second['rmse_sec'],comparison=comparison_record,
        tune={name:metric(fit_tune_labels,v) for name,v in tunes.items()},attribution=attribution,remove_top_gain_rows=removal,
        source_feature_params_and_purged_IDs_match=True,new_imported_encoder_hash_included=True,
        current_saved_replay_receipts_zero=True,no_independent_model_inference_performed=True,
        fit_steps={'deep225':old['fit']['steps'],'deep337':new['fit']['steps']},full_score_rows=len(y),missing_predictions_exact=True,
        meets_proposed_two_second_gate=first['rmse_sec']-second['rmse_sec']>=2,
        daily_and_top10_gain_robust=comparison_record['all_day_removals_improve'] and removal['10']['gain_seconds']>0)
    common.write_json(resultpath,result)
    print('DEEP_SEQUENCE_AUDIT',fold,result['gain_seconds'],result['daily_and_top10_gain_robust'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    for fold in args.folds:
        fold_audit(fold,meta)
    if all((OUT/(fold+'.json')).exists() for fold in ['F1','F3']):
        records={fold:common.read_json(OUT/(fold+'.json')) for fold in ['F1','F3']}
        summary=dict(status='passed',source_sha256=common.sha256(__file__),
            seasonal_rmse={name:season_score(records['F1'][name],records['F3'][name]) for name in ['old','new']},
            meets_fixed_proposal_gate=all(r['meets_proposed_two_second_gate'] and r['daily_and_top10_gain_robust'] for r in records.values()),
            no_new_composition_scored=True,
            caveat='Adaptive exposeddevelopment; original225 historicalencoderhashomission remains; new337 runner includesencoderhash. Sourcegap useshiddenlabeldiagnosticonly.')
        common.write_json(OUT/'summary.json',summary)
        print('SUMMARY',summary,flush=True)


if __name__=='__main__':
    main()
