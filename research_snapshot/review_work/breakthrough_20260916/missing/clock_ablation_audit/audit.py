"""Matched 600-tree225/337/313 full-cohort ablation audit, no fitting or inference."""
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
CACHE=BASE/'missing/sequence_flatten'
NEW=BASE/'models/sequence_clock_ablation'
OUT=BASE/'missing/clock_ablation_audit'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def checked(folder):
    record=common.read_json(folder/'manifest.json')
    assert record['status']=='complete', str(folder)
    for name,digest in record['outputs'].items():
        assert common.sha256(folder/name)==digest
    assert record['reload_max_abs_delta']==0
    assert record['fit']['steps']==record['refit']['steps']==600
    assert record['threads']==4 and record['seed']==20260916
    return record


def run_fold(fold,meta):
    destination=OUT/(fold+'.json')
    if destination.exists():
        result=common.read_json(destination)
        assert result['source_sha256']==common.sha256(__file__)
        return result
    registry={
        'base225':BASE/'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'/f'lightgbm_aobt_allfinite_{fold}_s20260916',
        'context337':CACHE/'models'/f'lightgbm_aobt_allfinite_{fold}_s20260916',
        'no_dep_clocks313':NEW/f'lightgbm_sequence_no_dep_clocks_aobt_allfinite_{fold}_s20260916'}
    records={name:checked(path) for name,path in registry.items()}
    old,full,new=[records[k] for k in registry]
    detail=common.read_json(NEW/'matched_protocol.json')
    assert new['anchor']['matched_protocol_sha256']==common.sha256(NEW/'matched_protocol.json')
    assert new['anchor']['sequence_clock_ablation']==detail
    assert new['source_hashes']==detail['source_hashes']
    snapshot=NEW/'source_snapshots/lightgbm_sequence_no_dep_clocks_aobt_allfinite_s20260916'
    for name,digest in new['source_hashes'].items():
        assert common.sha256(snapshot/Path(name).name)==digest
    protocol=NEW/'protocols/lightgbm_sequence_no_dep_clocks_aobt_allfinite_s20260916.json'
    assert common.sha256(protocol)==new['protocol_sha256']
    fields=['offset_AOBT_3_flt','offset_EOBT_1_flt','offset_IOBT_flt','offset_LOBT_flt','offset_SCHED_TIME_UTC_mvt','aobt_second']
    dropped=[f'flat_dep{rank}_{field}' for rank in range(1,5) for field in fields]
    assert detail['dropped_columns']==dropped and len(dropped)==24
    cm=common.read_json(CACHE/'manifest.json')
    cv=common.read_json(CACHE/'verification.json')
    assert cm['status']=='complete' and cv['status']=='passed'
    assert detail['cache_manifest_sha256']==cv['manifest_sha256']==common.sha256(CACHE/'manifest.json')
    assert detail['cache_verification_sha256']==common.sha256(CACHE/'verification.json')
    assert detail['cache_features_sha256']==cm['outputs']['training_features.parquet']==common.sha256(CACHE/'training_features.parquet')
    assert len(old['feature_columns'])==225
    assert full['feature_columns']==old['feature_columns']+cm['features']
    expected=old['feature_columns']+[c for c in cm['features'] if c not in dropped]
    assert new['feature_columns']==detail['final_columns']==expected and len(expected)==313
    assert old['fit']['params']==full['fit']['params']==new['fit']['params']
    assert old['fit_ids']==full['fit_ids']==new['fit_ids']
    assert old['anchor']['feature_receipts']==full['anchor']['feature_receipts']==new['anchor']['feature_receipts']
    assert detail['controls'][fold]['manifest_sha256']==common.sha256(registry['context337']/'manifest.json')
    index,split,_=common.fold_data(meta,fold,full=True)
    finite=np.isfinite(meta.proxy_sec.to_numpy(float))
    for rec in records.values():
        assert common.object_hash(rec['split'])==common.object_hash(split)
        for stage,pos in index.items():
            selected=pos[finite[pos]]
            assert rec['fit_ids'][stage]==dict(n=len(selected),hash=common.object_hash(meta.iloc[selected][ID].tolist()))
    reference,refrec=common.reference(fold)
    assert common.object_hash(split)==common.object_hash(refrec['split'])
    np.testing.assert_array_equal(reference[ID],meta.iloc[index['score']][ID])
    np.testing.assert_array_equal(reference[TARGET],meta.iloc[index['score']][TARGET])
    y=reference[TARGET].to_numpy(float)
    missing=~np.isfinite(reference.proxy_sec.to_numpy(float))
    predictions={}
    tune={}
    for name,path in registry.items():
        frame=pd.read_parquet(path/'candidate.parquet')
        np.testing.assert_array_equal(frame[ID],reference[ID])
        np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
        pred=frame.prediction_sec.to_numpy(float)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(pred[missing],reference.prediction_sec.to_numpy(float)[missing])
        predictions[name]=pred
        tune_frame=pd.read_parquet(path/'tune_predictions.parquet')
        positions=index['tune'][finite[index['tune']]]
        np.testing.assert_array_equal(tune_frame[ID],meta.iloc[positions][ID])
        tune[name]=metric(meta.iloc[positions][TARGET].to_numpy(float),tune_frame.prediction_sec.to_numpy(float))
    pairs={}
    for first,second in [('base225','context337'),('base225','no_dep_clocks313'),('no_dep_clocks313','context337')]:
        a,b=predictions[first],predictions[second]
        comp=comparison(y,a,b,reference.day.to_numpy())
        gain=(a-y)**2-(b-y)**2
        top=np.argsort(gain)[-10:]
        keep=np.ones(len(y),bool)
        keep[top]=False
        comp['remove_top10_gain_seconds']=metric(y,a,keep)['rmse_sec']-metric(y,b,keep)['rmse_sec']
        comp['airport_sse_gain']={str(ap):float(gain[reference.ADEP_mvt.eq(ap)].sum()) for ap in sorted(reference.ADEP_mvt.unique())}
        pairs[first+'_to_'+second]=comp
    result=dict(status='passed',fold=fold,source_sha256=common.sha256(__file__),
        manifests={name:dict(path=str(path/'manifest.json'),sha256=common.sha256(path/'manifest.json')) for name,path in registry.items()},
        schema_exact=True,only24_DEPclock_fields_removed=True,original225_and_allARR_fields_retained=True,
        original_IDs_labels_params_receipts_match=True,missing_V2_exact=True,saved_replay_receipts_zero=True,
        no_independent_model_inference=True,metrics={name:metric(y,p) for name,p in predictions.items()},tune=tune,comparisons=pairs)
    common.write_json(destination,result)
    print('ABLATION_AUDIT',fold,result['metrics'],flush=True)
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
        run_fold(fold,meta)
    if all((OUT/(fold+'.json')).exists() for fold in ['F1','F3']):
        results={fold:common.read_json(OUT/(fold+'.json')) for fold in ['F1','F3']}
        seasonal={name:season_score(results['F1']['metrics'][name],results['F3']['metrics'][name]) for name in results['F1']['metrics']}
        gain=seasonal['base225']-seasonal['context337']
        remaining=seasonal['base225']-seasonal['no_dep_clocks313']
        payload=dict(status='passed',source_sha256=common.sha256(__file__),seasonal_rmse=seasonal,
            ordered_context_gain_seconds=gain,gain_without_DEPclocks_seconds=remaining,
            lost_gain_after_ablation_seconds=gain-remaining,retained_RMSE_gain_fraction=remaining/gain,
            caveat='MatchedLGB600 ablation only; not a direct deep337/leaf63 ablation. Adaptive exposeddevelopment; neither feature importance nor ablation proves physical causation.')
        common.write_json(OUT/'summary.json',payload)
        print('ABLATION_SEASONAL',payload,flush=True)


if __name__=='__main__':
    main()
