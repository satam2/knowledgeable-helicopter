"""Independent missing-route chronology contracts and all-row native replay."""
from materialize_cohorts import ROOT,COLS,ID,TIME,sha,read,guard
from taxiout.artifacts import object_hash
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd

BASE=ROOT/'private_runs/tail240_20260916/robustness/tail/chronological/v1'
COHORTS=ROOT/'private_runs/tail240_20260916/robustness/validation/cohorts_v1'
OUT=ROOT/'private_runs/tail240_20260916/robustness/validation/missing'
TARGET='TAXITIME_SEC_mvt'


def check_protocol():
    record=read(BASE/'protocol.json')
    assert record['cohort_manifest_sha256']==sha(COHORTS/'manifest.json')
    assert record['ordinary_training_protocol_sha256']==sha(ROOT/'private_runs/tail240_20260916/robustness/chronological_v1/protocol.json')
    for path,digest in record['source_hashes'].items():assert sha(ROOT/path)==digest
    return record,read(COHORTS/'manifest.json')


def stage_meta(cohorts,fold,stage):
    rec=cohorts['folds'][fold]['stages'][stage];path=ROOT/rec['path']
    assert sha(path)==rec['sha256']
    frame=pd.read_parquet(path)
    return frame.loc[~np.isfinite(frame.proxy_sec)].reset_index(drop=True)


def stage_labels(cohorts,frame):
    path=ROOT/cohorts['metadata_path'];assert sha(path)==cohorts['metadata_sha256']
    values=pd.read_parquet(path,columns=[ID,TARGET],filters=[(TIME,'>=',frame[TIME].min()),(TIME,'<=',frame[TIME].max())]).set_index(ID)
    assert values.index.is_unique
    return values.loc[frame[ID],TARGET].to_numpy(float)


def model(fold,arm):
    protocol,cohort=check_protocol();folder=BASE/fold/arm;marker=read(folder/'manifest.json')
    assert marker['status']=='complete_predictions_frozen' and marker['fold']==fold and marker['arm']==arm
    assert marker['protocol_sha256']==sha(BASE/'protocol.json')
    assert marker['evaluation_labels_read'] is False and marker['calibration_labels_read'] is False
    assert marker['model_sha256']==sha(folder/'model.joblib')
    for name,digest in marker['outputs'].items():assert sha(folder/name)==digest
    permitted=['refit'] if arm.startswith('et_') else ['train','stop','refit']
    events=read(folder/'access.json');assert events==marker['target_access_events']
    assert [rec['kind'] for rec in events]==['label_read']*len(permitted)+['refit_frozen','all_predictions_frozen']
    assert pd.to_datetime([r['utc'] for r in events],utc=True).is_monotonic_increasing
    for stage,event in zip(permitted,events):
        frame=stage_meta(cohort,fold,stage);y=stage_labels(cohort,frame)
        assert event['stage']==stage and event['rows']==len(frame)
        assert event['id_hash']==object_hash(frame[ID].tolist()) and event['target_hash']==object_hash(y.tolist())
    refit=stage_meta(cohort,fold,'refit')
    assert marker['refit_rows']==len(refit) and marker['refit_id_hash']==object_hash(refit[ID].tolist())
    assert refit.FLIGHT_ID_mvt.dropna().is_unique
    for key,value in [('rows',len(refit)),('known_flight_ids',int(refit.FLIGHT_ID_mvt.notna().sum())),('null_flight_ids',int(refit.FLIGHT_ID_mvt.isna().sum()))]:
        assert marker['known_flight_uniqueness'][key]==value
    assert events[-2]['model_sha256']==marker['model_sha256']
    sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/robustness/tail/chronological'))
    import run_v1 as producer
    feature_path=ROOT/'private_runs/tail240_20260916/forensics/final_missing/v1/training_features.parquet'
    assert sha(feature_path)==protocol['feature_sha256']
    features=pd.read_parquet(feature_path)
    assert list(features)==protocol['feature_columns'] and len(features)==22470
    assert not {TARGET,'BLOCK_TIME_UTC_mvt'}.intersection(features.columns)
    bundle=joblib.load(folder/'model.joblib');evidence=read(folder/'fit_evidence.json')
    assert evidence['rows']==len(refit)
    if arm.startswith('et_'):
        leaf=1 if arm=='et_leaf1' else 20
        assert bundle['estimator'].n_estimators==300 and bundle['estimator'].min_samples_leaf==leaf
        assert bundle['steps']==leaf and bundle['prior'].history_n==len(refit)
        assert bundle['prior'].last_fit==refit[TIME].max()
        bundle['estimator'].n_jobs=1
    else:
        selection=read(folder/'selection.json')
        assert selection['steps']==evidence['steps']==bundle['steps']
        assert sha(folder/'selection.joblib')==selection['model_sha256']
        x=features.loc[refit[ID]];scale=producer.normalized.scale_of(x)
        np.testing.assert_allclose(np.mean(scale**2),evidence['normalizer'],rtol=1e-12)
        bundle['model'].reset_parameter({'num_threads':1})
    stages=['calibration']+['evaluation_'+m for m in protocol['panels'][fold]]
    assert set(stages)==set(marker['predictions']);checks={}
    for stage in stages:
        frame=stage_meta(cohort,fold,stage);entry=marker['predictions'][stage];path=folder/(stage+'.parquet')
        assert sha(path)==entry['sha256'];saved=pd.read_parquet(path)
        pd.testing.assert_frame_equal(saved[COLS].reset_index(drop=True),frame,check_exact=True)
        assert TARGET not in saved and len(saved)==entry['rows']
        assert object_hash(frame[ID].tolist())==entry['id_hash']
        x=features.loc[frame[ID]].copy()
        if arm.startswith('et_'):
            x[producer.forest.TIME_COLUMN]=frame[TIME].to_numpy()
            raw=producer.forest.input_frame(x)
            prior=bundle['prior'].transform(raw,pd.Series(frame[TIME].to_numpy(),index=x.index))
            constructed=pd.concat([raw,prior],axis=1)
            assert list(constructed)==bundle['feature_columns']
            prediction=bundle['estimator'].predict(bundle['encoder'].transform(constructed))
        else:
            encoded=bundle['encoder'].transform(x)
            scale=producer.normalized.scale_of(x)
            prediction=900.+scale*bundle['model'].predict(encoded,num_threads=1)
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction,saved.prediction_sec,rtol=0,atol=1e-9)
        assert 0<=entry['native_reload_max_abs_delta']<=1e-9
        checks[stage]=dict(rows=len(saved),max_abs_native_delta_sec=float(np.max(np.abs(prediction-saved.prediction_sec))))
        guard()
    result=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),
        expert_manifest_sha256=sha(folder/'manifest.json'),fold=fold,arm=arm,refit_rows=len(refit),
        exact_permitted_label_stages=permitted,predictions=checks,independent_allrow_native_replay=True,
        calibration_or_evaluation_labels_read=False,fit_used=False,gpu_used=False,peak_bytes=guard(),
        limitation='Known flight IDs are unique and purged across stages; unknown IDs cannot establish distinct real-world entities. Cached feature provenance and model preprocessing retain existing recipe.')
    OUT.mkdir(parents=True,exist_ok=True);dest=OUT/f'{fold}_{arm}.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,choices=['F1','F3'])
    p.add_argument('--arm',required=True,choices=['et_leaf1','et_leaf20','normalized_diagnostic'])
    args=p.parse_args();model(args.fold,args.arm)
