"""Independent completed-expert contracts; no fits or evaluation-label reads."""
from materialize_cohorts import ROOT,COLS,ID,TIME,sha,read,guard
from taxiout.artifacts import object_hash
import argparse
import json
import numpy as np
import pandas as pd

BASE=ROOT/'private_runs/tail240_20260916/robustness/chronological_v1'
COHORTS=ROOT/'private_runs/tail240_20260916/robustness/validation/cohorts_v1'
OUT=ROOT/'private_runs/tail240_20260916/robustness/validation/experts'
TARGET='TAXITIME_SEC_mvt'


def verify(fold,family,phase):
    protocol=read(BASE/'protocol.json');protocol_hash=sha(BASE/'protocol.json')
    assert protocol['cohort_manifest_sha256']==sha(COHORTS/'manifest.json')
    for relative,digest in protocol['sources'].items():assert sha(ROOT/relative)==digest,relative
    cohort=read(COHORTS/'manifest.json')
    folder=BASE/fold/family/phase;manifest=read(folder/'manifest.json')
    assert manifest['status']=='complete'
    assert (manifest['fold'],manifest['family'],manifest['phase'])==(fold,family,phase)
    assert manifest['protocol_sha256']==protocol_hash
    assert manifest['evaluation_labels_read'] is False
    assert manifest['features']==protocol['feature_columns']
    assert len(manifest['features'])==len(set(manifest['features']))==387
    for name,digest in manifest['outputs'].items():assert sha(folder/name)==digest,name
    assert manifest['model_sha256']==sha(folder/'model.joblib')
    launch=read(folder/'launch.json');assert launch['protocol_sha256']==protocol_hash and launch['phase']==phase
    stages=['train','stop'] if phase=='select' else ['refit','calibration']+sorted(
        name for name in cohort['folds'][fold]['stages'] if name.startswith('evaluation_'))
    frames={}
    for stage in stages:
        entry=cohort['folds'][fold]['stages'][stage];path=ROOT/entry['path']
        assert sha(path)==entry['sha256']
        frame=pd.read_parquet(path)
        frames[stage]=frame.loc[np.isfinite(frame.proxy_sec)].reset_index(drop=True)
    fit=frames[stages[0]]
    assert manifest['fit_rows']==len(fit)
    assert manifest['fit_id_hash']==object_hash(fit[ID].tolist())
    evidence=read(folder/'fit_evidence.json')
    assert evidence['rows']==len(fit) and evidence['steps']==manifest['selected_steps']
    assert 1<=manifest['selected_steps']<={'lgb':2500,'ple':40,'catboost':5000}[family]
    selection_binding=None
    if phase=='refit':
        selection=folder.parent/'select/manifest.json';selected=read(selection)
        assert selected['protocol_sha256']==protocol_hash and selected['status']=='complete'
        assert selected['fold']==fold and selected['family']==family and selected['phase']=='select'
        assert selected['selected_steps']==evidence['steps']
        selection_binding=dict(sha256=sha(selection),selected_steps=selected['selected_steps'])
    events=read(folder/'access.json')
    label_stages=['train','stop'] if phase=='select' else ['refit']
    assert [event['kind'] for event in events]==['expert_label_read']*len(label_stages)+['expert_frozen','predictions_frozen']
    assert pd.to_datetime([event['utc'] for event in events],utc=True).is_monotonic_increasing
    metadata=ROOT/cohort['metadata_path'];assert sha(metadata)==protocol['metadata_sha256']==cohort['metadata_sha256']
    for stage,event in zip(label_stages,events):
        assert event['stage']==stage
        frame=frames[stage]
        assert event['ordered_id_hash']==object_hash(frame[ID].tolist())
        low,high=frame[TIME].min(),frame[TIME].max()
        labels=pd.read_parquet(metadata,columns=[ID,TARGET],filters=[(TIME,'>=',low),(TIME,'<=',high)]).set_index(ID)
        assert labels.index.is_unique
        values=labels.loc[frame[ID],TARGET].to_numpy(float)
        assert np.isfinite(values).all() and event['target_hash']==object_hash(values.tolist())
    assert events[-2]['stage']==stages[0] and events[-2]['model_sha256']==manifest['model_sha256']
    assert set(manifest['predictions'])==set(stages[1:])
    reports={}
    for stage in stages[1:]:
        entry=manifest['predictions'][stage];path=folder/(stage+'.parquet')
        assert sha(path)==entry['sha256']
        predictions=pd.read_parquet(path)
        assert list(predictions)==COLS+['prediction_sec']
        pd.testing.assert_frame_equal(predictions[COLS],frames[stage],check_exact=True)
        assert predictions[ID].is_unique and np.isfinite(predictions.prediction_sec).all()
        assert len(predictions)==entry['rows'] and object_hash(predictions[ID].tolist())==entry['ordered_id_hash']
        assert 0<=entry['native_max_delta_sec']<=1e-6
        reports[stage]=dict(rows=len(predictions),ordered_id_hash=entry['ordered_id_hash'],
            sha256=entry['sha256'],producer_native_max_delta_sec=entry['native_max_delta_sec'])
    coverage=read(folder/'source_coverage.json')
    total=sum(len(frame) for frame in frames.values())
    assert coverage['status']=='passed' and coverage['requested_rows']==total
    assert coverage['columns']==protocol['feature_columns']
    assert all(record['rows']==total for record in coverage['coverage'].values())
    assert [rec['sha256'] for rec in coverage['sources']]==[rec['sha256'] for rec in manifest['feature_sources']]
    for record in coverage['sources']:assert sha(record['path'])==record['sha256']
    result=dict(status='passed',verifier_sha256=sha(__file__),protocol_sha256=protocol_hash,
        expert_manifest_sha256=sha(folder/'manifest.json'),fold=fold,family=family,phase=phase,
        fit_rows=len(fit),selected_steps=manifest['selected_steps'],selection_manifest_binding=selection_binding,
        exact_label_read_stages=label_stages,source_hashes_and_feature_coverage_verified=True,
        predictions=reports,evaluation_labels_read=False,fit_used=False,peak_bytes=guard(),
        limitation='Independent IDs, metadata, fit-label hashes, source hashes, selected-step transfer and receipt validation. Native replay values in this receipt are producer all-row reload evidence, not an independent model inference replay; feature matrix and encoder internals require separate verification.')
    OUT.mkdir(parents=True,exist_ok=True);dest=OUT/f'{fold}_{family}_{phase}.json'
    assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3'])
    parser.add_argument('--family',required=True,choices=['lgb','ple','catboost'])
    parser.add_argument('--phase',required=True,choices=['select','refit']);args=parser.parse_args()
    verify(args.fold,args.family,args.phase)
