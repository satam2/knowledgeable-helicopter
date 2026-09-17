"""Independent final-fit checks before any network upload."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, CatBoostClassifier

from run_screening import WORKSPACE, RAW
from taxiout.artifacts import read_json, write_json, sha256, source_hashes
from taxiout.io import training_paths, read_raw, concat_frames, verified_manifest
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, TARGET, PHASE
from taxiout.submission import validate_submission, serialized_values
from next230_common import config_for

OUT=external_path(WORKSPACE/'private_runs/submission_v2')


def main():
    protocol=read_json(OUT/'protocol.json')
    manifest=verified_manifest(config_for('baseline'))
    assert source_hashes()==protocol['source_hashes']
    assert sha256(WORKSPACE/'review_work/final_submission_v2.py')==protocol['runner_sha256']
    ready=read_json(OUT/'submission_ready.json')
    file=OUT/'knowledgeable-helicopter_v2.parquet'
    expected=pd.read_parquet(OUT/'ranking_predictions.parquet')
    template=pd.read_parquet(RAW/'submitting.parquet')
    saved=pd.read_parquet(file)
    assert np.array_equal(saved[ID],template[ID])
    ordered=expected.set_index(ID).loc[template[ID]].prediction_sec.to_numpy()
    assert np.array_equal(saved[TARGET],serialized_values(ordered))
    check=validate_submission(file,RAW/'submitting.parquet')
    assert check['sha256']==ready['sha256'] and check['rows']==344841
    assert ready['exact_batch_replay'] and ready['pipeline_verified']
    models={}
    for name,trees in {**protocol['final_trees'],'gate':300}.items():
        path=OUT/'components'/name
        record=read_json(path/'record.json')
        klass=CatBoostClassifier if name.endswith('_classifier') else CatBoostRegressor
        model=klass().load_model(str(path/'model.cbm'))
        assert model.tree_count_==trees
        assert model.feature_names_==record['feature_names']
        assert sha256(path/'model.cbm')==record['model_sha256']==ready['components'][name]
        assert record['status']=='complete' and record['reload_exact']
        models[name]={'rows':record['rows'],'trees':model.tree_count_,'passed':True}
    ranking=read_raw(RAW/'ranking.parquet',[ID,FLIGHT_ID,PHASE])
    train=concat_frames([read_raw(p,[ID,FLIGHT_ID,PHASE]) for p in training_paths(config_for('baseline'))])
    rank_dep=ranking.loc[ranking[PHASE].eq('DEP')]
    train_dep=train.loc[train[PHASE].eq('DEP')]
    movement_overlap=int(train_dep[ID].isin(rank_dep[ID]).sum())
    flight_overlap=int(train_dep[FLIGHT_ID].notna().mul(train_dep[FLIGHT_ID].isin(rank_dep[FLIGHT_ID].dropna())).sum())
    assert movement_overlap==0 and flight_overlap==0,(movement_overlap,flight_overlap)
    leaks=[]
    for repo in ['knowledgeable-helicopter','knowledgeable-helicopter-screening']:
        for path in (WORKSPACE/repo).rglob('*'):
            if path.is_file() and path.suffix in {'.parquet','.cbm','.pkl','.joblib','.npy','.npz'}:
                leaks.append(str(path))
    assert not leaks
    result={'passed':True,'raw_files_rehashed':len(manifest['files']),'submission':check,
        'models':models,'training_ranking_departure_id_overlap':movement_overlap,
        'training_ranking_flight_id_overlap':flight_overlap,'private_artifacts_in_repo':leaks,
        'all_frozen_sources_unchanged':True,'ready_manifest_sha256':sha256(OUT/'submission_ready.json')}
    write_json(OUT/'independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
