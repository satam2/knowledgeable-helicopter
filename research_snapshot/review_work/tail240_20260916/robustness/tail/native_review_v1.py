"""Independent source-cache reconstruction of a bounded LGB stopping subset."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import lightgbm
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/campaign_20260916'))
import common
ID = common.ID
FOLDER = ROOT/'private_runs/tail240_20260916/robustness/chronological_v1/F1/lgb/select'
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/robustness/tail/native_review/v1')
BASE_CATS = {'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt',
    'AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt',
    'airport_stand','airport_runway'}
pa.set_cpu_count(2)
pa.set_io_thread_count(1)
PEAK = 0


def guard():
    global PEAK
    info = psutil.Process().memory_info()
    PEAK = max(PEAK, getattr(info, 'peak_wset', info.rss))
    assert PEAK < 4*1024**3 and psutil.virtual_memory().available >= 8*1024**3


def main():
    started = time.monotonic()
    marker = common.read_json(FOLDER/'manifest.json')
    assert marker['status'] == 'complete' and marker['family']=='lgb' and marker['phase']=='select'
    protocol_path = FOLDER.parents[2]/'protocol.json'
    assert common.sha256(protocol_path)==marker['protocol_sha256']
    assert common.sha256(FOLDER/'model.joblib')==marker['model_sha256']
    assert common.sha256(FOLDER/'stop.parquet')==marker['predictions']['stop']['sha256']
    assert common.sha256(FOLDER/'source_coverage.json')==marker['outputs']['source_coverage.json']
    coverage = common.read_json(FOLDER/'source_coverage.json')
    pred = pd.read_parquet(FOLDER/'stop.parquet')
    assert 'TAXITIME_SEC_mvt' not in pred
    assert common.object_hash(pred[ID].tolist())==marker['predictions']['stop']['ordered_id_hash']
    model = joblib.load(FOLDER/'model.joblib')
    columns = marker['features']
    assert list(model['encoder'].columns)==columns
    nfit = marker['fit_rows']
    rawmatrix = np.memmap(FOLDER/'matrix.float32', mode='r', dtype='float32',
        shape=(nfit+len(pred),len(columns)), order='F')
    selected = set(range(64))
    unknown_examples = {}
    for name in model['encoder'].maps:
        index = columns.index(name)
        rows = np.flatnonzero(rawmatrix[nfit:,index]==1)
        if len(rows):
            unknown_examples[name] = int(rows[0])
            selected.add(int(rows[0]))
    positions = np.array(sorted(selected))
    query = pred.iloc[positions].copy()
    ids = pd.Index(query[ID])
    assert ids.is_unique and len(ids)<=64+len(model['encoder'].maps)
    values = {name:np.full(len(ids),None,dtype=object) if name in model['encoder'].maps else np.full(len(ids),np.nan,dtype=np.float32) for name in columns}
    seen = {name:np.zeros(len(ids),bool) for name in columns}
    reads = []
    for source in coverage['sources']:
        if source['requested_rows']==0:
            continue
        path = Path(source['path'])
        assert common.sha256(path)==source['sha256']
        parquet = pq.ParquetFile(path)
        found, groups = 0, 0
        for group in range(parquet.num_row_groups):
            key = parquet.read_row_group(group,columns=[ID],use_threads=False).to_pandas()[ID]
            loc = ids.get_indexer(key)
            mask = loc>=0
            if not mask.any():
                continue
            frame = parquet.read_row_group(group,columns=[ID,*source['columns']],use_threads=False).to_pandas().loc[mask]
            target = loc[mask]
            assert len(np.unique(target))==len(target)
            found += len(target)
            groups += 1
            for name in source['columns']:
                assert not seen[name][target].any(), (name,path)
                seen[name][target]=True
                if name in model['encoder'].maps:
                    values[name][target]=frame[name].astype('string').to_numpy(na_value=None)
                else:
                    v=pd.to_numeric(frame[name]).to_numpy(dtype=np.float32,na_value=np.nan)
                    v[~np.isfinite(v)]=np.nan
                    if source['fill']:
                        v[np.isnan(v)]=-999999.
                    values[name][target]=v
            guard()
        reads.append(dict(path=str(path),sha256=source['sha256'],selected_rows=found,feature_row_groups_read=groups))
    assert all(v.all() for v in seen.values())
    unknown_verified = {}
    for name in columns:
        if name in model['encoder'].maps:
            v=pd.Series(values[name],dtype='string')
            known=model['encoder'].maps[name]
            v=v.where(v.isna()|v.isin(known),'__UNSEEN_CONTEXT_VALUE__')
            if name not in BASE_CATS:
                v=v.fillna('MISSING')
            values[name]=pd.Categorical(v)
        else:
            np.testing.assert_array_equal(values[name], rawmatrix[nfit+positions,columns.index(name)])
    frame=pd.DataFrame(values,index=ids)
    encoded=model['encoder'].transform(frame)
    for name,row in unknown_examples.items():
        subset_position=int(np.flatnonzero(positions==row)[0])
        value=int(encoded[name].iloc[subset_position])
        assert value==1
        unknown_verified[name]=dict(stop_position=row,encoded_code=value)
    native=model['estimator'].booster_.predict(encoded,num_iteration=model['steps'],num_threads=2)
    predictions=np.asarray(native,float)+query.proxy_sec.to_numpy(float)
    difference=float(np.max(np.abs(predictions-query.prediction_sec.to_numpy(float))))
    assert difference<=1e-6
    OUT.mkdir(parents=True,exist_ok=False)
    query[[ID,'proxy_sec','prediction_sec']].assign(independent_prediction_sec=predictions).to_parquet(OUT/'replay.parquet',index=False)
    guard()
    receipt=dict(status='passed',source_sha256=common.sha256(__file__),producer_manifest_sha256=common.sha256(FOLDER/'manifest.json'),
        model_sha256=marker['model_sha256'],source_coverage_sha256=common.sha256(FOLDER/'source_coverage.json'),
        selected_rows=len(ids),selection='First64 stopping rows plus first stopping example with code1 for each categorical feature',
        selected_id_hash=common.object_hash(ids.tolist()),unknown_category_examples=unknown_verified,
        numeric_feature_cells_exact=True,categorical_maps='Reused saved fit-only encoder mappings; no vocabulary refit',
        source_reads=reads,steps=model['steps'],native_max_abs_delta_sec=difference,
        stopping_targets_read=False,evaluation_labels_read=False,new_fits=False,
        replay_sha256=common.sha256(OUT/'replay.parquet'),peak_bytes=PEAK,duration_sec=time.monotonic()-started,
        limitation='Bounded independent raw-cache input reconstruction and native Booster replay; not a full training reproduction or performance evaluation.')
    common.write_json(OUT/'receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
