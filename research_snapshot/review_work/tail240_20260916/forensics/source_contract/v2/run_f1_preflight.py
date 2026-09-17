"""Full F1 source contract, frozen-loader equality and native replay; no fit."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
import lightgbm as lgb
from pathlib import Path
import sys
import gc
import time

import contract
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/risk'))
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import run_risk as risk
import schema_discovery
common,ID = risk.common,risk.ID
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/forensics/source_contract/v2')
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    info = psutil.Process().memory_info()
    peak = max(info.rss,getattr(info,'peak_wset',info.rss))
    if peak >= 3*1024**3 or psutil.virtual_memory().available < 8*1024**3:
        raise MemoryError('Source-contract3GiB process/8GiB host-reserve budget')
    return peak


def main():
    assert psutil.virtual_memory().available >= 11*1024**3
    assert not OUT.exists()
    started = time.monotonic()
    reference = ROOT/'private_runs/tail240_20260916/state/learning_curve_design/v3/F1'
    native = ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    control = risk.union_folder('F1')
    marker = common.read_json(control/'manifest.json')
    original_native = common.read_json(native/'manifest.json')
    metadata = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(metadata) == common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta = pd.read_parquet(metadata,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'proxy_sec'])
    indices,split = common.make_fold(meta,common.load_config('configs/folds.yaml')['F1'])
    parts = {stage:meta.iloc[rows[np.isfinite(meta.iloc[rows].proxy_sec.to_numpy())]].copy()
             for stage,rows in indices.items() if stage in ['fit','tune']}
    for stage,frame in parts.items():
        assert dict(n=len(frame),hash=common.object_hash(frame[ID].tolist())) == marker['fit_ids'][stage]
    ids = pd.Index(pd.concat([parts['fit'][ID],parts['tune'][ID]],ignore_index=True))
    nfit = len(parts['fit'])
    del meta
    columns = marker['feature_columns']
    sources,discovery = schema_discovery.cached_discovery(risk.feature_sources,columns)
    reference_path = reference/'matrix.float32'
    assert reference_path.stat().st_size == len(ids)*len(columns)*4
    reference_sha = common.sha256(reference_path)
    OUT.mkdir(parents=True,exist_ok=False)
    source_paths = [Path(__file__),Path(contract.__file__),Path(__file__).with_name('test_contract.py'),
                    Path(risk.__file__),Path(schema_discovery.__file__)]
    protocol = dict(preserved_v1_protocol_sha256=common.sha256(OUT.parent/'v1/protocol.json'),
        operational_change='Wrapper enforces external_path and guards; streaming bytehash comparison replaces simultaneous matrix mappings; feature science unchanged',source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in source_paths},
        sources=[dict(path=str(p),sha256=h,fill=f,columns=c) for p,h,f,c in sources],
        control_manifest_sha256=common.sha256(control/'manifest.json'),
        native_manifest_sha256=common.sha256(native/'manifest.json'),
        reference_matrix_sha256=reference_sha,reference_matrix_path=str(reference_path),
        rows=len(ids),nfit=nfit,columns=columns,ids_hash=common.object_hash(ids.tolist()),
        scope='Source safety and exact existing matrix/native prediction parity; no labels read, no model fit, no accuracy claim',
        resources='1CPU;3GiB process historical/current peak;8GiB hostreserve;source/batch/loader guards;noGPU')
    common.write_json(OUT/'protocol.json',protocol)
    with threadpool_limits(1):
        matrix,vocab,receipts,check = contract.load_checked(risk,sources,ids,nfit,columns,OUT,guard=guard)
        common.write_json(OUT/'contract_receipt.json',check)
        assert common.sha256(native/'encoder.json') == original_native['outputs']['encoder.json']
        assert common.read_json(native/'encoder.json') == dict(columns=columns,vocab=vocab)
        matrix.flush()
        del matrix
        gc.collect()
        matrix_sha = common.sha256(OUT/'matrix.float32')
        assert matrix_sha == reference_sha
        assert common.sha256(reference_path) == reference_sha
        guard()
        matrix = np.memmap(OUT/'matrix.float32',mode='r',dtype='float32',shape=(len(ids),len(columns)),order='F')
        assert common.sha256(native/'model.txt') == original_native['outputs']['model.txt']
        model = lgb.Booster(model_file=str(native/'model.txt'))
        assert common.sha256(control/'tune_predictions.parquet') == marker['outputs']['tune_predictions.parquet']
        expected = pd.read_parquet(control/'tune_predictions.parquet')
        np.testing.assert_array_equal(expected[ID],parts['tune'][ID])
        prediction = np.empty(len(parts['tune']),dtype=float)
        for start in range(0,len(prediction),8192):
            stop = min(start+8192,len(prediction))
            prediction[start:stop] = model.predict(matrix[nfit+start:nfit+stop],num_threads=1)+parts['tune'].proxy_sec.to_numpy(float)[start:stop]
            guard()
        np.testing.assert_array_equal(prediction,expected.prediction_sec.to_numpy(float))
    common.write_json(OUT/'receipt.json',dict(status='passed',protocol_sha256=common.sha256(OUT/'protocol.json'),
        contract_receipt_sha256=common.sha256(OUT/'contract_receipt.json'),
        matrix_equal_including_nan=True,matrix_byte_sha256=matrix_sha,comparison='Streaming SHA256 bit-for-bit identity, including NaN bit patterns',feature_values_compared=len(ids)*len(columns),
        native_replay_max_abs_delta_sec=0.,native_replay_rows=len(prediction),
        feature_receipts=receipts,discovery=discovery,split=split,
        peak_bytes=guard(),elapsed_seconds=time.monotonic()-started,
        no_target_column_read=True,no_model_fit=True,no_gpu=True))
    print(common.read_json(OUT/'receipt.json'),flush=True)


if __name__ == '__main__':
    main()
