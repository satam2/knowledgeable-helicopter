"""CPU-native synthetic CatBoost387 canary validation; no new fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import lightgbm
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import catboost
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT, read, sha, write, guard


def main():
    source=ROOT/'review_work/tail240_20260916/forensics/catboost387_canary.py'
    spec=importlib.util.spec_from_file_location('cb387_canary_subject',source)
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    folder=subject.OUT;protocol=read(folder/'protocol.json');receipt=read(folder/'worker_receipt.json')
    watch=read(folder/'watchdog_receipt.json')
    assert subject.declaration()==protocol
    assert receipt['source_sha256']==sha(source) and receipt['protocol_sha256']==sha(folder/'protocol.json')
    assert receipt['status']==watch['status']=='passed' and watch['exit_code']==0
    for name,expected in receipt['outputs'].items():assert sha(folder/name)==expected
    frame,y=subject.generate(100000,protocol['columns'],protocol['synthetic_category_cardinalities'],seed=protocol['seed'])
    encoder=joblib.load(folder/'encoder.joblib')
    assert encoder.columns==protocol['columns'] and len(encoder.numeric)==372 and len(encoder.categories)==15
    for name,mapping in encoder.categories.items():
        expected={value:i+2 for i,value in enumerate(sorted(frame[name].astype('string').dropna().unique()))}
        assert mapping==expected
    probe=frame.iloc[:1024].copy()
    for name in encoder.categories:
        probe[name]=probe[name].astype('string')
        probe.loc[probe.index[0],name]='synthetic_unseen_probe'
        probe.loc[probe.index[1],name]=pd.NA
    encoded=encoder.transform(probe)
    for name in encoder.categories:
        assert encoded[name].iloc[0]==1 and encoded[name].iloc[1]==0
    model=catboost.CatBoostRegressor();model.load_model(str(folder/'model.cbm'),format='cbm')
    assert model.tree_count_==50 and model.feature_names_==protocol['columns']
    prediction=model.predict(encoded,task_type='CPU',thread_count=1)
    stored=np.load(folder/'probe_predictions.npy')
    np.testing.assert_array_equal(prediction,stored)
    assert np.isfinite(prediction).all()
    assert receipt['memory']['final']['peak']<4*1024**3
    assert watch['observed_child_peak_bytes']<10*1024**2
    out=ROOT/'private_runs/tail240_20260916/validation/catboost387_canary_result_v1'
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='native_cpu_probe_passed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(folder/'protocol.json'),
        worker_receipt_sha256=sha(folder/'worker_receipt.json'),watchdog_receipt_sha256=sha(folder/'watchdog_receipt.json'),
        synthetic_fit_vocabularies_exact=15,numeric_columns=372,rows=1024,native_replay_max_abs_delta=0.,
        worker_reported_OS_peak_bytes=receipt['memory']['final']['peak'],
        worker_reported_full100000_predictions_finite=receipt['finite_full_fit_predictions'],
        worker_full100000_predictions_independently_replayed=False,peak_bytes=guard(),no_fit_or_GPU=True,
        limitation='Parent100mswatchdog observedWindowslauncher only, not actualworker. Worker selfcheckpoints recordOShistoricalpeak<1GiB, but continuous100msactualworkerprotection notestablished. Synthetic50round canary does not guarantee full816060row/5000round resources.')
    write(out/'receipt.json',result);print(result,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
