"""Independent preparation checks for the frozen final missing route."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import lightgbm
from preflight import ROOT, OUT, RAW, ID, TARGET, read, sha
import gc
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits


def main():
    dest=OUT/'missing_preflight_receipt.json'
    assert not dest.exists()
    code=ROOT/'review_work/tail240_20260916/forensics/final_missing/run.py'
    sys.path.insert(0,str(code.parent))
    spec=importlib.util.spec_from_file_location('missing_release_review',code)
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    protocol,prep=read(subject.OUT/'protocol.json'),read(subject.OUT/'preparation.json')
    assert prep['status']=='passed' and prep['protocol_sha256']==sha(subject.OUT/'protocol.json')
    assert protocol['normalized_steps']==int(np.median(list(protocol['normalized_fold_steps'].values())))==111
    assert protocol['forest_leaf']==1 and protocol['forest_trees']==300
    for relative,digest in protocol['source_hashes'].items():assert sha(ROOT/relative)==digest
    for filename,digest in prep['outputs'].items():assert sha(subject.OUT/filename)==digest
    assert prep['training_rows']==22470 and prep['ranking_missing_rows']==5290
    assert all(r['normalized_max_abs_delta']==r['forest_max_abs_delta']==0 for r in prep['fold_replay'].values())
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    assert sha(RAW/'ranking.parquet')==protocol['raw_hashes']['ranking.parquet']
    raw=pq.read_table(RAW/'ranking.parquet',columns=subject.FIELDS,use_threads=False).to_pandas(strings_to_categorical=True)
    dep=raw.loc[raw.PHASE_mvt.eq('DEP')].copy(); del raw
    timecol=subject.TIME
    mask=dep.AOBT_3_flt.isna()
    rank=pd.read_parquet(subject.OUT/'ranking_missing_meta.parquet')
    np.testing.assert_array_equal(dep.loc[mask,ID],rank[ID]);assert len(rank)==5290
    peers=pd.read_parquet(subject.OUT/'ranking_peer_features.parquet')
    np.testing.assert_array_equal(peers.index,dep[ID])
    # Direct slices of ID-sorted groups independently check all six inputs used by all missing rows.
    dep['month_check']=pd.to_datetime(dep[timecol],utc=True).dt.strftime('%Y-%m')
    dep['seconds_check']=pd.to_datetime(dep[timecol],utc=True).dt.as_unit('ns').astype('int64')/1e9
    checked=0
    for _,group in dep.groupby(['ADEP_mvt','month_check'],observed=True):
        ordered=group.sort_values(ID)
        times=ordered.seconds_check.to_numpy()
        ids=ordered[ID].to_numpy()
        for position in np.flatnonzero(ordered.AOBT_3_flt.isna()):
            for width in (2,8,32):
                values=np.concatenate([times[max(0,position-width):position],times[position+1:position+width+1]])
                actual=peers.loc[ids[position],[f'id_peer_median_time_minus_query_w{width}',f'id_peer_time_spread_w{width}']].to_numpy(float)
                wanted=np.array([np.median(values)-times[position],np.quantile(values,.9)-np.quantile(values,.1)])
                np.testing.assert_array_equal(actual,wanted)
                checked+=2
    frame=subject.shared.base.airport_features(dep.loc[mask])
    times=pd.Series(dep.loc[mask,timecol].to_numpy(),index=frame.index)
    frame=pd.concat([frame,subject.shared.identity.id_context_features(frame,peers.loc[frame.index],times)],axis=1)
    pd.testing.assert_frame_equal(frame,pd.read_parquet(subject.OUT/'ranking_features.parquet'))
    del dep,peers,frame
    gc.collect()
    train=pd.read_parquet(subject.OUT/'training_features.parquet')
    meta=pd.read_parquet(subject.OUT/'training_meta.parquet')
    source_meta=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(source_meta)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][source_meta.name]
    full=pd.read_parquet(source_meta,columns=[ID,TARGET,timecol,'proxy_sec'])
    missing=full.loc[~np.isfinite(full.proxy_sec)].set_index(ID)
    np.testing.assert_array_equal(train.index,missing.index)
    pd.testing.assert_frame_equal(meta,missing[[TARGET,timecol,'proxy_sec']])
    assert missing[timecol].dt.year.eq(2025).all()
    assert set(missing[timecol].dt.month)==set(range(1,13))
    peak=psutil.Process().memory_info().peak_wset
    assert peak<3*1024**3
    receipt=dict(status='passed',protocol_sha256=sha(subject.OUT/'protocol.json'),preparation_sha256=sha(subject.OUT/'preparation.json'),
        all5290_missing_raw_ids_exact=True,independent_peer_values=checked,ranking76features_raw_replayed_exact=True,
        all22470_training_ids_labels_time_proxy_exact=True,all12training_months=True,
        producer_full_F1_F3_native_replays_bound=prep['fold_replay'],static_fitting_contract_review='111normalized rounds;300ET minleaf1;fit-only preprocessing;earlier-month train templates;all2025 ranking templates;nested25 blends',
        source_sha256=sha(__file__),peak_bytes=peak,gpu_used=False,fitting_used=False,network_used=False)
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
