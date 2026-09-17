"""Small no-fit reproductions of loader risks; no production feature values read."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import lightgbm
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT=v.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/risk'))
import run_risk as risk
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
from encoders import FrameEncoder
OUT=ROOT/'private_runs/tail240_20260916/validation/preprocessing_contract_probe_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    original=risk.feature_sources
    paths=[]
    for number in range(2):
        path=OUT/f'overlap_{number}.parquet'
        pd.DataFrame({v.ID:[1],'numeric':[100.+number]}).to_parquet(path,index=False)
        paths.append(path)
    risk.feature_sources=lambda _: [(path,v.sha256(path),False,['numeric']) for path in paths]
    matrix,vocab,receipts=risk.load_matrix(pd.Index([1,2]),1,['numeric'],OUT)
    coverage=dict(accepted=True,requested_ids=[1,2],source_ids=[[1],[1]],
        summed_rows=sum(r['rows'] for r in receipts),duplicate_id=1,missing_id=2,
        limitation='Synthetic only. Does not establish overlapping or absent IDs in frozen production caches.')
    del matrix
    risk.feature_sources=original
    values=np.array([1e40,-1e40,-999999.,-999998.99,16777217.,1750000000.],dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        cast=pd.to_numeric(pd.Series(values)).to_numpy(dtype=np.float32,na_value=np.nan)
    overflow=np.isfinite(values)&~np.isfinite(cast)
    transformed=cast.copy();transformed[~np.isfinite(transformed)]=np.nan
    assert int(overflow.sum())==2 and np.isnan(transformed[:2]).all()
    assert cast[2]==cast[3]==-999999. and cast[4]!=values[4]
    encoder=FrameEncoder().fit(pd.DataFrame({'raw_clock':[-999999.,0.]}))
    restored=encoder.transform(pd.DataFrame({'raw_clock':[-999999.,0.]}))
    assert np.isnan(restored.raw_clock.iloc[0])
    records=dict(status='completed_contract_probe',source_sha256=v.sha256(__file__),
        loader_sha256=v.sha256(risk.__file__),encoder_sha256=v.sha256(ROOT/'review_work/breakthrough_20260916/models/encoders.py'),
        cross_source_duplicate_can_mask_missing_id=coverage,
        cast_before_finite_check=dict(finite64_to_missing32_count=2,near_sentinel_rounding_collision=True,
            integer_example_before=float(values[4]),integer_example_after=float(cast[4]),
            epoch_seconds_float32_spacing=float(np.spacing(cast[5])),
            limitation='Synthetic contract behavior only; production value prevalence requires a separate pre-cast audit.'),
        blanket_FrameEncoder_raw_sentinel_becomes_missing=True,
        no_training=True,no_gpu=True,production_numeric_values_read=False,
        peak_bytes=psutil.Process().memory_info().peak_wset)
    assert records['peak_bytes']<2*1024**3
    v.write_json(OUT/'receipt.json',records)
    print(records,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
