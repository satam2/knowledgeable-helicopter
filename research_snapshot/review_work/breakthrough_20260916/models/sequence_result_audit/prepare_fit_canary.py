"""Bounded original-January-fit feature fixture for native dataset diagnostics."""
import lightgbm
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json,write_json,sha256,object_hash
from taxiout.schema import ID,TARGET,MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/models/sequence_result_audit/native_canary'


def selected_frame(path,columns,ids):
    parquet=pq.ParquetFile(path)
    pieces=[]
    for i in range(parquet.num_row_groups):
        key=parquet.read_row_group(i,columns=[ID],use_threads=False).column(0)
        selected=pc.is_in(key,value_set=pa.array(ids.to_numpy(),type=key.type))
        if pc.any(selected).as_py():
            pieces.append(parquet.read_row_group(i,columns=[ID,*columns],use_threads=False).filter(selected).to_pandas())
    result=pd.concat(pieces,ignore_index=True).set_index(ID)
    assert result.index.is_unique
    return result.loc[ids]


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(meta_path)==read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,TARGET,MOVEMENT,'proxy_sec'],filters=[(MOVEMENT,'>=',pd.Timestamp('2025-01-01',tz='UTC')),
                                                                                 (MOVEMENT,'<',pd.Timestamp('2025-02-01',tz='UTC'))])
    meta=meta.loc[np.isfinite(meta.proxy_sec)].set_index(ID)
    residual=meta[TARGET]-meta.proxy_sec
    ids=pd.Index(sorted(set(meta.index[:8192])|set(residual.abs().nlargest(32).index)))
    meta=meta.loc[ids]
    marker=read_json(ROOT/'private_runs/breakthrough_20260916/deeper_sequence/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916/manifest.json')
    assert all(n==0 for n in marker['split']['purged_related_departures'].values())
    desired=marker['feature_columns']
    base_path=ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418/training_2025-01-01_2025-02-01.parquet'
    assert sha256(base_path)==read_json(base_path.with_suffix('.json'))['sha256']
    basecols=[name for name in pq.read_schema(base_path).names if name!=ID]
    frame=selected_frame(base_path,basecols,ids)
    inputs=[('private_runs/mechanism_20260916/information/retrospective_v2','training_features.parquet'),
            ('private_runs/breakthrough_20260916/batch_context','training_features.parquet'),
            ('private_runs/breakthrough_20260916/missing/source_conventions','features.parquet'),
            ('private_runs/breakthrough_20260916/geometry_v2','training_features.parquet'),
            ('private_runs/breakthrough_20260916/weather','training_features.parquet'),
            ('private_runs/breakthrough_20260916/missing/sequence_flatten','training_features.parquet')]
    receipts=[]
    for directory,filename in inputs:
        root=ROOT/directory
        manifest=read_json(root/'manifest.json')
        path=root/filename
        expected=manifest['feature_sha256'] if filename=='features.parquet' else manifest['outputs'][filename]
        assert sha256(path)==expected
        available=pq.read_schema(path).names
        columns=[name for name in desired if name in available and name not in frame]
        extra=selected_frame(path,columns,ids)
        for name in columns:
            if pd.api.types.is_numeric_dtype(extra[name]):
                values=extra[name].replace([np.inf,-np.inf],np.nan)
                extra[name]=(values if 'sequence_flatten' in directory else values.fillna(-999999)).astype('float32')
            else:
                extra[name]=extra[name].astype('string').fillna('MISSING').astype('category')
        frame=pd.concat([frame,extra],axis=1)
        receipts.append({'file':str(path),'sha256':expected})
    frame=frame[desired]
    assert frame.shape==(len(ids),337)
    labels=(meta[TARGET]-meta.proxy_sec).to_numpy(float)
    assert np.isfinite(labels).all()
    frame.to_parquet(OUT/'features.parquet')
    meta[[TARGET,'proxy_sec']].to_parquet(OUT/'labels.parquet')
    record={'source_sha256':sha256(__file__),'rows':len(ids),'columns':337,'ids_hash':object_hash(ids.tolist()),
            'scope':'OnlyJanuary2025 originalpurgedFIT; first8192plus32largestabsresidualfitrows,diagnosticnotperformanceevaluation',
            'label_min':float(labels.min()),'label_max':float(labels.max()),'unmodified_labels':True,'source_inputs':receipts,
            'feature_sha256':sha256(OUT/'features.parquet'),'label_sha256':sha256(OUT/'labels.parquet'),
            'peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)}
    write_json(OUT/'fixture.json',record)
    print('FIT_CANARY_PREPARED',record['rows'],record['label_min'],record['label_max'],'peakMiB',record['peak_rss_bytes']/2**20,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
