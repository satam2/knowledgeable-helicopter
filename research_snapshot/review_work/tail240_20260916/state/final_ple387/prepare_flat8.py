"""Exact last-eight ranking extension from the bound last-sixteen event cache."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import prepare_ranking as shared

ROOT,common,ID=shared.ROOT,shared.common,shared.ID
flat8=shared.module('final_flatten8',ROOT/'review_work/breakthrough_20260916/missing/sequence_flatten8/build.py')
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/state/final_ple387/ranking_flat8_v1')


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();pa.set_cpu_count(1);pa.set_io_thread_count(1)
    original=common.read_json(ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten8/manifest.json')
    assert common.sha256(flat8.__file__)==original['source_sha256']
    marker=common.read_json(shared.OUT/'manifest.json')
    for name in ['ranking_events.parquet','ranking_queries.parquet','ranking_neighbors.npy']:
        assert common.sha256(shared.OUT/name)==marker['outputs'][name]
    protocol=dict(source_sha256=common.sha256(__file__),frozen_producer_sha256=original['source_sha256'],
        source_cache_manifest_sha256=common.sha256(shared.OUT/'manifest.json'),
        policy='Exact frozen Flatten8 on same savedrankingevents/query/neighbors, no newselection ormodel',resources='1CPU,<4GiB,8GiBreserve,noGPU')
    common.write_json(OUT/'protocol.json',protocol)
    events=pd.read_parquet(shared.OUT/'ranking_events.parquet')
    queries=pd.read_parquet(shared.OUT/'ranking_queries.parquet')
    neighbors=np.load(shared.OUT/'ranking_neighbors.npy',mmap_mode='r')
    engine=flat8.Flatten8(events,queries);writer=None
    try:
        for start in range(0,len(queries),4096):
            values=engine.batch(neighbors[start:start+4096],start)
            frame=pd.DataFrame(values,columns=flat8.COLUMNS)
            frame.insert(0,ID,queries[ID].iloc[start:start+len(frame)].to_numpy())
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(OUT/'ranking_flat224.parquet',table.schema,compression='zstd')
            writer.write_table(table);shared.guard()
    finally:
        if writer is not None:writer.close()
    result=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),rows=len(queries),columns=flat8.COLUMNS,
        id_hash=common.object_hash(queries[ID].tolist()),peak_bytes=shared.guard(),elapsed_seconds=time.monotonic()-started,
        outputs={'ranking_flat224.parquet':common.sha256(OUT/'ranking_flat224.parquet')})
    common.write_json(OUT/'manifest.json',result)
    print('COMPLETE',len(queries),result['elapsed_seconds'],result['peak_bytes'],flush=True)


if __name__=='__main__':run()
