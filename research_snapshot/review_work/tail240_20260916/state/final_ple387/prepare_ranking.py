"""Exact frozen ranking feature extensions for final PLE387 and sibling experts."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import gc
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/sequence_context'))
import cache as sequence


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


conv=module('final_conventions',ROOT/'review_work/breakthrough_20260916/missing/source_conventions/prepare_conventions.py')
flat=module('final_flatten',ROOT/'review_work/breakthrough_20260916/missing/sequence_flatten/build.py')
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/state/final_ple387/ranking_cache_v1')
RAW=ROOT/'data/09-15-2026-18-55-03_files_list/ranking.parquet'
ID=common.ID
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def guard():
    p=psutil.Process().memory_info()
    assert max(p.rss,p.peak_wset)<4*1024**3
    assert psutil.virtual_memory().available>=8*1024**3
    return int(p.peak_wset)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    frozen=common.read_json(ROOT/'private_runs/submission_v2/protocol.json')
    assert common.sha256(RAW)==frozen['raw_hashes']['ranking.parquet']
    binding=common.read_json(ROOT/'private_runs/submission_v2/ranking_inputs.json')
    for name,digest in binding['files'].items():assert common.sha256(ROOT/'private_runs/submission_v2'/name)==digest
    for producer,marker in [(conv,ROOT/'private_runs/breakthrough_20260916/missing/source_conventions/manifest.json'),
                            (sequence,ROOT/'private_runs/breakthrough_20260916/sequence_context/manifest.json')]:
        assert common.sha256(producer.__file__)==common.read_json(marker)['source_sha256']
    assert common.sha256(flat.__file__)==common.read_json(ROOT/'private_runs/breakthrough_20260916/missing/sequence_flatten/protocol.json')['source_sha256']
    protocol=dict(source_sha256=common.sha256(__file__),raw_sha256=common.sha256(RAW),
        sources={str(Path(p.__file__).relative_to(ROOT)):common.sha256(p.__file__) for p in [conv,sequence,flat]},
        policy='Same original producer functions, no changedfeature ortraining. Local rankingobservations only; ARR completion loaded withPHASE=ARR Arrowpredicate; no DEPblock/target.',
        outputs='Base30resetIDcolumn,sourceconventions85,flat112;preserve all344841 rankingDEP IDs andorder;sequence events/neighbors saved for independentreconstruction',
        resources='1CPU,<4GiBprocesspeak,8GiBhostreserve,noGPU')
    common.write_json(OUT/'protocol.json',protocol)
    meta=pd.read_parquet(ROOT/'private_runs/submission_v2/ranking_meta.parquet')
    base=pd.read_parquet(ROOT/'private_runs/submission_v2/ranking_base.parquet').reset_index()
    np.testing.assert_array_equal(base[ID],meta[ID]);assert len(meta)==344841
    base.to_parquet(OUT/'ranking_base.parquet',index=False)
    columns=[ID,conv.PHASE,conv.MOVEMENT,*conv.CLOCKS,'ADEP_mvt','ADEP_flt','ADES_mvt','ADES_flt','AIRCRAFT_TYPE_mvt','AIRCRAFT_TYPE_flt']
    raw=pq.read_table(RAW,columns=columns,use_threads=False).to_pandas(strings_to_categorical=True)
    obs,_,_=conv.make_observations(raw)
    dep=obs.loc[obs[conv.PHASE].eq('DEP')]
    features=conv.convention_features(dep).reset_index()
    np.testing.assert_array_equal(features[ID],meta[ID])
    for name in features.select_dtypes('category'):features[name]=features[name].astype('string')
    assert len(features.columns)==86
    features.to_parquet(OUT/'ranking_conventions.parquet',index=False)
    del raw,obs,dep,features,base
    gc.collect();guard()
    public=pq.read_table(RAW,columns=sequence.PUBLIC,use_threads=False).to_pandas()
    arrivals=pq.read_table(RAW,columns=[ID,sequence.BLOCK],filters=[(sequence.PHASE,'=','ARR')],use_threads=False).to_pandas()
    assert arrivals[ID].isin(public.loc[public[sequence.PHASE].eq('ARR'),ID]).all()
    queries=sequence.query_frame(public.loc[public[sequence.PHASE].eq('DEP')])
    np.testing.assert_array_equal(queries[ID],meta[ID])
    events=sequence.events_from_public(public,arrivals)
    neighbors=sequence.neighbor_indices(events,queries)
    assert neighbors.shape==(344841,32)
    events.to_parquet(OUT/'ranking_events.parquet',index=False)
    queries.to_parquet(OUT/'ranking_queries.parquet',index=False)
    np.save(OUT/'ranking_neighbors.npy',neighbors)
    del public,arrivals
    gc.collect();guard()
    transform=flat.Flatten(events,queries)
    writer=None
    try:
        for start in range(0,len(queries),4096):
            values=transform.batch(neighbors[start:start+4096],start)
            out=pd.DataFrame(values,columns=flat.COLUMNS)
            out.insert(0,ID,queries[ID].iloc[start:start+len(out)].to_numpy())
            table=pa.Table.from_pandas(out,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(OUT/'ranking_flat112.parquet',table.schema,compression='zstd')
            writer.write_table(table);guard()
    finally:
        if writer is not None:writer.close()
    report=dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),rows=len(meta),
        id_hash=common.object_hash(meta[ID].tolist()),finite_proxy_rows=int(np.isfinite(meta.proxy_sec).sum()),
        event_rows=len(events),flat_columns=flat.COLUMNS,peak_bytes=guard(),elapsed_seconds=time.monotonic()-started,
        outputs={p.name:common.sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    common.write_json(OUT/'manifest.json',report)
    print('COMPLETE',report['rows'],report['event_rows'],report['peak_bytes'],report['elapsed_seconds'],flush=True)


if __name__=='__main__':run()
