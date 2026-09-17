"""Fit-only cached feature missingness counts without target or model reads."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work'))
import next230_common as common
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
meta = pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',
    columns=[ID, TIME, 'proxy_sec'], use_threads=False).to_pandas()
fit = meta.loc[meta[TIME].ge(pd.Timestamp('2025-01-01',tz='UTC')) & meta[TIME].lt(pd.Timestamp('2025-06-01',tz='UTC'))]
ids = pd.Index(fit[ID])
binding = common.read_json(ROOT/'private_runs/next_230/models/clock_and_rome_ensemble_F1_s20260910/manifest.json')['split']
assert len(fit) == 822377 and common.object_hash(fit[ID].tolist()) == binding['stages']['fit']['id_hash']
assert all(v == 0 for v in binding['purged_related_departures'].values())
clocknames = ['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
cols = ['STAND_mvt','RUNWAY_mvt', *['takeoff_minus_'+c for c in clocknames]]
counts = {name: 0 for name in ['stand_missing','runway_missing','stand_empty','runway_empty','aobt_missing_all_other_nm_missing','aobt_missing_some_other_nm_present','any_clock_missing','finite_route_stand_missing','finite_route_runway_missing']}
patterns = {}
seen = np.zeros(len(fit),bool)
sources = {}
for path in sorted((ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418').glob('training_*.parquet')):
    marker = common.read_json(path.with_suffix('.json'))
    sources[str(path.relative_to(ROOT))] = dict(manifest_sha256=common.sha256(path.with_suffix('.json')), recorded_cache_sha256=marker['sha256'])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*cols],use_threads=False):
        x=batch.to_pandas()
        pos=ids.get_indexer(x[ID]); keep=pos>=0
        x=x.loc[keep]; pos=pos[keep]
        assert not seen[pos].any()
        seen[pos]=True
        finite=np.isfinite(fit.proxy_sec.iloc[pos].to_numpy())
        for field in ['stand','runway']:
            missing=x[field.upper()+'_mvt'].eq('m:').to_numpy()
            counts[field+'_missing'] += int(missing.sum())
            counts[field+'_empty'] += int(x[field.upper()+'_mvt'].eq('s0:').sum())
            counts['finite_route_'+field+'_missing'] += int((missing&finite).sum())
        values=x[cols[2:]].to_numpy(float)
        miss=(~np.isfinite(values))|(values==-999999.)
        counts['aobt_missing_all_other_nm_missing'] += int((miss[:,:4].all(axis=1)).sum())
        counts['aobt_missing_some_other_nm_present'] += int((miss[:,0]&(~miss[:,1:4].all(axis=1))).sum())
        counts['any_clock_missing'] += int(miss.any(axis=1).sum())
        for row in miss:
            key=''.join('0' if v else '1' for v in row)
            patterns[key]=patterns.get(key,0)+1
        assert psutil.Process().memory_info().peak_wset < 1024**3
assert seen.all()
manifest=common.read_json(ROOT/'private_runs/tail240_20260916/state/neural_context/v1/F1/manifest.json')
features=manifest['feature_columns']
groups=dict(stand=[c for c in features if c.startswith('batch_source_stand_')],
    query_runway=[c for c in features if c.startswith('batch_surface_') and '_runway_' in c and 'active_runways' not in c],
    arrival_runway=[c for c in features if c.startswith('retro_') and '_arr_runway_' in c],
    airport_active_runways=[c for c in features if 'observed_active_runways' in c])
receipt=dict(status='complete', scope='Original F1 fit only; cached whitelist, no target/raw/model/tune/score read',
    rows=len(fit), finite_proxy_rows=int(np.isfinite(fit.proxy_sec).sum()), fit_id_hash=binding['stages']['fit']['id_hash'],
    counts=counts, clock_presence_patterns=patterns, clock_pattern_order=clocknames,
    affected_feature_groups=groups, sources=sources, source_sha256=common.sha256(__file__),
    peak_rss_bytes=psutil.Process().memory_info().peak_wset,
    note='Cache identity bound by small manifests; full cache byte hashes not recomputed in this bounded count audit. Missing category token m: is source-null, s0: is empty string.')
folder=common.external_path(ROOT/'private_runs/tail240_20260916/state/preprocessing_semantics/v1')
folder.mkdir(parents=True,exist_ok=False)
common.write_json(folder/'receipt.json',receipt)
print(counts, patterns, {k:len(v) for k,v in groups.items()}, receipt['peak_rss_bytes'], flush=True)
