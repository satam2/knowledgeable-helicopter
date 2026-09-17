"""Count semantic location masks in exact F1 fit without outcomes."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from semantic_locations import known_location, feature_groups
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work'))
import next230_common as common
ID,TIME='MVT_ID_mvt','MVT_TIME_UTC_mvt'
meta=pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[ID,TIME,'proxy_sec'],use_threads=False).to_pandas()
fit=meta.loc[meta[TIME].ge(pd.Timestamp('2025-01-01',tz='UTC'))&meta[TIME].lt(pd.Timestamp('2025-06-01',tz='UTC'))]
ids=pd.Index(fit[ID]); assert len(ids)==822377
counts={}; masked={key:[] for key in ['stand','runway']}; seen=np.zeros(len(ids),bool)
for path in sorted((ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418').glob('training_*.parquet')):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,'ADEP_mvt','STAND_mvt','RUNWAY_mvt'],use_threads=False):
        x=batch.to_pandas(); pos=ids.get_indexer(x[ID]); keep=pos>=0; x=x.loc[keep]; pos=pos[keep]
        assert not seen[pos].any(); seen[pos]=True
        finite=np.isfinite(fit.proxy_sec.iloc[pos].to_numpy())
        for key in masked:
            token=x[key.upper()+'_mvt'].astype('string'); missing=~known_location(token,encoded=True)
            masked[key].extend(x.loc[missing,ID].tolist())
            for category in ['m:','s0:','s7:UNKNOWN','s2:NA']:
                flag=token.eq(category).to_numpy()
                for route,sub in [('all',flag),('finite',flag&finite)]:
                    name=f'{key}_{category}_{route}'
                    counts[name]=counts.get(name,0)+int(sub.sum())
assert seen.all()
manifest=common.read_json(ROOT/'private_runs/tail240_20260916/state/neural_context/v1/F1/manifest.json')
groups=feature_groups(manifest['feature_columns'])
affected={}
for cache,keys in [('private_runs/breakthrough_20260916/batch_context/training_features.parquet',['stand','query_runway']),('private_runs/breakthrough_20260916/retrospective_research/training_features.parquet',['arrival_runway']),('private_runs/breakthrough_20260916/missing/sequence_flatten/training_features.parquet',['stand_equality','runway_equality'])]:
    cols=[c for key in keys for c in groups[key]]
    for c in cols: affected[c]=dict(queries=0,finite=0,nonzero=0)
    for batch in pq.ParquetFile(ROOT/cache).iter_batches(batch_size=8192,columns=[ID,*cols],use_threads=False):
        x=batch.to_pandas()
        for key in keys:
            selector='stand' if key in ['stand','stand_equality'] else 'runway'
            sub=x.loc[x[ID].isin(masked[selector])]
            for c in groups[key]:
                values=sub[c].to_numpy(float); f=np.isfinite(values)
                affected[c]['queries']+=len(values); affected[c]['finite']+=int(f.sum()); affected[c]['nonzero']+=int((f&(values!=0)).sum())
        assert psutil.Process().memory_info().peak_wset<1024**3
folder=common.external_path(ROOT/'private_runs/tail240_20260916/state/preprocessing_semantics/v2')
folder.mkdir(parents=True,exist_ok=False)
result=dict(status='complete',rows=len(ids),fit_id_hash=common.object_hash(fit[ID].tolist()),counts=counts,
    mask_rows={k:len(v) for k,v in masked.items()},affected=affected,feature_groups=groups,
    source_sha256=common.sha256(__file__),helper_sha256=common.sha256(Path(__file__).with_name('semantic_locations.py')),
    peak_rss_bytes=psutil.Process().memory_info().peak_wset,
    interpretation='UNKNOWN stand and NA runway are observed field-specific semantic candidates, not organizer-codebook-confirmed null values; category values stay unchanged in proposed physicalmask.')
common.write_json(folder/'receipt.json',result)
print(result['counts'],result['mask_rows'],{k:len(v) for k,v in groups.items()},result['peak_rss_bytes'])
