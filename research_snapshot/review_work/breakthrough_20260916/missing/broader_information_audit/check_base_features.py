"""Hash/order supplement without loading the full feature matrix."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

out=ROOT/'private_runs/breakthrough_20260916/missing/broader_information_audit/base_features.json'
assert not out.exists()
frozen=common.read_json(ROOT/'private_runs/submission_v2/protocol.json')
assert common.source_hashes()==frozen['source_hashes']
root=ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
ids=[]
checks=[]
for path in sorted(root.glob('training_*.parquet')):
    marker=common.read_json(path.with_suffix('.json'))
    assert marker['identity']['inputs']==frozen['raw_hashes']
    assert marker['identity']['features']==frozen['base_config']['features']
    assert common.sha256(path)==marker['sha256']
    assert common.TARGET not in pq.read_schema(path).names and 'BLOCK_TIME_UTC_mvt' not in pq.read_schema(path).names
    ids.append(pq.read_table(path,columns=[common.ID],use_threads=False).column(0).to_numpy())
    checks.append(dict(file=path.name,sha256=marker['sha256'],rows=len(ids[-1])))
assert len(checks)==12
meta=pq.read_table(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[common.ID],use_threads=False).column(0).to_numpy()
np.testing.assert_array_equal(np.concatenate(ids),meta)
f2,_=common.reference('F2')
g1,_=common.reference('G1')
np.testing.assert_array_equal(f2[common.ID],g1[common.ID])
np.testing.assert_array_equal(f2[common.TARGET],g1[common.TARGET])
common.write_json(out,dict(status='passed',source_sha256=common.sha256(__file__),rows=len(meta),checks=checks,
    frozen_base_feature_config_verified=True,current_frozen_repository_source_hashes_match=True,F2_G1_same_score_ids_labels=True))
print('BASE_FEATURES_PASSED',len(meta),len(checks),flush=True)
