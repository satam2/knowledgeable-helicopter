"""Verify state fit/tune partitions cover the new complete finite population."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import validate_candidate as v

ROOT=v.ROOT
CACHE=ROOT/'private_runs/tail240_20260916/state/cache_v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/ordinary_state_preflight_v1'
MODEL=ROOT/'private_runs/tail240_20260916/models/ordinary_state_tune_v1'
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(MODEL/'protocol.json');cache=v.read_json(CACHE/'manifest.json')
    assert protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/ordinary_state_tune.py')
    assert protocol['cache_manifest_sha256']==v.sha256(CACHE/'manifest.json')
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[v.ID,'FLIGHT_ID_mvt',v.common.MOVEMENT,v.TARGET,'proxy_sec'])
    checks={}
    for fold in ['F1','F3']:
        indices,split,_=v.common.fold_data(meta,fold,full=True)
        assert v.object_hash(split)==v.object_hash(cache['folds'][fold]['split'])
        control=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        marker=v.read_json(control/'manifest.json')
        assert marker['feature_columns']==protocol['columns'][:387] and marker['fit']['params']==protocol['params']
        stageids={};checks[fold]={}
        for stage in ['fit','tune']:
            record=cache['folds'][fold]['stages'][stage];source=ROOT/record['path']
            assert v.sha256(source)==record['sha256']
            columns=pq.read_schema(source).names
            assert columns==[v.ID]+protocol['columns'][387:]
            assert not set(columns[1:])&set(marker['feature_columns'])
            ids=pd.read_parquet(source,columns=[v.ID])[v.ID]
            assert ids.is_unique
            np.testing.assert_array_equal(ids,meta.iloc[indices[stage]][v.ID])
            selected=meta.iloc[indices[stage]].loc[lambda x:np.isfinite(x.proxy_sec)]
            assert pd.Index(ids).get_indexer(selected[v.ID]).min()>=0
            stageids[stage]=pd.Index(ids)
            checks[fold][stage]={'all_rows':len(ids),'finite_rows':len(selected),'full_id_hash':v.object_hash(ids.tolist()),
                'finite_id_hash':v.object_hash(selected[v.ID].tolist()),'sha256':record['sha256']}
        assert not len(stageids['fit'].intersection(stageids['tune']))
    v.write_json(OUT/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),
        'producer_protocol_sha256':v.sha256(MODEL/'protocol.json'),'cache_manifest_sha256':v.sha256(CACHE/'manifest.json'),
        'checks':checks,'scope':'Exact original all-stage ID/order, disjoint stage IDs, full finite coverage, twelve float state columns distinct from union387, both-fold same control parameters/schema verified. History numerics rely on prior independent cache audit; no model fit or score access.',
        'peak_rss_bytes':v.guard()})
    print('PASSED',checks,flush=True)


if __name__=='__main__':main()
