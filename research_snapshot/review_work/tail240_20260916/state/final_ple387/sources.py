"""Bound full2025 plus ranking feature tuples for frozen225/387/449 recipes."""
from pathlib import Path
import sys
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
BASE=ROOT/'private_runs/breakthrough_20260916'
PREP=ROOT/'private_runs/tail240_20260916/state/final_ple387/ranking_cache_v1'
PREP8=PREP.parent/'ranking_flat8_v1'


def feature_sources(desired):
    if len(desired)!=len(set(desired)):raise ValueError('Repeated feature request')
    frozen=common.read_json(ROOT/'private_runs/submission_v2/protocol.json')
    prep=common.read_json(PREP/'manifest.json');assert prep['status']=='complete'
    candidates=[]
    for path in sorted((ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418').glob('training_*.parquet')):
        marker=common.read_json(path.with_suffix('.json'))
        assert marker['identity']['inputs']==frozen['raw_hashes']
        candidates.append((path,marker['sha256'],False))
    candidates.append((PREP/'ranking_base.parquet',prep['outputs']['ranking_base.parquet'],False))
    for folder in [ROOT/'private_runs/mechanism_20260916/information/retrospective_v2',BASE/'batch_context',BASE/'geometry_v2',BASE/'weather',BASE/'retrospective_research']:
        marker=common.read_json(folder/'manifest.json')
        for name in ['training_features.parquet','ranking_features.parquet']:
            candidates.append((folder/name,marker['outputs'][name],True))
    convention=BASE/'missing/source_conventions'
    marker=common.read_json(convention/'manifest.json')
    candidates.extend([(convention/'features.parquet',marker['feature_sha256'],True),
        (PREP/'ranking_conventions.parquet',prep['outputs']['ranking_conventions.parquet'],True)])
    use_eight=any(name.startswith(tuple(f'flat_{phase}{rank}_' for phase in ['dep','arr'] for rank in range(5,9))) for name in desired)
    folder=BASE/'missing'/('sequence_flatten8' if use_eight else 'sequence_flatten')
    marker=common.read_json(folder/'manifest.json')
    candidates.append((folder/'training_features.parquet',marker['outputs']['training_features.parquet'],False))
    if use_eight:
        marker8=common.read_json(PREP8/'manifest.json');assert marker8['status']=='complete'
        candidates.append((PREP8/'ranking_flat224.parquet',marker8['outputs']['ranking_flat224.parquet'],False))
    else:candidates.append((PREP/'ranking_flat112.parquet',prep['outputs']['ranking_flat112.parquet'],False))
    result=[]
    for path,digest,fill in candidates:
        available=set(pq.read_schema(path).names)
        columns=[c for c in desired if c in available]
        if columns:
            assert common.sha256(path)==digest,path
            result.append((path,digest,fill,columns))
    if set(desired)!={c for _,_,_,columns in result for c in columns}:raise ValueError('Missing feature source')
    return result
