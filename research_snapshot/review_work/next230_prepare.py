"""Single writer for label-free features; original data stays in place."""

import pandas as pd

from next230_common import OUT, config_for, load_data
from next230_features import runway_features, schedule_features
from taxiout.artifacts import read_json, sha256, write_json
from taxiout.availability import make_observations
from taxiout.io import concat_frames, read_raw, training_paths
from taxiout.schema import ID, MOVEMENT, PHASE
from pathlib import Path


def prepare():
    x, _, _ = load_data()
    path = OUT / 'features' / 'extensions.parquet'
    if path.exists():
        marker = read_json(path.parent / 'manifest.json')
        if marker['sha256'] != sha256(path):
            raise ValueError('Corrupt existing cache')
        print('FEATURE CACHE ALREADY COMPLETE', flush=True)
        return
    paths = training_paths(config_for('baseline'))
    context_cols = [ID, MOVEMENT, PHASE, 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt']
    context = make_observations(concat_frames([read_raw(p, context_cols) for p in paths]))[0]
    months = context[MOVEMENT].dt.year * 12 + context[MOVEMENT].dt.month
    frames = []
    for raw_path in paths:
        obs = make_observations(read_raw(raw_path, context_cols + ['STAND_mvt', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt']))[0]
        dep = obs.loc[obs[PHASE].eq('DEP')].set_index(ID, drop=False)
        query_months = (dep[MOVEMENT].dt.year * 12 + dep[MOVEMENT].dt.month).unique()
        ext = pd.concat([runway_features(dep, context.loc[months.isin(query_months)]), schedule_features(dep)], axis=1)
        frames.append(ext.reset_index())
        print(f'EXTENSION {raw_path.name}: {len(dep)} rows, {len(ext.columns)} columns', flush=True)
    result = concat_frames(frames)
    if not result[ID].is_unique or result[ID].tolist() != x.index.tolist():
        raise ValueError('Feature cohort mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    write_json(path.parent / 'manifest.json', {'rows': len(result), 'columns': list(result),
        'sha256': sha256(path), 'feature_source_sha256': sha256(Path(__file__).with_name('next230_features.py')),
        'raw_inputs': read_json(OUT / 'protocol.json')['raw_hashes'], 'policy': 'strict-prior, month-isolated, observation-only'})
    print('FEATURE CACHE COMPLETE', flush=True)


if __name__ == '__main__':
    prepare()
