"""Read-only frozen inputs and external campaign evidence."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
REPO = WORKSPACE / 'knowledgeable-helicopter-screening'
sys.path.insert(0, str(REPO / 'src'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import pandas as pd
from taxiout.artifacts import read_json, write_json, sha256, object_hash, source_hashes, utc_now
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, MOVEMENT
from taxiout.splits import make_fold
from taxiout.config import load_config
from taxiout.io import concat_frames

OUT = external_path(WORKSPACE / 'private_runs/campaign_20260916')
RAW = external_path(WORKSPACE / 'data/09-15-2026-18-55-03_files_list')
SEED = 20260916
SAMPLE = 200000


def checked_frame(path, manifest, key=None):
    expected = manifest['outputs'][key or path.name]
    if sha256(path) != expected:
        raise ValueError(f'Hash mismatch {path}')
    return pd.read_parquet(path)


def reference(fold):
    path = WORKSPACE / 'private_runs/next_230/models' / f'clock_and_rome_ensemble_{fold}_s20260910'
    record = read_json(path / 'manifest.json')
    assert record['status'] == 'complete'
    return checked_frame(path / 'score_predictions.parquet', record), record


def load_data():
    frozen = read_json(WORKSPACE / 'private_runs/submission_v2/protocol.json')
    if source_hashes() != frozen['source_hashes']:
        raise ValueError('Frozen sources changed')
    old = WORKSPACE / 'private_runs/screening_230'
    audit = read_json(old / 'reports/data_audit.json')
    meta_path = old / 'data/interim/audit/departures.parquet'
    assert sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path)
    cache = old / 'data/interim/features/a2a101f52a0aa418'
    frames = []
    for p in sorted(cache.glob('training_*.parquet')):
        marker = read_json(p.with_suffix('.json'))
        assert marker['identity']['inputs'] == frozen['raw_hashes']
        assert marker['identity']['features'] == frozen['base_config']['features']
        assert sha256(p) == marker['sha256']
        frames.append(pd.read_parquet(p))
    assert len(frames) == 12
    x = concat_frames(frames).set_index(ID)
    assert np.array_equal(x.index, meta[ID]) and len(x) == 2085047
    assert TARGET not in x and 'BLOCK_TIME_UTC_mvt' not in x
    return x, meta


def fold_data(meta, fold, full=False):
    idx, evidence = make_fold(meta.drop(columns=TARGET), load_config('configs/folds.yaml')[fold])
    selected = dict(idx)
    for stage in ['fit', 'refit']:
        rows = idx[stage]
        if not full and len(rows) > SAMPLE:
            selected[stage] = np.sort(np.random.default_rng(SEED).choice(rows, SAMPLE, replace=False))
    return selected, evidence, {k: {'n': len(v), 'id_hash': object_hash(meta.iloc[v][ID].tolist())} for k,v in selected.items()}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        return read_json(p)
    frozen = read_json(WORKSPACE / 'private_runs/submission_v2/protocol.json')
    record = dict(created_utc=utc_now(), plan_sha256=sha256(HERE/'PLAN.md'),
        reference='clock_and_rome_ensemble', reference_seasonal_rmse_sec=292.8133464057956,
        official_reference_rmse_sec=294.626, sample=SAMPLE, seed=SEED,
        folds=['F1','F3'], broader_folds=['F2','G1'], source_hashes=frozen['source_hashes'],
        raw_hashes=frozen['raw_hashes'], blend_weight=.25,
        selection='development only; all folds previously exposed',
        registry=[{'family':f,'target':t,'features':'base','training':'sample200k'}
                  for f,ts in [('catboost',['direct','correction']),('lightgbm',['direct','correction']),
                               ('xgboost',['direct','correction']),('ridge',['correction']),('tabm',['correction'])]
                  for t in ts])
    write_json(p, record)
    return record


if __name__ == '__main__':
    freeze()
    x,meta=load_data()
    records={}
    for fold in ['F1','F2','F3','G1']:
        idx,split,sampled=fold_data(meta,fold)
        ref,rec=reference(fold)
        assert object_hash(split) == object_hash(rec['split'])
        assert np.array_equal(ref[ID],meta.iloc[idx['score']][ID])
        assert np.array_equal(ref[TARGET],meta.iloc[idx['score']][TARGET])
        records[fold]={'split':split,'sampled':sampled,'reference_sha256':object_hash(rec)}
    write_json(OUT/'contract.json',dict(rows=len(x),features=list(x),folds=records,
        common_sha256=sha256(__file__),created_utc=utc_now()))
    print('Contract verified',len(x),len(x.columns),flush=True)
