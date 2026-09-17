"""New retrospective ID-context ablation; original missing-model runs stay frozen."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import run_missing_models as base
from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.paths import external_path
from taxiout.schema import ID, MOVEMENT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/id_context_v1')
CACHE = external_path(ROOT / 'private_runs/mechanism_20260916/id_neighbors')
ARMS = ['direct_airport', 'historical_template']


def id_context_features(x, peer, timestamps):
    if not x.index.equals(peer.index) or not x.index.equals(timestamps.index):
        raise ValueError('ID-context inputs must have exact matching ID order')
    result = peer.astype('float32').copy()
    query = pd.to_datetime(timestamps, utc=True)
    schedule_seconds = x.schedule_proxy_sec.to_numpy(float)
    valid_schedule = np.isfinite(schedule_seconds) & (schedule_seconds != -999999)
    schedule = query - pd.to_timedelta(np.where(valid_schedule, schedule_seconds, np.nan), unit='s')
    ages = []
    for width in [2, 8, 32]:
        delta = peer[f'id_peer_median_time_minus_query_w{width}'].to_numpy(float)
        spread = peer[f'id_peer_time_spread_w{width}'].to_numpy(float)
        inferred = query + pd.to_timedelta(delta, unit='s')
        age = -delta
        result[f'idctx_peer_age_w{width}_sec'] = age
        result[f'idctx_peer_minus_schedule_w{width}_sec'] = np.where(valid_schedule, delta + schedule_seconds, np.nan)
        result[f'idctx_peer_age_signedlog_w{width}'] = np.sign(age) * np.log1p(np.abs(age))
        result[f'idctx_peer_age_modday_w{width}'] = np.mod(age, 86400)
        result[f'idctx_peer_age_spread_ratio_w{width}'] = age / (np.abs(spread) + 60)
        result[f'idctx_peer_query_day_delta_w{width}'] = (inferred.dt.normalize() - query.dt.normalize()).dt.total_seconds().to_numpy() / 86400
        result[f'idctx_peer_schedule_day_delta_w{width}'] = (inferred.dt.normalize() - schedule.dt.normalize()).dt.total_seconds().to_numpy() / 86400
        result[f'idctx_peer_schedule_same_day_w{width}'] = np.where(valid_schedule, inferred.dt.normalize().eq(schedule.dt.normalize()), np.nan)
        result[f'idctx_peer_hour_w{width}'] = inferred.dt.hour.to_numpy()
        result[f'idctx_peer_weekday_w{width}'] = inferred.dt.dayofweek.to_numpy()
        ages.append(age)
    result['idctx_w2_minus_w32_age_sec'] = ages[0] - ages[2]
    result['idctx_w8_minus_w32_age_sec'] = ages[1] - ages[2]
    return result.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')


def declare(seed, threads):
    original = read_json(ROOT / 'private_runs/breakthrough_20260916/missing/protocol.json')
    for filename, digest in original['declaration']['source_hashes'].items():
        assert sha256(ROOT / filename) == digest
    cache = read_json(CACHE / 'audit.json')
    assert sha256(CACHE / 'features.parquet') == cache['feature_sha256']
    assert sha256(ROOT / 'review_work/mechanism_20260916/id_neighborhood.py') == cache['script_sha256']
    declaration = {'arms': ARMS, 'seed': seed, 'threads': threads,
        'base_protocol_sha256': sha256(ROOT / 'private_runs/breakthrough_20260916/missing/protocol.json'),
        'base_runner_sha256': sha256(HERE / 'run_missing_models.py'), 'wrapper_sha256': sha256(__file__),
        'id_context_audit_sha256': sha256(CACHE / 'audit.json'), 'id_context_feature_sha256': cache['feature_sha256'],
        'availability': 'RETROSPECTIVE supplied DEP record order; same airport+UTC movement month; own movement excluded; may contain future departures. This is not a causal event-time feature block.',
        'features': 'Frozen peer medians/spreads w2/8/32 plus peer-minus-schedule, peer age signedlog/modday, calendar-day alignment, inferred peerhour/weekday, crosswindow agreement.',
        'source_semantics': 'Peer median timestamps are record-order signatures, not authenticated source-system timestamp or aircraft rotation.',
        'learning': 'Same original model configs and full missing-source folds. Every predictive threshold learned only in fit/tune; no hand-selected score-tail thresholds.',
        'variants': ['candidate', 'blend25'], 'blend_weight': .25,
        'selection': 'Adaptive information ablation after original missing-model results; exposed score folds, not fresh validation.'}
    path = OUT / 'protocol.json'
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        record = read_json(path)
        if record['declaration'] != declaration:
            raise ValueError('ID-context protocol changed')
        return record
    result = {'created_utc': utc_now(), 'declaration': declaration}
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare(args.seed, args.threads)
    if args.declare_only:
        print('Declared retrospective ID-context ablation; no training launched', flush=True)
        return
    x, meta = base.load_data()
    peer = pd.read_parquet(CACHE / 'features.parquet').set_index(ID)
    assert np.array_equal(peer.index, meta[ID])
    peer = peer.loc[x.index]
    times = meta.set_index(ID).loc[x.index, MOVEMENT]
    extension = id_context_features(x, peer, times)
    x = pd.concat([x, extension], axis=1)
    write_json(OUT / 'feature_manifest.json', {'columns': list(x), 'extension_columns': list(extension),
        'rows': len(x), 'all_extension_finite': bool(np.isfinite(extension.to_numpy()).all()),
        'source_sha256': sha256(__file__), 'feature_cache_sha256': sha256(CACHE / 'features.parquet')})
    base.OUT = OUT
    for arm in args.arms:
        for fold in args.folds:
            base.run_arm(arm, fold, x, meta, args.seed, args.threads)


if __name__ == '__main__':
    main()
