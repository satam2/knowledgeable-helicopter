"""Test hidden off-block equality to prior stand arrival events, with controls."""

import os
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.io import concat_frames
from taxiout.paths import external_path
from taxiout.schema import BLOCK, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, utc

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/mechanism_20260916/clock_forensics')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
COLS = [ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, TARGET, 'AOBT_3_flt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt']


def ns(values):
    return utc(values).dt.as_unit('ns').astype('int64').to_numpy()


def event_candidates(times, flights, query_times, hidden_block, query_flights):
    """Nearest-to-hidden diagnostic and latest observable candidate, strict prior.

    Same-flight counterparts are excluded, including repeats. This does not
    identify aircraft rotations. Hidden block is used only in nearest diagnostics.
    """
    distance = np.full(len(query_times), np.inf)
    latest_age = np.full(len(query_times), np.nan)
    if not len(times):
        return distance, latest_age
    high = np.searchsorted(times, query_times, side='left')
    insertion = np.searchsorted(times, hidden_block, side='left')
    left = np.minimum(insertion - 1, high - 1)
    right = np.minimum(insertion, high)
    latest = high - 1
    for indices, direction in [(left, -1), (right, 1), (latest, -1)]:
        while True:
            valid = (indices >= 0) & (indices < high) & (indices < len(times))
            same = np.zeros(len(indices), bool)
            same[valid] = np.isfinite(query_flights[valid]) & (flights[indices[valid]] == query_flights[valid])
            if not same.any():
                break
            indices[same] += direction
    for indices in [left, right]:
        valid = (indices >= 0) & (indices < high) & (indices < len(times))
        distance[valid] = np.minimum(distance[valid], np.abs(times[indices[valid]] - hidden_block[valid]) / 1e9)
    valid = (latest >= 0) & (latest < high)
    latest_age[valid] = (query_times[valid] - times[latest[valid]]) / 1e9
    return distance, latest_age


def test_event_candidates():
    times = np.array([10, 20, 30, 40], np.int64) * 10**9
    flights = np.array([1., 2., 3., 4.])
    distance, latest = event_candidates(times, flights, np.array([30, 40, 45], dtype=np.int64) * 10**9,
        np.array([30, 20, 40], dtype=np.int64) * 10**9, np.array([99., 2., 4.]))
    np.testing.assert_equal(distance, [10., 10., 10.])
    np.testing.assert_equal(latest, [10., 10., 15.])
    changed = times.copy()
    changed[-1] = 999 * 10**9
    old = event_candidates(times, flights, np.array([30], dtype=np.int64) * 10**9, np.array([10], dtype=np.int64) * 10**9, np.array([99.]))
    new = event_candidates(changed, flights, np.array([30], dtype=np.int64) * 10**9, np.array([10], dtype=np.int64) * 10**9, np.array([99.]))
    np.testing.assert_equal(old, new)


def summarize(dep, result, score_map):
    label = dep[TARGET].to_numpy(float)
    proxy = (utc(dep[MOVEMENT]) - utc(dep.AOBT_3_flt)).dt.total_seconds().to_numpy()
    missing = ~np.isfinite(proxy)
    giant = label > 3600
    gap = np.abs(label - proxy) > 1800
    masks = {'all': np.ones(len(dep), bool), 'ordinary': ~missing & ~giant & ~gap,
        'missing_nm': missing, 'large_gap': gap, 'giant_target': giant,
        'missing_giant': missing & giant, 'rome_missing': missing & dep.ADEP_mvt.eq('LIRF').to_numpy(),
        'rome_missing_giant': missing & giant & dep.ADEP_mvt.eq('LIRF').to_numpy()}
    output = {}
    for name, mask in masks.items():
        ids = dep.loc[mask, ID]
        error = score_map.reindex(ids).to_numpy()
        item = {'n': int(mask.sum()), 'methods': {}}
        for method, (distance, latest) in result.items():
            item['methods'][method] = {'prior_event_available_n': int(np.isfinite(latest[mask]).sum()),
                'within_hidden_block': {str(tol): {'n': int((distance[mask] <= tol).sum()),
                    'fraction': float((distance[mask] <= tol).mean()) if mask.any() else None,
                    'reference_sse_selected': float(np.nansum(error[distance[mask] <= tol]))} for tol in [0, 1, 5, 30, 60, 300]},
                'latest_prior_candidate_within60_n': int((np.abs(latest[mask] - label[mask]) <= 60).sum()),
                'latest_prior_age_median_sec': float(np.nanmedian(latest[mask])) if np.isfinite(latest[mask]).any() else None}
        output[name] = item
    return output


def main():
    test_event_candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    expected = read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    paths = sorted(RAW.glob('training_*.parquet'))
    arrivals = []
    for path in paths:
        assert sha256(path) == expected[path.name]
        raw = pq.read_table(path, columns=[ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, 'ADES_mvt', 'STAND_mvt'], use_threads=False).to_pandas(strings_to_categorical=True)
        arrivals.append(raw.loc[raw[PHASE].eq('ARR')].drop(columns=PHASE))
    arrivals = concat_frames(arrivals)
    arrivals['month'] = utc(arrivals[MOVEMENT]).dt.strftime('%Y-%m')
    arrivals['stand'] = arrivals.STAND_mvt.astype('string').fillna('<missing>')
    arrivals['airport'] = arrivals.ADES_mvt.astype('string')
    event_tables = {}
    for event_name, column in [('landing', MOVEMENT), ('completion', BLOCK)]:
        valid = arrivals.loc[arrivals[column].notna() & arrivals.STAND_mvt.notna()]
        if event_name == 'completion':
            valid = valid.loc[utc(valid[BLOCK]).ge(utc(valid[MOVEMENT]))]
        event_tables[event_name] = {}
        for key, sub in valid.groupby(['airport', 'stand', 'month'], observed=True, sort=False):
            order = np.argsort(ns(sub[column]), kind='stable')
            event_tables[event_name][key] = (ns(sub[column])[order], sub[FLIGHT_ID].to_numpy()[order])
    random_stand = {}
    rng = np.random.default_rng(20260916)
    for airport, sub in arrivals.groupby('airport', observed=True):
        stands = np.array(sorted(set(sub.stand) - {'<missing>'}))
        shuffled = rng.permutation(stands)
        for original, other in zip(shuffled, np.roll(shuffled, 1)):
            if original != other:
                random_stand[(str(airport), str(original))] = str(other)
    ref_frames = [common.reference(fold)[0][[ID, 'squared_error']] for fold in ['F1', 'F3']]
    score_map = pd.concat(ref_frames).set_index(ID).squared_error
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'months': {},
        'policy': 'Same UTC landing month; event strictly before query takeoff; same flight excluded. Hidden block nearest event is diagnostic only.',
        'control': 'Fixed seed20260916 cyclic shuffled stand mapping per airport; ordinary departures and shuffled stands are negative controls.',
        'rotation_warning': 'No registration, no aircraft rotation attribution. Shared stand alone does not establish same aircraft.',
        'tests': 'strict ties/self-flight exclusion/future-event mutation synthetic tests passed'}
    for path in paths:
        raw = pq.read_table(path, columns=COLS, use_threads=False).to_pandas(strings_to_categorical=True)
        dep = raw.loc[raw[PHASE].eq('DEP')].copy().reset_index(drop=True)
        dep['month'] = utc(dep[MOVEMENT]).dt.strftime('%Y-%m')
        dep['stand'] = dep.STAND_mvt.astype('string').fillna('<missing>')
        q, block, flights = ns(dep[MOVEMENT]), ns(dep[BLOCK]), dep[FLIGHT_ID].to_numpy()
        result = {method: (np.full(len(dep), np.inf), np.full(len(dep), np.nan)) for method in ['same_landing', 'same_completion', 'shuffled_landing', 'shuffled_completion']}
        for key, indices in dep.groupby(['ADEP_mvt', 'stand', 'month'], observed=True, sort=False).indices.items():
            airport, stand, month = map(str, key)
            if stand == '<missing>':
                continue
            for control, selected_stand in [('same', stand), ('shuffled', random_stand.get((airport, stand)))]:
                if selected_stand is None:
                    continue
                for event_name in ['landing', 'completion']:
                    events = event_tables[event_name].get((airport, selected_stand, month))
                    if events is None:
                        continue
                    distance, latest = event_candidates(*events, q[indices], block[indices], flights[indices])
                    result[f'{control}_{event_name}'][0][indices] = distance
                    result[f'{control}_{event_name}'][1][indices] = latest
        month = path.name.split('_')[1][:7]
        report['months'][month] = summarize(dep, result, score_map)
        write_json(OUT / 'stand_event_progress.json', report)
        group = report['months'][month]['missing_giant']
        print(month, 'missing_giant_n', group['n'], 'same_completion_within60', group['methods']['same_completion']['within_hidden_block']['60']['n'], flush=True)
    report['runtime_sec'] = time.monotonic() - started
    report['peak_rss_bytes'] = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
    write_json(OUT / 'stand_events.json', report)
    print('DONE stand-event diagnostics', report['runtime_sec'], flush=True)


if __name__ == '__main__':
    main()
