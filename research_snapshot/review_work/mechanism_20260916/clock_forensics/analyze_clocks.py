"""Aggregate clock-mechanism forensics; no raw copies, label-based routing or fit."""

import os
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'

import sys
import time
import gc
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
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.config import load_config
from taxiout.io import concat_frames
from taxiout.paths import external_path
from taxiout.schema import BLOCK, CLOCKS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, utc
from taxiout.splits import make_fold

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/mechanism_20260916/clock_forensics')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
FIELDS = [ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, TARGET, *CLOCKS,
          'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_OPERATOR_flt',
          'AIRCRAFT_TYPE_mvt', 'FLIGHT_TYPE_flt', 'ADEP_flt', 'ADES_flt']
CLOCK_NAMES = ['nm', 'estimated', 'initial', 'last', 'schedule']


def prepare():
    frozen = read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    frames, hashes = [], {}
    for path in sorted(RAW.glob('training_*.parquet')):
        digest = sha256(path)
        assert digest == frozen['raw_hashes'][path.name]
        hashes[path.name] = digest
        raw = pq.read_table(path, columns=FIELDS, use_threads=False).to_pandas(strings_to_categorical=True)
        dep = raw.loc[raw[PHASE].eq('DEP')].copy()
        data = dep[[ID, FLIGHT_ID, MOVEMENT, TARGET, 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt',
                    'RUNWAY_mvt', 'AIRCRAFT_OPERATOR_flt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_TYPE_flt']].reset_index(drop=True)
        move, block = utc(dep[MOVEMENT], required=True), utc(dep[BLOCK], required=True)
        assert np.array_equal(data[TARGET], (move - block).dt.total_seconds())
        data['month'] = move.dt.month.to_numpy(dtype=np.int16)
        data['hour'] = move.dt.hour.to_numpy(dtype=np.int16)
        data['day'] = pd.Categorical(move.dt.strftime('%Y-%m-%d'))
        data['block_second'] = block.dt.second.to_numpy(dtype=np.int16)
        data['block_minute'] = block.dt.minute.to_numpy(dtype=np.int16)
        for name, clock in zip(CLOCK_NAMES, CLOCKS):
            observed = utc(dep[clock])
            data[f'proxy_{name}'] = (move - observed).dt.total_seconds().to_numpy(dtype=np.float32)
            data[f'{name}_second'] = observed.dt.second.to_numpy(dtype=np.float32)
            data[f'{name}_minute'] = observed.dt.minute.to_numpy(dtype=np.float32)
        data['source_airport_disagree'] = (dep.ADEP_mvt.notna() & dep.ADEP_flt.notna() & dep.ADEP_mvt.astype('string').ne(dep.ADEP_flt.astype('string'))).to_numpy()
        frames.append(data)
        print(f'Read {path.name}: {len(data):,} departures', flush=True)
    frame = concat_frames(frames)
    del frames
    gc.collect()
    frame['gap'] = frame[TARGET] - frame.proxy_nm
    frame['missing_nm'] = ~np.isfinite(frame.proxy_nm)
    frame['gap_gt1800'] = np.abs(frame.gap) > 1800
    frame['giant_target'] = frame[TARGET] > 3600
    return frame, hashes


def rate(mask):
    return float(np.mean(mask)) if len(mask) else None


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def nearest_multiple(values, step, tolerance, nonzero=True):
    values = np.asarray(values, float)
    rounded = np.rint(values / step)
    return np.isfinite(values) & (np.abs(values - rounded * step) <= tolerance) & ((rounded != 0) if nonzero else True)


def profile(frame, scored=False):
    y = frame[TARGET].to_numpy(float)
    report = {'n': len(frame), 'missing_nm_n': int(frame.missing_nm.sum()),
        'gap_gt1800_n': int(frame.gap_gt1800.sum()), 'target_gt3600_n': int((y > 3600).sum()),
        'target_gt7200_n': int((y > 7200).sum()), 'target_negative_n': int((y < 0).sum()),
        'target_quantiles_sec': dict(zip(['p50', 'p95', 'p99', 'p999', 'max'], np.quantile(y, [.5, .95, .99, .999, 1]).tolist()))}
    clock_report = {}
    masks = {'all': np.ones(len(frame), bool), 'missing_nm': frame.missing_nm.to_numpy(),
             'large_gap': frame.gap_gt1800.to_numpy(), 'giant_target': y > 3600}
    for group, mask in masks.items():
        selected = frame.loc[mask]
        stats = {'n': len(selected)}
        if scored:
            stats['reference_sse_share'] = float(selected.squared_error.sum() / frame.squared_error.sum())
            stats['reference_rmse_sec'] = float(np.sqrt(selected.squared_error.mean())) if len(selected) else None
        for name in CLOCK_NAMES:
            offset = selected[TARGET].to_numpy(float) - selected[f'proxy_{name}'].to_numpy(float)
            valid = np.isfinite(offset)
            stats[name] = {'available_n': int(valid.sum()), 'exact_target_n': int((valid & (offset == 0)).sum()),
                'within60_n': int((valid & (np.abs(offset) <= 60)).sum()),
                'gap_median_sec': float(np.median(offset[valid])) if valid.any() else None,
                'gap_abs_p95_sec': float(np.quantile(np.abs(offset[valid]), .95)) if valid.any() else None,
                'minute_aligned_pct': 100 * rate(selected[f'{name}_second'].fillna(-1).eq(0)) if len(selected) else None}
        stats['airport_block_minute_aligned_pct'] = 100 * rate(selected.block_second.eq(0)) if len(selected) else None
        stats['airport_block_five_min_aligned_pct'] = 100 * rate(selected.block_second.eq(0) & selected.block_minute.mod(5).eq(0)) if len(selected) else None
        clock_report[group] = stats
    report['clock_groups'] = clock_report
    large = frame.loc[frame.gap_gt1800]
    gaps = large.gap.to_numpy(float)
    report['large_gap_quantization'] = {str(step): {str(tol): {'n': int(nearest_multiple(gaps, step, tol).sum()),
        'fraction_large_gap': rate(nearest_multiple(gaps, step, tol)),
        **({'reference_sse_share_all': float(large.loc[nearest_multiple(gaps, step, tol), 'squared_error'].sum() / frame.squared_error.sum())} if scored else {})}
        for tol in [0, 1, 5, 30, 60]} for step in [60, 1800, 3600, 86400]}
    report['large_gap_nearest_hour_modes'] = [{'hour_offset': int(key), 'n': int(count)} for key, count in pd.Series(np.rint(gaps / 3600)).value_counts().head(12).items()]
    report['large_gap_exact_second_modes'] = [{'gap_sec': float(key), 'n': int(count)} for key, count in large.gap.value_counts().head(15).items()]
    return report


def grouped(frame, columns, scored=False, maximum=20):
    data = frame.copy(deep=False)
    groups = data.groupby(columns, observed=True, dropna=False)
    summary = groups.agg(n=(ID, 'size'), missing_n=('missing_nm', 'sum'), gap_n=('gap_gt1800', 'sum'),
                         giant_n=('giant_target', 'sum'), mean_target=(TARGET, 'mean'))
    summary['missing_rate'] = summary.missing_n / summary.n
    summary['gap_rate'] = summary.gap_n / summary.n
    summary['giant_rate'] = summary.giant_n / summary.n
    if scored:
        summary['sse'] = groups.squared_error.sum()
        summary['sse_share'] = summary.sse / data.squared_error.sum()
        summary = summary.sort_values('sse', ascending=False)
    else:
        summary = summary.sort_values('gap_n', ascending=False)
    return summary.head(maximum).reset_index().replace({np.nan: None}).to_dict('records')


def observable_regimes(frame, scored=False):
    delta = frame.proxy_schedule - frame.proxy_nm
    disagreement = frame.proxy_initial - frame.proxy_last
    masks = {'nm_missing_schedule_missing': frame.missing_nm & frame.proxy_schedule.isna(),
        'nm_missing_schedule_negative': frame.missing_nm & frame.proxy_schedule.lt(0),
        'nm_missing_schedule_0_30m': frame.missing_nm & frame.proxy_schedule.between(0, 1800),
        'nm_missing_schedule_30m_2h': frame.missing_nm & frame.proxy_schedule.between(1800, 7200, inclusive='right'),
        'nm_missing_schedule_over2h': frame.missing_nm & frame.proxy_schedule.gt(7200),
        'nm_proxy_negative': frame.proxy_nm.lt(0), 'nm_proxy_over2h': frame.proxy_nm.gt(7200),
        'nm_second_zero': frame.nm_second.eq(0),
        'nm_schedule_disagree_gt30m': delta.abs().gt(1800),
        'initial_last_disagree_gt30m': disagreement.abs().gt(1800),
        'nm_schedule_near_hour_nonzero': nearest_multiple(delta, 3600, 60),
        'nm_schedule_near_day_nonzero': nearest_multiple(delta, 86400, 60)}
    rows = []
    for name, mask in masks.items():
        selected = frame.loc[mask]
        record = {'name': name, 'n': len(selected), 'fraction_rows': len(selected) / len(frame),
                  'large_gap_fraction': rate(selected.gap_gt1800), 'giant_target_fraction': rate(selected.giant_target)}
        if scored:
            record['reference_sse_share'] = float(selected.squared_error.sum() / frame.squared_error.sum())
        rows.append(record)
    return rows


def cluster_transfer(fit, score, keys):
    history = fit.groupby(keys, observed=True, dropna=False).agg(fit_n=(ID, 'size'),
        fit_gap_rate=('gap_gt1800', 'mean'), fit_missing_rate=('missing_nm', 'mean'),
        fit_giant_rate=('giant_target', 'mean'), fit_y_mean=(TARGET, 'mean'))
    target = score.groupby(keys, observed=True, dropna=False).agg(score_n=(ID, 'size'),
        score_gap_n=('gap_gt1800', 'sum'), score_missing_n=('missing_nm', 'sum'),
        score_giant_n=('giant_target', 'sum'), score_sse=('squared_error', 'sum'), score_y_mean=(TARGET, 'mean'))
    paired = target.join(history, how='left')
    paired['score_sse_share'] = paired.score_sse / score.squared_error.sum()
    return paired.sort_values('score_sse', ascending=False).head(20).reset_index().replace({np.nan: None}).to_dict('records')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    data, hashes = prepare()
    meta = data[[ID, FLIGHT_ID, MOVEMENT]]
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'raw_hashes': hashes,
        'scope': 'Aggregate retrospective diagnosis only. Target-defined slices are never inference routes. No model/rule pilots yet.',
        'availability': 'No dedicated source-system identifier or aircraft registration; observed field names and reporting precision are proxies, not authenticated source provenance.',
        'training_all': profile(data), 'monthly': [], 'folds': {}}
    for month, subset in data.groupby('month', sort=True):
        report['monthly'].append({'month': int(month), **profile(subset)})
    for fold in ['F1', 'F3']:
        idx, split = make_fold(meta, load_config('configs/folds.yaml')[fold])
        reference, ref_record = common.reference(fold)
        assert object_hash(split) == object_hash(ref_record['split'])
        score = data.iloc[idx['score']].copy()
        assert np.array_equal(score[ID], reference[ID])
        assert np.array_equal(score[TARGET], reference[TARGET])
        score['prediction'] = reference.prediction_sec.to_numpy()
        score['squared_error'] = reference.squared_error.to_numpy()
        fit, tune = [data.iloc[idx[stage]] for stage in ['fit', 'tune']]
        details = {'fit': profile(fit), 'tune': profile(tune), 'score': profile(score, True),
            'observable_fit': observable_regimes(fit), 'observable_score': observable_regimes(score, True),
            'clusters': {}, 'transfer': {}, 'complete_score_n': len(score), 'split_hash': object_hash(split)}
        for label, keys in [('airport', ['ADEP_mvt']), ('airport_operator', ['ADEP_mvt', 'AIRCRAFT_OPERATOR_flt']),
                            ('airport_stand', ['ADEP_mvt', 'STAND_mvt']), ('airport_hour', ['ADEP_mvt', 'hour']),
                            ('day_airport', ['day', 'ADEP_mvt'])]:
            details['clusters'][label] = grouped(score, keys, True)
            if label != 'day_airport':
                details['transfer'][label] = cluster_transfer(fit, score, keys)
        details['large_gap_clusters'] = grouped(score.loc[score.gap_gt1800], ['ADEP_mvt', 'AIRCRAFT_OPERATOR_flt'], True)
        details['missing_clusters'] = grouped(score.loc[score.missing_nm], ['ADEP_mvt', 'AIRCRAFT_OPERATOR_flt'], True)
        # Diagnostic ceilings only: what error share is associated with offset lattices?
        gap = score.gap.to_numpy(float)
        details['hour_lattice_label_diagnostic'] = {'n': int((score.gap_gt1800 & nearest_multiple(gap, 3600, 60)).sum()),
            'sse_share': float(score.loc[score.gap_gt1800 & nearest_multiple(gap, 3600, 60), 'squared_error'].sum() / score.squared_error.sum())}
        report['folds'][fold] = details
        print(f'{fold}: complete {len(score):,} score rows; aggregates built', flush=True)
    report['runtime_sec'] = time.monotonic() - began
    report['peak_rss_bytes'] = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
    if report['peak_rss_bytes'] > 4 * 1024**3:
        raise MemoryError('Aggregate audit exceeded revised approved 4 GiB peak')
    write_json(OUT / 'aggregate.json', json_finite(report))
    print('DONE', report['runtime_sec'], report['peak_rss_bytes'], flush=True)


if __name__ == '__main__':
    main()
