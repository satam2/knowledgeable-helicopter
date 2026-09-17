"""Identify which observed clock aligns with supplied record ordering."""

import os
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, BLOCK, TARGET, PHASE, utc

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/record_order')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')


def seconds(series):
    converted = utc(series)
    result = converted.dt.as_unit('ns').astype('int64').to_numpy(dtype=float) / 1e9
    result[converted.isna().to_numpy()] = np.nan
    return result


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {'n': len(values), 'median': float(np.median(values)) if len(values) else None,
        'mae_sec': float(np.abs(values).mean()) if len(values) else None,
        'rmse_sec': float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
        'within60_pct': float(np.mean(np.abs(values) <= 60) * 100) if len(values) else None,
        'abs_p90': float(np.quantile(np.abs(values), .9)) if len(values) else None}


def nanmedian_peers(values, width):
    padded = np.pad(values, (width, width), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
    peers = np.concatenate([windows[:, :width], windows[:, width+1:]], axis=1)
    count = np.sum(np.isfinite(peers), axis=1)
    med = np.full(len(values), np.nan)
    valid = count > 0
    med[valid] = np.nanmedian(peers[valid], axis=1)
    return med, count


def finite(value):
    if isinstance(value, dict):
        return {key: finite(v) for key, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    started = time.monotonic()
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'groups': [], 'neighbor_diagnostics': [],
        'scope': 'Retrospective source-order audit, hidden departure block used only to assess mechanism. No fitted model or feature cache.',
        'prior': 'Same-airport/month ARR and DEP ID ranges do not bracket, as verified in information_audit. This audit compares source clock ordering and observed DEP-peer anchors.'}
    for path in sorted(RAW.glob('training_*.parquet')):
        assert sha256(path) == expected[path.name]
        cols = [ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, TARGET, 'AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt',
                'SCHED_TIME_UTC_mvt', 'ARVT_3_flt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt']
        raw = pq.read_table(path, columns=cols, use_threads=False).to_pandas(strings_to_categorical=True)
        raw['airport'] = np.where(raw[PHASE].eq('DEP'), raw.ADEP_mvt.astype('string'), raw.ADES_mvt.astype('string'))
        raw['month'] = utc(raw[MOVEMENT]).dt.strftime('%Y-%m')
        raw['day'] = utc(raw[MOVEMENT]).dt.strftime('%Y-%m-%d')
        for key, part in raw.groupby(['airport', PHASE, 'month'], observed=True, sort=False):
            airport, phase, month = key
            if len(part) < 20:
                continue
            part = part.sort_values(ID, kind='stable')
            ids = part[ID].to_numpy()
            gap = np.diff(ids)
            record = {'airport': str(airport), 'phase': str(phase), 'month': str(month), 'n': len(part),
                'id_step_one_pct': float(np.mean(gap == 1) * 100),
                'id_gap_quantiles': np.quantile(gap, [0, .5, .9, .99, 1]).tolist(), 'clocks': {}}
            clocks = {'movement': MOVEMENT, 'airport_block': BLOCK, 'schedule': 'SCHED_TIME_UTC_mvt',
                      'nm_actual': 'AOBT_3_flt' if phase == 'DEP' else 'ARVT_3_flt'}
            if phase == 'DEP':
                clocks.update(nm_estimated='EOBT_1_flt', nm_initial='IOBT_flt')
            same_day = part.day.iloc[1:].to_numpy() == part.day.iloc[:-1].to_numpy()
            for name, column in clocks.items():
                stamp = seconds(part[column])
                dt = np.diff(stamp)
                valid = same_day & np.isfinite(dt)
                rho = []
                for _, sub in part.groupby('day', observed=True):
                    if len(sub) >= 10 and sub[column].notna().sum() >= 10:
                        rho.append(sub[ID].corr(pd.Series(seconds(sub[column]), index=sub.index), method='spearman'))
                record['clocks'][name] = {'within_day_spearman_mean': float(np.nanmean(rho)) if rho else None,
                    'within_day_adjacent_reversal_pct': float(np.mean(dt[valid] < 0) * 100) if valid.any() else None,
                    'within_day_adjacent_gap': stats(dt[valid]),
                    'hour_boundary_jump_pct': float(np.mean(np.abs(dt[valid]) > 3600) * 100) if valid.any() else None}
            report['groups'].append(record)
            if phase == 'DEP':
                hidden = seconds(part[BLOCK])
                missing = part.AOBT_3_flt.isna().to_numpy()
                target = part[TARGET].to_numpy(float)
                diagnostic = {'airport': str(airport), 'month': str(month), 'n': len(part), 'anchors': {}}
                for name, column in [('peer_nm_actual', 'AOBT_3_flt'), ('peer_nm_initial', 'IOBT_flt'), ('peer_schedule', 'SCHED_TIME_UTC_mvt'), ('peer_takeoff', MOVEMENT)]:
                    estimates, count = nanmedian_peers(seconds(part[column]), 8)
                    difference = estimates - hidden
                    diagnostic['anchors'][name] = {group: stats(difference[mask]) for group, mask in
                        [('all', np.ones(len(part), bool)), ('missing', missing), ('missing_over12h', missing & (target > 43200)),
                         ('missing_day_plus_2h', missing & (target >= 86400) & (target <= 93600))]}
                    diagnostic['anchors'][name]['missing_with_support_n'] = int((missing & (count > 0)).sum())
                report['neighbor_diagnostics'].append(diagnostic)
        print(path.name, 'order mechanism audited', flush=True)
    report['runtime_sec'] = time.monotonic() - started
    write_json(OUT / 'audit.json', finite(report))
    print('DONE record order', report['runtime_sec'], flush=True)


if __name__ == '__main__':
    main()
