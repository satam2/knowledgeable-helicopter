"""Aggregate observation distribution shifts; no departure target/block reads."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.schema import ID, PHASE, MOVEMENT, CLOCKS
from taxiout.paths import external_path
BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = external_path(BASE / 'retrospective_research/validation_transfer')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
COLS = [ID, PHASE, MOVEMENT, 'ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', *CLOCKS]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def guard():
    if psutil.Process().memory_info().rss > 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Transfer audit resource guard')


def hist(values):
    return {str(k): int(v) for k, v in pd.Series(values).value_counts(dropna=False).items()}


def aggregate(raw):
    q = pd.to_datetime(raw[MOVEMENT], utc=True)
    times = [pd.to_datetime(raw[c], utc=True) for c in CLOCKS]
    proxies = np.column_stack([(q - t).dt.total_seconds() for t in times])
    nm = proxies[:, 0]
    status = np.select([~np.isfinite(nm), nm < 0, nm > 7200], ['missing', 'negative', 'gt7200'], default='ordinary')
    bins = pd.cut(nm, [-np.inf, 0, 300, 600, 900, 1200, 1500, 1800, 3600, 7200, np.inf], right=False).astype(str)
    order = pd.DataFrame(proxies).rank(axis=1, method='dense').fillna(0).astype(int).astype(str).agg('|'.join, axis=1).to_numpy()
    airport = raw.ADEP_mvt.astype('string').fillna('<missing>').to_numpy()
    daydelta = (q.dt.normalize() - times[0].dt.normalize()).dt.total_seconds().to_numpy() / 86400
    same_planning = np.isfinite(proxies[:, 1:4]).all(axis=1) & (proxies[:, 1] == proxies[:, 2]) & (proxies[:, 2] == proxies[:, 3])
    nm_aligned = times[0].notna() & times[0].dt.second.eq(0) & times[0].dt.microsecond.eq(0)
    perairport = {}
    for name in sorted(np.unique(airport)):
        mask = airport == name
        finite = mask & np.isfinite(nm)
        perairport[str(name)] = {'rows': int(mask.sum()), 'share': float(mask.mean()), 'status': hist(status[mask]),
            'missing_pct': float(100 * np.mean(~np.isfinite(nm[mask]))),
            'proxy_median_sec': float(np.median(nm[finite])) if finite.any() else None,
            'planning_equal_pct': float(100 * same_planning[mask].mean()),
            'nm_minute_aligned_pct': float(100 * nm_aligned.to_numpy()[mask].mean()),
            'order_histogram': hist(order[mask])}
    return {'rows': len(raw), 'airport': hist(airport), 'proxy_status': hist(status), 'proxy_bins': hist(bins),
        'clock_order': hist(order), 'hour': hist(q.dt.hour), 'weekday': hist(q.dt.weekday), 'nm_day_delta': hist(daydelta),
        'all_nm_planning_missing_rows': int((~np.isfinite(proxies[:, :4])).all(axis=1).sum()),
        'planning_equal_pct': float(100 * same_planning.mean()), 'nm_minute_aligned_pct': float(100 * nm_aligned.mean()),
        'per_airport': perairport, 'time_min': str(q.min()), 'time_max': str(q.max())}


def fractions(record, key):
    return {k: v / record['rows'] for k, v in record[key].items()}


def compare(a, b):
    result = {}
    for key in ['airport', 'proxy_status', 'proxy_bins', 'clock_order', 'hour', 'weekday', 'nm_day_delta']:
        p, q = fractions(a, key), fractions(b, key)
        deltas = {k: q.get(k, 0) - p.get(k, 0) for k in set(p) | set(q)}
        result[key] = {'total_variation': .5 * sum(abs(v) for v in deltas.values()),
            'largest_fraction_deltas_b_minus_a': dict(sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12])}
    result['missing_pct_a'] = 100 * a['proxy_status'].get('missing', 0) / a['rows']
    result['missing_pct_b'] = 100 * b['proxy_status'].get('missing', 0) / b['rows']
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'aggregate.json').exists():
        raise ValueError('Preserve completed transfer audit')
    frozen = json.loads((ROOT / 'private_runs/submission_v2/protocol.json').read_text())
    selected = {month: [] for month in ['2025-01', '2025-07', '2025-11']}
    receipts = []
    for path in sorted(RAW.glob('training_*.parquet')):
        guard()
        assert sha(path) == frozen['raw_hashes'][path.name]
        observed = pq.read_table(path, columns=COLS, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
        months = pd.to_datetime(observed[MOVEMENT], utc=True).dt.strftime('%Y-%m')
        for month in selected:
            part = observed.loc[months.eq(month)].copy()
            if len(part):
                selected[month].append(part)
        receipts.append({'file': path.name, 'sha256': frozen['raw_hashes'][path.name]})
        del observed, part
        gc.collect()
    records = {}
    for month in list(selected):
        raw = pd.concat(selected.pop(month), ignore_index=True)
        records[month] = aggregate(raw)
        del raw
        gc.collect()
    guard()
    path = RAW / 'ranking.parquet'
    assert sha(path) == frozen['raw_hashes'][path.name]
    raw = pq.read_table(path, columns=COLS, filters=[(PHASE, '=', 'DEP')], use_threads=False).to_pandas()
    months = pd.to_datetime(raw[MOVEMENT], utc=True).dt.strftime('%Y-%m')
    for month in sorted(months.unique()):
        records[month] = aggregate(raw.loc[months.eq(month)])
    records['ranking_all'] = aggregate(raw)
    receipts.append({'file': path.name, 'sha256': frozen['raw_hashes'][path.name]})
    # A synthetic weighted mixture of observed local distributions, not new rows or scores.
    weighted = {'rows': 344841}
    for key in ['airport', 'proxy_status', 'proxy_bins', 'clock_order', 'hour', 'weekday', 'nm_day_delta']:
        a, b = records['2025-07'], records['2025-11']
        weighted[key] = {k: 192122 * a[key].get(k, 0) / a['rows'] + 152719 * b[key].get(k, 0) / b['rows'] for k in set(a[key]) | set(b[key])}
    records['local_weighted_mixture'] = weighted
    comparisons = {name: compare(records[a], records[b]) for name, a, b in [
        ('local_weighted_to_ranking', 'local_weighted_mixture', 'ranking_all'),
        ('July2025_to_July2026', '2025-07', '2026-07'),
        ('November2025_to_January2026', '2025-11', '2026-01'),
        ('January2025_to_January2026', '2025-01', '2026-01')]}
    original = json.loads((ROOT / 'knowledgeable-helicopter/reports/comparison.json').read_text())['candidates']['residual_long_proxy_specialist']['folds']
    old_local = ((192122 * original['F1']['rmse_sec']**2 + 152719 * original['F3']['rmse_sec']**2) / 344841)**.5
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__), 'columns_read': COLS,
        'privacy': 'No departure block/target column read; aggregate-only output; no network/models/new scores',
        'raw_receipts': receipts, 'records': records, 'comparisons': comparisons,
        'calibration_evidence': {'V2': {'local_seasonal': 292.8133464057957, 'official': 294.626, 'official_minus_local': 294.626 - 292.8133464057957,
            'local_method': 'F1July/F3November2025 rankingmonthcount-weighted MSE; final model trained on full2025 differs fromfoldfits'},
            'original_release': {'local_seasonal': old_local, 'candidate': 'residual_long_proxy_specialist',
                'release_submission_sha256': '823408382b15b41a5b353b219ef7e5ebbfb07e81e8c3c833f3ebf66295c88e13',
                'V1_official': 348.0775, 'identity_link': 'No inspectedreceipt binds uploadedV1 bytes to originalreleasehash; do nottreatasverified calibrationpair',
                'conditional_difference_if_same_recipe': 348.0775 - old_local}},
        'rss_bytes': psutil.Process().memory_info().rss, 'limitations': 'Marginal covariate shifts do notidentify hidden conditionalerror shift or rankordering; no claimedofficialscoreforecast'}
    guard()
    (OUT / 'aggregate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'comparisons': comparisons, 'calibration': result['calibration_evidence'], 'rss_bytes': result['rss_bytes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
