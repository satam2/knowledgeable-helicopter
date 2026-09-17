"""ARR-only raw calendar-carry audit; no departure labels or block reads."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
OUT = external_path(ROOT / 'private_runs/tail240_20260916/state/arrival_calendar/v1')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'PHASE_mvt',
        'ADES_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
COUNTS = ['rows', 'finite_duration', 'known_three_clocks', 'negative', 'abs_gt_12h',
          'negative_near_day', 'positive_near_day', 'utc_block_schedule_date_carry',
          'utc_carry_negative', 'utc_carry_extreme', 'rome_block_schedule_date_carry',
          'rome_carry_negative', 'rome_carry_extreme']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if max(info.rss, peak) > 2 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('ARR calendar audit exceeded 2GiB process / 8GiB host reserve')


def derive(frame):
    if not frame.PHASE_mvt.eq('ARR').all():
        raise ValueError('Only ARR block timestamps permitted')
    f = frame.copy()
    t, b, s = [pd.to_datetime(f[c], utc=True) for c in COLS[-3:]]
    d = (b - t).dt.total_seconds()
    known = t.notna() & b.notna() & s.notna()
    utc = known & b.dt.normalize().eq(s.dt.normalize()) & b.dt.normalize().ne(t.dt.normalize())
    local = [x.dt.tz_convert('Europe/Rome').dt.normalize() for x in (t, b, s)]
    rome = known & f.ADES_mvt.isin(['LIRF', 'LIRA']) & local[1].eq(local[2]) & local[1].ne(local[0])
    f['airport'] = f.ADES_mvt.fillna('MISSING')
    f['prefix'] = f.FLIGHT_mvt.astype('string').str.extract(r'^([A-Za-z]+)', expand=False).str.upper().fillna('MISSING')
    f['month'] = t.dt.strftime('%Y-%m').fillna('MISSING')
    f['duration_sec'] = d
    f['block_minus_movement_days_utc'] = (b.dt.normalize() - t.dt.normalize()).dt.total_seconds() / 86400
    flags = [np.ones(len(f), dtype=bool), np.isfinite(d), known, d.lt(0), d.abs().gt(43200),
             d.between(-86400, -79200), d.between(86400, 93600), utc,
             utc & d.lt(0), utc & d.abs().gt(43200), rome,
             rome & d.lt(0), rome & d.abs().gt(43200)]
    for name, value in zip(COUNTS, flags):
        f[name] = np.asarray(value, dtype=np.int64)
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve immutable prior run'
    frozen = json.loads((ROOT / 'private_runs/submission_v2/protocol.json').read_text())['raw_hashes']
    paths = sorted(RAW.glob('training_*.parquet')) + [RAW / 'ranking.parquet']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        columns=COLS, raw_hashes={p.name: frozen[p.name] for p in paths},
        policy='ARR predicate before block projection; raw rows preserved including duplicates/anomalies. Calendar relation is evidence, not a label correction. Ranking ARR allowed; no DEP block, target, or score labels read. All raw months for source forensics, no model selection.',
        resources='1CPU,2GiB sampled RSS plus historical peak_wset checkpoints,8GiB host reserve; noGPU.',
        definitions=dict(negative_near_day='-86400 <= raw B-T <= -79200', positive_near_day='86400 <= raw B-T <= 93600',
             date_carry='all B/T/S known; calendar date(B)==date(S)!=date(T)',
             rome='Europe/Rome timezone, LIRF/LIRA only; UTC counts cover every airport'))
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    groups, anomalies, file_counts = [], [], []
    for path in paths:
        guard()
        actual = sha(path)
        assert actual == frozen[path.name], path
        scanner = ds.dataset(path, format='parquet').scanner(columns=COLS, filter=ds.field('PHASE_mvt') == 'ARR',
            batch_size=16384, batch_readahead=1, fragment_readahead=1, use_threads=False)
        total = np.zeros(len(COUNTS), dtype=np.int64)
        for batch in scanner.to_batches():
            f = derive(batch.to_pandas())
            f['source'] = path.name
            total += f[COUNTS].sum().to_numpy(np.int64)
            groups.append(f.groupby(['source', 'month', 'airport', 'prefix'], dropna=False)[COUNTS].sum())
            unusual = f.negative.eq(1) | f.abs_gt_12h.eq(1) | f.utc_block_schedule_date_carry.eq(1) | f.rome_block_schedule_date_carry.eq(1)
            if unusual.any():
                anomalies.append(f.loc[unusual])
            del f
            guard()
        file_counts.append(dict(source=path.name, **dict(zip(COUNTS, map(int, total)))))
        print(json.dumps(file_counts[-1]), flush=True)
    aggregate = pd.concat(groups).groupby(level=[0, 1, 2, 3]).sum().reset_index()
    aggregate.to_parquet(OUT / 'airport_prefix_month.parquet', index=False)
    pd.concat(anomalies, ignore_index=True).to_parquet(OUT / 'anomalous_arrivals.parquet', index=False)
    pd.DataFrame(file_counts).to_csv(OUT / 'file_counts.csv', index=False)
    guard()
    info = psutil.Process().memory_info()
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(OUT / 'protocol.json'),
        totals=pd.DataFrame(file_counts)[COUNTS].sum().to_dict(), rss_bytes=info.rss,
        peak_wset_bytes=getattr(info, 'peak_wset', info.rss),
        artifacts={p.name: sha(p) for p in OUT.iterdir() if p.name != 'receipt.json'})
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
