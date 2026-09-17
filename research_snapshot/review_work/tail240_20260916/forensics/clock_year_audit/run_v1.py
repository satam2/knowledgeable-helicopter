"""All twelve observed 2025 packs; unchanged original seven-clock arithmetic."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from datetime import date
import time
import lightgbm
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common

OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/forensics/clock_year_audit/v1')
RAW = ROOT / 'data/09-15-2026-18-55-03_files_list'
BASE = ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
ID, TIME = common.ID, common.MOVEMENT
CLOCKS = ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt']
FEATURES = ['takeoff_minus_'+c for c in CLOCKS] + ['nm_actual_minus_estimated', 'last_minus_initial']


def main():
    started = time.monotonic()
    assert psutil.virtual_memory().available >= 10*1024**3
    OUT.mkdir(parents=True, exist_ok=False)
    raw_hashes = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    paths = [RAW / f'training_{date(2025,m,1).isoformat()}_{(date(2026,1,1) if m==12 else date(2025,m+1,1)).isoformat()}.parquet' for m in range(1, 13)]
    prior = ROOT / 'private_runs/tail240_20260916/source_distinctions/clock_preprocessing_audit_v2'
    common.write_json(OUT / 'protocol.json', dict(source_sha256=common.sha256(__file__),
        preserved_original_protocol_sha256=common.sha256(prior / 'protocol.json'),
        preserved_original_receipt_sha256=common.sha256(prior / 'receipt.json'),
        preserved_original_source_sha256=common.sha256(ROOT / 'review_work/tail240_20260916/source_distinctions/audit_clock_preprocessing_v2.py'),
        base_pipeline_sha256=common.sha256(ROOT / 'knowledgeable-helicopter-screening/src/taxiout/features/pipeline.py'),
        encoding='Basepipeline fillna(-999999) is already stored in cache even when riskloader sourcefill=False. Preserve this representation for parity, count missing vs actualsentinel independently.',
        input_columns=[ID, 'PHASE_mvt', TIME, *CLOCKS],
        scope='Every Jan-Dec2025training departure observed clocks only; no DEP target/block or ranking fields. Includes observed June/July/October/November/December clock fields, never their labels. Raw timestamp conversion independent of original duration helper. Compare7baseclock fields exactly afterfloat32; quantify parser coercion and representational rounding without changing any value.',
        raw_files={p.name: raw_hashes[p.name] for p in paths}, resources='1CPU/2GiBpeak/8GiBreserve'))
    records = []
    for path in paths:
        assert common.sha256(path) == raw_hashes[path.name]
        cache = BASE / path.name
        marker = common.read_json(cache.with_suffix('.json'))
        assert common.sha256(cache) == marker['sha256']
        raw = pq.read_table(path, columns=[ID, 'PHASE_mvt', TIME, *CLOCKS],
            filters=[('PHASE_mvt', '=', 'DEP')], use_threads=False).to_pandas()
        frame = pd.read_parquet(cache, columns=[ID, *FEATURES]).set_index(ID)
        assert frame.index.is_unique and raw[ID].is_unique and set(frame.index) == set(raw[ID])
        frame = frame.loc[raw[ID]]
        times, clock_stats = {}, {}
        for column in [TIME, *CLOCKS]:
            original = raw[column]
            converted = pd.to_datetime(original, utc=True, errors='coerce')
            times[column] = converted
            lost = original.notna() & converted.isna()
            clock_stats[column] = dict(source_dtype=str(original.dtype), original_nonnull=int(original.notna().sum()),
                coerced_nonnull=int(lost.sum()), minimum=str(converted.min()), maximum=str(converted.max()))
        pairs = [(name, TIME, clock) for name, clock in zip(FEATURES[:5], CLOCKS)]
        pairs += [('nm_actual_minus_estimated', 'AOBT_3_flt', 'EOBT_1_flt'), ('last_minus_initial', 'LOBT_flt', 'IOBT_flt')]
        comparisons = {}
        for name, end, start in pairs:
            seconds = (times[end]-times[start]).dt.total_seconds().to_numpy(float)
            actual = frame[name].to_numpy(float)
            rounded = seconds.astype(np.float32).astype(float)
            expected = np.where(np.isfinite(rounded), rounded, -999999.)
            equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
            assert equal.all(), (path.name, name, int((~equal).sum()))
            valid = np.isfinite(seconds)
            comparisons[name] = dict(rows=len(seconds), finite=int(valid.sum()),
                mismatches=0, finite_to_nonfinite=int(np.sum(valid & ~np.isfinite(rounded))),
                encoded_missing=int(np.sum(~valid)),
                rounded_to_sentinel=int(np.sum(valid & (seconds != -999999.) & (rounded == -999999.))),
                sentinel_collision=int(np.sum(seconds == -999999.)),
                within_one_float32_ulp_of_sentinel=int(np.sum(valid & (seconds != -999999.) & (np.abs(seconds+999999.) <= float(np.spacing(np.float32(999999.)))))),
                negative_values=int(np.sum(seconds < 0)), absolute_over_day=int(np.sum(np.abs(seconds) > 86400)),
                max_float32_rounding_sec=float(np.max(np.abs(rounded[valid]-seconds[valid]))) if valid.any() else 0.)
        info = psutil.Process().memory_info()
        peak = getattr(info, 'peak_wset', info.rss)
        assert peak < 2*1024**3 and psutil.virtual_memory().available >= 8*1024**3
        records.append(dict(file=path.name, rows=len(raw), raw_ids_hash=common.object_hash(raw[ID].tolist()),
            cache_ids_after_alignment_hash=common.object_hash(frame.index.tolist()), cache_sha256=marker['sha256'], clocks=clock_stats,
            comparisons=comparisons, peak_bytes=peak))
        print('CLOCK_AUDIT', path.name, len(raw), 'exact', flush=True)
    previous = common.read_json(prior / 'receipt.json')
    for actual, original in zip(records[:5], previous['records']):
        assert {k:actual[k] for k in ['file','rows','cache_sha256','clocks']} == {k:original[k] for k in ['file','rows','cache_sha256','clocks']}
        for name, expected in original['comparisons'].items():
            assert {k:actual['comparisons'][name][k] for k in expected} == expected
    assert sum(r['rows'] for r in records[:5]) == previous['departure_rows'] == 822377
    common.write_json(OUT / 'receipt.json', dict(status='passed',
        first_five_month_original_fields_exact=True, runtime_sec=time.monotonic()-started,
        peak_bytes=max(r['peak_bytes'] for r in records),
        protocol_sha256=common.sha256(OUT / 'protocol.json'), records=records,
        departure_rows=sum(r['rows'] for r in records), features_checked=7,
        conclusion='No raw-to-base clock arithmetic mismatch found. Negative/long intervals are preserved observations, not automatically invalid. This audit does not prove timestamp operational truth or inspect every derived context feature.'))


if __name__ == '__main__':
    main()
