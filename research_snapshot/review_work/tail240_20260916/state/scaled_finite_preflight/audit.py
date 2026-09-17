"""Label-free full-finite scale/weight diagnostics before model fitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/models'))
import scaled_finite_tune as subject
common, ID = subject.common, subject.ID
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/scaled_finite_preflight/v1')
META = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
CACHE = ROOT / 'private_runs/breakthrough_20260916/missing/source_conventions/features.parquet'
META_COLUMNS = [ID, 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'ADEP_mvt', 'proxy_sec', 'schedule_sec']
CACHE_COLUMNS = [ID, subject.GAP, subject.MISSING]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def guard():
    info = psutil.Process().memory_info()
    assert max(info.rss, info.peak_wset) < 2 * 1024**3
    assert psutil.virtual_memory().available >= 8 * 1024**3


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = common.read_json(CACHE.with_name('manifest.json'))
    metadata_hash = common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][META.name]
    assert common.sha256(META) == metadata_hash
    assert common.sha256(CACHE) == manifest['feature_sha256']
    declaration = dict(source_sha256=common.sha256(__file__), scaled_source_sha256=common.sha256(subject.__file__),
        metadata_sha256=metadata_hash, cache_sha256=manifest['feature_sha256'],
        metadata_columns=META_COLUMNS, cache_columns=CACHE_COLUMNS,
        boundary='No target or DEP block reads; no fitting, formula selection, or score evaluation. Original make_fold purges and exact original finite fit/tune ID hashes.',
        resources='1CPU,<2GiB historicalpeak checkpoints,8GiBhostreserve; noGPU.')
    common.write_json(OUT / 'protocol.json', declaration)
    meta = pd.read_parquet(META, columns=META_COLUMNS)
    cache = pd.read_parquet(CACHE, columns=CACHE_COLUMNS)
    assert np.array_equal(meta[ID], cache[ID]) and meta[ID].is_unique
    gap = cache[subject.GAP].to_numpy(float)
    missing = cache[subject.MISSING].to_numpy(float)
    raw_gap = meta.schedule_sec.to_numpy(float) - meta.proxy_sec.to_numpy(float)
    expected_missing = ~np.isfinite(raw_gap)
    expected_gap = np.where(expected_missing, -999999., raw_gap).astype('float32').astype(float)
    np.testing.assert_array_equal(gap, expected_gap)
    np.testing.assert_array_equal(missing, expected_missing.astype(float))
    scales = subject.scale_of(gap, missing)
    expected_scale = np.sqrt(900.**2 + np.where((missing == 0) & np.isfinite(gap) & (gap != -999999.), gap, 0.)**2)
    np.testing.assert_allclose(scales, expected_scale, rtol=2e-16, atol=1e-10)
    assert np.isfinite(scales).all() and (scales >= 900).all()
    guard()
    report = dict(formula_max_abs_error=float(np.max(np.abs(scales-expected_scale))),
        cache_gap_exact_raw_clock_difference_float32=True, missing_flags_exact_raw_clocks=True,
        valid_gap_sentinel_collisions=int(np.sum((missing == 0) & (gap == -999999.))), folds={})
    for fold in ['F1', 'F3']:
        indices, split = common.make_fold(meta, common.load_config('configs/folds.yaml')[fold])
        control = common.read_json(subject.risk.union_folder(fold) / 'manifest.json')
        rows = {stage: idx[np.isfinite(meta.iloc[idx].proxy_sec.to_numpy())] for stage, idx in indices.items() if stage in ('fit', 'tune')}
        normalizer = float(np.mean(scales[rows['fit']]**2))
        parts = {}
        for stage, idx in rows.items():
            assert common.object_hash(meta.iloc[idx][ID].tolist()) == control['fit_ids'][stage]['hash']
            assert len(idx) == control['fit_ids'][stage]['n']
            value = scales[idx]
            weights = value**2 / normalizer
            order = np.argsort(-weights, kind='stable')
            f = meta.iloc[idx][[ID, common.MOVEMENT, 'ADEP_mvt', 'proxy_sec', 'schedule_sec']].copy()
            f['gap_sec'], f['scale'], f['weight'] = gap[idx], value, weights
            f.iloc[order[:20]].to_parquet(OUT / f'{fold}_{stage}_top20_observed_weights.parquet', index=False)
            daily = f.assign(day=f[common.MOVEMENT].dt.floor('D')).groupby('day').weight.sum()
            parts[stage] = dict(rows=len(idx), id_hash=common.object_hash(f[ID].tolist()),
                scale_quantiles=np.quantile(value, [0,.5,.9,.99,.999,1]).tolist(),
                effective_sample_size=float(weights.sum()**2 / np.dot(weights, weights)),
                ess_fraction=float(weights.sum()**2 / np.dot(weights, weights) / len(idx)),
                max_weight_share=float(weights.max()/weights.sum()),
                top10_weight_share=float(weights[order[:10]].sum()/weights.sum()),
                top01pct_weight_share=float(weights[order[:max(1, int(np.ceil(len(idx)*.001)))]].sum()/weights.sum()),
                max_day_weight_share=float(daily.max()/daily.sum()),
                missing_clock_rows=int(np.sum(missing[idx] != 0)),
                missing_scale_exact900=bool(np.all(value[missing[idx] != 0] == 900.)))
            guard()
        report['folds'][fold] = dict(fit_scale_squared_normalizer=normalizer, stages=parts,
            split_hash=common.object_hash(split), control_manifest_sha256=common.sha256(subject.risk.union_folder(fold) / 'manifest.json'))
    info = psutil.Process().memory_info()
    report['peak_wset_bytes'] = info.peak_wset
    report['protocol_sha256'] = common.sha256(OUT / 'protocol.json')
    common.write_json(OUT / 'receipt.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
