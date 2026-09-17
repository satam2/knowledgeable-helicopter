"""Read-only diagnostic of existing masked-known OPDI fit/tune samples."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
ID, TIME, TARGET = common.ID, common.MOVEMENT, common.TARGET
CACHE = ROOT / 'private_runs/tail240_20260916/source_distinctions/opdi_dates_v1'
ORACLE = ROOT / 'private_runs/tail240_20260916/validation/opdi_dates_oracle_v3'
OUT = ROOT / 'private_runs/tail240_20260916/forensics/opdi_known_sample_v1'


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    return dict(n=len(values), mean=float(values.mean()),
        quantiles={str(q): float(v) for q, v in zip([0, .01, .1, .5, .9, .99, 1], np.quantile(values, [0, .01, .1, .5, .9, .99, 1]))}) if len(values) else dict(n=0)


def summarize(query, joined, mode):
    prefix = f'opdi_{mode}_day1_'
    offset = query[prefix + 'offset_sec']
    matched = offset.notna()
    groups = {}
    for label, mask in [('resolved', matched), ('unique', matched & query[prefix + 'count_30min'].eq(1))]:
        selected = query.loc[mask]
        offset_values = selected[prefix + 'offset_sec'].to_numpy(float)
        y = selected[TARGET].to_numpy(float)
        proxy = selected.proxy_sec.to_numpy(float)
        nm_error = y - proxy
        opdi_error = y - offset_values
        gap = np.abs(nm_error) > 1800
        groups[label] = dict(n=len(selected), fraction=float(len(selected) / len(query)),
            offset_sec=distribution(offset_values), nm_proxy_sec=distribution(proxy), raw_y_sec=distribution(y),
            y_minus_nm=distribution(nm_error), y_minus_opdi_offset=distribution(opdi_error),
            nm_rmse=float(np.sqrt(np.mean(nm_error**2))) if len(y) else None,
            opdi_offset_rmse=float(np.sqrt(np.mean(opdi_error**2))) if len(y) else None,
            nm_mae=float(np.mean(np.abs(nm_error))) if len(y) else None,
            opdi_offset_mae=float(np.mean(np.abs(opdi_error))) if len(y) else None,
            first_seen_more_than600sec_before_takeoff=int((offset_values > 600).sum()),
            first_seen_more_than600sec_after_takeoff=int((offset_values < -600).sum()),
            offset_within600sec=int((np.abs(offset_values) <= 600).sum()),
            offset_gt600_and_before_nm_aobt=int(((offset_values > 600) & (offset_values > proxy)).sum()),
            opdi_offset_closer_to_y=int((np.abs(opdi_error) < np.abs(nm_error)).sum()),
            raw_nm_gap_gt1800=int(gap.sum()), gap_and_opdi_closer=int((gap & (np.abs(opdi_error) < np.abs(nm_error))).sum()),
            duration_sec=distribution(selected[prefix + 'duration_sec']))
        witnesses = joined.loc[(joined['mode'] == mode) & joined[ID].isin(selected[ID])]
        assert len(witnesses) == len(selected)
    return dict(queries=len(query), coverage_count_gt0=int(query[prefix + 'count_30min'].gt(0).sum()),
        resolved_count=int(matched.sum()), ambiguous_candidates=int(query[prefix + 'count_30min'].gt(1).sum()), groups=groups)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    marker = common.read_json(CACHE / 'manifest.json')
    for name, expected in marker['outputs'].items():
        assert common.sha256(CACHE / name) == expected
    oracle = common.read_json(ORACLE / 'receipt.json')
    exact_path = ORACLE / 'recovered_exact_witness_ids.parquet'
    assert common.sha256(exact_path) == oracle['recovered_exact_witness_ids_sha256']
    protocol = dict(source_sha256=common.sha256(__file__), cache_manifest_sha256=common.sha256(CACHE / 'manifest.json'),
        oracle_receipt_sha256=common.sha256(ORACLE / 'receipt.json'),
        scope='Existing deterministic masked-known500/month sample only, retained original full fold fit/tune IDs after flight-ID purges. No score-stage analysis, new matching, downloads, fits, or selected date offsets.',
        primary='Same-day shift0 raw/learned/lexical observations, report resolved and uniquely matched separately. All finite NM sample retained; ordinary proxy[0,7200] reported separately.',
        comparisons='Paired raw Y-minus-OPDI offset versus raw Y-minus-NM proxy on identical resolved rows, for information value only. T-first_seen is a surveillance boundary offset, not offblock.',
        limitation='Small deterministic ID-spaced sample is not random; cannot establish rare-gap capture, complete known-flight coverage, exact identity or ground-phase coverage.')
    common.write_json(OUT / 'protocol.json', protocol)
    query = pd.read_parquet(CACHE / 'queries.parquet')
    known = query.loc[query.AOBT_3_flt.notna()].copy()
    assert len(known) == 6000 and known[ID].is_unique
    features = pd.read_parquet(CACHE / 'features.parquet')
    known = known.merge(features, on=ID, how='left', validate='one_to_one')
    witnesses = pd.read_parquet(CACHE / 'witnesses.parquet')
    witnesses = witnesses.loc[witnesses['shift'].eq(0) & witnesses[ID].isin(known[ID])].copy()
    exact = pd.read_parquet(exact_path)
    witnesses = witnesses.merge(exact, on=[ID, 'mode', 'shift'], how='left', validate='one_to_one')
    assert witnesses.public_id_exact.notna().all()
    witness_queries = witnesses.merge(known[[ID, TIME]], on=ID, validate='many_to_one')
    np.testing.assert_array_equal((witness_queries[TIME] - witness_queries.first_seen).dt.total_seconds(), witness_queries.offset_sec)
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, 'FLIGHT_ID_mvt', TIME, 'ADEP_mvt', 'proxy_sec'])
    meta[TARGET] = np.nan
    selections, splits = {}, {}
    allowed = set()
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        splits[fold] = split
        for stage in ('fit', 'tune'):
            ids = meta.iloc[idx[stage]][ID]
            ids = known.loc[known[ID].isin(ids), ID]
            selections[(fold, stage)] = ids
            allowed.update(ids)
    # Labels are loaded only for the union of explicitly eligible sample IDs.
    labels = pd.read_parquet(meta_path, columns=[ID, TARGET, 'proxy_sec'], filters=[(ID, 'in', sorted(allowed))])
    known = known.loc[known[ID].isin(allowed)].merge(labels, on=ID, validate='one_to_one')
    assert np.isfinite(known.proxy_sec).all()
    np.testing.assert_array_equal((known[TIME] - known.AOBT_3_flt).dt.total_seconds(), known.proxy_sec)
    results = {}
    for (fold, stage), ids in selections.items():
        rows = known.loc[known[ID].isin(ids)].copy()
        entry = {}
        for slice_name, frame in [('all_finite', rows), ('ordinary', rows.loc[rows.proxy_sec.between(0, 7200)])]:
            entry[slice_name] = {mode: summarize(frame, witnesses, mode) for mode in ('raw', 'learned', 'lexical')}
        entry['month_coverage'] = []
        for month, group in rows.groupby('month'):
            item = dict(month=month, queries=len(group))
            for mode in ('raw', 'learned', 'lexical'):
                offsets = group[f'opdi_{mode}_day1_offset_sec']
                item[mode] = dict(resolved=int(offsets.notna().sum()), before_gt600=int(offsets.gt(600).sum()))
            entry['month_coverage'].append(item)
        results[f'{fold}_{stage}'] = entry
        print(fold, stage, {mode: entry['ordinary'][mode]['groups']['resolved'] for mode in ('raw', 'learned', 'lexical')}, flush=True)
    known.to_parquet(OUT / 'eligible_known_rows.parquet', index=False)
    witnesses.loc[witnesses[ID].isin(allowed)].to_parquet(OUT / 'eligible_exact_witnesses.parquet', index=False)
    memory = psutil.Process().memory_info()
    assert getattr(memory, 'peak_wset', memory.rss) < 2 * 1024**3 and psutil.virtual_memory().available >= 8 * 1024**3
    common.write_json(OUT / 'summary.json', dict(status='complete', source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT / 'protocol.json'), eligible_unique_known=len(known), results=results, splits=splits,
        peak_bytes=getattr(memory, 'peak_wset', memory.rss), score_stage_analyzed=False, no_external_requests=True,
        outputs={name: common.sha256(OUT / name) for name in ['eligible_known_rows.parquet', 'eligible_exact_witnesses.parquet']}))


if __name__ == '__main__':
    main()
