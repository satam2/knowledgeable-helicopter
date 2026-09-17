"""Covered-row NM-error oracle budget; no model or matching changes."""
import surface_diagnostic_v1 as metrics
import numpy as np
import pandas as pd

ROOT, common, ID, TARGET, TIME = metrics.ROOT, metrics.common, metrics.ID, metrics.TARGET, metrics.TIME
BASE = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot'
OUT = BASE / 'oracle_budget_v1'


def summarize(frame):
    finite = np.isfinite(frame.proxy_sec.to_numpy(float))
    covered = finite & frame.unique_exit_offset_sec.notna().to_numpy()
    errors = frame[TARGET].to_numpy(float) - frame.proxy_sec.to_numpy(float)
    total = float(np.sum(errors[finite] ** 2))
    removable = float(np.sum(errors[covered] ** 2))
    n = int(finite.sum())
    baseline = float(np.sqrt(total / n)) if n else None
    oracle = float(np.sqrt(max(0., total - removable) / n)) if n else None
    return dict(queries=len(frame), finite_nm=n, missing_nm=int((~finite).sum()),
        covered_finite=int(covered.sum()), covered_missing=int((~finite & frame.unique_exit_offset_sec.notna().to_numpy()).sum()),
        nm_sse=total, removable_nm_sse=removable, removable_fraction=removable/total if total else None,
        nm_rmse=baseline, oracle_rmse=oracle, oracle_gain=baseline-oracle if n else None)


def main():
    assert not OUT.exists()
    assert metrics.psutil.virtual_memory().available >= 10 * 1024**3
    source = BASE / 'dual_diagnostic_v2'
    marker = common.read_json(source / 'summary.json')
    assert marker['status'] == 'complete'
    queries_path = BASE / 'join_v2/queries.parquet'
    query_marker = common.read_json(BASE / 'join_v2/manifest.json')
    assert common.sha256(queries_path) == query_marker['outputs']['queries.parquet']
    queries = pd.read_parquet(queries_path, columns=[ID, TIME])
    results = {}
    for arm in ['first_seen', 'takeoff_event']:
        path = source / (arm + '_rows.parquet')
        assert common.sha256(path) == marker['outputs'][path.name]
        rows = pd.read_parquet(path).merge(queries, on=ID, validate='one_to_one')
        assert len(rows) == 49522
        rows['utc_day'] = rows[TIME].dt.strftime('%Y-%m-%d')
        results[arm] = dict(all=summarize(rows), airports={a:summarize(g) for a,g in rows.groupby('ADEP_mvt')},
            days={d:summarize(g) for d,g in rows.groupby('utc_day')},
            airport_days={a+'|'+d:summarize(g) for (a,d),g in rows.groupby(['ADEP_mvt','utc_day'])},
            missing_covered=rows.loc[rows.proxy_sec.isna() & rows.unique_exit_offset_sec.notna(),
                [ID,'ADEP_mvt','utc_day',TARGET,'unique_exit_offset_sec']].to_dict('records'))
    OUT.mkdir()
    common.write_json(OUT / 'receipt.json', dict(status='complete', source_sha256=common.sha256(__file__),
        input_summary_sha256=common.sha256(source/'summary.json'), inputs=marker['outputs'], query_sha256=common.sha256(queries_path),
        method='Oracle replaces NM with exact raw Y only on frozen unique-exit finite-NM rows, leaves every other finite-NM row unchanged. Upper bound relative to raw NM, not a trained-model result. Missing NM rows retained separately.',
        results=results, peak_bytes=metrics.guard(), no_model=True, no_new_match=True, no_score=True))
    print({arm:r['all'] for arm,r in results.items()}, flush=True)


if __name__ == '__main__':
    main()
