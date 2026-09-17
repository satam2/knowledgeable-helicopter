"""Independent raw-record oracle and UTC-month boundary audit; no labels read."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'private_runs/tail240_20260916/source_distinctions/alias_v1'
OUT = ROOT / 'private_runs/tail240_20260916/validation/alias_oracle_v3'
ID, FID, PHASE, TIME = 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt'
FIELDS = [ID, FID, PHASE, TIME, 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'ADES_mvt', 'AOBT_3_flt']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def norm(series):
    clean = series.astype('string').str.upper().str.replace(r'\s+', '', regex=True)
    parts = clean.str.extract(r'^([A-Z]+)([0-9]+)([A-Z]*)$')
    parts[1] = parts[1].str.lstrip('0').replace('', '0')
    return clean, parts


def mapping(data):
    data = data.drop_duplicates([FID, 'mp', 'cp'])
    counts = data.groupby(['mp', 'cp']).size()
    answer = {}
    for prefix in counts.index.get_level_values(0).unique():
        row = counts.loc[prefix]
        if row.sum() >= 20 and row.max() / row.sum() >= .98:
            answer[prefix] = row.idxmax()
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = json.loads((BASE / 'protocol.json').read_text())
    summary = json.loads((BASE / 'summary.json').read_text())
    assert sha(BASE / 'selected.parquet') == summary['output_sha256']
    selected = pd.read_parquet(BASE / 'selected.parquet')
    assert not selected.duplicated([ID + '_query', 'match_mode']).any()
    history, results = [], {}
    samples_checked = 0
    for filename, report in summary['months'].items():
        path = ROOT / 'data/09-15-2026-18-55-03_files_list' / filename
        assert sha(path) == protocol['raw_hashes'][filename]
        raw = pq.read_table(path, columns=FIELDS, use_threads=False).to_pandas()
        assert raw[ID].is_unique
        raw[TIME] = pd.to_datetime(raw[TIME], utc=True)
        raw['month'] = raw[TIME].dt.strftime('%Y-%m')
        raw['mc'], mp = norm(raw.FLIGHT_mvt)
        raw['cc'], cp = norm(raw.CALLSIGN_flt)
        for prefix, parts in [('m', mp), ('c', cp)]:
            for field, num in [('p', 0), ('n', 1), ('s', 2)]:
                raw[prefix + field] = parts[num]
        dep = raw.loc[raw[PHASE].eq('DEP')].set_index(ID)
        arr = raw.loc[raw[PHASE].eq('ARR')].set_index(ID)
        pack = selected.loc[selected.raw_file.eq(filename)]
        q, a = dep.loc[pack[ID+'_query']].reset_index(), arr.loc[pack[ID+'_arrival']].reset_index()
        for key in ('ADEP_mvt', 'ADES_mvt'):
            assert q[key].equals(a[key])
        np.testing.assert_array_equal(a[FID], pack[FID])
        np.testing.assert_array_equal(a.AOBT_3_flt, pack.AOBT_3_flt)
        np.testing.assert_array_equal(q.AOBT_3_flt.isna(), pack.query_nm_missing)
        airborne = (a[TIME] - q[TIME]).dt.total_seconds()
        proxy = (q[TIME] - a.AOBT_3_flt).dt.total_seconds()
        np.testing.assert_array_equal(airborne, pack.airborne_sec)
        np.testing.assert_array_equal(proxy, pack.candidate_proxy_sec)
        assert airborne.between(0, 64800).all() and proxy.between(0, 172800).all()
        exact = pack.match_mode.eq('exact').to_numpy()
        assert q.loc[exact, 'mc'].equals(a.loc[exact, 'mc'])
        alias = ~exact
        for left, right in [('mp', 'cp'), ('mn', 'cn'), ('ms', 'cs')]:
            actual = q.loc[alias, left].map(report['mapping']) if left == 'mp' else q.loc[alias, left]
            np.testing.assert_array_equal(actual.to_numpy(), a.loc[alias, right].to_numpy())
        row = {'selected_rows': len(pack), 'dep_movement_months': dep.month.value_counts().to_dict(),
               'arr_movement_months': arr.month.value_counts().to_dict(),
               'selected_query_outside_pack_month': int(q.month.ne(filename[9:16]).sum()),
               'selected_arrival_other_query_month': int(q.month.ne(a.month).sum()),
               'chronology_differences': {}}
        if history:
            old = pd.concat(history, ignore_index=True)
            for month in dep.month.unique():
                disallowed = old.month.ge(month)
                if disallowed.any():
                    strict = mapping(old.loc[~disallowed])
                    affected = alias & q.month.eq(month).to_numpy()
                    mapped = q.loc[affected, 'mp'].map(strict)
                    changed = mapped.ne(a.loc[affected, 'cp']).fillna(True)
                    row['chronology_differences'][month] = {'history_pair_rows_not_earlier_utc_month': int(disallowed.sum()),
                        'selected_alias_rows': int(affected.sum()), 'strict_mapping_changed_selected_rows': int(changed.sum())}
            del old
        pool = arr.reset_index().dropna(subset=[FID, 'AOBT_3_flt', 'ADEP_mvt', 'ADES_mvt']).sort_values(ID).drop_duplicates([FID, 'AOBT_3_flt'])
        for _, group in pack.groupby(['match_mode', 'query_nm_missing']):
            positions = np.unique(np.linspace(0, len(group)-1, min(4, len(group)), dtype=int))
            for _, chosen in group.iloc[positions].iterrows():
                query = dep.loc[chosen[ID+'_query']]
                possible = pool.loc[pool.ADEP_mvt.eq(query.ADEP_mvt) & pool.ADES_mvt.eq(query.ADES_mvt)]
                if chosen.match_mode == 'exact':
                    mask = possible.mc.eq(query.mc)
                else:
                    mask = possible.cp.eq(report['mapping'].get(query.mp)) & possible.cn.eq(query.mn) & possible.cs.eq(query.ms)
                airborne = (possible[TIME] - query[TIME]).dt.total_seconds()
                proxy = (query[TIME] - possible.AOBT_3_flt).dt.total_seconds()
                day_delta = (possible[TIME].dt.floor('D') - query[TIME].floor('D')).dt.days
                found = possible.loc[mask & airborne.between(0,64800) & proxy.between(0,172800) & day_delta.isin([0,1])]
                assert len(found) == 1 and found.iloc[0][ID] == chosen[ID+'_arrival']
                samples_checked += 1
        same = raw.mn.eq(raw.cn).fillna(False) & raw.ms.eq(raw.cs).fillna(False)
        known = raw.loc[same & raw[FID].notna(), [FID, 'mp', 'cp', 'month']].dropna().drop_duplicates()
        conflicts = known.groupby([FID, 'mp']).cp.nunique()
        row['within_pack_fid_prefix_with_conflicting_callsign_prefix'] = int(conflicts.gt(1).sum())
        history.append(known)
        results[filename] = row
        assert psutil.Process().memory_info().rss < 4*1024**3
        print(filename, row['selected_rows'], row['chronology_differences'], flush=True)
        del raw, dep, arr, pool, q, a
    receipt = {'status': 'passed_raw_matching_oracle_with_protocol_caveat', 'source_sha256': sha(__file__),
        'audited_source_sha256': protocol['source_sha256'], 'audit_protocol_sha256': sha(BASE/'protocol.json'),
        'audit_output_sha256': sha(BASE/'selected.parquet'), 'selected_rows_checked': len(selected),
        'independent_brute_force_uniqueness_samples': samples_checked, 'months': results,
        'runtime_sec': time.monotonic()-start, 'rss_bytes': psutil.Process().memory_info().rss,
        'limits': 'No target labels loaded. Candidate completeness tested only for deterministic samples within original raw packs. Prefix chronology counted independently; file packs are not strict UTC months. Missing-source correctness and predictive benefit not established.'}
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETE', receipt['selected_rows_checked'], samples_checked, flush=True)


if __name__ == '__main__':
    main()
