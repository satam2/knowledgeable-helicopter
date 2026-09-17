"""Masked-source clock recovery through earlier observed flight-code aliases."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.schema import ID, FLIGHT_ID, PHASE, MOVEMENT

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
FIELDS = [ID, FLIGHT_ID, PHASE, MOVEMENT, 'FLIGHT_mvt', 'CALLSIGN_flt',
          'ADEP_mvt', 'ADES_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'SCHED_TIME_UTC_mvt']


def parts(series):
    cleaned = series.astype('string').str.upper().str.replace(r'\s+', '', regex=True)
    split = cleaned.str.extract(r'^([A-Z]+)([0-9]+)([A-Z]*)$')
    split.columns = ['prefix', 'number', 'suffix']
    split['number'] = split['number'].str.lstrip('0').replace('', '0')
    return cleaned, split


def prefix_table(history):
    if not history:
        return {}
    data = pd.concat(history, ignore_index=True).drop_duplicates([FLIGHT_ID, 'mp', 'cp'])
    counts = data.groupby(['mp', 'cp'], observed=True).size()
    result = {}
    for prefix in counts.index.get_level_values(0).unique():
        row = counts.loc[prefix]
        winner = row.idxmax()
        if row.sum() >= 20 and row.loc[winner] / row.sum() >= .98:
            result[str(prefix)] = str(winner)
    return result


def prepare(frame):
    result = frame.copy()
    for source, stem in [('FLIGHT_mvt', 'm'), ('CALLSIGN_flt', 'c')]:
        clean, split = parts(result[source])
        result[stem + 'clean'] = clean
        for part, suffix in [('prefix', 'p'), ('number', 'n'), ('suffix', 's')]:
            result[stem + suffix] = split[part]
    return result


def candidates(query, arrivals, mapping):
    q = query[[ID, MOVEMENT, 'SCHED_TIME_UTC_mvt', 'ADEP_mvt', 'ADES_mvt',
               'mclean', 'mp', 'mn', 'ms']].copy()
    q['alias_prefix'] = q.mp.map(mapping)
    q['day'] = q[MOVEMENT].dt.floor('D')
    a = arrivals.dropna(subset=[FLIGHT_ID, 'AOBT_3_flt', 'ADEP_mvt', 'ADES_mvt']).copy()
    a = a.sort_values(ID).drop_duplicates([FLIGHT_ID, 'AOBT_3_flt'])
    a['arrival_day'] = a[MOVEMENT].dt.floor('D')
    joined = []
    for mode in ('exact', 'alias'):
        if mode == 'exact':
            left = ['ADEP_mvt', 'ADES_mvt', 'mclean', 'day']
            right = ['ADEP_mvt', 'ADES_mvt', 'mclean', 'day']
        else:
            left = ['ADEP_mvt', 'ADES_mvt', 'alias_prefix', 'mn', 'ms', 'day']
            right = ['ADEP_mvt', 'ADES_mvt', 'cp', 'cn', 'cs', 'day']
        for shift in (0, -1):
            shifted = a.copy()
            shifted['day'] = shifted.arrival_day + pd.Timedelta(days=shift)
            pair = q.dropna(subset=left).merge(shifted.dropna(subset=right),
                left_on=left, right_on=right, suffixes=('_query', '_arrival'), how='inner')
            if not len(pair):
                continue
            airborne = (pair[MOVEMENT + '_arrival'] - pair[MOVEMENT + '_query']).dt.total_seconds()
            proxy = (pair[MOVEMENT + '_query'] - pair['AOBT_3_flt']).dt.total_seconds()
            valid = airborne.between(0, 18 * 3600) & proxy.between(0, 48 * 3600)
            pair = pair.loc[valid].copy()
            pair['candidate_proxy_sec'] = proxy.loc[valid]
            pair['airborne_sec'] = airborne.loc[valid]
            pair['match_mode'] = mode
            joined.append(pair)
    if not joined:
        return pd.DataFrame()
    return pd.concat(joined, ignore_index=True).drop_duplicates(
        [ID + '_query', FLIGHT_ID, 'AOBT_3_flt', 'match_mode'])


def summarize(frame, pairs, mode):
    subset = pairs.loc[pairs.match_mode.eq(mode)].copy() if len(pairs) else pairs
    counts = subset.groupby(ID + '_query').size() if len(subset) else pd.Series(dtype=int)
    selected = subset.loc[subset[ID + '_query'].isin(counts.index[counts.eq(1)])].copy() if len(subset) else subset
    source = frame.set_index(ID)
    report = {}
    for missing in (True, False):
        ids = source.index[source.AOBT_3_flt.isna().eq(missing)]
        kept = selected.loc[selected[ID + '_query'].isin(ids)].copy() if len(selected) else selected
        stats = {'rows': len(ids), 'candidate_rows': int(counts.reindex(ids, fill_value=0).gt(0).sum()),
                 'unique_rows': len(kept), 'ambiguous_rows': int(counts.reindex(ids, fill_value=0).gt(1).sum())}
        if len(kept):
            query_ids = kept[ID + '_query']
            stats['by_airport'] = kept.groupby('ADEP_mvt', observed=True).size().to_dict()
            if not missing:
                original = source.loc[query_ids, 'AOBT_3_flt'].reset_index(drop=True)
                recovered = kept.AOBT_3_flt.reset_index(drop=True)
                delta = (recovered - original).dt.total_seconds().to_numpy()
                stats.update(exact_aobt_fraction=float(np.mean(delta == 0)),
                             within60_aobt_fraction=float(np.mean(np.abs(delta) <= 60)),
                             aobt_error_rmse=float(np.sqrt(np.mean(delta ** 2))),
                             exact_flight_id_fraction=float(np.mean(source.loc[query_ids, FLIGHT_ID].to_numpy() == kept[FLIGHT_ID].to_numpy())))
        report['missing' if missing else 'known_masked'] = stats
    return report, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = common.external_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    raw = common.WORKSPACE / 'data/09-15-2026-18-55-03_files_list'
    frozen = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    protocol = {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'fields': FIELDS, 'hidden_columns_loaded': [], 'mapping': 'Earlier UTC training months only; same normalized digit/suffix in observed flight+callsign pair; >=20 distinct NM flights and >=98% dominant prefix mapping.',
        'matching': 'Route+normalized number+suffix+learned callsign prefix; same-month arrivals, arrival after takeoff within18h, candidate takeoff-minus-NMoffblock0..48h; exactlyonejourney only.',
        'control': 'Same constraints with exact normalized raw airportflight string.',
        'availability': 'Retrospective arrivals may occur after query; no query NMidentity/clocks used in construction. Known clocks used only in masked validation afterward.',
        'selection': 'No taxi labels or score feedback used. Clock precision on masked known source does not establish missing-source correctness.',
        'raw_hashes': frozen, 'prediction_model': False}
    common.write_json(out / 'protocol.json', protocol)
    history, results, selections = [], {}, []
    began = time.monotonic()
    for path in sorted(raw.glob('training_*.parquet')):
        assert common.sha256(path) == frozen[path.name]
        frame = prepare(pq.read_table(path, columns=FIELDS, use_threads=False).to_pandas())
        mapping = prefix_table(history)
        dep = frame.loc[frame[PHASE].eq('DEP')]
        arr = frame.loc[frame[PHASE].eq('ARR')]
        pairs = candidates(dep, arr, mapping)
        report = {'mapping': mapping, 'mapping_count': len(mapping)}
        for mode in ('exact', 'alias'):
            report[mode], selected = summarize(dep, pairs, mode)
            if len(selected):
                selected['raw_file'] = path.name
                selected['query_nm_missing'] = selected[ID + '_query'].map(dep.set_index(ID).AOBT_3_flt.isna())
                selections.append(selected[[ID + '_query', ID + '_arrival', FLIGHT_ID, 'AOBT_3_flt',
                    'candidate_proxy_sec', 'airborne_sec', 'match_mode', 'raw_file', 'query_nm_missing']])
        same_number = frame.mn.eq(frame.cn).fillna(False) & frame.ms.eq(frame.cs).fillna(False)
        known = frame.loc[same_number & frame[FLIGHT_ID].notna(), [FLIGHT_ID, 'mp', 'cp']].dropna()
        history.append(known)
        results[path.name] = report
        print(path.name, {mode: report[mode] for mode in ('exact', 'alias')}, flush=True)
    if selections:
        pd.concat(selections, ignore_index=True).to_parquet(out / 'selected.parquet', index=False)
    common.write_json(out / 'summary.json', {'status': 'complete', 'protocol_sha256': common.sha256(out / 'protocol.json'),
        'runtime_sec': time.monotonic() - began, 'months': results,
        'output_sha256': common.sha256(out / 'selected.parquet') if selections else None})


if __name__ == '__main__':
    main()
