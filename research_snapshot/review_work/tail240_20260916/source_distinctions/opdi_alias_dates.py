"""Public observation offsets under explicit aliases and date hypotheses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import sys
import gc
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import audit_alias as alias

ROOT = alias.ROOT
common = alias.common
ID, TIME = alias.ID, alias.MOVEMENT
BASE = ROOT / 'private_runs/breakthrough_20260916/missing/opdi_rotation'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
QUERY_FIELDS = [ID, alias.FLIGHT_ID, alias.PHASE, TIME, 'FLIGHT_mvt', 'CALLSIGN_flt',
                'ADEP_mvt', 'ADES_mvt', 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt']
PUBLIC_FIELDS = ['id', 'icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen']


def lexical_prefix(prefix):
    if not isinstance(prefix, str):
        return None
    if len(prefix) > 3:
        first, tail = prefix[:3], prefix[3:]
        if len(tail) <= 3 and first.endswith(tail):
            return first
    return prefix


def query_codes(frame, mapping):
    clean, parts = alias.parts(frame.FLIGHT_mvt)
    learned = parts.prefix.map(mapping).fillna(parts.prefix)
    lexical = parts.prefix.map(lexical_prefix)
    result = {}
    for name, prefix in [('raw', parts.prefix), ('learned', learned), ('lexical', lexical)]:
        result[name] = prefix + parts.number + parts.suffix
    return pd.DataFrame(result, index=frame.index)


def normalize(series):
    return series.astype('string').fillna('').str.upper().str.replace(r'\s+', '', regex=True)


def emit_features(queries, public, mapping):
    pub = public.copy()
    for column in ('first_seen', 'last_seen'):
        pub[column] = pd.to_datetime(pub[column], utc=True, errors='coerce')
    for column in ('adep', 'ades', 'icao24'):
        pub[column] = normalize(pub[column])
    _, codes = alias.parts(pub.flt_id)
    pub['code'] = codes.prefix + codes.number + codes.suffix
    pub = pub.loc[pub.first_seen.notna() & pub.last_seen.gt(pub.first_seen) & pub.code.notna() & pub.icao24.ne('')]
    pub = pub.drop_duplicates(['code', 'adep', 'ades', 'first_seen', 'last_seen', 'icao24'])
    names = query_codes(queries, mapping)
    all_codes = set(pd.concat([names[c] for c in names]).dropna())
    pub = pub.loc[pub.code.isin(all_codes)]
    groups = {key: g.sort_values('first_seen') for key, g in pub.groupby(['code', 'adep', 'ades'], sort=False)}
    records, witnesses = [], []
    for pos, query in enumerate(queries.itertuples(index=False)):
        q = query._asdict()
        moment = q[TIME]
        record = {ID: q[ID]}
        for mode in names:
            code = names.iloc[pos][mode]
            key = (code, str(q['ADEP_mvt']).upper(), str(q['ADES_mvt']).upper())
            group = groups.get(key)
            for shift in (-1, 0, 1):
                prefix = f'opdi_{mode}_day{shift + 1}'
                record[prefix + '_count_30min'] = 0.
                for name in ('offset_sec', 'shift_residual_sec', 'duration_sec', 'candidate_margin_sec'):
                    record[prefix + '_' + name] = np.nan
                if group is None:
                    continue
                offset = (moment - group.first_seen).dt.total_seconds().to_numpy()
                distance = np.abs(offset - shift * 86400.)
                order = np.argsort(distance, kind='stable')
                within = distance <= 1800
                record[prefix + '_count_30min'] = float(within.sum())
                if len(order) == 0 or distance[order[0]] > 1800:
                    continue
                hit = group.iloc[order[0]]
                margin = distance[order[1]] - distance[order[0]] if len(order) > 1 else np.nan
                if len(order) > 1 and margin == 0:
                    continue
                record[prefix + '_offset_sec'] = offset[order[0]]
                record[prefix + '_shift_residual_sec'] = offset[order[0]] - shift * 86400
                record[prefix + '_duration_sec'] = (hit.last_seen - hit.first_seen).total_seconds()
                record[prefix + '_candidate_margin_sec'] = margin
                witnesses.append({ID: q[ID], 'mode': mode, 'shift': shift, 'public_id': hit['id'],
                    'code': code, 'first_seen': hit.first_seen, 'last_seen': hit.last_seen,
                    'offset_sec': offset[order[0]], 'count': int(within.sum()), 'margin_sec': margin})
        records.append(record)
    return pd.DataFrame(records).set_index(ID).astype('float32'), pd.DataFrame(witnesses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = common.external_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    acquisition = common.read_json(BASE / 'acquisition.json')
    assert acquisition['status'] == 'complete'
    public_paths = {row['month']: row for row in acquisition['sources']}
    raw_root = ROOT / 'data/09-15-2026-18-55-03_files_list'
    raw_hashes = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    protocol = {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'alias_source_sha256': common.sha256(alias.__file__), 'query_fields': QUERY_FIELDS,
        'hidden_columns_loaded': [], 'scope': 'All trainingmissingNM and500deterministicmaskedknownDEP per logicalUTCmonth. No ranking targets.',
        'modes': ['raw normalized code', 'earlierUTCmonths observedprefixmap', 'explicit repeatedsuffix lexicalcollapse'],
        'prefix_mapping': 'Earlier logicalUTCmonth known observations only,20distinctpairedFLIGHTIDs,98%dominance, same digit/suffix.',
        'offsets': 'For each mode emit nearestpublicflight firstseen about T+24h,T,T-24h within30min; keep all three hypotheses and support/margin, no scorechoice.',
        'available_public_context': 'All already downloaded public2025months; querymatches only within25h ofsuppliedtakeoff; public files straddling monthboundaries allowed. Retrospective batch, not realtime.',
        'first_seen_semantics': 'Observed-flight start, not offblock time. No true identity or date-shift claim from matching alone.',
        'licensing': 'Previously attributed local noncommercial research; prize/commercial reuse eligibility unresolved.',
        'acquisition_sha256': common.sha256(BASE / 'acquisition.json'), 'raw_hashes': raw_hashes,
        'prediction_model': False}
    common.write_json(out / 'protocol.json', protocol)
    frames = []
    for path in sorted(raw_root.glob('training_*.parquet')):
        assert common.sha256(path) == raw_hashes[path.name]
        frame = pq.read_table(path, columns=QUERY_FIELDS, use_threads=False).to_pandas()
        frame[TIME] = pd.to_datetime(frame[TIME], utc=True)
        frames.append(frame)
    observed = pd.concat(frames, ignore_index=True)
    del frames
    observed['month'] = observed[TIME].dt.strftime('%Y%m')
    observed = alias.prepare(observed)
    history, results, chunks, witnesses, query_rows = [], {}, [], [], []
    started = time.monotonic()
    for month in sorted(observed.month.unique()):
        part = observed.loc[observed.month.eq(month)]
        deps = part.loc[part[alias.PHASE].eq('DEP')]
        missing = deps.AOBT_3_flt.isna()
        known = deps.loc[~missing].sort_values(ID)
        sampled = known.iloc[np.linspace(0, len(known) - 1, min(500, len(known)), dtype=int)]
        query = pd.concat([deps.loc[missing], sampled]).sort_values(ID).reset_index(drop=True)
        mapping = alias.prefix_table(history)
        month_start = pd.Timestamp(month + '01', tz='UTC')
        wanted = [(month_start - pd.Timedelta(days=1)).strftime('%Y%m'), month,
                  (month_start + pd.offsets.MonthBegin(1)).strftime('%Y%m')]
        public = []
        for m in wanted:
            if m not in public_paths or not m.startswith('2025'):
                continue
            src = public_paths[m]
            path = Path(src['local_path'])
            assert common.sha256(path) == src['sha256']
            public.append(pq.read_table(path, columns=PUBLIC_FIELDS, use_threads=False).to_pandas())
        features, links = emit_features(query, pd.concat(public, ignore_index=True), mapping)
        assert features.index.is_unique and np.array_equal(features.index, query[ID])
        chunks.append(features)
        witnesses.append(links)
        query_rows.append(query[[ID, 'month', 'ADEP_mvt', 'FLIGHT_mvt', TIME, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt']])
        stats = {'queries': len(query), 'missing_queries': int(query.AOBT_3_flt.isna().sum()),
                 'mapping': mapping, 'coverage': {c: int(features[c].gt(0).sum()) for c in features if c.endswith('_count_30min')}}
        results[month] = stats
        print(month, {k: v for k, v in stats.items() if k != 'mapping'}, flush=True)
        same = part.mn.eq(part.cn).fillna(False) & part.ms.eq(part.cs).fillna(False)
        history.append(part.loc[same & part[alias.FLIGHT_ID].notna(), [alias.FLIGHT_ID, 'mp', 'cp']].dropna())
        del public, features, links
        gc.collect()
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise MemoryError('Host memory reserve')
    pd.concat(chunks).reset_index().to_parquet(out / 'features.parquet', index=False)
    pd.concat(witnesses, ignore_index=True).to_parquet(out / 'witnesses.parquet', index=False)
    pd.concat(query_rows, ignore_index=True).to_parquet(out / 'queries.parquet', index=False)
    common.write_json(out / 'manifest.json', {'status': 'complete', 'protocol_sha256': common.sha256(out / 'protocol.json'),
        'source_sha256': common.sha256(__file__), 'alias_source_sha256': common.sha256(alias.__file__),
        'months': results, 'features': list(chunks[0]), 'runtime_sec': time.monotonic() - started,
        'outputs': {name: common.sha256(out / name) for name in ('features.parquet', 'witnesses.parquet', 'queries.parquet')}})


if __name__ == '__main__':
    main()
