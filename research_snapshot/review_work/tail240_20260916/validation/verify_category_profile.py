"""Independent bounded category-profile checks; no producer imports or models."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'review_work/tail240_20260916/forensics/preprocessing_audit'
AUDIT = ROOT / 'private_runs/tail240_20260916/forensics/preprocessing_audit/v1'
OUT = ROOT / 'private_runs/tail240_20260916/validation/category_profile_v1'
PROJECTION = ['PHASE_mvt', 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt']
FIELDS = ['ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
          'FLIGHT_RULE_mvt', 'FLIGHT_mvt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt',
          'AIRCRAFT_TYPE_flt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_RULE_flt',
          'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt', 'WK_TBL_CAT_flt', 'CALLSIGN_flt']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def guard():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, 'peak_wset', memory.rss))
    assert peak < 2 * 1024**3
    assert psutil.virtual_memory().available >= 8 * 1024**3
    return peak


def check_summary(rows, expected):
    assert len({(r['kind'], r['raw']) for r in rows}) == len(rows)
    assert all(isinstance(r['rows'], int) and r['rows'] > 0 for r in rows)
    assert all(r['kind'] in ('str', 'null') for r in rows)
    assert sum(r['rows'] for r in rows) == expected['rows']
    assert sum(r['rows'] for r in rows if r['kind'] == 'null') == expected['null_rows']
    valid = [r for r in rows if r['kind'] != 'null']
    assert len(valid) == expected['distinct_nonnull']
    kinds = Counter()
    for row in rows:
        kinds[row['kind']] += row['rows']
    assert dict(kinds) == expected['python_types']
    transformations = {
        'trim': str.strip,
        'ascii_case': lambda s: s.upper() if s.isascii() else s,
        'nfc': lambda s: unicodedata.normalize('NFC', s),
        'nfkc': lambda s: unicodedata.normalize('NFKC', s),
        'remove_whitespace': lambda s: ''.join(s.split()),
        'numeric_hypothesis': lambda s: str(int(s.split('.')[0])) if re.fullmatch(r'[0-9]+(?:\.0+)?', s) else s,
    }
    for name, transform in transformations.items():
        groups = defaultdict(list)
        changed = 0
        for row in valid:
            value = transform(row['raw'])
            changed += row['rows'] if value != row['raw'] else 0
            groups[value].append(row)
        collisions = [{'normalized': value, 'forms': forms, 'rows': sum(r['rows'] for r in forms)}
                      for value, forms in groups.items() if len(forms) > 1]
        collisions.sort(key=lambda r: (-r['rows'], r['normalized']))
        calculated = dict(changed_rows=changed, collision_groups=len(collisions),
                          collision_rows=sum(r['rows'] for r in collisions), largest_collisions=collisions[:30])
        assert calculated == expected['normalizations'][name], name


def entries(counts):
    return [dict(kind='null' if value is None else 'str', raw='' if value is None else value, rows=count)
            for value, count in counts.items()]


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    guard()
    receipt, protocol, summary = [read(AUDIT / name) for name in ('receipt.json', 'protocol.json', 'summary.json')]
    counts = read(AUDIT / 'raw_category_counts.json')
    assert protocol['fields'] == ['PHASE_mvt', *FIELDS]
    assert digest(SOURCE / 'run_v1.py') == protocol['source_sha256'] == receipt['source_sha256']
    assert digest(SOURCE / 'test_v1.py') == protocol['test_sha256']
    assert digest(AUDIT / 'protocol.json') == receipt['protocol_sha256'] == summary['protocol_sha256']
    for name, expected in receipt['outputs'].items():
        assert digest(AUDIT / name) == expected, name
    assert set(counts) == set(FIELDS) == set(summary['fields'])
    for field in FIELDS:
        check_summary(counts[field], summary['fields'][field])
    frozen = read(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    total = {name: Counter() for name in PROJECTION[1:]}
    by_airport = defaultdict(Counter)
    pack_counts = []
    for pack in protocol['packs']:
        path = ROOT / 'data/09-15-2026-18-55-03_files_list' / pack
        actual_hash = digest(path)
        assert actual_hash == protocol['raw_hashes'][pack] == frozen[pack]
        file = pq.ParquetFile(path)
        assert all(str(file.schema_arrow.field(field).type) == 'string' for field in FIELDS)
        n = 0
        local = {name: Counter() for name in PROJECTION[1:]}
        for batch in file.iter_batches(batch_size=8192, columns=PROJECTION, use_threads=False):
            columns = batch.to_pydict()
            for phase, airport, stand, runway in zip(*(columns[name] for name in PROJECTION)):
                if phase != 'DEP':
                    continue
                n += 1
                for field, value in [('ADEP_mvt', airport), ('STAND_mvt', stand), ('RUNWAY_mvt', runway)]:
                    local[field][value] += 1
                    total[field][value] += 1
                    if field != 'ADEP_mvt':
                        by_airport[(field, str(airport))][value] += 1
            guard()
        pack_summary = read(AUDIT / (path.stem + '_profile.json'))
        assert n == pack_summary['dep_rows']
        for field in local:
            check_summary(entries(local[field]), pack_summary['fields'][field])
        pack_counts.append(n)
        print(pack, n, flush=True)
    assert pack_counts == [row['dep_rows'] for row in summary['packs']]
    assert sum(pack_counts) == summary['total_departures'] == 822377
    for field, actual in total.items():
        expected = {None if row['kind'] == 'null' else row['raw']: row['rows'] for row in counts[field]}
        assert dict(actual) == expected, field
    assert len(by_airport) == len(summary['airport_locations'])
    for item in summary['airport_locations']:
        check_summary(entries(by_airport[(item['field'], item['airport'])]), item['summary'])
    literals = dict(stand_null=total['STAND_mvt'][None], stand_UNKNOWN=total['STAND_mvt']['UNKNOWN'],
                    runway_null=total['RUNWAY_mvt'][None], runway_NA=total['RUNWAY_mvt']['NA'])
    assert literals == dict(stand_null=16, stand_UNKNOWN=622, runway_null=0, runway_NA=4)
    OUT.mkdir(parents=True, exist_ok=False)
    result = dict(status='passed', verifier_sha256=digest(Path(__file__)),
                  audit_receipt_sha256=digest(AUDIT / 'receipt.json'), source_sha256=receipt['source_sha256'],
                  protocol_sha256=receipt['protocol_sha256'], raw_hashes=protocol['raw_hashes'],
                  outputs=receipt['outputs'], raw_projection=PROJECTION, pack_dep_counts=pack_counts,
                  all_17_stored_count_normalizations_exact=True, raw_three_fields_exact=True,
                  airport_location_summaries_exact=True, literals=literals, peak_bytes=guard(),
                  no_model_or_gpu=True, no_labels_ids_or_clocks_projected=True,
                  scope='Five physical Jan-May packs, DEP only. No logical-time or fit-purge selection. Literal meanings remain unconfirmed.')
    (OUT / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
