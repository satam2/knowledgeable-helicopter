"""Observed categorical profile in five early packs; no targets or clocks."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[4]
RAW=ROOT/'data/09-15-2026-18-55-03_files_list'
OUT=ROOT/'private_runs/tail240_20260916/forensics/preprocessing_audit/v1'
FIELDS=['ADEP_mvt','ADES_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_RULE_mvt',
        'FLIGHT_mvt','ADEP_flt','ADES_flt','ADES_FILED_flt','AIRCRAFT_TYPE_flt',
        'AIRCRAFT_OPERATOR_flt','FLIGHT_RULE_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt',
        'WK_TBL_CAT_flt','CALLSIGN_flt']
MISSING_STRINGS={'','na','n/a','nan','null','none','missing','unknown','unk','-','--','?','not available'}
CAP=3*1024**3


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')


def guard():
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    assert max(peak,info.rss)<CAP,'3GiB process ceiling'
    assert psutil.virtual_memory().available>=8*1024**3,'Host reserve below8GiB'
    return int(peak)


def typed(value):
    return (type(value).__name__,str(value)) if value is not None else ('null','')


def normalize(text,mode):
    if mode=='trim': return text.strip()
    if mode=='ascii_case': return text.upper() if text.isascii() else text
    if mode=='nfc': return unicodedata.normalize('NFC',text)
    if mode=='nfkc': return unicodedata.normalize('NFKC',text)
    if mode=='remove_whitespace': return ''.join(text.split())
    if mode=='numeric_hypothesis':
        match=re.fullmatch(r'([0-9]+)(?:\.0+)?',text)
        return str(int(match[1])) if match else text
    raise ValueError(mode)


def summarize(counts):
    nonnull={key:n for key,n in counts.items() if key[0]!='null'}
    type_counts=Counter()
    for (kind,_),count in counts.items():
        type_counts[kind]+=count
    flags=Counter()
    witnesses=defaultdict(list)
    for (kind,text),count in nonnull.items():
        found=[]
        if text!=text.strip(): found.append('edge_whitespace')
        if any(ch.isspace() for ch in text): found.append('any_whitespace')
        if any(ch.islower() for ch in text): found.append('lowercase')
        if not text.isascii(): found.append('non_ascii')
        if any(unicodedata.category(ch).startswith('C') for ch in text): found.append('control_or_format')
        if text.strip().casefold() in MISSING_STRINGS: found.append('missing_like_literal_unconfirmed')
        if re.fullmatch(r'0[0-9]+',text): found.append('integer_leading_zero')
        if re.fullmatch(r'[0-9]+\.0+',text): found.append('integer_decimal_spelling')
        if text!=unicodedata.normalize('NFC',text): found.append('nfc_changes')
        if text!=unicodedata.normalize('NFKC',text): found.append('nfkc_changes')
        for flag in found:
            flags[flag]+=count
            if len(witnesses[flag])<20: witnesses[flag].append(dict(kind=kind,raw=text,rows=count))
    results={}
    for mode in ['trim','ascii_case','nfc','nfkc','remove_whitespace','numeric_hypothesis']:
        groups=defaultdict(list)
        changed_rows=0
        for (kind,text),count in nonnull.items():
            transformed=normalize(text,mode)
            changed_rows+=count*(text!=transformed)
            groups[transformed].append(dict(kind=kind,raw=text,rows=count))
        collisions=[dict(normalized=key,forms=values,rows=sum(v['rows'] for v in values))
                    for key,values in groups.items() if len(values)>1]
        collisions.sort(key=lambda value:(-value['rows'],value['normalized']))
        results[mode]=dict(changed_rows=int(changed_rows),collision_groups=len(collisions),
            collision_rows=sum(value['rows'] for value in collisions),largest_collisions=collisions[:30])
    return dict(rows=sum(counts.values()),null_rows=counts.get(('null',''),0),distinct_nonnull=len(nonnull),
                python_types=dict(type_counts),
                flags=dict(flags),witnesses=dict(witnesses),normalizations=results)


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    frozen=json.loads((ROOT/'private_runs/submission_v2/protocol.json').read_text())
    paths=sorted(RAW.glob('training_2025-0[1-5]-01_*.parquet'))
    assert len(paths)==5
    assert not set(FIELDS).intersection({'TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt'})
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=dict(source_sha256=sha(__file__),test_sha256=sha(Path(__file__).with_name('test_v1.py')),
        packs=[p.name for p in paths],raw_hashes={p.name:frozen['raw_hashes'][p.name] for p in paths},
        fields=['PHASE_mvt',*FIELDS],selection='DEP rows in firstfive physicaltrainingpacks only; no outcome/IDpurge filtering, no logical-date claim, no tune/rankingpacks.',
        scope='Observed categoricalvalues only. No label,block,movementclock,model,prediction,trainingoracquisition.',
        resources='1CPU,Arrow8192batch,current/OShistoricalpeak<3GiB,hostreserve8GiB.',
        transformations='Diagnostic only: trim,ASCIIuppercase,NFC,NFKC,whitespaceremoval,numeric-equivalence hypotheses. NO edits/merges; numeric leadingzeros and missing-like literals unconfirmed.',
        location_scope='Stand/runwaycollision reports also calculated separatelywithin departureairport to avoid crossairport falseequivalence.')
    write(OUT/'protocol.json',protocol)
    started=time.monotonic()
    totals={field:Counter() for field in FIELDS}
    locations=defaultdict(Counter)
    packs=[]
    for path in paths:
        assert sha(path)==protocol['raw_hashes'][path.name]
        file=pq.ParquetFile(path)
        schemas={field:str(file.schema_arrow.field(field).type) for field in FIELDS}
        local={field:Counter() for field in FIELDS}
        dep_rows=0
        for batch in file.iter_batches(batch_size=8192,columns=protocol['fields'],use_threads=False):
            table=pa.Table.from_batches([batch])
            table=table.filter(pc.equal(table['PHASE_mvt'],'DEP'))
            dep_rows+=len(table)
            airport=table['ADEP_mvt'].to_pylist()
            for field in FIELDS:
                values=table[field].to_pylist()
                counts=Counter(typed(value) for value in values)
                totals[field].update(counts)
                local[field].update(counts)
                if field in ['STAND_mvt','RUNWAY_mvt']:
                    for apt,value in zip(airport,values):
                        locations[(field,str(apt))][typed(value)]+=1
            guard()
        entry=dict(path=path.name,sha256=protocol['raw_hashes'][path.name],dep_rows=dep_rows,schemas=schemas,
            fields={field:summarize(counts) for field,counts in local.items()})
        write(OUT/(path.stem+'_profile.json'),entry)
        packs.append(dict(path=path.name,dep_rows=dep_rows,schemas=schemas))
        print('PACK',path.name,'DEP',dep_rows,'peak',guard(),flush=True)
    complete=dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),
        packs=packs,total_departures=sum(p['dep_rows'] for p in packs),fields={field:summarize(counts) for field,counts in totals.items()},
        airport_locations=[dict(field=field,airport=apt,summary=summarize(counts)) for (field,apt),counts in sorted(locations.items())],
        elapsed_seconds=time.monotonic()-started,peak_bytes=guard(),no_labels_or_clocks_read=True,no_transform_applied=True)
    write(OUT/'summary.json',complete)
    write(OUT/'raw_category_counts.json',{field:[dict(kind=kind,raw=value,rows=count) for (kind,value),count in sorted(counts.items())] for field,counts in totals.items()})
    write(OUT/'receipt.json',dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},peak_bytes=guard(),no_model_fit=True))
    print('DONE',complete['total_departures'],'peak',guard(),'seconds',complete['elapsed_seconds'],flush=True)


if __name__=='__main__': main()
