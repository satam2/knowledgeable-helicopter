"""Bounded independent oracle for emitted public flight/date hypotheses."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import hashlib
import re
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'private_runs/tail240_20260916/source_distinctions/opdi_dates_v1'
OUT = ROOT/'private_runs/tail240_20260916/validation/opdi_dates_oracle_v1'
ID, TIME = 'MVT_ID_mvt','MVT_TIME_UTC_mvt'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def code_parts(value):
    if pd.isna(value):
        return None
    clean = re.sub(r'\s+','',str(value).upper())
    matched = re.fullmatch(r'([A-Z]+)([0-9]+)([A-Z]*)',clean)
    if matched is None:
        return None
    prefix, number, suffix = matched.groups()
    return prefix, number.lstrip('0') or '0', suffix


def query_code(value, mode, mapping):
    parts = code_parts(value)
    if parts is None:
        return None
    prefix, number, suffix = parts
    if mode == 'learned':
        prefix = mapping.get(prefix,prefix)
    elif mode == 'lexical' and len(prefix)>3 and len(prefix[3:])<=3 and prefix[:3].endswith(prefix[3:]):
        prefix = prefix[:3]
    return prefix+number+suffix


def guard():
    rss = psutil.Process().memory_info().rss
    assert rss < 4*1024**3 and psutil.virtual_memory().available > 8*1024**3
    return rss


def main():
    manifest = json.loads((BASE/'manifest.json').read_text())
    protocol = json.loads((BASE/'protocol.json').read_text())
    assert manifest['status']=='complete' and sha(BASE/'protocol.json')==manifest['protocol_sha256']
    for name,digest in manifest['outputs'].items():
        assert sha(BASE/name)==digest
    assert sha(ROOT/'review_work/tail240_20260916/source_distinctions/opdi_alias_dates.py')==manifest['source_sha256']
    assert sha(ROOT/'review_work/tail240_20260916/source_distinctions/audit_alias.py')==manifest['alias_source_sha256']
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    x=pd.read_parquet(BASE/'features.parquet').set_index(ID)
    q=pd.read_parquet(BASE/'queries.parquet').set_index(ID)
    w=pd.read_parquet(BASE/'witnesses.parquet')
    assert x.index.is_unique and q.index.is_unique and not w.duplicated([ID,'mode','shift']).any()
    np.testing.assert_array_equal(x.index,q.index)
    assert q.AOBT_3_flt.isna().sum()==22470
    assert len(q)==28470
    assert list(x)==manifest['features'] and len(x.columns)==45
    rawparts=[]
    fields=[ID,'PHASE_mvt',TIME,'FLIGHT_mvt','ADEP_mvt','ADES_mvt','AOBT_3_flt','SCHED_TIME_UTC_mvt']
    for path in sorted((ROOT/'data/09-15-2026-18-55-03_files_list').glob('training_*.parquet')):
        assert sha(path)==protocol['raw_hashes'][path.name]
        raw=pq.read_table(path,columns=fields,use_threads=False).to_pandas()
        rawparts.append(raw.loc[raw[ID].isin(q.index)].copy())
        del raw
        guard()
    raw=pd.concat(rawparts,ignore_index=True).set_index(ID).loc[q.index]
    assert raw.PHASE_mvt.eq('DEP').all()
    for column in (TIME,'FLIGHT_mvt','ADEP_mvt','AOBT_3_flt','SCHED_TIME_UTC_mvt'):
        pd.testing.assert_series_equal(raw[column],q[column],check_names=False,check_dtype=False)
    q['ADES_mvt']=raw.ADES_mvt
    assert q[TIME].dt.strftime('%Y%m').equals(q.month)
    sample_ids=[]
    for month,part in q.groupby('month',sort=False):
        assert len(part)==manifest['months'][month]['queries']
        for missing in (True,False):
            rows=part.loc[part.AOBT_3_flt.isna().eq(missing)]
            if len(rows):
                positions=np.unique(np.linspace(0,len(rows)-1,min(3,len(rows)),dtype=int))
                sample_ids.extend(rows.iloc[positions].index.tolist())
    sample=q.loc[sample_ids]
    wanted_codes=set()
    for _,row in sample.iterrows():
        for mode in ('raw','learned','lexical'):
            value=query_code(row.FLIGHT_mvt,mode,manifest['months'][row.month]['mapping'])
            if value is not None:
                wanted_codes.add(value)
    acquisition_path=ROOT/'private_runs/breakthrough_20260916/missing/opdi_rotation/acquisition.json'
    assert sha(acquisition_path)==protocol['acquisition_sha256']
    acquisition=json.loads(acquisition_path.read_text())
    public_samples=[]
    witness_rows=[]
    wanted_ids=set(w.public_id.tolist())
    fields=['id','icao24','flt_id','adep','ades','first_seen','last_seen']
    for source in acquisition['sources']:
        path=Path(source['local_path'])
        assert sha(path)==source['sha256']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=fields,use_threads=False):
            frame=batch.to_pandas()
            frame['public_month']=source['month']
            witness_rows.append(frame.loc[frame.id.isin(wanted_ids)])
            parts=frame.flt_id.astype('string').str.upper().str.replace(r'\s+','',regex=True).str.extract(r'^([A-Z]+)([0-9]+)([A-Z]*)$')
            frame['code']=parts[0]+parts[1].str.lstrip('0').replace('','0')+parts[2]
            keep=frame.code.isin(wanted_codes)
            if keep.any():
                public_samples.append(frame.loc[keep])
            guard()
        print('PUBLIC_READ',source['month'],flush=True)
    public=pd.concat(public_samples,ignore_index=True)
    witnessed=pd.concat(witness_rows,ignore_index=True).drop_duplicates()
    for frame in (public,witnessed):
        for column in ('first_seen','last_seen'):
            frame[column]=pd.to_datetime(frame[column],utc=True)
        for column in ('adep','ades','icao24'):
            frame[column]=frame[column].astype('string').fillna('').str.upper().str.replace(r'\s+','',regex=True)
    verified_witnesses=0
    witness_lookup={key:group for key,group in witnessed.groupby('id',sort=False)}
    assert int(x[[c for c in x if c.endswith('_offset_sec')]].notna().sum().sum())==len(w)
    for row in w.itertuples(index=False):
        entry=row._asdict()
        query=q.loc[entry[ID]]
        pub=witness_lookup[row.public_id]
        pub=pub.loc[pub.first_seen.eq(row.first_seen)&pub.last_seen.eq(row.last_seen)]
        assert len(pub)>=1
        pub=pub.iloc[0]
        assert pub.adep==str(query.ADEP_mvt).upper() and pub.ades==str(query.ADES_mvt).upper()
        assert ''.join(code_parts(pub.flt_id))==row.code
        assert query_code(query.FLIGHT_mvt,row.mode,manifest['months'][query.month]['mapping'])==row.code
        offset=(query[TIME]-pub.first_seen).total_seconds()
        assert offset==row.offset_sec and abs(offset-row.shift*86400)<=1800
        prefix=f'opdi_{row.mode}_day{row.shift+1}'
        assert x.loc[entry[ID],prefix+'_offset_sec']==np.float32(offset)
        assert x.loc[entry[ID],prefix+'_shift_residual_sec']==np.float32(offset-row.shift*86400)
        assert x.loc[entry[ID],prefix+'_duration_sec']==np.float32((pub.last_seen-pub.first_seen).total_seconds())
        assert x.loc[entry[ID],prefix+'_count_30min']==row.count
        verified_witnesses+=1
    hypotheses=0
    for movement_id,query in sample.iterrows():
        start=pd.Timestamp(query.month+'01',tz='UTC')
        months={(start-pd.Timedelta(days=1)).strftime('%Y%m'),query.month,(start+pd.offsets.MonthBegin(1)).strftime('%Y%m')}
        for mode in ('raw','learned','lexical'):
            code=query_code(query.FLIGHT_mvt,mode,manifest['months'][query.month]['mapping'])
            group=public.loc[public.public_month.isin(months)&public.code.eq(code)&public.adep.eq(str(query.ADEP_mvt).upper())&public.ades.eq(str(query.ADES_mvt).upper())]
            group=group.loc[group.first_seen.notna()&group.last_seen.gt(group.first_seen)&group.icao24.ne('')]
            group=group.drop_duplicates(['code','adep','ades','first_seen','last_seen','icao24']).sort_values('first_seen')
            for shift in (-1,0,1):
                prefix=f'opdi_{mode}_day{shift+1}'
                offset=(query[TIME]-group.first_seen).dt.total_seconds().to_numpy()
                distances=np.abs(offset-shift*86400)
                order=np.argsort(distances,kind='stable')
                expected_count=int((distances<=1800).sum())
                assert x.loc[movement_id,prefix+'_count_30min']==expected_count
                match=len(order)>0 and distances[order[0]]<=1800 and (len(order)<2 or distances[order[1]]>distances[order[0]])
                fields=['offset_sec','shift_residual_sec','duration_sec','candidate_margin_sec']
                if match:
                    hit=group.iloc[order[0]]
                    expected=[offset[order[0]],offset[order[0]]-shift*86400,(hit.last_seen-hit.first_seen).total_seconds(),distances[order[1]]-distances[order[0]] if len(order)>1 else np.nan]
                else:
                    expected=[np.nan]*4
                np.testing.assert_allclose(x.loc[movement_id,[prefix+'_'+f for f in fields]].to_numpy(),np.array(expected,dtype='float32'),rtol=0,atol=0,equal_nan=True)
                hypotheses+=1
    receipt={'status':'passed','source_sha256':sha(__file__),'producer_manifest_sha256':sha(BASE/'manifest.json'),
        'query_rows_raw_verified':len(q),'missing_queries':22470,'masked_known_queries':6000,'witness_rows_raw_verified':verified_witnesses,
        'brute_force_query_samples':len(sample),'brute_force_mode_date_hypotheses':hypotheses,
        'runtime_sec':time.monotonic()-started,'rss_bytes':guard(),
        'limits':'No taxi labels/private block clocks read. All witness arithmetic checked; candidate completeness and abstentions brute-forced for deterministic samples only. Mapping chronology source-reviewed; no independent prefix-support recount. Observedtrackstart is not offblock or proof of identity/date correction.'}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
