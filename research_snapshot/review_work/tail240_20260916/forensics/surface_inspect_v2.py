"""Public-only streaming event/flight key, coverage and chronology inspection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from collections import Counter
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from surface_acquire_v1 import OUT as ACQUIRE, NAME, guard
OUT = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot/inspection_v2'
AIRPORTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LTAI','LTFM','LSZH']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def nearest_airport(latitude, longitude, coordinates):
    lat, lon = np.radians(latitude), np.radians(longitude)
    best = np.full(len(lat), np.inf)
    name = np.full(len(lat), '', dtype=object)
    for airport, alat, alon in coordinates:
        a, b = np.radians(alat), np.radians(alon)
        distance = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(np.sin((lat-a)/2)**2 + np.cos(lat)*np.cos(a)*np.sin((lon-b)/2)**2, 0, 1)))
        closer = np.isfinite(distance) & (distance < best) & (distance <= 10000)
        best[closer], name[closer] = distance[closer], airport
    return name, best


def main():
    assert psutil.virtual_memory().available >= 10 * 1024**3
    OUT.mkdir(parents=True, exist_ok=False)
    acquired = common.read_json(ACQUIRE / 'manifest.json')
    event_path = ACQUIRE / NAME
    assert common.sha256(event_path) == acquired['sha256']
    acquisition = ROOT / 'private_runs/breakthrough_20260916/missing/opdi_rotation/acquisition.json'
    flight_receipt = next(row for row in common.read_json(acquisition)['sources'] if row['month'] == '202501')
    flight_path = Path(flight_receipt['local_path'])
    assert common.sha256(flight_path) == flight_receipt['sha256']
    airport_dir = ROOT / 'output/breakthrough_20260916/weather_wind/public'
    airports_path = airport_dir / 'airports.csv'
    assert common.sha256(airports_path) == common.read_json(airport_dir / 'receipt.json')['airports.csv']['sha256']
    coordinates = pd.read_csv(airports_path, usecols=['ident','latitude_deg','longitude_deg'])
    coordinates = coordinates.loc[coordinates.ident.isin(AIRPORTS)]
    assert set(coordinates.ident) == set(AIRPORTS) and coordinates.ident.is_unique
    protocol = dict(source_sha256=common.sha256(__file__), acquisition_manifest_sha256=common.sha256(ACQUIRE / 'manifest.json'),
        event_sha256=acquired['sha256'], flight_list_sha256=flight_receipt['sha256'], airports_sha256=common.sha256(airports_path),
        scope='Public-only fullonefile inspection; all event IDs and FK keys retain exactuint64 then decimalstrings, nofloatconversion.',
        coverage='Closest of11fixedairportcenters within10km Haversine; all eventcategories counted, source/versions retained. Geographiccounts are not challengeflightcoverage.',
        bounds='Collapse exact duplicate public projections only; conflicting IDs stop. Exact EVENT.flight_id to January FLIGHT.id. Compare event_time against public first_seen/last_seen, interpreting all source-naive timestamps consistently as UTC.',
        resources='1CPU,<2GiBOSpeak,currenthostreserve>=8GiB,start>=10GiBavailable;8192rowbatches.',
        private_inputs_read=False, raw_labels_read=False, model_fit=False)
    common.write_json(OUT / 'protocol.json', protocol)
    schema = pq.read_schema(event_path)
    assert pa.types.is_uint64(schema.field('id').type) and pa.types.is_uint64(schema.field('flight_id').type)
    public = pd.read_parquet(flight_path, columns=['id','icao24','flt_id','adep','ades','first_seen','last_seen','version'])
    assert public.id.dtype == np.dtype('uint64')
    public_rows_before = len(public)
    public = public.drop_duplicates()
    public_exact_duplicate_rows = public_rows_before - len(public)
    assert public.id.is_unique, 'Conflicting public flight IDs require abstention policy; stop before join'
    public['flight_id_exact'] = public.id.astype('string')
    public = public.drop(columns='id').rename(columns={'version':'flight_version'})
    for column in ['first_seen','last_seen']:
        public[column] = pd.to_datetime(public[column], utc=True)
    public = public.set_index('flight_id_exact', verify_integrity=True)
    public.loc[public.adep.isin(AIRPORTS)].reset_index().to_parquet(OUT / 'public_origin_flights.parquet', index=False)
    counters = {key:Counter() for key in ['type','source','version','flight_version','bounds_by_type','airport_type','airport_origin_type']}
    event_ids = []
    total, matched, before, after, retained, invalid_time = 0, 0, 0, 0, 0, 0
    writer = None
    exceptions = []
    for batch in pq.ParquetFile(event_path).iter_batches(batch_size=8192, use_threads=False):
        frame = batch.to_pandas()
        assert frame.id.dtype == np.dtype('uint64') and frame.flight_id.dtype == np.dtype('uint64')
        event_ids.append(frame.id.to_numpy(copy=True))
        frame['event_id_exact'] = frame.id.astype('string')
        frame['flight_id_exact'] = frame.flight_id.astype('string')
        frame = frame.drop(columns=['id','flight_id'])
        frame['event_time'] = pd.to_datetime(frame.event_time, utc=True)
        linked = public.reindex(frame.flight_id_exact)
        linked.index = frame.index
        joined = pd.concat([frame, linked], axis=1)
        has_flight = joined.first_seen.notna() & joined.last_seen.notna()
        early = has_flight & joined.event_time.lt(joined.first_seen)
        late = has_flight & joined.event_time.gt(joined.last_seen)
        total += len(joined)
        matched += int(has_flight.sum())
        before += int(early.sum())
        after += int(late.sum())
        invalid_time += int(joined.event_time.isna().sum())
        for key in ['type','source','version','flight_version']:
            counters[key].update({str(k):int(v) for k,v in joined[key].value_counts(dropna=False).items()})
        for event_type in joined.type.unique():
            mask = joined.type.eq(event_type)
            counters['bounds_by_type'][(str(event_type),'total')] += int(mask.sum())
            counters['bounds_by_type'][(str(event_type),'matched')] += int((mask & has_flight).sum())
            counters['bounds_by_type'][(str(event_type),'before_first_seen')] += int((mask & early).sum())
            counters['bounds_by_type'][(str(event_type),'after_last_seen')] += int((mask & late).sum())
        airport, distance = nearest_airport(joined.latitude.to_numpy(), joined.longitude.to_numpy(), coordinates.itertuples(index=False,name=None))
        joined['event_airport'], joined['distance_to_airport_m'] = airport, distance
        nearby = joined.loc[joined.event_airport.ne('')].copy()
        for (airport_name,event_type), count in nearby.groupby(['event_airport','type']).size().items():
            counters['airport_type'][(airport_name,event_type)] += int(count)
        for (airport_name,event_type), count in nearby.loc[nearby.event_airport.eq(nearby.adep)].groupby(['event_airport','type']).size().items():
            counters['airport_origin_type'][(airport_name,event_type)] += int(count)
        retained += len(nearby)
        for column in ['type','source','version','info','event_id_exact','flight_id_exact','icao24','flt_id','adep','ades','flight_version','event_airport']:
            nearby[column] = nearby[column].astype('string')
        table = pa.Table.from_pandas(nearby, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT / 'airport_events.parquet', table.schema, compression='zstd')
        writer.write_table(table)
        bad = joined.loc[~has_flight | early | late]
        if len(exceptions) < 100:
            for row in bad.head(100-len(exceptions)).itertuples(index=False):
                exceptions.append({key:str(getattr(row,key)) for key in ['event_id_exact','flight_id_exact','type','event_time','first_seen','last_seen','version','flight_version']})
        guard()
    if writer is not None:
        writer.close()
    all_ids = np.concatenate(event_ids)
    duplicate_event_ids = len(all_ids) - len(np.unique(all_ids))
    del all_ids, event_ids
    assert total == acquired['rows']
    records = {name:([dict(key=list(key),n=value) for key,value in sorted(counter.items())] if any(isinstance(key,tuple) for key in counter) else dict(counter)) for name,counter in counters.items()}
    common.write_json(OUT / 'manifest.json', dict(status='complete', source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT / 'protocol.json'),
        public_flight_rows_before=public_rows_before, public_exact_duplicate_projection_rows=public_exact_duplicate_rows,
        public_unique_ids=len(public), rows=total, matched_public_flight_rows=matched, unmatched_or_invalid_bound_rows=total-matched, invalid_event_times=invalid_time,
        event_before_first_seen=before,event_after_last_seen=after,duplicate_event_ids=duplicate_event_ids,
        retained_airport_events=retained, counts=records, exception_examples=exceptions,
        peak_bytes=guard(), private_inputs_read=False, raw_labels_read=False,
        outputs={name:common.sha256(OUT/name) for name in ['airport_events.parquet','public_origin_flights.parquet']}))
    print('SURFACE_INSPECT', total,'FKmatched',matched,'before',before,'after',after,'duplicates',duplicate_event_ids,'near_airports',retained,'peak',guard(), flush=True)
    print('PARKING_EXITS', [row for row in records['airport_type'] if row['key'][1]=='exit-parking_position'], flush=True)


if __name__ == '__main__':
    main()
