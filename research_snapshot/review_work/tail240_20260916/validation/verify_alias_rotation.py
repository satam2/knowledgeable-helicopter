"""Full small-pilot oracle for strict alias aircraft previous-leg features."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import validate_candidate as validation
from verify_opdi_dates_v3 import query_code

ROOT=validation.ROOT
BASE=ROOT/'private_runs/tail240_20260916/forensics/alias_rotation/pilot_v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/alias_rotation_v1'
PUBLIC=ROOT/'private_runs/breakthrough_20260916/missing/opdi_rotation'
ID,TIME='MVT_ID_mvt','MVT_TIME_UTC_mvt'
read,sha,write=validation.read_json,validation.sha256,validation.write_json
SUFFIXES=['match','match_ambiguous','match_offset_sec','previous_leg_available','previous_same_airport','ground_interval_sec','previous_leg_duration_sec','previous_leg_age_at_takeoff_sec']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def normalized(value):
    return '' if pd.isna(value) else re.sub(r'\s+','',str(value).upper())


def main():
    manifest=read(BASE/'manifest.json')
    protocol=read(BASE/'protocol.json')['declaration']
    assert manifest['status']=='complete' and manifest['protocol_sha256']==sha(BASE/'protocol.json')
    assert manifest['source_sha256']==sha(ROOT/'review_work/tail240_20260916/forensics/alias_rotation.py')
    for name,digest in manifest['outputs'].items():
        assert sha(BASE/name)==digest
    for name,digest in protocol['dependency_sources'].items():
        assert sha(ROOT/name)==digest
    mapping_path=ROOT/'private_runs/tail240_20260916/source_distinctions/opdi_dates_v1/manifest.json'
    assert sha(mapping_path)==protocol['mapping_manifest_sha256']
    mappings=read(mapping_path)['months']
    assert sha(PUBLIC/'acquisition.json')==protocol['acquisition_sha256']
    sources={r['month']:r for r in read(PUBLIC/'acquisition.json')['sources']}
    queries=pd.read_parquet(BASE/'queries.parquet').set_index(ID)
    features=pd.read_parquet(BASE/'features.parquet').set_index(ID)
    witness=pd.read_parquet(BASE/'witnesses.parquet').set_index([ID,'mode'])
    assert queries.index.is_unique and features.index.is_unique and witness.index.is_unique
    assert set(queries.index)==set(features.index) and len(queries)==manifest['queries']==1488
    assert queries.PHASE_mvt.eq('DEP').all() and queries.ADEP_mvt.eq('LIRF').all() and queries.AOBT_3_flt.isna().all()
    oldmarker=read(PUBLIC/'manifest.json')
    assert sha(PUBLIC/'manifest.json')==protocol['old_manifest_sha256']
    assert sha(PUBLIC/'training_features.parquet')==oldmarker['outputs']['training_features.parquet']
    old=pd.read_parquet(PUBLIC/'training_features.parquet').set_index(ID)
    np.testing.assert_array_equal(features[[f'alias_raw_{s}' for s in SUFFIXES]],old.loc[features.index,[f'opdi_{s}' for s in SUFFIXES]])
    reconstructed=pd.DataFrame(np.nan,index=features.index,columns=features.columns,dtype='float32')
    for mode in ('raw','learned','lexical'):
        for suffix in ('match','match_ambiguous','previous_leg_available','previous_same_airport'):
            reconstructed[f'alias_{mode}_{suffix}']=np.float32(0)
    verified=0
    for month,rows in queries.groupby('month'):
        source=sources[month]
        path=Path(source['local_path'])
        assert sha(path)==source['sha256']
        public=pd.read_parquet(path,columns=['id','icao24','flt_id','adep','ades','first_seen','last_seen'])
        public['id_exact']=public.id.map(lambda value:str(int(value)))
        for column in ('first_seen','last_seen'):
            public[column]=pd.to_datetime(public[column],utc=True)
        for column in ('icao24','flt_id','adep','ades'):
            public[column]=public[column].map(normalized)
        public=public.loc[public.first_seen.dt.strftime('%Y%m').eq(month)&public.last_seen.gt(public.first_seen)&public.icao24.ne('')]
        wanted={query_code(value,mode,mappings[month]['mapping']) for value in rows.FLIGHT_mvt for mode in ('learned','lexical')}
        wanted.update(rows.FLIGHT_mvt.map(normalized))
        groups={key:group for key,group in public.loc[public.flt_id.isin(wanted)].groupby(['flt_id','adep','ades'])}
        tails={key:group for key,group in public.groupby('icao24')}
        for movement_id,query in rows.iterrows():
            for mode in ('raw','learned','lexical'):
                code=normalized(query.FLIGHT_mvt) if mode=='raw' else query_code(query.FLIGHT_mvt,mode,mappings[month]['mapping'])
                prefix=f'alias_{mode}_'
                group=groups.get((code,normalized(query.ADEP_mvt),normalized(query.ADES_mvt)))
                if group is None:
                    assert (movement_id,mode) not in witness.index
                    continue
                hits=group.loc[(query[TIME]-group.first_seen).dt.total_seconds().abs().le(600)]
                if len(hits)>1:
                    reconstructed.loc[movement_id,prefix+'match_ambiguous']=1
                if len(hits)!=1:
                    assert (movement_id,mode) not in witness.index
                    continue
                hit=hits.iloc[0]
                trace=witness.loc[(movement_id,mode)]
                assert trace.public_id==hit.id_exact and trace.icao24==hit.icao24
                assert trace.public_first_seen==hit.first_seen and trace.public_last_seen==hit.last_seen
                reconstructed.loc[movement_id,prefix+'match']=1
                reconstructed.loc[movement_id,prefix+'match_offset_sec']=(query[TIME]-hit.first_seen).total_seconds()
                before=tails[hit.icao24].loc[lambda frame:frame.first_seen.lt(hit.first_seen)]
                if before.last_seen.ge(hit.first_seen).any():
                    assert trace.rejection=='overlapping_earlier_leg'
                else:
                    eligible=before.loc[before.last_seen.lt(hit.first_seen)&before.last_seen.lt(query[TIME])]
                    latest=eligible.loc[eligible.last_seen.eq(eligible.last_seen.max())]
                    if len(latest)!=1:
                        assert trace.rejection=='no_unique_previous_leg'
                    else:
                        previous=latest.iloc[0]
                        assert trace.previous_public_id==previous.id_exact and trace.previous_last_seen==previous.last_seen
                        reconstructed.loc[movement_id,prefix+'previous_leg_available']=1
                        if previous.ades!=normalized(query.ADEP_mvt):
                            assert trace.rejection=='previous_destination_mismatch'
                        else:
                            assert pd.isna(trace.rejection)
                            reconstructed.loc[movement_id,prefix+'previous_same_airport']=1
                            reconstructed.loc[movement_id,prefix+'ground_interval_sec']=(hit.first_seen-previous.last_seen).total_seconds()
                            reconstructed.loc[movement_id,prefix+'previous_leg_duration_sec']=(previous.last_seen-previous.first_seen).total_seconds()
                            reconstructed.loc[movement_id,prefix+'previous_leg_age_at_takeoff_sec']=(query[TIME]-previous.last_seen).total_seconds()
                verified+=1
        validation.guard()
        print('MONTH_VERIFIED',month,len(rows),flush=True)
    np.testing.assert_array_equal(reconstructed,features)
    assert verified==len(witness)
    coverage={'raw_matches':int(features.alias_raw_match.sum()),'raw_rotations':int(features.alias_raw_previous_same_airport.sum())}
    for mode in ('learned','lexical'):
        coverage[mode+'_new_matches']=int((features[f'alias_{mode}_match'].eq(1)&features.alias_raw_match.eq(0)).sum())
        coverage[mode+'_new_rotations']=int((features[f'alias_{mode}_previous_same_airport'].eq(1)&features.alias_raw_previous_same_airport.eq(0)).sum())
        assert coverage[mode+'_new_matches']==manifest['gate_additional_matches'][mode]
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'receipt.json',{'status':'passed','source_sha256':sha(__file__),'producer_manifest_sha256':sha(BASE/'manifest.json'),
        'pilot_queries':len(queries),'all_mode_queries_checked':len(queries)*3,'all_numeric_columns_checked':len(features.columns),
        'raw_control_exact':True,'exact_public_witnesses_verified':verified,'coverage':coverage,'peak_rss_bytes':validation.guard(),
        'limits':'Outcome-free fullsmallpilot publicquery/pastleg matching replay, rawcontrol exact. Queryrawrowwhitelist source-reviewed and priormappingcache verified; missingquery selection not reread independently here. Addedcoverage alone does not establish correctNMidentity or taxi prediction benefit.'})
    print(coverage,flush=True)


if __name__=='__main__':
    main()
