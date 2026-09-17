"""Schema-only audit of documented provenance and existing feature mappings."""
from pathlib import Path
import re
import sys
import hashlib
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
OUT = ROOT / 'private_runs/tail240_20260916/forensics/source_provenance_v1'
DOC = ROOT / 'private_runs/aviation_spike/sources/challenge_data.txt'
UNION = ROOT / 'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916/manifest.json'
MAPPING = {
    'MVT_ID_mvt': ('Movement record identifier; not a source-system identifier', [], 'alignment and stable ordering; not raw numeric predictor'),
    'FLIGHT_ID_mvt': ('NM flight identifier if matched', ['source_nm_record_missing'], 'fold purges/linkage; missing-record flag uses AOBT missingness, not exact ID'),
    'FLIGHT_mvt': ('Airport-reported flight number', ['source_callsign_flight_exact_equal'], 'comparison flag; exact string not native union categorical'),
    'FLIGHT_RULE_mvt': ('Airport flight rules I/V/unknown', ['source_airport_rule_ifr', 'source_airport_rule_vfr', 'source_airport_rule_missing'], 'represented categories'),
    'ADEP_mvt': ('Airport departure aerodrome', ['ADEP_mvt', 'source_adep_agreement', 'source_adep_disagreement'], 'direct and cross-source consistency'),
    'ADES_mvt': ('Airport destination aerodrome', ['ADES_mvt', 'source_ades_agreement', 'source_ades_disagreement'], 'direct and cross-source consistency'),
    'PHASE_mvt': ('Arrival or departure movement phase', [], 'cohort/context selection; constant DEP on prediction rows'),
    'MVT_TIME_UTC_mvt': ('Best available takeoff or landing time by phase', ['utc_hour', 'minute', 'weekday', 'month'], 'time reference for clock offsets and context; no per-row best-source flag supplied'),
    'BLOCK_TIME_UTC_mvt': ('Offblock or inblock by phase', [], 'hidden for ranking DEP; excluded from DEP predictors'),
    'SCHED_TIME_UTC_mvt': ('Scheduled departure or arrival time by phase', ['takeoff_minus_SCHED_TIME_UTC_mvt', 'conv_sched_second', 'conv_sched_calendar_day_delta'], 'clock offsets and formatting/date consistency'),
    'AIRCRAFT_TYPE_mvt': ('Airport aircraft type', ['AIRCRAFT_TYPE_mvt', 'source_aircraft_type_agreement', 'source_aircraft_type_disagreement'], 'direct and source consistency'),
    'RUNWAY_mvt': ('Departure or arrival runway by phase', ['RUNWAY_mvt', 'airport_runway'], 'direct and grouped context'),
    'STAND_mvt': ('Departure or arrival stand by phase', ['STAND_mvt', 'airport_stand'], 'direct and grouped context'),
    'TAXITIME_SEC_mvt': ('Taxi-out or taxi-in seconds by phase', [], 'DEP target; hidden for ranking DEP and excluded from DEP predictors'),
    'LOBT_flt': ('Last known off-block time', ['takeoff_minus_LOBT_flt', 'last_minus_initial', 'conv_last_calendar_day_delta'], 'clock snapshot, not message revision status'),
    'CALLSIGN_flt': ('NM flight callsign', ['source_callsign_flight_exact_equal'], 'comparison flag; exact string not native union categorical'),
    'ADEP_flt': ('NM departure aerodrome', ['source_adep_agreement', 'source_adep_disagreement'], 'source consistency; separate NM airport raw category not native union input'),
    'ADES_flt': ('NM destination aerodrome', ['source_ades_agreement', 'source_ades_disagreement', 'source_filed_destination_changed'], 'source consistency and diversion'),
    'ADES_FILED_flt': ('Initially filed destination; differing final destination indicates diversion', ['source_filed_destination_changed'], 'diversion flag; not update timestamp or message revision code'),
    'MARKET_SEGMENT_flt': ('NM market segment', ['MARKET_SEGMENT_flt'], 'direct category; not data-source provenance'),
    'IOBT_flt': ('Initial off-block time', ['takeoff_minus_IOBT_flt', 'last_minus_initial', 'conv_init_calendar_day_delta'], 'clock snapshot, not a sequence of updates'),
    'FLIGHT_RULE_flt': ('NM flight rules I/V/Y/Z', ['source_nm_rule_missing'], 'only missingness represented here; raw rule distinctions omitted from union, but are operating rules, not source-event provenance'),
    'FLIGHT_TYPE_flt': ('NM flight type S/N/G/M/X', ['FLIGHT_TYPE_flt'], 'direct flight-type category, not event type'),
    'AIRCRAFT_TYPE_flt': ('NM aircraft type', ['source_aircraft_type_agreement', 'source_aircraft_type_disagreement'], 'source consistency; separate NM type raw category not native union input'),
    'WK_TBL_CAT_flt': ('Wake turbulence category', ['WK_TBL_CAT_flt'], 'direct category'),
    'AIRCRAFT_OPERATOR_flt': ('Anonymized aircraft operator', ['AIRCRAFT_OPERATOR_flt'], 'direct category; Arrow R extension metadata describes storage, not an observed update flag'),
    'EOBT_1_flt': ('Estimated offblock time for FPL-based M1 trajectory', ['takeoff_minus_EOBT_1_flt', 'nm_actual_minus_estimated', 'conv_est_calendar_day_delta'], 'M1 semantic lineage fixed by field definition'),
    'ARVT_1_flt': ('Arrival time for FPL-based M1 trajectory', ['retro_own_arvt3_minus_arvt1_sec', 'retro_own_planned_duration_sec'], 'M1 arrival context; no source publication time supplied'),
    'AOBT_3_flt': ('Actual offblock time for flown M3 trajectory', ['takeoff_minus_AOBT_3_flt', 'nm_actual_minus_estimated', 'conv_nm_calendar_day_delta'], 'M3 semantic lineage fixed by field definition'),
    'ARVT_3_flt': ('Arrival time for flown M3 trajectory', ['retro_own_arvt3_minus_takeoff_sec', 'retro_own_arvt3_minus_arvt1_sec', 'retro_own_flown_duration_sec'], 'M3 arrival context; no source publication time supplied'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    dictionary = set(re.findall(r'\b[A-Z][A-Z0-9_]*_(?:mvt|flt)\b', DOC.read_text(encoding='utf-8')))
    raw_hashes = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    columns = common.read_json(UNION)['feature_columns']
    files = []
    for path in [*sorted(common.RAW.glob('training_*.parquet')), common.RAW / 'ranking.parquet']:
        assert common.sha256(path) == raw_hashes[path.name]
        source = pq.ParquetFile(path)
        schema = source.schema_arrow
        assert len(schema) == 30 and set(schema.names) == set(MAPPING) == dictionary
        fields = []
        for field in schema:
            meta = field.metadata or {}
            fields.append(dict(name=field.name, type=str(field.type), metadata={
                key.decode(): dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for key, value in meta.items()}))
        files.append(dict(filename=path.name, sha256=raw_hashes[path.name], rows=source.metadata.num_rows,
            columns=len(schema), fields=fields, schema_metadata_keys=[key.decode() for key in (schema.metadata or {})],
            file_metadata_keys=[key.decode() for key in (source.metadata.metadata or {})], created_by=source.metadata.created_by))
    mapping = []
    for name, (meaning, features, handling) in MAPPING.items():
        assert all(feature in columns for feature in features)
        mapping.append(dict(field=name, dictionary_meaning=meaning, union_features=features, handling=handling))
    docs = [DOC, ROOT / 'private_runs/aviation_spike/sources/challenge_data.html',
        ROOT / 'output/breakthrough_20260916/research_gap/primary/sources/eurocontrol_dpi.txt',
        ROOT / 'output/breakthrough_20260916/research_gap/primary/sources/eurocontrol_structure.txt']
    report = dict(status='complete_negative_for_explicit_omitted_provenance', source_sha256=common.sha256(__file__),
        files=files, official_dictionary_field_count=len(dictionary), exact_dictionary_schema_match=True,
        source_documents={str(path.relative_to(ROOT)):common.sha256(path) for path in docs},
        union_manifest_sha256=common.sha256(UNION), union_columns=len(columns), field_mapping=mapping,
        unavailable=['source feed/system identifier', 'best-available movement source flag', 'event message type',
            'update/revision status or sequence', 'message emission/reception time', 'operational flight-date assignment provenance',
            'actual clock precision/source-quality flag'],
        distinction='Field-level M1/M3 and movement-vs-flight provenance is documented and represented. Record-level update/source/date provenance is not supplied. Not every nominal raw category is fully represented in union387, but no omitted category is documented as the requested provenance.',
        raw_value_columns_loaded=[], private_target_values_read=False, private_dep_block_values_read=False,
        raw_metadata_only=True, external_requests=False, model_fit=False)
    common.write_json(OUT / 'receipt.json', report)
    print('PROVENANCE_NEGATIVE', len(files), 'files', len(dictionary), 'dictionary/schema fields, all mapped; no explicit omitted provenance', flush=True)


if __name__ == '__main__':
    main()
