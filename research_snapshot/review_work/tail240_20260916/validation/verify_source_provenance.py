"""Independent supplied-schema and storage-metadata reconstruction."""
import hashlib
import re
import pyarrow.parquet as pq
import validate_candidate as v


def main():
    folder=v.ROOT/'private_runs/tail240_20260916/forensics/source_provenance_v1'
    receipt=v.read_json(folder/'receipt.json')
    out=v.ROOT/'private_runs/tail240_20260916/validation/source_provenance_v1'
    out.mkdir(parents=True,exist_ok=False)
    assert receipt['source_sha256']==v.sha256(v.ROOT/'review_work/tail240_20260916/forensics/source_provenance_v1.py')
    for path,digest in receipt['source_documents'].items():assert v.sha256(v.ROOT/path)==digest
    doc=v.ROOT/'private_runs/aviation_spike/sources/challenge_data.txt'
    names=set(re.findall(r'\b[A-Z][A-Z0-9_]*_(?:mvt|flt)\b',doc.read_text(encoding='utf-8')))
    assert len(names)==30 and {m['field'] for m in receipt['field_mapping']}==names
    union=v.ROOT/'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916/manifest.json'
    assert v.sha256(union)==receipt['union_manifest_sha256']
    features=v.read_json(union)['feature_columns']
    for mapping in receipt['field_mapping']:assert set(mapping['union_features']).issubset(features)
    frozen=v.read_json(v.ROOT/'private_runs/submission_v2/protocol.json')['raw_hashes']
    for reported in receipt['files']:
        path=v.common.RAW/reported['filename']
        assert v.sha256(path)==reported['sha256']==frozen[path.name]
        parquet=pq.ParquetFile(path)
        schema=parquet.schema_arrow
        assert len(schema)==30 and set(schema.names)==names
        fields=[]
        for field in schema:
            fields.append(dict(name=field.name,type=str(field.type),metadata={key.decode():dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for key,value in (field.metadata or {}).items()}))
        actual=dict(filename=path.name,sha256=frozen[path.name],rows=parquet.metadata.num_rows,columns=len(schema),fields=fields,schema_metadata_keys=[k.decode() for k in (schema.metadata or {})],file_metadata_keys=[k.decode() for k in (parquet.metadata.metadata or {})],created_by=parquet.metadata.created_by)
        assert actual==reported
    result=dict(status='passed',source_sha256=v.sha256(__file__),producer_receipt_sha256=v.sha256(folder/'receipt.json'),files=len(receipt['files']),dictionary_fields=len(names),all_schema_fields_types_metadata_exact=True,all_listed_union_features_exist=True,raw_values_loaded=False,private_targets_read=False,private_dep_block_read=False,limitation='Confirms supplied dictionary field set and exact schema/metadata; does not prove all useful feature encodings exhausted or authenticate hidden source-system semantics.')
    v.write_json(out/'receipt.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
