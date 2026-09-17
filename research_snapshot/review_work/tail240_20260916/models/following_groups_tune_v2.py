"""F3-only continuation with unchanged models and bounded schema discovery."""
import following_groups_tune as frozen
import schema_discovery
from pathlib import Path
import argparse

OLD=frozen.OUT
frozen.OUT=frozen.common.external_path(frozen.ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v2')
ORIGINAL=frozen.ORIGINAL_SOURCES
DISCOVERY=[]


def cached_sources(columns):
    result,receipt=schema_discovery.cached_discovery(ORIGINAL,columns)
    DISCOVERY.append(receipt)
    frozen.common.write_json(frozen.OUT/'schema_discovery_receipts.json',DISCOVERY)
    return result


def declaration():
    common=frozen.common
    old=common.read_json(OLD/'protocol.json')
    assert old['source_sha256']==common.sha256(frozen.__file__)
    record=dict(old)
    record['runtime_fix']=dict(wrapper_sha256=common.sha256(__file__),helper_sha256=common.sha256(schema_discovery.__file__),
        original_protocol_sha256=common.sha256(OLD/'protocol.json'),
        scope='F3only. Synchronous schema metadata memoization preserves exact column selection/order/hashes, all rows/modelparams/stopping. No changes to F1 or completed sources.',
        independent_evidence='Validation neural v4 reduced source-discovery peak from16.06GB to1.03GB by same once-per-path cache; original discovery re-read every schema once per desired column.')
    record['sources']=dict(old['sources'])
    for path in [__file__,frozen.__file__,schema_discovery.__file__]:
        record['sources'][str(Path(path).relative_to(frozen.ROOT))]=common.sha256(path)
    frozen.OUT.mkdir(parents=True,exist_ok=True)
    path=frozen.OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


frozen.ORIGINAL_SOURCES=cached_sources
frozen.declare=declaration

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        frozen.run('F3')
