"""Bound source review and synthetic hidden-column invariance checks; no raw labels."""
from materialize_cohorts import ROOT,sha,read,guard
import importlib.util
import json
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch


def load(name,relative):
    path=ROOT/relative
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    protocolpath=ROOT/'private_runs/tail240_20260916/robustness/chronological_v1/protocol.json'
    protocol=read(protocolpath)
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest
    blocked={'MVT_ID_mvt','FLIGHT_ID_mvt','TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'}
    assert not blocked.intersection(protocol['feature_columns'])
    assert len(protocol['feature_columns'])==len(set(protocol['feature_columns']))==387
    sources={}
    for name,script,folder in [
        ('retrospective_peer','review_work/breakthrough_20260916/models_retrieval/retrospective_research/build.py','private_runs/breakthrough_20260916/retrospective_research'),
        ('batch_context','review_work/breakthrough_20260916/batch_context/build_batch_context.py','private_runs/breakthrough_20260916/batch_context')]:
        producer=read(ROOT/folder/'manifest.json')
        assert sha(ROOT/script)==producer['source_sha256']
        module=load('boundary_'+name,script)
        assert not {'TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'}.intersection(module.COLS)
        module.tests()
        sources[name]=dict(source_sha256=sha(ROOT/script),producer_manifest_sha256=sha(ROOT/folder/'manifest.json'),
            observed_projection=module.COLS,synthetic_hidden_departure_column_poison_tests_passed=True)
    sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/robustness'))
    import train
    now=pd.Timestamp('2025-04-01',tz='UTC')
    frame=pd.DataFrame({train.ID:[3.,1.],train.MOVEMENT:[now+pd.Timedelta(seconds=2),now]})
    all_labels=pd.DataFrame({train.ID:[1.,2.,3.],train.TARGET:[100.,200.,300.]})
    calls=[]
    def synthetic_read(path,columns=None,filters=None):
        assert path==train.META and columns==[train.ID,train.TARGET]
        assert filters==[(train.MOVEMENT,'>=',now),(train.MOVEMENT,'<=',now+pd.Timedelta(seconds=2))]
        calls.append(dict(columns=columns,filters=[(c,o,str(v)) for c,o,v in filters]))
        return all_labels.copy()
    with patch.object(train.pd,'read_parquet',side_effect=synthetic_read):
        np.testing.assert_array_equal(train.read_labels(frame),[300.,100.])
    assert len(calls)==1
    from taxiout.features.traffic import prior_counts
    query=pd.Series(pd.to_datetime(['2025-01-01T00:15:00Z']))
    events=pd.Series(pd.to_datetime(['2025-01-01T00:00:00Z','2025-01-01T00:14:59Z','2025-01-01T00:15:00Z','2025-01-01T00:15:01Z']))
    np.testing.assert_array_equal(prior_counts(events,query,15),[2])
    result=dict(status='passed',verifier_sha256=sha(__file__),training_protocol_sha256=sha(protocolpath),
        source_producer_bindings=sources,features_unique_387_and_direct_hidden_fields_excluded=True,
        supervised_label_read_projection_and_time_predicate_synthetic_check=calls,
        prior_named_features_are_observation_counts=['dep_prior_15m','dep_prior_60m','arr_prior_15m','arr_prior_60m'],
        traffic_source_sha256=sha(ROOT/'knowledgeable-helicopter-screening/src/taxiout/features/traffic.py'),
        arrival_boundary='Retrospective peer and sequence source readers project completion BLOCK only with PHASE=ARR predicate; peer builder rejects non-ARR completion inputs. No departure target/block predictor.',
        supplied_batch_limitation='Retrospective future/same-day clock peers and completed arrivals remain available inputs. This verifies supervised label-stage chronology, not real-time event availability. Cached feature discovery scans year files but projects declared observed predictors only; vocabulary is restricted to train/refit prefix.',
        exposure_limitation='Chronological new model fits cannot erase prior outcome-driven architecture and feature selection on2025. All evaluation months remain exposed development.',
        actual_label_reads=False,model_fits=False,gpu_used=False,peak_bytes=guard())
    out=ROOT/'private_runs/tail240_20260916/robustness/validation/source_boundary_review.json'
    assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
